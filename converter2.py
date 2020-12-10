import random
import time

import click
from mido import MidiFile, MidiTrack, second2tick
import torch


MIDI_CHUNK = 128

NO_NOTE = -1
SOS = -2
EOS = -3

# by eliminating chords completely, we can drastically simplify the
# input/output language

lang_forward = {(x, ): i for i, x in enumerate(
    [NO_NOTE, SOS, EOS] + list(range(128))
)}
lang_backward = {i: (x, ) for i, x in enumerate(
    [NO_NOTE, SOS, EOS] + list(range(128))
)}


@click.group()
def main():
    pass


def to_vector(mid):
    """
    Produce vector format from midi, removing percussion
    """
    new_mid = MidiFile()
    for track in mid.tracks:
        new_track = MidiTrack()
        for msg in track:
            if msg.type == 'note_on' and msg.channel != 9:
                new_track.append(msg)
        new_mid.tracks.append(new_track[:])

    v = [[]]
    j = 0
    for msg in new_mid:
        if msg.type != 'note_on':
            continue

        dt = int(second2tick(msg.time, new_mid.ticks_per_beat, 500000))
        for __ in range(dt):
            v.append([])
            j += 1
        v[j].append(msg.note)

    result = []
    for x in v:
        if len(x) == 0:
            result.append((NO_NOTE, ))
        elif len(x) > 3:
            result.append(tuple(random.sample(x, 3)))
        else:
            result.append(tuple(x))
    # return [tuple(x) if len(x) > 0 else (-1,) for x in v]
    if len(result) > MIDI_CHUNK:
        x = random.randint(0, len(result) - MIDI_CHUNK)
        return result[x:(x + MIDI_CHUNK)]
    else:
        return result


def find_beat(v):
    """
    find the beat of a vector format midi using the autocorrelation function
    """
    # binary vector for testing autocorrelation
    v2 = [0 if x[0] == -1 else 1 for x in v]
    result = []
    # no need to check more than 24*4 = 96
    # i.e. 4 quarter notes of standard midi
    for lag in range(96):
        s = 0
        for i in range(len(v2)):
            if v2[i] > 0 and v2[(i + lag) % len(v2)] > 0:
                s += 1
        result.append((lag, s))
    k = 1
    srt = sorted(result, key=lambda x: x[1])
    while srt[-k][0] == 0:
        k += 1
    return srt[-k][0]


def simplify(v):
    """
    Produce simplifyed melody from full format
    """
    # simplify melody
    # i.e. remove chords by selecting a random note
    s = []
    for x in v:
        if len(x) > 1:
            s.append((random.choice(x), ))
        else:
            s.append(x)

    # simplify rhythm
    # i.e. keep only the first note in each beat length segment
    beat = find_beat(s)
    # print("major beat is", beat)
    for i, note in enumerate(s):
        if i % beat == 0:
            found = False
        if note[0] != NO_NOTE:
            if not found:
                found = True
            else:
                s[i] = (NO_NOTE, )

    return s


def get_satb(v):
    """
    Take a vector format file and return four separate melodies
    """
    s = []
    a = []
    t = []
    b = []
    for x in v:
        if len(x) > 1:
            c = sorted(list(x))
            s.append((c[-1], ))
            a.append((random.choice(x), ))
            t.append((random.choice(x), ))
            b.append((c[0], ))
        else:
            s.append(x)
            a.append(x)
            t.append(x)
            b.append(x)
    return s, a, t, b


def transposed(v, k):
    t = []
    for msg in v:
        z = []
        for x in msg:
            if x < 0:
                z.append(x)
            else:
                tn = x + k
                if tn >= 0 and tn < 128:
                    z.append(tn)
        if len(z) == 0:
            z.append(NO_NOTE)
        t.append(tuple(z))
    return t


def get_data(infiles, device):
    """
    Dynamically produce a pair of simple/full tensors for training
    """
    infiles = list(infiles)
    random.shuffle(infiles)
    while True:
        for infile in infiles:
            print(infile)
            mid = MidiFile(infile)
            v_full = to_vector(mid)
            v_simple = simplify(v_full)

            t = random.randint(-12, 12)
            t_simple = transposed(v_simple, t)
            t_full = transposed(v_full, t)
            s, a, t, b = get_satb(t_full)

            src_tensor = torch.tensor(
                [lang_forward[x] for x in t_simple],
                dtype=torch.long, device=device
            ).view(-1, 1)
            trg_tensor_s = torch.tensor(
                [lang_forward[x] for x in s],
                dtype=torch.long, device=device
            ).view(-1, 1)
            trg_tensor_a = torch.tensor(
                [lang_forward[x] for x in a],
                dtype=torch.long, device=device
            ).view(-1, 1)
            trg_tensor_t = torch.tensor(
                [lang_forward[x] for x in t],
                dtype=torch.long, device=device
            ).view(-1, 1)
            trg_tensor_b = torch.tensor(
                [lang_forward[x] for x in b],
                dtype=torch.long, device=device
            ).view(-1, 1)

            if tuple(src_tensor.shape) != (128, 1):
                continue

            print(
                src_tensor.shape,
                trg_tensor_s.shape,
                trg_tensor_a.shape,
                trg_tensor_t.shape,
                trg_tensor_b.shape,
            )

            yield [{
                'src': src_tensor,
                'trg_s': trg_tensor_s,
                'trg_a': trg_tensor_a,
                'trg_t': trg_tensor_t,
                'trg_b': trg_tensor_b
            }]


class Encoder(torch.nn.Module):
    def __init__(self, inp_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = torch.nn.Embedding(inp_dim, emb_dim)
        self.rnn = torch.nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder(torch.nn.Module):
    def __init__(self, out_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = torch.nn.Embedding(out_dim, emb_dim)
        self.rnn = torch.nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = torch.nn.Linear(hid_dim, out_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, inp, hidden, cell):
        inp = inp.unsqueeze(0)
        embedded = self.dropout(self.embedding(inp))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell


class Seq2Seq(torch.nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim
        assert encoder.n_layers == decoder.n_layers

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.out_dim

        outputs = torch.zeros(
            trg_len, batch_size, trg_vocab_size
        ).to(self.device)

        hidden, cell = self.encoder(src)
        inp = trg[0, :]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(inp, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            inp = trg[t] if teacher_force else top1

        return outputs


INPUT_DIM = len(lang_forward)
OUTPUT_DIM = len(lang_forward)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5


def init_weights(m):
    for name, param in m.named_parameters():
        torch.nn.init.uniform_(param.data, -0.08, 0.08)


def _train(models, data, optimizers, criterion, clip):
    epoch_loss = 0
    for i, (model, optimizer) in enumerate(zip(models, optimizers)):
        model.train()
        for i, d in enumerate(data):
            src = d['src']
            trg = d['trg_' + ['s', 'a', 't', 'b'][i]]

            optimizer.zero_grad()
            output = model(src, trg)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            epoch_loss += loss.item()

    return epoch_loss/len(data)


def _evaluate(models, data, criterion):
    epoch_loss = 0
    for i, model in enumerate(models):
        model.eval()
        with torch.no_grad():

            for i, d in enumerate(data):
                src = d['src']
                trg = d['trg_' + ['s', 'a', 't', 'b'][i]]

                output = model(src, trg, 0)  # no forcing
                output_dim = output.shape[-1]
                output = output[1:].view(-1, output_dim)
                trg = trg[1:].view(-1)

                loss = criterion(output, trg)

                epoch_loss += loss.item()

    return epoch_loss/len(data)


@main.command()
@click.argument('infiles', nargs=-1, type=click.Path())
def train(infiles):
    device = torch.device('cpu')

    encs = []
    decs = []
    models = []

    for i in range(4):
        encs.append(
            Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
        )
        decs.append(
            Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
        )
        models.append(Seq2Seq(encs[-1], decs[-1], device).to(device))

    for model in models:
        model.apply(init_weights)
        print(model)

    optimizers = [torch.optim.Adam(model.parameters()) for model in models]
    criterion = torch.nn.CrossEntropyLoss()

    N_EPOCHS = 3000
    CLIP = 1

    best_valid_loss = float('inf')
    print('epoch\ttime\ttrain_loss\tvalid_loss')

    epoch = 0
    start_time = time.time()
    for train_data, eval_data in zip(
            get_data(infiles, device),
            get_data(infiles, device)
    ):
        train_loss = _train(models, train_data, optimizers, criterion, CLIP)
        valid_loss = _evaluate(models, eval_data, criterion)
        end_time = time.time()

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            for i, s in enumerate(['s', 'a', 't', 'b']):
                torch.save(models[i].state_dict(), 'state_dict_' + s + '.pt')
                torch.save(models[i], 'model_' + s + '.pt')

        print(epoch, end_time - start_time, train_loss, valid_loss, sep='\t')
        epoch += 1
        if epoch > N_EPOCHS:
            break

    # load best model and test
    # model.load_state_dict(torch.load('minimal.pt'))
    # test_loss = _evaluate(model, data, criterion)
    # print('test_loss', test_loss)


def _generate(models, data):
    result = []
    for i, model in enumerate(models):
        model.eval()
        with torch.no_grad():

            output = model(data, data, 0)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            result.append(list(output))

    return result


def to_midi(vs):
    """
    Produce midi format from vectors
    """
    mid = MidiFile()
    for v in vs:
        track = MidiTrack()
        for msg in v:
            topv, topi = msg.topk(1)
            print(topv, topi)

        mid.tracks.append(track[:])
    return mid


@main.command()
@click.argument('infile', type=click.Path())
def test(infile):
    device = torch.device('cpu')
    models = []
    for i, s in enumerate(['s', 'a', 't', 'b']):
        models.append(torch.load('model_' + s + '.pt'))

    mid = MidiFile(infile)
    v_full = to_vector(mid)
    v_simple = simplify(v_full)
    src_tensor = torch.tensor(
        [lang_forward[x] for x in v_simple],
        dtype=torch.long, device=device
    ).view(-1, 1)

    full = _generate(models, src_tensor)
    res = to_midi(full)


if __name__ == '__main__':
    main()
