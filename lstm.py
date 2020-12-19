import click
from mido import MidiFile, MidiTrack, second2tick, Message
import numpy as np
import torch
import random
import time


MIDI_CHUNK = 4096


@click.group()
def main():
    pass


def clean_midi(mid):
    """
    remove everything but note_on, note_off
    """
    new_mid = MidiFile()
    for track in mid.tracks:
        new_track = MidiTrack()
        for msg in track:
            if msg.type in ['note_on', 'note_off'] and msg.channel != 9:
                new_track.append(msg)
        new_mid.tracks.append(new_track[:])
    return new_mid


def to_vector(mid):
    """
    produce 2d array form of midi.
    """
    # note that not all my training data has note_on - note_off
    # some files only report note on
    # to make it sensible, make all notes a max length
    MAX_NOTE_LEN = 96
    mid = clean_midi(mid)
    steps = int(second2tick(mid.length, mid.ticks_per_beat, 500000)) + 1
    steps = max(steps, MIDI_CHUNK)
    note_active = np.zeros((128))
    v = np.zeros((128, steps))
    j = 0
    for msg in mid:
        if msg.type in ['note_on', 'note_off']:
            dt = int(second2tick(msg.time, mid.ticks_per_beat, 500000))
            for __ in range(dt):
                v[:, j] = note_active
                for i in range(128):
                    if note_active[i] > 0:
                        note_active[i] -= 1
                j += 1
            if msg.type == 'note_on':
                note_active[msg.note] = MAX_NOTE_LEN
            else:
                note_active[msg.note] = 0
    return np.clip(v, 0, 1)


def to_simple_vector(mid):
    """
    produce 2d array form of midi.
    identical to to_vector but allows only a single note at a time
    """
    NOTE_LEN = 96
    mid = clean_midi(mid)
    steps = int(second2tick(mid.length, mid.ticks_per_beat, 500000)) + 1
    steps = max(steps, MIDI_CHUNK)
    note_active = np.zeros((128))
    note_on = False
    v = np.zeros((128, steps))
    j = 0
    for msg in mid:
        if msg.type in ['note_on']:
            dt = int(second2tick(msg.time, mid.ticks_per_beat, 500000))
            for __ in range(dt):
                v[:, j] = note_active
                for i in range(128):
                    if note_active[i] > 0:
                        note_active[i] -= 1
                j += 1
            if np.all(note_active == 0):
                note_on = False
            if msg.type == 'note_on' and not note_on:
                note_active[msg.note] = NOTE_LEN
                note_on = True
    return np.clip(v, 0, 1)


def load_data(midis):
    """
    generate a batch of training data from some midi files
    """
    MELODY_SAMPLES = 32
    x = np.zeros((MELODY_SAMPLES*len(midis), MIDI_CHUNK, 128), dtype='float32')
    y = np.zeros((MELODY_SAMPLES*len(midis), MIDI_CHUNK, 128), dtype='float32')
    j = 0
    for infile in midis:
        mid = MidiFile(infile)
        v = to_vector(mid)
        vs = to_simple_vector(mid)
        k = v.shape[1]
        for i in range(MELODY_SAMPLES):
            # take a number of random samples from the song
            # each sample is MIDI_CHUNK long
            z = random.randint(0, k - MIDI_CHUNK)
            x[j] = np.transpose(v[:, z:(z + MIDI_CHUNK)])
            y[j] = np.transpose(vs[:, z:(z + MIDI_CHUNK)])
            j += 1
    return {'src': y, 'trg': x}


def test_save_vector_as_midi(v):
    """
    save 2d array as midi for playback
    """
    mid = MidiFile()
    track = MidiTrack()
    k = 0
    for i in range(1, v.shape[1]):
        for j in range(128):
            if v[j, i - 1] == 0 and v[j, i] == 1:
                track.append(Message(
                    'note_on', note=j, velocity=127, time=i - k
                ))
                k = i
            elif v[j, i - 1] == 1 and v[j, i] == 0:
                track.append(Message(
                    'note_off', note=j, velocity=127, time=i - k
                ))
                k = i
    mid.tracks.append(track)
    mid.save('test.mid')


def plot_vector(v):
    print('plotting')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots()
    print(v[0, :, :].shape)
    axs.imshow(np.transpose(v[0, :, :]), aspect=15/1)
    fig.set_size_inches(15, 2)
    plt.show()


LSTM_SIZE = 128
LSTM_LAYERS = 1
LSTM_HIDDEN = 128


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = torch.nn.LSTM(
            LSTM_SIZE,
            LSTM_HIDDEN,
            LSTM_LAYERS,
            batch_first=True
        )
        self.fc = torch.nn.Sigmoid()

    def forward(self, x, prev_state):
        out, state = self.lstm(x, prev_state)
        __, state = self.lstm(out, prev_state)
        out = self.fc(out)
        return out, state

    def init_state(self, sequence_length):
        return (torch.zeros(LSTM_LAYERS, sequence_length, LSTM_SIZE),
                torch.zeros(LSTM_LAYERS, sequence_length, LSTM_SIZE))


@main.command()
@click.argument('infiles', type=click.Path(), nargs=-1)
def test(infiles):
    # import matplotlib.pyplot as plt
    # mid = MidiFile(infile)
    # v = to_vector(mid)[:, :4096]
    # fig, axs = plt.subplots()
    # axs.imshow(v, aspect=15/1)
    # fig.set_size_inches(15, 2)
    # plt.show()
    # test_save_vector_as_midi(v)
    d = load_data(infiles)
    v = d['trg']
    s = d['src']
    print(v.shape)
    print(s.shape)
    plot_vector(v)
    plot_vector(s)

    net = Net()
    net = net.float()
    state = net.init_state(32*len(infiles))
    # t = torch.tensor(v, device=device)
    t = torch.from_numpy(v).to(torch.float)
    # t = torch.unsqueeze(t, 1)
    print(t)
    print(t.shape)
    out, state = net.forward(t, state)
    print(out.shape)
    print(out)


def init_weights(m):
    for name, param in m.named_parameters():
        torch.nn.init.uniform_(param.data, -0.08, 0.08)


def _train(model, data, optimizer, criterion, clip):
    epoch_loss = 0
    model.train()
    src = torch.from_numpy(data).to(torch.float)

    optimizer.zero_grad()
    output = model(src)

    loss = criterion(output, src)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()

    epoch_loss += loss.item()

    return epoch_loss/len(data)


@main.command()
@click.argument('infiles', nargs=-1, type=click.Path())
def train(infiles):
    model = Net().float()
    model.apply(init_weights)
    print(model)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.BCELoss()

    N_EPOCHS = 512
    CLIP = 1
    FILE_BATCH = 4

    best_valid_loss = float('inf')
    print('epoch\ttime\ttrain_loss\tvalid_loss')

    start_time = time.time()
    for epoch in range(N_EPOCHS):
        train_data = load_data(random.sample(infiles, FILE_BATCH))
        # eval_data = load_data(random.sample(infiles, FILE_BATCH))
        train_loss = _train(model, train_data, optimizer, criterion, CLIP)
        valid_loss = train_loss
        # valid_loss = _evaluate(models, eval_data, criterion)
        end_time = time.time()

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'convolver_state_dict.pt')

        print(epoch, end_time - start_time, train_loss, valid_loss, sep='\t')


@main.command()
@click.argument('infiles', type=click.Path(), nargs=-1)
def test_generate(infiles):
    # generate some music based on input data and self feedback
    model = Net().float()
    model.load_state_dict(torch.load('convolver_state_dict.pt'))
    model.eval()

    THRESHOLD = 0.01

    data = load_data(random.sample(infiles, 1))['src'][0:1, :, :, :]
    # src = torch.from_numpy(data).to(torch.float)
    src = torch.tensor(data, requires_grad=False, dtype=torch.float)
    print(type(data), data.shape)
    for i in range(MIDI_CHUNK):
        print(i)
        res = model(src)
        print(res)
        nres = res.detach().numpy()
        nres[nres >= THRESHOLD] = 1.0
        nres[nres < THRESHOLD] = 0
        # src[:, :, :, :-1] = src[:, :, :, 1:]
        newsrc = np.zeros((1, 1, 128, 4096))
        newsrc[:, :, :, :-1] = src.detach().numpy()[:, :, :, 1:]
        newsrc[:, :, :, -1] = nres
        # src[:, :, :, -1] = res
        src = torch.tensor(newsrc, requires_grad=False, dtype=torch.float)

    plot_vector(src.numpy())
    test_save_vector_as_midi(src.numpy())



if __name__ == '__main__':
    main()
