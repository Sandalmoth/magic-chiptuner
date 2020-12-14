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


def simplify(v):
    """
    return simplified last time slice of a 128xMIDI_CHUNK
    """
    if np.sum(v[: -1]) == 0:
        return np.zeros(128)
    x = random.choices(list(range(128)), v[:, -1])
    y = np.zeros(128)
    y[x] = 1
    return y


def load_data(midis):
    """
    generate a batch of training data from some midi files
    """
    MELODY_SAMPLES = 32
    x = np.zeros((MELODY_SAMPLES*len(midis), 1, 128, MIDI_CHUNK), dtype='float32')
    s = np.zeros((MELODY_SAMPLES*len(midis), 128), dtype='float32')
    j = 0
    for infile in midis:
        mid = MidiFile(infile)
        v = to_vector(mid)
        k = v.shape[1]
        for i in range(MELODY_SAMPLES):
            # take a number of random samples from the song
            # each sample is MIDI_CHUNK long
            z = random.randint(0, k - MIDI_CHUNK)
            x[j, 0] = v[:, z:(z + MIDI_CHUNK)]
            s[j] = v[:, z:(z + MIDI_CHUNK)][:, -1]
            x[j, 0, :, -1] = simplify(v[:, z:(z + MIDI_CHUNK)])
            j += 1
    return {'src': x, 'trg': s}


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


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 2, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(2, 4, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 8, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.drop_out = torch.nn.Dropout()
        self.fc1 = torch.nn.Linear(65536, 8192)
        self.fc2 = torch.nn.Linear(8192, 1024)
        self.fc3 = torch.nn.Linear(1024, 128)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


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
    v = load_data(infiles)['src']
    print(v.shape)


    net = Net()
    net = net.float()
    # t = torch.tensor(v, device=device)
    t = torch.from_numpy(v).to(torch.float)
    # t = torch.unsqueeze(t, 1)
    print(t)
    print(t.shape)
    out = net.forward(t)
    print(out.shape)
    print(out)


def init_weights(m):
    for name, param in m.named_parameters():
        torch.nn.init.uniform_(param.data, -0.08, 0.08)


def _train(model, data, optimizer, criterion, clip):
    epoch_loss = 0
    model.train()
    src = torch.from_numpy(data['src']).to(torch.float)
    trg = torch.from_numpy(data['trg']).to(torch.float)

    optimizer.zero_grad()
    output = model(src)

    loss = criterion(output, trg)
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
    criterion = torch.nn.MSELoss()

    N_EPOCHS = 248
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


if __name__ == '__main__':
    main()
