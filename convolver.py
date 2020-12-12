import click
from mido import MidiFile, MidiTrack, second2tick, Message
import numpy as np
import torch


MIDI_CHUNK = 4096  # leads to nice 128x128 states


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
    note_active = np.zeros((128))
    v = np.zeros((128, steps))
    j = 0
    print(v.shape)
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
            torch.nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.drop_out = torch.nn.Dropout()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        print(out.shape)
        out = self.drop_out(out)


@main.command()
@click.argument('infile', type=click.Path())
def test(infile):
    import matplotlib.pyplot as plt
    mid = MidiFile(infile)
    v = to_vector(mid)[:, :4096]
    fig, axs = plt.subplots()
    axs.imshow(v, aspect=15/1)
    fig.set_size_inches(15, 2)
    plt.show()
    test_save_vector_as_midi(v)

    device = torch.device('cpu')
    net = Net()
    t = torch.tensor(v, device=device)
    net.forward(t)


if __name__ == '__main__':
    main()
