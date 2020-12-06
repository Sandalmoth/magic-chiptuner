import random
import os
import numpy as np

import click
from mido import MidiFile, MidiTrack, second2tick
import torch


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
            result.append(NO_NOTE, )
        elif len(x) > 3:
            result.append(tuple(random.sample(x, 3)))
        else:
            result.append(tuple(x))
    # return [tuple(x) if len(x) > 0 else (-1,) for x in v]
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
    print("major beat is", beat)
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


@main.command()
@click.argument('infiles', nargs=-1, type=click.Path())
def from_midi(infiles):
    """
    Convert an input midi file to a simplified vector format.
    Standardizes the rhythm, and strips any percussion tracks.
    Generates an output with all notes, and a simplifyied melody.
    """

    for infile in infiles:
        print('parsing', infile)
        mid = MidiFile(infile)
        v_full = to_vector(mid)
        v_simple = simplify(v_full)

        outfile = 'data/txt/' + os.path.splitext(os.path.basename(infile))[0]
        for t in range(-12, 12):
            t_simple = transposed(v_simple, t)
            # add_to_simple_dict(t_simple)
            t_full = transposed(v_full, t)
            # add_to_full_dict(t_full)

            a_simple = np.array([lang_forward[x] for x in t_simple])
            np.save(outfile + '_' + str(t) + '.simple', a_simple)
            a_full = np.array([lang_forward[x] for x in t_full])
            np.save(outfile + '_' + str(t) + '.full', a_full)


def get_data(infiles, device):
    """
    Dynamically produce a pair of simple/full tensors for training
    """
    random.shuffle(infiles)
    while True:
        for infile in infiles:
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

            yield {
                'src': src_tensor,
                'trg_s': trg_tensor_s,
                'trg_a': trg_tensor_a,
                'trg_t': trg_tensor_t,
                'trg_b': trg_tensor_b
            }


if __name__ == '__main__':
    main()
