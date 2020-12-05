import random
import os
import numpy as np
import pickle
import gc

import click
from mido import MidiFile, MidiTrack, second2tick
import torch


NO_NOTE = -1
SOS = -2
EOS = -3


simple_forward = {(x, ): i for i, x in enumerate([NO_NOTE, SOS, EOS])}
simple_backward = {i: (x, ) for i, x in enumerate([NO_NOTE, SOS, EOS])}
full_forward = {(x, ): i for i, x in enumerate([NO_NOTE, SOS, EOS])}
full_backward = {i: (x, ) for i, x in enumerate([NO_NOTE, SOS, EOS])}


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

    return [tuple(x) if len(x) > 0 else (-1,) for x in v]


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
        s.append((random.choice(x), ))

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


def add_to_simple_dict(v):
    for token in v:
        simple_forward[token] = len(simple_forward)
        simple_backward[len(simple_backward)] = token


def add_to_full_dict(v):
    for token in v:
        full_forward[token] = len(full_forward)
        full_backward[len(full_backward)] = token


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
            add_to_simple_dict(t_simple)
            t_full = transposed(v_full, t)
            add_to_full_dict(t_full)

            a_simple = np.array([simple_forward[x] for x in t_simple])
            np.save(outfile + '_' + str(t) + '.simple', a_simple)
            a_full = np.array([full_forward[x] for x in t_full])
            np.save(outfile + '_' + str(t) + '.full', a_full)

    # save dictionaries so we can get midi back later
    with open('simple_forward.dict', 'wb') as f:
        pickle.dump(simple_forward, f)
    with open('simple_backward.dict', 'wb') as f:
        pickle.dump(simple_backward, f)
    with open('full_forward.dict', 'wb') as f:
        pickle.dump(full_forward, f)
    with open('full_backward.dict', 'wb') as f:
        pickle.dump(full_backward, f)


@main.command()
@click.argument('infiles', nargs=-1, type=click.Path())
def build_dicts(infiles):
    for infile in infiles:
        print('parsing', infile)
        mid = MidiFile(infile)
        v_full = to_vector(mid)
        v_simple = simplify(v_full)

        for t in range(-12, 13):
            t_simple = transposed(v_simple, t)
            add_to_simple_dict(t_simple)
            t_full = transposed(v_full, t)
            add_to_full_dict(t_full)

        gc.collect()
        print(len(full_forward))

    with open('simple_forward.dict', 'wb') as f:
        pickle.dump(simple_forward, f)
    with open('simple_backward.dict', 'wb') as f:
        pickle.dump(simple_backward, f)
    with open('full_forward.dict', 'wb') as f:
        pickle.dump(full_forward, f)
    with open('full_backward.dict', 'wb') as f:
        pickle.dump(full_backward, f)



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

            src_tensor = torch.tensor(
                [simple_forward[x] for x in t_simple],
                dtype=torch.long, device=device
            ).view(-1, 1)
            trg_tensor = torch.tensor(
                [full_forward[x] for x in t_full],
                dtype=torch.long, device=device
            ).view(-1, 1)

            # yield {'src': src_tensor, 'trg': trg_tensor}
            yield src_tensor, trg_tensor


if __name__ == '__main__':
    main()
