import random
import pickle
import os

import click
from mido import MidiFile, MidiTrack, second2tick


NO_NOTE = -1
SOS = -2
EOS = -3


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

        outfile = 'data/txt/' + \
                  os.path.splitext(os.path.basename(infile))[0] + '.bin'
        with open(outfile, 'wb') as f:
            pickle.dump({'src': v_simple, 'trg': v_full}, f)


if __name__ == '__main__':
    main()
