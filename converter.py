try:
    import cPickle as pickle
except:
    import pickle

import random
import os

import click
from mido import MidiFile, MidiTrack, second2tick


@click.group()
def main():
    pass


def standardize_rhythm(mid):
    # time signature is made of
    # - numerator
    # - denominator
    # this is like the 4/4 or 3/4 or 6/8 thing fromm sheet music
    # - midi clocks per metronome click
    # so basically midi clocks per numerator click?
    # - number of 32nd notes in a quarter note
    # so a midi can be like divided on triplets rather than 8ths and stuff
    # if time_signature meta message isn't present then 4/4 is default
    # NOTE
    # it may not be necessary to do this. Midi defaults to 24 possible
    # per quarter note anyway
    pass


def find_beat(v):
    """
    find the beat of a vector format midi using the autocorrelation function
    """

    v2 = [0 for __ in range(len(v))]
    j = 0
    for note in v:
        if note == 0:
            j += 1
        else:
            v2[j] = 1


    result = []
    # no need to check more than 24*4 = 96
    # i.e. 4 quarter notes of standard midi
    for lag in range(96):
        s = 0
        for i in range(len(v2)):
            if v2[i] > 0 and v2[(i + lag) % len(v2)] > 0:
                s += 1
        result.append((lag, s))
    # print(result)
    # print(sorted(result, key=lambda x: x[1]))
    return sorted(result, key=lambda x: x[1])[-2][0]


def simplify(v):
    """
    simplify vector format file to a suitable simple melody
    """
    # TODO
    # idea:
    #   find a common beat frequency (with fourier transform?)
    #   remove notes not on the common beat if they have neighbours that are
    #   remove chords by selecting a (random?) note
    simple = []
    chord = []
    # remove chords
    for note in v:
        # may be a note, or an end of chord
        # first, gather up everything played simultaneously
        if note == 0:
            # if the chord ends, add a note at random
            if len(chord) > 0:
                simple.append(random.choice(chord))
            simple.append(0)
            chord = []
        else:
            chord.append(note)
    simple2 = []
    # simplify rhythm
    beat = find_beat(simple)
    print(beat)
    # simply make sure there is no more than 1 note in each beat segment
    j = 0
    found = False
    for note in simple:
        if note == 0:
            j += 1
            simple2.append(0)
            if j % beat == 0:
                found = False
        else:
            if not found:
                simple2.append(note)
                found = True

    return simple2


# so, midi notes seem to be in the [0, 127] range
# I could convert that to strings, but I figure
# since neural nets work with vectors anyway
# it kinda makes more sense to convert to a string of bytes
# something like
#  0 - end timestep
#  1 - end song
#  128-255 - play note
# then it would turn into sequences like:
# 200, 0, 0, 0, 0, 205, 200, 0, 0, 0, 0, 1
# which would be note, wait a bit, chord, wait a bit, end song
# That sequence could easily be saved as bytes or whatever

SONG_END = 1 # is this really needed?
STEP_END = 0

def vectorize(mid):
    """
    turn midi into byte string
    """
    v = []
    for msg in mid:
        # who needs note of message anyway
        if msg.type == 'note_on':
            # on a serious note, I think they will just complicate the training
            # and it's better to just approximate them when restorign to midi
            dt = second2tick(msg.time, mid.ticks_per_beat, 500000)
            for __ in range(int(dt)):
                v.append(STEP_END)
            v.append(msg.note + 128)
    v.append(SONG_END)
    return v


def transpose(v, offset):
    """
    transpose vector format midi
    discards note that go out of the 0, 127 range
    """
    t = []
    for note in v:
        if note >= 128:
            tn = note + offset
            if tn >= 0 and tn < 128:
                t.append(tn)
        else:
            t.append(note)
    return tn



@main.command()
# @click.option('-i', '--infiles', type=click.Path(), multiple=True)
# @click.option('-o', '--outfulls', type=click.Path(), multiple=True)
# @click.option('-s', '--outsimples', type=click.Path(), multiple=True)
@click.argument('infiles', nargs=-1, type=click.Path())
def to_text(infiles):
    """
    Convert an input midi file to a textbased output file.
    Standardizes the rhythm, and strips any percussion tracks.
    Generates an output with all notes, and a simplifyied melody.
    """
    # TODO make a function that lives up to that description
    # outfull = 'data/txt/tmp.mid'
    # outsimple = 'data/txt/tmp2.mid'

    mn = 999
    mx = 0

    for infile in infiles:
        print('parsing:', infile)
        mid_in = MidiFile(infile)
        print(mid_in, mid_in.tracks)

        # remove every message on channel 9 (standard percussion channel)
        # also extract every message on channel 0 (standard melody (?))
        mid_full = MidiFile()
        mid_melody = MidiFile()
        for track in mid_in.tracks:
            new_track = MidiTrack()
            melody_track = MidiTrack()
            for msg in track:
                if msg.type in ['note_on']:  # , 'note_off']:
                    if msg.channel == 0:
                        melody_track.append(msg)
                    if msg.channel != 9:
                        new_track.append(msg)
                # else:
                #     print(msg)
            mid_full.tracks.append(new_track[:])
            mid_melody.tracks.append(melody_track[:])

        # save tracks for testing
        # mid_full.save(outfull)
        # mid_melody.save(outsimple)

        # TODO
        # normalize to identical time signature/bpm somehow
        # clues:
        #   look at set_tempo meta message
        #   look at time_signature meta message
        # idea:
        #   subdivide each quarter note into some default number of segments
        #   each segment gets an output line in text for timing
        #   maybe 12 segments good? (16th notes and triplets possible)
        #   maybe 24 would be better to allow for 32th notes?
        # NOTE
        # 24 is actually standard for midi
        # maybe no standardization is needed

        v_full = vectorize(mid_full)
        v_melody = vectorize(mid_melody)
        v_melody = simplify(v_melody)
        print(v_full.count(0), v_melody.count(0))
        # assert v_full.count(0) == v_melody.count(0)  # same number of steps

        outfile = os.path.splitext(os.path.basename(infile))[0]
        for i in range(-12, 13):
            tfull = transpose(v_full, i)
            tmelody = transpose(v_melody, i)
            outfull = 'data/txt/' + outfile + '_' + str(i) + '.full'
            outsimple = 'data/txt/' + outfile + '_' + str(i) + '.simple'
            with open(outfull, 'wb') as f:
                pickle.dump(tfull, f)
            with open(outsimple, 'wb') as f:
                pickle.dump(tmelody, f)




if __name__ == '__main__':
    main()
