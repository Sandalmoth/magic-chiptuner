import click
from mido import MidiFile, MidiTrack


@click.group()
def main():
    pass


@main.command()
@click.argument('infile', type=click.Path())
@click.argument('outfull', type=click.Path())
@click.argument('outsimple', type=click.Path())
def to_text(infile, outfull, outsimple):
    """
    Convert an input midi file to a textbased output file.
    Standardizes the rhythm, and strips any percussion tracks.
    Generates an output with all notes, and a simplifyied melody.
    """
    # TODO make a function that lives up to that description
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
            if msg.type in ['note_on', 'note_off']:
                if msg.channel == 0:
                    melody_track.append(msg)
                if msg.channel != 9:
                    new_track.append(msg)
            else:
                print(msg)
        mid_full.tracks.append(new_track[:])
        mid_melody.tracks.append(melody_track[:])

    # save tracks for testing
    mid_full.save(outfull)
    mid_melody.save(outsimple)

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


if __name__ == '__main__':
    main()
