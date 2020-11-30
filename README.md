# magic-chiptuner
Improving simple melodies realtime!

## Idea
When we play guitar hero or similar, there's a really simple music representation
in the five buttons + rhythm thing, yet it still feels like we're playing.
It also clearly has some kind of relationship to the melody. What if we let an
AI reconstruct music from a simplifyed representation like that? So we could
play a really basic melody, and it tries to complete the rest.

Since data availability for guitar hero levels + sheet music/midi isn't great,
I will instead focus on transforming a simple midi melody (e.g. played on the 
keyboard) to full music! This should be possible, since there is plenty of free
midi music available for training data.

## Basic design plan/TODO
- parse midi into some ~~text based format~~ vector of bytes
- ~~standardize rhythms~~ just use midi quantization
- generate simple melody version of training data by stripping chords, drums, fast notes
- transpose to all 12 keys
- train some kind of network on generating complex data from simplifyed (seq2seq?)
- implement streaming midi to text conversion
- feed into network
- restore output to midi

### Work log ###
_2020-11-26_  
Read midi, remove percussion, extract melody (? on last one).

_2020-11-27_  
Remove broken files from training data. Discover training data uses full range
of notes from 0 to 127. Implement conversion to string of bytes.

_2020-11-28_  
Simplify melody by removing chords. Implement autocorrelation for finding beat
though performance is not amazing (though maybe ok enough) and simpified rhythm
to have no more than one note per calculated beat.

_2020-11-29_  
Transpose and save files with sensible filename. Set the transposition to
-5 -> +6 semitones, to ensure that there should be good data for most melodies.
I would do more, but had to save on harddrive space... We'll see if it's enough.

_2019-11-30_  
Implement some seq2seq (according to a reference, see code). also implement loading
data and parsing into torch tensors.

## Preemptive QnA
_Why chiptuner?_  
The training data I have is gameboy/nes music, so, that's what it's gonna learn.

