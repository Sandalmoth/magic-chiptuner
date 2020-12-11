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
- train seq2seq model on generating complex data from simplifyed
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

_2020-11-30_  
Implement some seq2seq (according to a reference, see code). also implement loading
data and parsing into torch tensors.

_2020-12-01_  
Implement model training (same reference). Probably need to read up a bit more on
exactly how torch wants the data, because there's a lot of errors. Either way,
I can probably get rid of the encoding layer? Also: #devember I guess.

_2020-12-02_  
Time to jump back and start with a minimal working example of pytorch seq2seq!
The minimal example seems to have worked, though I changed the data representation
slightly, making each token a rhythmic time step. This way, the rhythm of the
input should be better preserved during the transformation.

_2020-12-03_  
Initial work on redoing file conversion to be compatible with the new code. I think
I should do the translation dictionaries right away so that I dont have to save
any intermediate (large) tuple formats.

_2020-12-04_  
New conversion up and running, but in the interest of saving hard drive space
I'm gonna have to switch to dynamically generating the training tensors on demand
rather than precomputing and storing them.

_2020-12-05_  
Dynamic generation has a separate issue, in that we need to process everything
so that we can accurately set the output dimension of the network. I think that
using a single network to produce all the output instruments necessitates an
excessively large output language that just won't work. New idea: Use a single
input network and several output networks that play together.

_2020-12-06_  
New training data generation, intended to train four separate seq2seq networks
simultaneously. I have a suspicion that this four-brained-ness won't sound very
good, but I'm gonna try it before I trash it.

_2020-12-07_  
Model training setup with the new data. Chunked the data a bit for performance.
It is still outrageously slow however. I may have to switch approach again,
though I kinda wanna try training it overnight to see what it learns.

_2020-12-09_  
No progress yesterday, as I messed up and crashed the training by giving it an
html file among the midis. Anyway... I'm gonna train once more but also save the
whole model, not just the state dict for easier loading (I think). Also added
some machinery needed to playing the result as sound.

_2020-12-10_  
It learned to always be silent. _Nice._ Time for a new approach.

_2020-12-10_  
I am going to try a convolutional approach. Basically, represent music as a 2d
vector of time and pitch. The last (time) column holds just the current simplifyed
melody, the rest holds the all the notes played so far. The output is the full
melody at the current step (i.e. what notes are currently playing). So, task
number one will be to convert midi to such a 2d format (and back).

## Preemptive QnA
_Why chiptuner?_  
The training data I have is gameboy/nes music, so, that's what it's gonna learn.
