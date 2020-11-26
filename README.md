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
- parse midi into some text based format
- standardize rhythms
- generate simple melody version of training data by stripping chords, drums, fast notes
- transpose to all 12 keys
- train some kind of network on generating complex data from simplifyed (seq2seq?)
- implement streaming midi to text conversion
- feed into network
- restore output to midi
