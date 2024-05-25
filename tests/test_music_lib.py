"""Read and explore a midi file"""

from music21 import converter, instrument, note, chord, midi
import pygame


file_ = './data/chopin_midi/test/ballade1.mid'
notes_to_parse = None
notes = []

# read file
mid = converter.parse(file_)

parts = instrument.partitionByInstrument(mid)    

if parts: # file has instrument parts
    notes_to_parse = parts.parts[0].recurse()
else: # file has notes in a flat structure
    notes_to_parse = midi.flat.notes 
    
for element in notes_to_parse:
    if isinstance(element, note.Note):
        notes.append(str(element.pitch))
        print(element.duration.quarterLength)
    elif isinstance(element, chord.Chord):
        notes.append('.'.join(str(n) for n in element.normalOrder))    

