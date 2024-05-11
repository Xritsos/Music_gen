"""This modules reads the midi files and saves their notes and chords 
to be used later for training"""

import glob
import pickle
from music21 import converter, instrument, note, chord


def save_notes_chords(dir: str = ''):
    """Get notes and chords for all midi files in the specified directory.
    Data are saved on the data directory as a pickle object.

    Args:
        dir (str, required): directory to read files from. Defaults to ''.
    """
    
    notes = []
    count_tracks = 0
    print()
    print("Reading files...")
    for file in glob.glob(f"{dir}/*.mid"):
        try:
            midi = converter.parse(file)
        except Exception as ex:
            print()
            print(f"Failed to parse {file} due to: {ex}")
            print("Aborting...")
            exit()

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
                
        count_tracks += 1
        
    print()
    print(f"Total number of tracks parsed: {count_tracks}")

    print()
    print("Saving file...")
    with open('./data/notes', 'wb') as filepath:
        try:
            pickle.dump(notes, filepath)
        except Exception as ex:
            print()
            print(f"Failed to write {filepath} due to: {ex}")
            print("Exiting...")
            exit()
        else:
            print()
            print(f"File {filepath} saved successfully !")
            
            
if __name__ == "__main__":
    directory = './data/'
    
    save_notes_chords(directory)
    