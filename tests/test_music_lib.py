"""Read and explore a midi file"""

from music21 import converter, instrument, note, chord, midi
import pygame


file_ = '/home/akahige/Python Work/Music_gen/tests/EyesOnMePiano.mid'
notes_to_parse = None
notes = []

# read file
mid = converter.parse(file_)

def play(music_file):
    clock = pygame.time.Clock()
    try:
        pygame.mixer.music.load(music_file)
        print("Music file %s loaded!" % music_file)
    except pygame.error:
        print("File %s not found! (%s)" % (music_file, pygame.get_error()))
        return
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        # check if playback has finished
        clock.tick(30)

pygame.mixer.pre_init(44100, 16, 2, 4096) #frequency, size, channels, buffersize
pygame.init() #turn all of pygame on.
    
try:
    # use the midi file you just saved
    play(file_)
except KeyboardInterrupt:
    # if user hits Ctrl/C then exit
    # (works only in console mode)
    pygame.mixer.music.fadeout(1000)
    pygame.mixer.music.stop()
    raise SystemExit
# parts = instrument.partitionByInstrument(mid)

# if parts: # file has instrument parts
#     notes_to_parse = parts.parts[0].recurse()
# else: # file has notes in a flat structure
#     notes_to_parse = mid.flat.notes



# for element in notes_to_parse:
#     print(element.__dict__)
    # if isinstance(element, note.Note):
    #     notes.append(str(element.pitch))
    # elif isinstance(element, chord.Chord):
    #     notes.append('.'.join(str(n) for n in element.normalOrder))


# sp = midi.realtime.StreamPlayer(mid)
# sp.play()
