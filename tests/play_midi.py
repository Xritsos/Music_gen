"""Load and play a midi file using pygame (installation of Timidity++ is required)
"""
    
import pygame


def play(file_path):
    pygame.mixer.pre_init(44100, 16, 2, 4096) # frequency, size, channels, buffersize
    pygame.init() # turn all of pygame on.
    
    clock = pygame.time.Clock()
    try:
        pygame.mixer.music.load(file_path)
    except pygame.error as er:
        print(f"Could not load file due to: {er}")
        exit()
    
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        # check if playback has finished
        clock.tick(30)
        
      
if __name__ == "__main__":
    file_path = './tests/test_file.mid'

    try:
        play(file_path)
    except KeyboardInterrupt:
        # if user hits Ctrl/C then exit
        # (works only in console mode)
        pygame.mixer.music.fadeout(1000)
        pygame.mixer.music.stop()
        
        raise SystemExit
    