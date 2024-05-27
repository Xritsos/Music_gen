[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-360/) 
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)

# Music generation using LSTMs
This is an attempt to generate music based on Deep Learning techniques, using Chopin's piano pieces. The process is simplified by only extracting notes and chords from the midi files, provided by [3]. The goal is to explore RNNs and create a melody rather than an actual Chopin-style music composition. It is left for the reader to try incorporating rests, note durations and volumes, which is a more complex task.

## Files
The Chopin piano pieces can be found, in MIDI fomat, inside `data/chopin_midi/train`, but the extracted notes and chords are also provided in `data/notes_chopin_train`. The whole exploration phase is documented in the `steps_log.odt` file and all the tests performed can be found on `tests.csv` file. For all the tests provided there, in the `logs` folder there are the corresponding epoch-loss information, that one can visualize using the `source/plot_loss.py` module. The generated outputs for each test id can be accessed in MIDI format, in folder `outputs`.   

In `source/data_modules` there are the necessary modules for extracting notes/chords from a midi file and creating sequences to feed the model. Regarding training, using the `tests/main.py` you can run a training session which is based on the `tests.csv` file. There exists the `test_models.py` module that takes the trained module and creates an output midi file. If you want to hear your output, using the `tests/play_midi.py` module you can specify the midi track of your choice.  

## Requirements
All the python packages required are saved in the `requirements.txt` file. The experiments were held on a Linux machine. For playing the midi files, as mentioned above, installation of Timidity++ is required.

## Result
Below you can hear the result of test id 74, as described in the `tests.csv` file.

https://github.com/Xritsos/Music_gen/assets/57326163/d0e796c9-e1a9-4619-a841-613afb1d6268

## Acknowledgements
Special thanks to Sigurður Skúli, for his article in music generation, on top of which this implementation was based.

Results presented in this work have been produced using the AUTH Compute Infrastructure and Resources. The author would like to acknowledge the support provided by the Scientific Computing Office throughout the progress of this research work.

## Refences
[1] Sigurður Skúli, https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5  
[2] Juyee Sabade, https://medium.com/@sabadejuyee21/music-generation-using-deep-learning-7d3dbb2254af  
[3] MIDIWORLD, https://www.midiworld.com/chopin.htm

## Contact
e-mail: chrispsyc@yahoo.com
