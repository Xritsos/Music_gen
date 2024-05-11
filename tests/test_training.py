"""Test the training process as proposed in:
https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5
"""

import os
import sys
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.callbacks import ModelCheckpoint

sys.path.append('./')

from source.data_modules import sequence
from source.models import base_model


def train():
    data_path = './data/notes'
    
    try:
        with open(data_path, 'rb') as d:
            notes = pickle.load(d)
    except Exception as ex:
        print()
        print(f"Failed to load {data_path} due to: {ex}")
        print("Exiting...")
        exit()
    else:
        print()
        print("Notes loaded successfully !")
        
    # get amount of pitch names
    n_vocab = len(set(notes))

    network_input, network_output = sequence.prepare_sequences(notes, n_vocab)

    model = base_model.get_base_model(network_input, n_vocab)
    
    checkpoint_filepath = './model_ckpts/test_model.keras'
    checkpoint = ModelCheckpoint(checkpoint_filepath,
                                 monitor='loss', 
                                 verbose=0, 
                                 save_best_only=True, 
                                 mode='min')
    
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, 
              epochs=2, 
              batch_size=16, 
              callbacks=callbacks_list)
    
    
if __name__ == "__main__":
    
    train()

