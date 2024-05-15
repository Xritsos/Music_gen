"""Test the training process as proposed in:
https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5
"""

import os
import sys
import pickle
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

sys.path.append('./')

from source.data_modules import sequence
from source.models import base_model


def train(test_id):
    tf.random.set_seed(11)
    
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
        
    df = pd.read_csv('./tests.csv')
    
    n_epochs = int(df['epochs'][test_id-1])
    batch_size = int(df['batch_size'][test_id-1])
    learning_rate = float(df['learning_rate'][test_id-1])
    sequence_length = int(df['sequence_length'][test_id-1])
    weight_decay = float(df['weight_decay'][test_id-1])
    optim = df['optimizer'][test_id-1]
        
    # get amount of pitch names
    n_vocab = len(set(notes))

    network_input, network_output = sequence.prepare_sequences(notes, n_vocab, sequence_length)

    model = base_model.get_base_model(network_input, n_vocab)
    
    if optim == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                             weight_decay=weight_decay)
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    
    checkpoint_filepath = f'./model_ckpts/{test_id}_model.keras'
    checkpoint = ModelCheckpoint(checkpoint_filepath,
                                 monitor='loss', 
                                 verbose=0, 
                                 save_best_only=True, 
                                 mode='min')
    
    callbacks_list = [checkpoint]

    history = model.fit(network_input, network_output, 
                        epochs=n_epochs, 
                        batch_size=batch_size, 
                        callbacks=callbacks_list)
    
    min_loss = min(history.history['loss'])
    
    df.at[test_id-1, 'loss'] = min_loss
    df.to_csv('./tests.csv', index=False)
    
    epochs = [i for i in range(1, n_epochs+1)]
    to_file = {'epoch': epochs, 'loss': history.history['loss']}
    
    log_df = pd.DataFrame(to_file)
    log_df.to_csv(f'./logs/{test_id}.csv', index=False)
    
    
if __name__ == "__main__":
    print(tf.config.list_physical_devices('GPU'))
    
    test_id = 1
    

    with tf.device('/device:GPU:0'):
        train(test_id)

