"""Test the training process as proposed in:
https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5
"""

import os
import sys
import time
import pickle
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

sys.path.append('./')

from source.data_modules import sequence
from source.models import base_model, bidi_model


def train(test_id):
    start_time = time.time()
    
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
    drop_factor = float(df['dropout'][test_id-1])
    
    print(f"============== Parameters for test {test_id} ===============")
    print(f"Batch size: {batch_size}")
    print(f"Learning_rate: {learning_rate}")
    print(f"Sequence length: {sequence_length}")
    print(f"Weight decay: {weight_decay}")
    print(f"Optimizer: {optim}")
    print(f"Dropout factor: {drop_factor}")
    print(f"Num. of epochs: {n_epochs}")
    print("=========================================")
            
    # get amount of pitch names
    n_vocab = len(set(notes))

    network_input, network_output = sequence.prepare_sequences(notes, n_vocab, sequence_length)

    model = bidi_model.get_bidi_model(network_input, n_vocab, drop_factor)
    
    if optim == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                             decay=weight_decay)
    
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
    
    end_time = time.time()
    runtime = round((end_time-start_time)/60, 2)
    
    min_loss = min(history.history['loss'])
    
    df.at[test_id-1, 'loss'] = min_loss
    df.at[test_id-1, 'runtime(min.)'] = runtime
    df.to_csv('./tests.csv', index=False)
    
    epochs = [i for i in range(1, n_epochs+1)]
    to_file = {'epoch': epochs, 'loss': history.history['loss']}
    
    log_df = pd.DataFrame(to_file)
    log_df.to_csv(f'./logs/{test_id}.csv', index=False)
