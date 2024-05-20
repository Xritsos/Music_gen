"""This is an implementation of the model suggested on 
https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization as BatchNorm
from tensorflow.keras import initializers


def get_base_model(network_input, n_vocab, drop_factor):
    tf.random.set_seed(11)
    tf.keras.utils.set_random_seed(11)
    model = Sequential()
    model.add(LSTM(512, 
                   input_shape=(network_input.shape[1], network_input.shape[2]), 
                   recurrent_dropout=drop_factor, 
                   return_sequences=True,
                   kernel_initializer=initializers.glorot_uniform(seed=0)))
    
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=drop_factor, kernel_initializer=initializers.glorot_uniform(seed=0)))
    model.add(LSTM(512, kernel_initializer=initializers.glorot_uniform(seed=0)))
    model.add(BatchNorm())
    model.add(Dropout(drop_factor))
    model.add(Dense(256, kernel_initializer=initializers.glorot_uniform(seed=0)))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(drop_factor))
    model.add(Dense(n_vocab, kernel_initializer=initializers.glorot_uniform(seed=0)))
    model.add(Activation('softmax'))

    return model
