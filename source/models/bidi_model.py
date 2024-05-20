"""This is an implementation of the model suggested on 
https://medium.com/@sabadejuyee21/music-generation-using-deep-learning-7d3dbb2254af
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from keras_self_attention import SeqSelfAttention
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.layers import Activation
from tensorflow.keras import initializers
from tensorflow.keras.layers import BatchNormalization as BatchNorm


def get_bidi_model(network_input, n_vocab, drop_factor):
    tf.random.set_seed(11)
    tf.keras.utils.set_random_seed(11)
    
    model = Sequential()
    model.add(Bidirectional(LSTM(512, input_shape=(network_input.shape[1], 
                                                   network_input.shape[2]), 
                                 return_sequences=True,
                                 kernel_initializer=initializers.glorot_uniform(seed=0))))
    
    model.add(SeqSelfAttention(attention_activation='sigmoid', kernel_initializer=initializers.glorot_uniform(seed=0)))
    model.add(Dropout(drop_factor)) 
    model.add(GlobalMaxPooling1D()) 
    model.add(Dense(n_vocab, kernel_initializer=initializers.glorot_uniform(seed=0)))
    model.add(Activation('softmax')) 

    return model
