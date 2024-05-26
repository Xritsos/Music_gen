import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from test_training import train
from test_models import generate


if __name__ == "__main__":
    
    print(tf.config.list_physical_devices('GPU'))

    for test_id in [74]:
        with tf.device('/device:GPU:0'):
            train(test_id)
        
            generate(test_id)