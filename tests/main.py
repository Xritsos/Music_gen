from test_training import train
from test_base_model import generate



if __name__ == "__main__":
    
    print(tf.config.list_physical_devices('GPU'))

    test_id = 10

    with tf.device('/device:GPU:0'):
        train(test_id)
        
        generate(test_id)