"""
In this module it is implemented a simple plotting function that reads from 
the logs for each test run and plotts the loss during training.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot(test_id):
    df = pd.read_csv(f'./logs/{test_id}.csv')
    
    epochs = df['epoch']
    loss = df['loss']
    
    if max(epochs) <= 200:
        x_ticks = [i for i in range(0, max(epochs)+20, 20)]
        x_ticks[0] = 1
    else:
        x_ticks = [i for i in range(0, max(epochs)+50, 50)]
        x_ticks[0] = 1
        
    y_ticks = np.linspace(min(loss), max(loss), 10)
    
    plt.plot(epochs, loss)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.show()
    

if __name__ == "__main__":
    
    plot(74)
    