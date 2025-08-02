import numpy as np
from tensorflow.keras.utils import to_categorical
from models.NNModel import NeuralNetwork
from models.NPModel import NPNeuralNetwork
from Stats import ( process_and_save_results, plot_final_metrics)
import pandas as pd
import os

#datasets
from tensorflow.keras.datasets import mnist
def ensure_directories():
    if not os.path.exists('Images'):
        os.makedirs('Images')
    if not os.path.exists('artifacts'):
        os.makedirs('artifacts')
    if not os.path.exists('logs'):
        os.makedirs('logs')

def load_data_MNIST():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 784) / 255.0
    y_train = to_categorical(y_train, 10)
    X_test = X_test.reshape(-1, 784) / 255.0
    y_test_orig = y_test.flatten() if y_test.ndim > 1 else y_test
    y_test = to_categorical(y_test, 10)
    
    return X_train, y_train, X_test, y_test, y_test_orig

no_of_test = 10
epochs = [50, 100, 200, 300]
architectures = [
    [784, 256, 128, 10],
    [784, 256, 128, 10],
    [3072, 512, 256, 128, 10],
    [3072, 1024, 512, 256, 128, 10]
]


def handle_test_and_train(i, j, X_train, y_train, X_test, y_test, y_test_orig):
    e = epochs[i]
    architecture = architectures[i]
    idn = i*10 + j
    print(f"Training standard neural network - IDN: {idn}...")
    std_model = NeuralNetwork(architecture)
    std_model.train(X_train, y_train, epochs=e)
    std_model.save_model(idn)

    print(f"\nTraining neuroplastic neural network - IDN: {idn}...")
    np_model = NPNeuralNetwork(architecture)
    np_model.train(X_train, y_train, epochs=e)
    np_model.save_model(idn)

    process_and_save_results(idn, std_model, np_model, X_test, y_test, y_test_orig)


#main loop
def main():
    ensure_directories()
    plot_final_metrics()

if __name__ == "__main__":
    main()