import numpy as np
from tensorflow.keras.utils import to_categorical
from models.NNModel import NeuralNetwork
from models.NPModel import NPNeuralNetwork
from Stats import ( process_and_save_results, plot_final_metrics)
import pandas as pd
import os

#datasets
from tensorflow.keras.datasets import (
    mnist, fashion_mnist, cifar10, cifar100
) 

def ensure_directories():
    if not os.path.exists('Images'):
        os.makedirs('Images')
    if not os.path.exists('artifacts'):
        os.makedirs('artifacts')
    if not os.path.exists('logs'):
        os.makedirs('logs')

def load_data_MNIST():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_test = X_test.reshape(-1, 784) / 255.0
    X_train = X_train.reshape(-1, 784) / 255.0
    y_train = to_categorical(y_train, 10)
    y_test_orig = y_test.flatten() if y_test.ndim > 1 else y_test
    y_test = to_categorical(y_test, 10)
    
    return X_train, y_train, X_test, y_test, y_test_orig

def load_data_fashion():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_test = X_test.reshape(-1, 784) / 255.0
    X_train = X_train.reshape(-1, 784) / 255.0
    y_train = to_categorical(y_train, 10)
    y_test_orig = y_test.flatten() if y_test.ndim > 1 else y_test
    y_test = to_categorical(y_test, 10)
    
    return X_train, y_train, X_test, y_test, y_test_orig

def load_data_CIFAR10():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_test = X_test.reshape(-1, 3072) / 255.0
    X_train = X_train.reshape(-1, 3072) / 255.0
    y_train = to_categorical(y_train, 10)
    y_test_orig = y_test.flatten() if y_test.ndim > 1 else y_test
    y_test = to_categorical(y_test, 10)
    
    return X_train, y_train, X_test, y_test, y_test_orig

def load_data_CIFAR100():
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()
    X_test = X_test.reshape(-1, 3072) / 255.0
    X_train = X_train.reshape(-1, 3072) / 255.0
    y_train = to_categorical(y_train, 100)
    y_test_orig = y_test.flatten() if y_test.ndim > 1 else y_test
    y_test = to_categorical(y_test, 100)
    
    return X_train, y_train, X_test, y_test, y_test_orig

no_of_test = 10
epochs = [100, 100, 200, 300]
architectures = [
    [784, 256, 128, 10],
    [784, 256, 128, 10],
    [3072, 512, 256, 128, 10],
    [3072, 1024, 512, 256, 128, 100]
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
    
    for i in range(0, 4):
        if i == 0:
            X_train, y_train, X_test, y_test, y_test_orig = load_data_MNIST()
            print(f"\n\n\n TRAINING ON MNIST FOR {no_of_test} TESTS \n\n\n")
        elif i == 1:
            X_train, y_train, X_test, y_test, y_test_orig = load_data_fashion()
            print(f"\n\n\n TRAINING ON FASHION MNIST FOR {no_of_test} TESTS \n\n\n")
        elif i == 2:
            X_train, y_train, X_test, y_test, y_test_orig = load_data_CIFAR10()
            print(f"\n\n\n TRAINING ON CIFAR FOR {no_of_test} TESTS \n\n\n")
        else:
            X_train, y_train, X_test, y_test, y_test_orig = load_data_CIFAR100()
            print(f"\n\n\n TRAINING ON CIFAR(advanced) FOR {no_of_test} TESTS \n\n\n")

        for j in range(no_of_test):
            handle_test_and_train(i, j, X_train, y_train, X_test, y_test, y_test_orig)

        print("\n\n\n TRAIN AND TEST LOOP FINISHED \n\n\n")

    plot_final_metrics()

if __name__ == "__main__":
    main()