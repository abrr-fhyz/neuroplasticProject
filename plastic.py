from tensorflow.keras.utils import to_categorical
from models.NNGPU import NeuralNetwork
from models.NPGPU import NPNeuralNetwork
from Stats import ( process_and_save_results, plot_final_metrics)
import numpy as np
import os

#datasets
from tensorflow.keras.datasets import (
    mnist, fashion_mnist
) 

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

initial_test = 0
final_test = 1
epochs = [50, 60]
architectures = [
    [784, 256, 128, 10],
    [784, 256, 128, 10]
]

def handle_test_and_train(i, j, X_train, y_train, X_test, y_test, y_test_orig, k):
    e = epochs[k]
    architecture = architectures[k]
    idn = i*10 + j
    if i == 0:
        print(f"Training standard neural network - IDN: {idn}...")
        std_model = NeuralNetwork(architecture)
        std_model.train(X_train, y_train, epochs=e)
        std_model.save_model(idn)
    #std_model.load_model(idn)

    if i == 1:
        print(f"\nTraining neuroplastic neural network - IDN: {idn}...")
        np_model = NPNeuralNetwork(architecture)
        np_model.train(X_train, y_train, epochs=e)
        np_model.save_model(idn)
    #np_model.load_model(idn)

def handle_test_and_retrain(i, j, X_train, y_train, X_test, y_test, y_test_orig, k):
    e = epochs[k]
    architecture = architectures[k]
    idn = i*10 + j
    if i == 0:
        print(f"Training standard neural network - IDN: {idn}...")
        std_model = NeuralNetwork(architecture)
        std_model.load_model(idn)
        std_model.train(X_train, y_train, epochs=e)
        std_model.save_model(idn)

    if i == 1:
        print(f"\nTraining neuroplastic neural network - IDN: {idn}...")
        np_model = NPNeuralNetwork(architecture)
        np_model.load_model(idn)
        np_model.train(X_train, y_train, epochs=e)
        np_model.save_model(idn)


def tabulate_results(i, j):
    idn = (i)*10 + j
    std_model = NeuralNetwork(architectures[0])
    std_model.load_model(idn)
    np_model = NPNeuralNetwork(architectures[0])
    np_model.load_model(idn + 10)
    _, _, X_test_2, y_test_2, y_test_orig_2 = load_data_fashion()
    process_and_save_results(idn, std_model, np_model, X_test_2, y_test_2, y_test_orig_2)

#main loop
def main():
    for i in range(2, 2):
        if i == 0:
            print(f"\n\n\nTRAINING Standard NN FOR {final_test - initial_test} TESTS \n\n\n")
        elif i == 1:
            print(f"\n\n\nTRAINING NPNN NN FOR {final_test - initial_test} TESTS \n\n\n")

        X_train, y_train, X_test, y_test, y_test_orig = load_data_fashion()
        for j in range(initial_test, final_test):
            handle_test_and_train(i, j, X_train, y_train, X_test, y_test, y_test_orig, 1)
        X_train, y_train, X_test, y_test, y_test_orig = load_data_MNIST()
        for j in range(initial_test, final_test):
            handle_test_and_retrain(i, j, X_train, y_train, X_test, y_test, y_test_orig, 0)
        print("\n\n\nTRAIN AND TEST LOOP FINISHED \n\n\n")
        
    tabulate_results(0, 0)
    #plot_final_metrics()

if __name__ == "__main__":
    main()