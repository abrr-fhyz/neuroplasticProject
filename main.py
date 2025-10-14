from tensorflow.keras.utils import to_categorical
from models.NNGPU import NeuralNetwork
from models.NPGPU import NPNeuralNetwork
from Stats import ( process_and_save_results, plot_final_metrics)
import numpy as np
import os

#datasets
from tensorflow.keras.datasets import (
    mnist, fashion_mnist, cifar10
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

def load_CIFAR_adv():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train_img = X_train.reshape(-1, 32, 32, 3)
    X_test_img = X_test.reshape(-1, 32, 32, 3)
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])
    X_train_img = (X_train_img / 255.0 - mean) / std
    X_test_img = (X_test_img / 255.0 - mean) / std
    X_train = X_train_img.reshape(-1, 3072)
    X_test = X_test_img.reshape(-1, 3072)
    y_train = to_categorical(y_train, 10)
    y_test_orig = y_test.flatten() if y_test.ndim > 1 else y_test
    y_test = to_categorical(y_test, 10)
    
    return X_train, y_train, X_test, y_test, y_test_orig

initial_test = 21
final_test = 26
epochs = [50, 80, 120, 300]
architectures = [
    [784, 256, 128, 10],
    [784, 256, 128, 10],
    [3072, 512, 256, 128, 10],
    [3072, 512, 256, 128, 100]
]


def handle_test_and_train(i, j, X_train, y_train, X_test, y_test, y_test_orig):
    e = epochs[i]
    architecture = architectures[i]
    idn = i*10 + j
    print(f"Training standard neural network - IDN: {idn}...")
    std_model = NeuralNetwork(architecture)
    #std_model.train(X_train, y_train, epochs=e)
    #std_model.save_model(idn)
    #std_model.load_model(idn)

    print(f"\nTraining neuroplastic neural network - IDN: {idn}...")
    np_model = NPNeuralNetwork(architecture)
    np_model.train(X_train, y_train, X_test, y_test, epochs=e)
    np_model.save_model(idn)
    #np_model.load_model(idn)

    #process_and_save_results(idn, std_model, np_model, X_test, y_test, y_test_orig)


#main loop
def main():
    ensure_directories()    
    for i in range(1, 2):
        if i == 0:
            X_train, y_train, X_test, y_test, y_test_orig = load_data_MNIST()
            print(f"\n\n\nTRAINING ON MNIST FOR {final_test - initial_test} TESTS \n\n\n")
        elif i == 1:
            X_train, y_train, X_test, y_test, y_test_orig = load_data_fashion()
            print(f"\n\n\nTRAINING ON FASHION MNIST FOR {final_test - initial_test} TESTS \n\n\n")
        elif i == 2:
            X_train, y_train, X_test, y_test, y_test_orig = load_data_CIFAR10()
            print(f"\n\n\nTRAINING ON CIFAR FOR {final_test - initial_test} TESTS \n\n\n")
        else:
            X_train, y_train, X_test, y_test, y_test_orig = load_CIFAR_adv()
            print(f"\n\n\nTRAINING ON CIFAR(advanced) FOR {final_test - initial_test} TESTS \n\n\n")

        for j in range(initial_test, final_test):
            handle_test_and_train(i, j, X_train, y_train, X_test, y_test, y_test_orig)

        print("\n\n\nTRAIN AND TEST LOOP FINISHED \n\n\n")

    plot_final_metrics()

if __name__ == "__main__":
    main()