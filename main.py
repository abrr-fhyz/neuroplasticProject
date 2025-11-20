from tensorflow.keras.utils import to_categorical
from models.NNGPU import NeuralNetwork
from models.Prototype import NPNeuralNetwork
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from Stats import ( process_and_save_results, plot_final_metrics)
import numpy as np
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

def load_data_CIFAR10_sub(train_per_class = 1000, test_per_class = 250, random_state = 42):
    X_train, y_train_1, X_test, y_test_1, _ = load_data_CIFAR10()
    y_train = np.argmax(y_train_1, axis=1)
    y_test = np.argmax(y_test_1, axis=1)
    train_idx = []
    test_idx = []
    np.random.seed(random_state)
    for class_id in range(10):
        class_train = np.where(y_train == class_id)[0]
        class_test = np.where(y_test == class_id)[0]
        train_idx.extend(np.random.choice(class_train, train_per_class, replace=False))
        test_idx.extend(np.random.choice(class_test, test_per_class, replace=False))
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    return X_train[train_idx], y_train_1[train_idx], X_test[test_idx], y_test_1[test_idx], y_test[test_idx]

def load_data_CIFAR10_stan():
    X_train, y_train, X_test, y_test, y_test_orig = load_data_CIFAR10()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, y_train, X_test, y_test, y_test_orig

def load_data_CIFAR100():
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()
    X_test = X_test.reshape(-1, 3072) / 255.0
    X_train = X_train.reshape(-1, 3072) / 255.0
    y_train = to_categorical(y_train, 100)
    y_test_orig = y_test.flatten() if y_test.ndim > 1 else y_test
    y_test = to_categorical(y_test, 100)   
    return X_train, y_train, X_test, y_test, y_test_orig

def load_data_CIFAR100_stan():
    X_train, y_train, X_test, y_test, y_test_orig = load_data_CIFAR100()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, y_train, X_test, y_test, y_test_orig


initial_test = 0
final_test = 10
epochs = [50, 60, 120, 60, 20, 70]
architectures = [
    [784, 256, 128, 10],
    [784, 256, 128, 10],
    [3072, 512, 256, 128, 10],
    [3072, 512, 256, 128, 10],
    [3072, 512, 256, 128, 10],
    [3072, 512, 256, 128, 100]
]


def handle_test_and_train(i, j, X_train, y_train, X_test, y_test, y_test_orig):
    e = epochs[i]
    architecture = architectures[i]
    idn = i*10 + j
    print(f"Training standard neural network - IDN: {idn}...")
    std_model = NeuralNetwork(architecture)
    std_model.train(X_train, y_train, epochs=e)
    std_model.save_model(idn)
    #std_model.load_model(idn)

    print(f"\nTraining neuroplastic neural network - IDN: {idn}...")
    np_model = NPNeuralNetwork(architecture)
    np_model.train(X_train, y_train, epochs=e)
    np_model.save_model(idn)
    #np_model.load_model(idn)

    process_and_save_results(idn, std_model, np_model, X_test, y_test, y_test_orig)


#main loop
def main():
    ensure_directories()    
    for i in range(0, 6):
        if i == 0:
            X_train, y_train, X_test, y_test, y_test_orig = load_data_MNIST()
            print(f"\n\n\nTRAINING ON MNIST FOR {final_test - initial_test} TESTS \n\n\n")
        elif i == 1:
            X_train, y_train, X_test, y_test, y_test_orig = load_data_fashion()
            print(f"\n\n\nTRAINING ON FASHION MNIST FOR {final_test - initial_test} TESTS \n\n\n")
        elif i == 2:
            X_train, y_train, X_test, y_test, y_test_orig = load_data_CIFAR10()
            print(f"\n\n\nTRAINING ON CIFAR FOR {final_test - initial_test} TESTS \n\n\n")
        elif i == 3:
            X_train, y_train, X_test, y_test, y_test_orig = load_data_CIFAR10_stan()
            print(f"\n\n\nTRAINING ON CIFAR(Standard) FOR {final_test - initial_test} TESTS \n\n\n")
        elif i == 4:
            X_train, y_train, X_test, y_test, y_test_orig = load_data_CIFAR10()
            print(f"\n\n\nTRAINING ON CIFAR(Plateau) FOR {final_test - initial_test} TESTS \n\n\n")
        else:
            X_train, y_train, X_test, y_test, y_test_orig = load_data_CIFAR100_stan()
            print(f"\n\n\nTRAINING ON CIFAR100 FOR {final_test - initial_test} TESTS \n\n\n")

        for j in range(initial_test, final_test):
            handle_test_and_train(i, j, X_train, y_train, X_test, y_test, y_test_orig)

        print("\n\n\nTRAIN AND TEST LOOP FINISHED \n\n\n")

    plot_final_metrics()

if __name__ == "__main__":
    main()