from tensorflow.keras.utils import to_categorical
from models.NPGPU import NPNeuralNetwork

#datasets
from tensorflow.keras.datasets import (
    fashion_mnist, cifar10
) 

from main import load_data_CIFAR100_stan, load_data_CIFAR10, load_data_fashion

no_of_test = 5
epochs = [60, 120, 70]
architectures = [
    [784, 256, 128, 10],
    [3072, 512, 256, 128, 10],
    [3072, 512, 256, 128, 100]
]

def handle_ablation_testing(i, j, X_train, y_train, X_test, y_test, y_test_orig):
    e = epochs[i]
    architecture = architectures[i]
    idn = i*100 + j
    print(f"Training NPNN with only Connection + Prunning Neurogenesis - IDN: {idn}...")
    np_model_1 = NPNeuralNetwork(architecture, en_hebbian=False, en_adaptive_lr=False)
    np_model_1.train(X_train, y_train, epochs=e)
    np_model_1.save_model(1000 + idn)

    print(f"\nTraining NPNN with only Adaptive Learning Rate - IDN: {idn}...")
    np_model_2 = NPNeuralNetwork(architecture, en_hebbian=False, en_plasticity=False)
    np_model_2.train(X_train, y_train, epochs=e)
    np_model_2.save_model(2000 + idn)

    print(f"\nTraining NPNN with Hebbian Updates - IDN: {idn}...")
    np_model_3 = NPNeuralNetwork(architecture, en_adaptive_lr=False, en_plasticity=False)
    np_model_3.train(X_train, y_train, epochs=e)
    np_model_3.save_model(3000 + idn)

def main():
    for i in range(0, 1):
        if i == 0:
            X_train, y_train, X_test, y_test, y_test_orig = load_data_fashion()
            print(f"\n\n\n TRAINING ON Fashion MNIST FOR {no_of_test} TESTS \n\n\n")
        elif i == 1:
            X_train, y_train, X_test, y_test, y_test_orig = load_data_CIFAR10()
            print(f"\n\n\n TRAINING ON CIFAR FOR {no_of_test} TESTS \n\n\n")
        else:
            X_train, y_train, X_test, y_test, y_test_orig = load_data_CIFAR100_stan()
            print(f"\n\n\n TRAINING ON CIFAR100 FOR {no_of_test} TESTS \n\n\n")

        for j in range(no_of_test):
            handle_ablation_testing(i, j, X_train, y_train, X_test, y_test, y_test_orig)

    print("\n\n\n TRAIN AND TEST LOOP FINISHED \n\n\n")

if __name__ == "__main__":
    main()