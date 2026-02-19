from models.CNN import train_test_CNN, CNN_test_only, preprocess
from models.CNN_module import ConvPoolModule
from models.NNGPU import NeuralNetwork
from models.NPGPU import NPNeuralNetwork
from tensorflow.keras.datasets import cifar10

(X_train_np, y_train_np), (X_test_np, y_test_np) = cifar10.load_data()
X_train, y_train = preprocess(X_train_np, y_train_np)
X_test,  y_test  = preprocess(X_test_np,  y_test_np)
TEST_BATCH = 256
for idx in range(0, 10):
    cpm_name = 'artifacts/conv_'
    cpm_name_1 = cpm_name + 'nn_' + str(idx) + '.npz'
    cpm_name_2 = cpm_name + 'np_' + str(idx) + '.npz'
    fc_name_1 = 'nn_' + str(idx)
    fc_name_2 = 'np_' + str(idx)
    cpm = ConvPoolModule([
        {"in_channels":  3, "out_channels":  32, "kernel_size": 3, "pad": 1, "pool_size": 2},
        {"in_channels": 32, "out_channels":  64, "kernel_size": 3, "pad": 1, "pool_size": 2},
        {"in_channels": 64, "out_channels": 128, "kernel_size": 3, "pad": 1, "pool_size": 2},
    ])
    cpm.load(path=cpm_name_1) 
    flat_size = cpm.output_size(32, 32)
    fc = NeuralNetwork([flat_size, 256, 128, 10])
    fc.load_model(fc_name_1)
    CNN_test_only(cpm, fc, X_test, y_test, TEST_BATCH)
    cpm.load(path=cpm_name_2) 
    flat_size = cpm.output_size(32, 32)
    fc = NPNeuralNetwork([flat_size, 256, 128, 10])
    fc.load_model(fc_name_2)
    CNN_test_only(cpm, fc, X_test, y_test, TEST_BATCH)