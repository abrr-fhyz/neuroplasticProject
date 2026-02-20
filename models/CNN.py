import cupy as cp
import numpy as np
from models.CNN_module import ConvPoolModule
from models.NNGPU import NeuralNetwork
from models.NPGPU import NPNeuralNetwork
from tensorflow.keras.datasets import cifar10, cifar100

def preprocess(X, y, num_classes=10):
    X = X.astype(np.float32) / 255.0            
    X = X.transpose(0, 3, 1, 2)                 
    y = y.flatten()
    y_oh = np.zeros((len(y), num_classes), dtype=np.float32)
    y_oh[np.arange(len(y)), y] = 1.0
    return cp.asarray(X), cp.asarray(y_oh)

def fc_input_gradient(fc_model, X_flat, y):
    activations, zs = fc_model.feed_forward(X_flat)
    delta = NeuralNetwork.cross_entropy_dydx(y, activations[-1])  

    for i in range(2, fc_model.no_of_layers):
        sp    = NeuralNetwork.sigmoid_dydx(zs[-i])
        delta = cp.dot(delta, fc_model.weights[-i+1].T) * sp

    return cp.dot(delta, fc_model.weights[0].T)

def np_fc_input_gradient(fc_model, X_flat, y):
    activations, zs = fc_model.feed_forward(X_flat)
    delta = NPNeuralNetwork.cross_entropy_dydx(y, activations[-1])

    for i in range(2, fc_model.no_of_layers):
        sp    = NPNeuralNetwork.sigmoid_dydx(zs[-i])
        delta = cp.dot(delta, (fc_model.weights[-i+1] * fc_model.masks[-i+1]).T) * sp

    return cp.dot(delta, (fc_model.weights[0] * fc_model.masks[0]).T)

def nn_CNN_train(cpm, fc, EPOCHS, X_train, y_train, N, BATCH_SIZE, LR_FC, LR_CONV):
    for epoch in range(1, EPOCHS + 1):
        idx = cp.random.permutation(N)
        X_s, y_s = X_train[idx], y_train[idx]

        epoch_loss = 0.0
        epoch_acc  = 0.0
        n_batches  = 0

        for start in range(0, N, BATCH_SIZE):
            Xb = X_s[start : start + BATCH_SIZE]   
            yb = y_s[start : start + BATCH_SIZE]   
            flat = cpm.forward(Xb)                  
            fc.back_propagate(flat, yb, LR_FC)
            d_flat = fc_input_gradient(fc, flat, yb)
            cpm.backward(d_flat, LR_CONV)
            with cp.cuda.Device(0):
                y_pred   = fc.predict(flat)
                loss     = float(NeuralNetwork.cross_entropy_loss(yb, y_pred))
                pred_cls = cp.argmax(y_pred, axis=1)
                true_cls = cp.argmax(yb,     axis=1)
                acc      = float(cp.mean(pred_cls == true_cls))

            epoch_loss += loss
            epoch_acc  += acc
            n_batches  += 1

        print(f"Epoch {epoch:>3}/{EPOCHS}\tLoss: {epoch_loss/n_batches:.4f}\tTrain Acc: {epoch_acc/n_batches*100:.2f}%")
    return cpm, fc

def np_CNN_train(cpm, fc, EPOCHS, X_train, y_train, N, BATCH_SIZE, LR_CONV):
    for epoch in range(1, EPOCHS + 1):
        idx = cp.random.permutation(N)
        X_s, y_s = X_train[idx], y_train[idx]

        epoch_loss = 0.0
        epoch_acc  = 0.0
        n_batches  = 0

        for start in range(0, N, BATCH_SIZE):
            Xb = X_s[start : start + BATCH_SIZE]  
            yb = y_s[start : start + BATCH_SIZE]   
            flat = cpm.forward(Xb)                  
            fc.epoch_count = epoch                
            fc.back_propagate(flat, yb)
            d_flat = np_fc_input_gradient(fc, flat, yb)
            cpm.backward(d_flat, LR_CONV)
            y_pred   = fc.predict(flat)
            loss     = float(NPNeuralNetwork.cross_entropy_loss(yb, y_pred))
            pred_cls = cp.argmax(y_pred, axis=1)
            true_cls = cp.argmax(yb,     axis=1)
            acc      = float(cp.mean(pred_cls == true_cls))

            epoch_loss += loss
            epoch_acc  += acc
            n_batches  += 1

        avg_loss = epoch_loss / n_batches
        avg_acc  = epoch_acc  / n_batches
        fc.plasticity_update(avg_loss)

        active_pct = float(cp.mean(cp.stack([m.mean() for m in fc.masks]))) * 100
        print(f"Epoch {epoch:>3}/{EPOCHS}\tLoss: {avg_loss:.4f}\tTrain Acc: {avg_acc*100:.2f}%\tLR: {fc.lr:.7f}\tActive connections: {active_pct:.1f}%")
    return cpm, fc

def CNN_test(cpm, fc, X_test, y_test, TEST_BATCH, sl, fc_type):
    CNN_test_only(cpm, fc, X_test, y_test, TEST_BATCH)
    model_ext = fc_type + "_" + str(sl)
    cp_name = "artifacts/conv_" + model_ext + ".npz"
    cpm.save(cp_name)
    fc.save_model(model_ext)

def CNN_test_only(cpm, fc, X_test, y_test, TEST_BATCH = 256):
    correct, total = 0, 0

    for start in range(0, X_test.shape[0], TEST_BATCH):
        Xb = X_test[start : start + TEST_BATCH]
        yb = y_test[start : start + TEST_BATCH]

        flat     = cpm.forward(Xb)
        y_pred   = fc.predict(flat)
        pred_cls = cp.argmax(y_pred, axis=1)
        true_cls = cp.argmax(yb,     axis=1)
        correct += int(cp.sum(pred_cls == true_cls))
        total   += Xb.shape[0]

    print(f"\nTest Accuracy: {correct/total*100:.2f}%  ({correct}/{total})")

def train_test_CNN(fc_type, sl, EPOCHS = 25, BATCH_SIZE = 64, LR_CONV = 0.001, LR_FC = 0.001, TEST_BATCH = 256, dataset = 'cifar10'):
    if dataset == 'cifar10':
        (X_train_np, y_train_np), (X_test_np, y_test_np) = cifar10.load_data()
        X_train, y_train = preprocess(X_train_np, y_train_np)
        X_test,  y_test  = preprocess(X_test_np,  y_test_np)
    elif dataset == 'cifar100':
        (X_train_np, y_train_np), (X_test_np, y_test_np) = cifar100.load_data()
        X_train, y_train = preprocess(X_train_np, y_train_np, num_classes=100)
        X_test,  y_test  = preprocess(X_test_np,  y_test_np, num_classes=100)
    else:
        print("Sorry")
        return
    N = X_train.shape[0]
    cpm = ConvPoolModule([
        {"in_channels":  3, "out_channels":  32, "kernel_size": 3, "pad": 1, "pool_size": 2},
        {"in_channels": 32, "out_channels":  64, "kernel_size": 3, "pad": 1, "pool_size": 2},
        {"in_channels": 64, "out_channels": 128, "kernel_size": 3, "pad": 1, "pool_size": 2},
    ])
    flat_size = cpm.output_size(32, 32)
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")
    print(f"Flat feature size into FC: {flat_size}")
    
    if dataset == 'cifar10':
        arch = [flat_size, 256, 128, 10]
    elif dataset == 'cifar100':
        arch = [flat_size, 256, 128, 100]
    else:
        print("Sorry")
        return

    if(fc_type == 'nn'):
        fc = NeuralNetwork(arch, gpu=True)
        cpm, fc = nn_CNN_train(cpm, fc, EPOCHS, X_train, y_train, N, BATCH_SIZE, LR_FC, LR_CONV)
        CNN_test(cpm, fc, X_test, y_test, TEST_BATCH, sl, fc_type)
    elif(fc_type == 'np'):
        fc = NPNeuralNetwork(
            layers         = arch,
            initial_lr     = 0.001,
            en_hebbian     = True,
            en_adaptive_lr = True,
            en_plasticity  = True,
            gpu            = True,
        )
        cpm, fc = np_CNN_train(cpm, fc, EPOCHS, X_train, y_train, N, BATCH_SIZE, LR_CONV)
        CNN_test(cpm, fc, X_test, y_test, TEST_BATCH, sl, fc_type)
    else:
        print("Sorry")

def special_test(idx, EPOCHS = 50, BATCH_SIZE = 64, LR_CONV = 0.001):
    (X_train_np, y_train_np), (X_test_np, y_test_np) = cifar100.load_data()
    X_train, y_train = preprocess(X_train_np, y_train_np, num_classes=100)
    X_test,  y_test  = preprocess(X_test_np,  y_test_np, num_classes=100)
    N = X_train.shape[0]
    cpm = ConvPoolModule([
        {"in_channels":  3, "out_channels":  32, "kernel_size": 3, "pad": 1, "pool_size": 2},
        {"in_channels": 32, "out_channels":  64, "kernel_size": 3, "pad": 1, "pool_size": 2},
        {"in_channels": 64, "out_channels": 128, "kernel_size": 3, "pad": 1, "pool_size": 2},
    ])
    cpm.load(f"artifacts/conv_np_{idx}.npz")
    fc = NPNeuralNetwork([2048, 256, 128, 100])
    for epoch in range(1, EPOCHS + 1):
        idx = cp.random.permutation(N)
        X_s, y_s = X_train[idx], y_train[idx]
        epoch_loss = 0.0
        epoch_acc  = 0.0
        n_batches  = 0

        for start in range(0, N, BATCH_SIZE):
            Xb = X_s[start : start + BATCH_SIZE]  
            yb = y_s[start : start + BATCH_SIZE]   
            flat = cpm.forward(Xb)                  
            fc.epoch_count = epoch                
            fc.back_propagate(flat, yb)
            d_flat = np_fc_input_gradient(fc, flat, yb)
            cpm.backward(d_flat, LR_CONV)
            y_pred   = fc.predict(flat)
            loss     = float(NPNeuralNetwork.cross_entropy_loss(yb, y_pred))
            pred_cls = cp.argmax(y_pred, axis=1)
            true_cls = cp.argmax(yb,     axis=1)
            acc      = float(cp.mean(pred_cls == true_cls))

            epoch_loss += loss
            epoch_acc  += acc
            n_batches  += 1

        avg_loss = epoch_loss / n_batches
        avg_acc  = epoch_acc  / n_batches
        fc.plasticity_update(avg_loss)

        active_pct = float(cp.mean(cp.stack([m.mean() for m in fc.masks]))) * 100
        print(f"Epoch {epoch:>3}/{EPOCHS}\tLoss: {avg_loss:.4f}\tTrain Acc: {avg_acc*100:.2f}%\tLR: {fc.lr:.7f}\tActive connections: {active_pct:.1f}%")
        if epoch%5 == 0:
            CNN_test_only(cpm, fc, X_test, y_test)
    CNN_test(cpm, fc, X_test, y_test, TEST_BATCH = 256, sl = idx, fc_type = 'np')