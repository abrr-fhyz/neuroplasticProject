import numpy as np

class NeuralNetwork:
    def __init__(self, layers, gpu = True):
        self.filename = "artifacts/model_"

        if gpu:
            global np
            try:
                import cupy as np
                GPU_AVAILABLE = True
                print("GPU available")
            except ImportError:
                GPU_AVAILABLE = False
                print("GPU not available")

        self.layers = layers
        self.weights = []
        self.biases = []
        self.no_of_layers = len(self.layers)

        for i in range(self.no_of_layers-1):
            self.weights.append(np.random.randn(self.layers[i], self.layers[i+1]) * 0.1)
            self.biases.append(np.zeros((1, self.layers[i+1])))

        self.acc_stat = []
        self.loss_stat = []

    @staticmethod
    def sigmoid(x):
        return 1 / (1+np.exp(-x))

    @staticmethod
    def sigmoid_dydx(x):
        sx = NeuralNetwork.sigmoid(x)
        return sx * (1-sx)

    @staticmethod
    def cross_entropy_loss(y_true, y_pred):
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1-eps)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis = 1))

    @staticmethod
    def cross_entropy_dydx(y_true, y_pred):
        return y_pred - y_true

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis = 1, keepdims = True))
        return e_x / np.sum(e_x, axis = 1, keepdims = True)

    def feed_forward(self, X):
        activations = [X]
        zs = []

        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            zs.append(z)
            if i == len(self.weights)-1:
                a = NeuralNetwork.softmax(z)
            else:
                a = NeuralNetwork.sigmoid(z)
            activations.append(a)

        return activations, zs

    def back_propagate(self, X, y, l_r):
        activations, zs = self.feed_forward(X)
        delta = NeuralNetwork.cross_entropy_dydx(y, activations[-1])
        nabla_w = [0] * (self.no_of_layers-1)
        nabla_b = [0] * (self.no_of_layers-1)
        nabla_w[-1] = np.dot(activations[-2].T, delta)
        nabla_b[-1] = np.sum(delta, axis = 0, keepdims = True)

        for i in range(2, self.no_of_layers):
            z = zs[-i]
            sp = NeuralNetwork.sigmoid_dydx(z)
            delta = np.dot(delta, self.weights[-i+1].T) * sp
            nabla_w[-i] = np.dot(activations[-i-1].T, delta)
            nabla_b[-i] = np.sum(delta, axis = 0, keepdims = True)

        for i in range(len(self.weights)):
            self.weights[i] -= l_r * nabla_w[i]
            self.biases[i] -= l_r * nabla_b[i]

    def predict(self, X):
        X = np.asarray(X)
        activations, temp = self.feed_forward(X)
        return activations[-1]

    def train(self, X, y, epochs = 1000, l_r = 0.001, batch_size = 32):
        X = np.asarray(X)
        y = np.asarray(y)
        n = X.shape[0]

        for epoch in range(epochs):
            indices = np.arange(n)
            np.random.shuffle(indices)
            X, y  = X[indices], y[indices]
            for i in range(0, n, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                self.back_propagate(X_batch, y_batch, l_r)

            y_pred = self.predict(X)
            loss = NeuralNetwork.cross_entropy_loss(y, y_pred)
            predicted = np.argmax(y_pred, axis = 1)
            true = np.argmax(y, axis = 1)
            acc = np.mean(predicted == true)
            print(f"Epoch {epoch+1}, Loss: {loss:.5f}, Accuracy: {acc * 100:.2f}%")
            self.acc_stat.append(float(acc))
            self.loss_stat.append(float(loss))

    def get_stats(self):
        return self.acc_stat, self.loss_stat

    def save_model(self, idn):
        import numpy as numpy_lib
        save_dict = {
            "layers": self.layers,
            "accuracy": self.acc_stat,
            "loss": self.loss_stat
        }
        for idx, w in enumerate(self.weights):
            save_dict[f"weight_{idx}"] = np.asnumpy(w) if GPU_AVAILABLE else w
        for idx, b in enumerate(self.biases):
            save_dict[f"bias_{idx}"] = np.asnumpy(b) if GPU_AVAILABLE else b
        modelName = f"{self.filename}{idn}.npz"
        numpy_lib.savez(modelName, **save_dict)
        print("NN model saved successfully as " + modelName)

    def load_model(self, idn):
        import numpy as numpy_lib
        modelName = f"{self.filename}{idn}.npz"
        data = numpy_lib.load(modelName, allow_pickle=True)
        self.layers = list(data["layers"])
        self.no_of_layers = len(self.layers)
        self.weights = [np.asarray(data[f"weight_{i}"]) for i in range(self.no_of_layers - 1)]
        self.biases = [np.asarray(data[f"bias_{i}"]) for i in range(self.no_of_layers - 1)]
        self.acc_stat = list(data["accuracy"])
        self.loss_stat = list(data["loss"])