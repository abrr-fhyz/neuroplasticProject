import numpy as np
gpu_active = False

class NPNeuralNetwork:
    def __init__(self, layers, initial_lr=0.001, gpu = True):
        self.filename = "artifacts/model_np_"
        self.layers = layers
        self.no_of_layers = len(layers)

        if gpu:
            global np
            global gpu_active
            try:
                import cupy as np
                gpu_active = True
                print("GPU available")
            except ImportError:
                print("GPU not available")

        self.improvement_threshold = 0.001 
        self.patience = 5 
        self.patience_counter = 0
        self.lr = initial_lr
        self.lr_min = 1e-6 
        self.lr_max = 0.1  
        self.lr_decay_factor = 0.975 
        self.lr_growth_factor = 1.1
        self.hebbian_rate = 1e-4 
        self.bcm_const = 3000 
        self.plasticity_factor = 1.2
        self.prune_threshold = 1e-4 
        self.requalify_threshold = 1e-2 
        self.plasticity_interval = 3
        self.epoch_count = 0

        self.weights = []
        self.biases = []
        self.masks = [] 
        self.cum_weight_updates = [] 
        self.bcm_thresholds = [] 
        self.prev_loss = None
        self.smoothed_loss = None
        self.clip_value = 3.0

        for i in range(self.no_of_layers-1):
            w = np.random.randn(layers[i], layers[i+1]) * 0.1
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
            self.masks.append(np.ones_like(w))
            self.cum_weight_updates.append(np.zeros_like(w))
            self.bcm_thresholds.append(np.ones((1, layers[i+1])) * 0.1)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_dydx(x):
        sx = NPNeuralNetwork.sigmoid(x)
        return sx * (1 - sx)

    @staticmethod
    def cross_entropy_loss(y_true, y_pred):
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    @staticmethod
    def cross_entropy_dydx(y_true, y_pred):
        return y_pred - y_true

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def feed_forward(self, X):
        activations = [X]
        zs = []
        for i in range(len(self.weights)):
            current_weight = self.weights[i] * self.masks[i]
            z = np.dot(activations[-1], current_weight) + self.biases[i]
            zs.append(z)
            if i == len(self.weights)-1:
                a = NPNeuralNetwork.softmax(z)
            else:
                a = NPNeuralNetwork.sigmoid(z)
            activations.append(a)
        return activations, zs

    def back_propagate(self, X, y):
        activations, zs = self.feed_forward(X)
        delta = NPNeuralNetwork.cross_entropy_dydx(y, activations[-1])
        nabla_w = [None] * (self.no_of_layers-1)
        nabla_b = [None] * (self.no_of_layers-1)
        nabla_w[-1] = np.dot(activations[-2].T, delta)
        nabla_b[-1] = np.sum(delta, axis=0, keepdims=True)

        for i in range(2, self.no_of_layers):
            z = zs[-i]
            sp = NPNeuralNetwork.sigmoid_dydx(z)
            delta = np.dot(delta, (self.weights[-i+1] * self.masks[-i+1]).T) * sp
            nabla_w[-i] = np.dot(activations[-i-1].T, delta)
            nabla_b[-i] = np.sum(delta, axis=0, keepdims=True)

        current_hebbian = self.hebbian_rate
        plasticity_event = (self.epoch_count % self.plasticity_interval == 0) 
        if plasticity_event:
            current_hebbian *= self.plasticity_factor
        for i in range(len(nabla_w)):
            post_squared_avg = np.mean(activations[i+1]**2, axis=0, keepdims=True)
            self.bcm_thresholds[i] = (
                (self.bcm_const - 1) / self.bcm_const * self.bcm_thresholds[i] + 
                1 / self.bcm_const * post_squared_avg
            )  
            bcm_modulation = activations[i+1] * (activations[i+1] - self.bcm_thresholds[i])
            bcm_update = current_hebbian * np.dot(activations[i].T, bcm_modulation)
            nabla_w[i] += bcm_update
            hebbian_term = self.hebbian_rate * np.dot(activations[i].T, activations[i+1])
            nabla_w[i] += hebbian_term

        for i in range(len(nabla_w)):
            nabla_w[i] = np.clip(nabla_w[i], -self.clip_value, self.clip_value)
            nabla_b[i] = np.clip(nabla_b[i], -self.clip_value, self.clip_value)

        for i in range(len(self.weights)):
            update = self.lr * nabla_w[i]
            self.weights[i] -= update * self.masks[i]
            self.biases[i] -= self.lr * nabla_b[i]
            self.cum_weight_updates[i] += np.abs(update)

    def predict(self, X):
        X = np.asarray(X)
        activations, _ = self.feed_forward(X)
        return activations[-1]

    def plasticity_update(self, current_loss):
        smoothing_rate = 0.1
        if self.smoothed_loss is None:
            self.smoothed_loss = current_loss
        else:
            self.smoothed_loss = smoothing_rate * current_loss + (1 - smoothing_rate) * self.smoothed_loss

        if self.epoch_count % self.plasticity_interval == 0:
            for i in range(len(self.weights)):
                prune_candidates = self.cum_weight_updates[i] < self.prune_threshold
                self.masks[i][prune_candidates] = 0
                if (self.prev_loss is not None) and (self.smoothed_loss > self.prev_loss):
                    requalify_candidates = (self.masks[i] == 0) & (np.abs(self.weights[i]) < self.requalify_threshold)
                    self.weights[i][requalify_candidates] = np.random.randn(*self.weights[i][requalify_candidates].shape) * 0.1
                    self.masks[i][requalify_candidates] = 1
                self.cum_weight_updates[i] = np.zeros_like(self.weights[i])

        if self.prev_loss is not None:
            relative_improvement = (self.prev_loss - self.smoothed_loss) / self.prev_loss

            if relative_improvement > self.improvement_threshold:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    self.lr *= self.lr_growth_factor
                    self.patience_counter = 0
            else:
                self.lr *= self.lr_decay_factor
                self.patience_counter = 0

            self.lr = min(max(self.lr, self.lr_min), self.lr_max)
        self.prev_loss = self.smoothed_loss

    def train(self, X, y, epochs=1000, batch_size=32):
        X = np.asarray(X)
        y = np.asarray(y)
        n = X.shape[0]
        
        for epoch in range(epochs):
            self.epoch_count += 1
            indices = np.arange(n)
            np.random.shuffle(indices)
            X, y = X[indices], y[indices]

            for i in range(0, n, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                self.back_propagate(X_batch, y_batch)

            y_pred = self.predict(X)
            loss = NPNeuralNetwork.cross_entropy_loss(y, y_pred)
            predicted = np.argmax(y_pred, axis=1)
            true = np.argmax(y, axis=1)
            acc = np.mean(predicted == true)
            print(f"Epoch {epoch+1}, Loss: {loss:.5f}, Accuracy: {acc*100:.2f}%, LR: {self.lr:.8f}")
            self.plasticity_update(loss)

    def save_model(self, idn):
        import numpy as numpy_lib
        save_dict = {
            "layers": self.layers
        }
        for idx, w in enumerate(self.weights):
            save_dict[f"weight_{idx}"] = np.asnumpy(w) if gpu_active else w
        for idx, b in enumerate(self.biases):
            save_dict[f"bias_{idx}"] = np.asnumpy(b) if gpu_active else b
        for idx, m in enumerate(self.masks):
            save_dict[f"masks_{idx}"] = np.asnumpy(m) if gpu_active else m
        for idx, t in enumerate(self.bcm_thresholds):
            save_dict[f"bcm_threshold_{idx}"] = np.asnumpy(t) if gpu_active else t
        modelName = f"{self.filename}{idn}.npz"
        numpy_lib.savez(modelName, **save_dict)
        print(f"NP model saved successfully as {modelName}")

    def load_model(self, idn):
        import numpy as numpy_lib
        modelName = f"{self.filename}{idn}.npz"
        data = numpy_lib.load(modelName, allow_pickle=True)
        self.layers = list(data["layers"])
        self.no_of_layers = len(self.layers)
        self.weights = [np.asarray(data[f"weight_{i}"]) for i in range(self.no_of_layers - 1)]
        self.biases = [np.asarray(data[f"bias_{i}"]) for i in range(self.no_of_layers - 1)]
        self.masks = [np.asarray(data[f"masks_{i}"]) for i in range(self.no_of_layers - 1)]
        self.bcm_thresholds = [np.asarray(data[f"bcm_threshold_{i}"]) for i in range(self.no_of_layers - 1)]
        print(f"NP model loaded successfully as {modelName}")