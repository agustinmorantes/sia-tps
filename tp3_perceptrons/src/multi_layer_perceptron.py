import numpy as np

class ActivationFunction:
    def __init__(self, fn, derivative):
        self.fn = fn
        self.derivative = derivative

sigmoid = ActivationFunction(
    fn=lambda x: 1 / (1 + np.exp(-x)),
    derivative=lambda y: y * (1 - y)
)

tanh = ActivationFunction(
    fn=np.tanh,
    derivative=lambda y: 1 - y**2
)

class MultiLayerPerceptron:
    def __init__(self, layer_sizes, activation="tanh", eta=0.05, alpha=0.9, optimizer='sgd', batch_size=None, seed=123):
        if seed is not None:
            np.random.seed(seed)

        if activation == "tanh":
            self.activation = tanh
        elif activation == "sigmoid":
            self.activation = sigmoid
        else:
            raise ValueError("Unsupported activation function. Use 'tanh' or 'sigmoid'.")

        self.eta = eta
        self.alpha = alpha  # solo se usa para momentum
        self.optimizer = optimizer
        self.batch_size = batch_size  # None = batch, 1 = online, >1 = minibatch
        self.n_layers = len(layer_sizes) - 1

        # Inicialización de pesos y bias
        self.weights = [np.random.uniform(-0.5, 0.5, (layer_sizes[i], layer_sizes[i+1])) for i in range(self.n_layers)]
        self.biases = [np.random.uniform(-0.5, 0.5, (1, layer_sizes[i+1])) for i in range(self.n_layers)]

        # Variables auxiliares
        self.deltaW_prev = [np.zeros_like(w) for w in self.weights]
        self.deltaB_prev = [np.zeros_like(b) for b in self.biases]

        # Adam
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.t = 0

    def forward(self, X):
        activations = [X]
        for i in range(self.n_layers):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = self.activation.fn(z)
            activations.append(a)
        return activations

    def backward(self, activations, Y):
        deltas = []

        delta_L = (activations[-1] - Y) * self.activation.derivative(activations[-1])
        deltas.insert(0, delta_L)
        
        for i in reversed(range(self.n_layers - 1)):
            delta = np.dot(deltas[0], self.weights[i+1].T) * self.activation.derivative(activations[i+1])
            deltas.insert(0, delta)

        return deltas

    def update_weights(self, activations, deltas):
        self.t += 1
        for i in range(self.n_layers):
            grad_w = np.dot(activations[i].T, deltas[i])
            grad_b = np.sum(deltas[i], axis=0, keepdims=True)

            if self.optimizer == 'sgd':
                # Descenso de gradiente clásico
                self.weights[i] -= self.eta * grad_w
                self.biases[i]  -= self.eta * grad_b

            elif self.optimizer == 'momentum':
                # Actualización con Momentum
                deltaW = self.eta * grad_w + self.alpha * self.deltaW_prev[i]
                deltaB = self.eta * grad_b + self.alpha * self.deltaB_prev[i]

                self.weights[i] -= deltaW
                self.biases[i]  -= deltaB

                self.deltaW_prev[i] = deltaW
                self.deltaB_prev[i] = deltaB

            elif self.optimizer == 'adam':
                # Actualización con Adam
                beta1, beta2, eps = 0.9, 0.999, 1e-8
                self.m_w[i] = beta1 * self.m_w[i] + (1 - beta1) * grad_w
                self.v_w[i] = beta2 * self.v_w[i] + (1 - beta2) * (grad_w**2)
                m_hat_w = self.m_w[i] / (1 - beta1**self.t)
                v_hat_w = self.v_w[i] / (1 - beta2**self.t)
                self.weights[i] -= self.eta * m_hat_w / (np.sqrt(v_hat_w) + eps)

                self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * grad_b
                self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * (grad_b**2)
                m_hat_b = self.m_b[i] / (1 - beta1**self.t)
                v_hat_b = self.v_b[i] / (1 - beta2**self.t)
                self.biases[i] -= self.eta * m_hat_b / (np.sqrt(v_hat_b) + eps)

    def train(self, X, Y, epochs=10000, epsilon=1e-5):
        n_samples = X.shape[0]

        if self.batch_size is None:
            # Batch mode
            effective_batch_size = n_samples
        else:
            # Online (batch_size=1) o Minibatch (batch_size>1)
            effective_batch_size = self.batch_size

        for epoch in range(epochs):
            # Shuffle del dataset al inicio de cada época
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]

            epoch_mse = 0
            n_batches = 0

            # Procesar por batches
            for start_idx in range(0, n_samples, effective_batch_size):
                end_idx = min(start_idx + effective_batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                Y_batch = Y_shuffled[start_idx:end_idx]

                # Forward pass
                activations = self.forward(X_batch)

                # Backward pass
                deltas = self.backward(activations, Y_batch)

                # Update weights
                self.update_weights(activations, deltas)

                # Calcular MSE del batch
                batch_mse = 0.5 * np.mean((Y_batch - activations[-1])**2)
                epoch_mse += batch_mse
                n_batches += 1

            # MSE promedio de la época
            epoch_mse /= n_batches

            if epoch % 500 == 0:
                print(f"Época {epoch}, MSE = {epoch_mse:.6f}")

            if epoch_mse < epsilon:
                print(f"Convergencia alcanzada en época {epoch}, MSE={epoch_mse:.6f}")
                break

    def predict(self, X):
        return self.forward(X)[-1]