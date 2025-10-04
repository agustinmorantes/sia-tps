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
    def __init__(self, layer_sizes, activation=tanh, eta=0.05, alpha=0.9, optimizer='sgd'):
        self.activation = activation
        self.eta = eta
        self.alpha = alpha  # solo se usa para momentum
        self.optimizer = optimizer
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
        for epoch in range(epochs):
            activations = self.forward(X)
            deltas = self.backward(activations, Y)
            self.update_weights(activations, deltas)

            mse = 0.5 * np.mean((Y - activations[-1])**2)

            if epoch % 500 == 0:
                print(f"Época {epoch}, MSE = {mse:.6f}")
                
            if mse < epsilon:
                print(f"Convergencia alcanzada en época {epoch}, MSE={mse:.6f}")
                break

    def predict(self, X):
        return self.forward(X)[-1]