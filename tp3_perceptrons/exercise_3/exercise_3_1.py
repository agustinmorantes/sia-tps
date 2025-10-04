from src.multi_layer_perceptron import MultiLayerPerceptron, sigmoid, tanh
import numpy as np

X = np.array([
    [-1,  1],
    [ 1, -1],
    [-1, -1],
    [ 1,  1]
])

Y = np.array([
    [ 1],
    [ 1],
    [-1],
    [-1]
])

layer_sizes = [2, 2, 1]

mlp = MultiLayerPerceptron(
    layer_sizes=layer_sizes,
    activation=tanh,
    eta=0.05,
    optimizer='adam'
)

mlp.train(X, Y, epochs=10000, epsilon=1e-5)

print("\n--- Resultados XOR ---")
for x, y in zip(X, Y):
    output = mlp.predict(x.reshape(1, -1))
    print(f"Entrada: {x}, Esperado: {y[0]}, Salida: {output[0][0]:.6f}")
