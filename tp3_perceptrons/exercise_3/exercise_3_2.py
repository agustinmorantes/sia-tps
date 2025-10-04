import numpy as np
from src.multi_layer_perceptron import MultiLayerPerceptron, tanh

data = np.loadtxt("./resources/TP3-ej3-digitos.txt", dtype=int)

X = data.reshape(-1, 35)

X = X * 2 - 1

n_digits = 10
n_repeats = X.shape[0] // n_digits
digit_labels = np.tile(np.arange(n_digits), n_repeats)
Y = np.array([1 if d % 2 else -1 for d in digit_labels]).reshape(-1, 1)

n_inputs = X.shape[1]
layer_sizes = [n_inputs, 15, 1]

mlp = MultiLayerPerceptron(
    layer_sizes=layer_sizes,
    activation=tanh,
    eta=0.05
)

mlp.train(X, Y, epochs=5000, epsilon=1e-4)

print("\n--- Resultados: Clasificación de Paridad ---")
correctos = 0
for x_sample, y_sample in zip(X, Y):
    output = mlp.predict(x_sample.reshape(1, -1))
    pred = 1 if output[0][0] > 0 else -1
    print(f"Esperado: {y_sample[0]}, Predicción: {pred}, Salida red: {output[0][0]:.6f}")
    if pred == y_sample[0]:
        correctos += 1

print(f"\nPrecisión total: {correctos}/{len(Y)} = {correctos / len(Y) * 100:.2f}%")
