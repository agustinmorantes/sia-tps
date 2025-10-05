import numpy as np
from src.multi_layer_perceptron import MultiLayerPerceptron, sigmoid, tanh

np.set_printoptions(suppress=True, precision=6, floatmode='fixed')

data = np.loadtxt("./resources/TP3-ej3-digitos.txt", dtype=int)
X = data.reshape(-1, 35)
X = X.astype(float)

n_digits = 10
n_repeats = X.shape[0] // n_digits
digit_labels = np.tile(np.arange(n_digits), n_repeats)

Y = np.eye(n_digits)[digit_labels]

layer_sizes = [35, 50, 10]

mlp = MultiLayerPerceptron(
    layer_sizes=layer_sizes,
    activation=sigmoid,
    eta=0.02,
    alpha=0.9,
    optimizer='adam',
    batch_size=1 # online
)

print("Entrenando red para reconocimiento de dígitos...")
mlp.train(X, Y, epochs=20000, epsilon=1e-6)

# Sin Ruido
predictions = mlp.predict(X)
predicted_digits = np.argmax(predictions, axis=1)
accuracy = np.mean(predicted_digits == digit_labels)

print(f"\nAccuracy final: {accuracy*100:.2f}%")

for real, pred, out in zip(digit_labels, predicted_digits, predictions):
    print(f"Dígito real: {real}, Predicción: {pred}, Valores red: {np.round(out, 6)}")

