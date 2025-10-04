import numpy as np
from src.multi_layer_perceptron import MultiLayerPerceptron, sigmoid, tanh

data = np.loadtxt("./resources/TP3-ej3-digitos.txt", dtype=int)
X = data.reshape(-1, 35)
X = X * 2 - 1             # Normalizamos a [-1, 1]

n_digits = 10
n_repeats = X.shape[0] // n_digits
digit_labels = np.tile(np.arange(n_digits), n_repeats)

Y = np.eye(n_digits)[digit_labels]

layer_sizes = [35, 50, 10]

mlp = MultiLayerPerceptron(
    layer_sizes=layer_sizes,
    activation=sigmoid,
    eta=0.05,
    optimizer='adam'
)

print("Entrenando red para reconocimiento de dígitos...")
mlp.train(X, Y, epochs=10000, epsilon=1e-5)

# Sin Ruido
predictions = mlp.predict(X)
predicted_digits = np.argmax(predictions, axis=1)
accuracy = np.mean(predicted_digits == digit_labels)

print(f"\nAccuracy final: {accuracy*100:.2f}%")

for real, pred, out in zip(digit_labels, predicted_digits, predictions):
    print(f"Dígito real: {real}, Predicción: {pred}, Valores red: {np.round(out, 6)}")

# Con Ruido
print("\nProbando con ruido en las imágenes...")

# Agregar ruido gaussiano a las entradas
noise_level = 0.2
X_noisy = X + np.random.normal(0, noise_level, X.shape)
X_noisy = np.clip(X_noisy, -1, 1)

predictions_noisy = mlp.predict(X_noisy)
predicted_noisy = np.argmax(predictions_noisy, axis=1)
accuracy_noisy = np.mean(predicted_noisy == digit_labels)

print(f"Accuracy con ruido (σ={noise_level}): {accuracy_noisy*100:.2f}%")

for real, pred, out in zip(digit_labels, predicted_noisy, predictions_noisy):
    print(f"Dígito real: {real}, Predicción: {pred}, Valores red: {np.round(out, 6)}")
