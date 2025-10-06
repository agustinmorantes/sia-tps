import json
import os

import numpy as np
from src.multi_layer_perceptron import MultiLayerPerceptron, sigmoid, tanh

def main(config: dict):
    np.set_printoptions(suppress=True, precision=6, floatmode='fixed')

    txt_path = os.path.join(os.path.dirname(__file__), "resources/TP3-ej3-digitos.txt")
    data = np.loadtxt(txt_path, dtype=int)

    X = data.reshape(-1, 35)
    X = X.astype(float)

    n_digits = 10
    n_repeats = X.shape[0] // n_digits
    digit_labels = np.tile(np.arange(n_digits), n_repeats)

    Y = np.eye(n_digits)[digit_labels]

    mlp = MultiLayerPerceptron(**config)

    print("Entrenando red para reconocimiento de dígitos...")
    mlp.train(X, Y, epochs=20000, epsilon=1e-6)

    # Sin Ruido
    predictions = mlp.predict(X)
    predicted_digits = np.argmax(predictions, axis=1)
    accuracy = np.mean(predicted_digits == digit_labels)

    print(f"\nAccuracy final: {accuracy*100:.2f}%")

    for real, pred, out in zip(digit_labels, predicted_digits, predictions):
        print(f"Dígito real: {real}, Predicción: {pred}, Valores red: {np.round(out, 6)}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, "config/exercise_3_3.json"), 'r') as f:
        config = json.load(f)["config"]
    main(config)
