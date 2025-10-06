import json
import os

from src.multi_layer_perceptron import MultiLayerPerceptron, sigmoid, tanh
import numpy as np

def main(config: dict):
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


    mlp = MultiLayerPerceptron(**config)

    mlp.train(X, Y, epochs=10000, epsilon=1e-5)

    print("\n--- Resultados XOR ---")
    for x, y in zip(X, Y):
        output = mlp.predict(x.reshape(1, -1))
        print(f"Entrada: {x}, Esperado: {y[0]}, Salida: {output[0][0]:.6f}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, "config/exercise_3_1.json"), 'r') as f:
        config = json.load(f)["config"]
    main(config)
