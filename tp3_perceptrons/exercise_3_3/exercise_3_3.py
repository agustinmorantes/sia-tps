import json
import os

import numpy as np
from src.multi_layer_perceptron import MultiLayerPerceptron, sigmoid, tanh

def main(config: dict, experiment_name: str = ""):
    np.set_printoptions(suppress=True, precision=6, floatmode='fixed')

    noise_scale = config['noise_scale'] if 'noise_scale' in config else 0.5
    # remove noise_scale from config to avoid issues in MLP init
    if 'noise_scale' in config:
        del config['noise_scale']


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

    # Verificación con ruido - múltiples corridas
    n_runs = 100
    all_accuracies = []
    all_runs_data = []

    for i in range(n_runs):
        noise = np.random.normal(0, noise_scale, X.shape)
        X_noisy = np.clip(X + noise, 0, 1)
        predictions = mlp.predict(X_noisy)
        predicted_digits = np.argmax(predictions, axis=1)
        accuracy = np.mean(predicted_digits == digit_labels)

        all_accuracies.append(accuracy)
        all_runs_data.append({
            "run": i + 1,
            "accuracy": float(accuracy),
            "predicted_digits": [int(x) for x in predicted_digits],
            "predictions": [list(map(float, out)) for out in predictions]
        })

    # Calcular estadísticas agregadas
    mean_accuracy = float(np.mean(all_accuracies))
    std_accuracy = float(np.std(all_accuracies))
    min_accuracy = float(np.min(all_accuracies))
    max_accuracy = float(np.max(all_accuracies))

    results: dict = {
        "noise_scale": float(noise_scale),
        "n_runs": n_runs,
        "digit_labels": [int(x) for x in digit_labels],
        "aggregate_statistics": {
            "mean_accuracy": mean_accuracy,
            "std_accuracy": std_accuracy,
            "min_accuracy": min_accuracy,
            "max_accuracy": max_accuracy,
            "all_accuracies": [float(x) for x in all_accuracies]
        },
        "individual_runs": all_runs_data
    }

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    output_filepath = os.path.join(results_dir, f"exercise_3_3_{experiment_name}.json")
    with open(output_filepath, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nResultados de {n_runs} corridas con ruido (scale={noise_scale}):")
    print(f"Accuracy promedio: {mean_accuracy*100:.2f}% ± {std_accuracy*100:.2f}%")
    print(f"Accuracy mínimo: {min_accuracy*100:.2f}%")
    print(f"Accuracy máximo: {max_accuracy*100:.2f}%")

    # Mostrar detalles de la mejor corrida
    best_run_idx = np.argmax(all_accuracies)
    best_run = all_runs_data[best_run_idx]
    print(f"\nMejor corrida (#{best_run_idx + 1}):")
    for real, pred, out in zip(digit_labels, best_run['predicted_digits'], best_run['predictions']):
        print(f"Dígito real: {real}, Predicción: {pred}, Valores red: {np.round(out, 6)}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, "config/exercise_3_3.json"), 'r') as f:
        config = json.load(f)["config"]
    main(config)
