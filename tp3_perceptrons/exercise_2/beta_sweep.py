import os
import json
import csv
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from src.exercise2.training_config import PerceptronTrainingConfig
from src.exercise2.data_handler import Exercise2DataHandler
from src.exercise2.activation_functions import ActivationFunction
from src.exercise2.gradient_optimizer import GradientDescentOptimizer
from src.exercise2.perceptron_model import SingleLayerPerceptronModel


def run_beta_sweep(betas: List[float]) -> List[Tuple[float, float, float]]:
    """Ejecuta K-Fold para varios valores de beta en modo batch.

    Retorna una lista de tuplas (beta, train_mse, test_mse).
    """
    # Cargar configuración base LOGISTIC
    config_path = os.path.join(os.path.dirname(__file__), "config", "ej2_logistic.json")
    with open(config_path, "r") as f:
        base_config = json.load(f)

    # Inicializar configuración de entrenamiento (singleton)
    PerceptronTrainingConfig(
        epsilon=base_config["epsilon"],
        seed=base_config.get("seed", None),
        maxEpochs=base_config["maxEpochs"],
    )

    # Ruta del dataset
    dataset_path = os.path.join(os.path.dirname(__file__), "resources", "set.csv")

    # Cargar dataset y preparar splits una sola vez
    data_handler = Exercise2DataHandler(dataset_path)
    target_min = np.min(data_handler.target_values)
    target_max = np.max(data_handler.target_values)
    fold_splits = data_handler.create_k_fold_splits(k_folds=7)

    results: List[Tuple[float, float, float]] = []
    for beta in betas:
        # Crear función de activación LOGISTIC con el beta deseado
        activation = ActivationFunction.create_activation_function(
            "LOGISTIC",
            {"beta": float(beta)}
        )
        activation.configure_output_normalization(target_min, target_max)

        # Crear optimizador
        optimizer_class = GradientDescentOptimizer.create_optimizer(
            base_config["learning"]["optimizer"]["type"]
        )
        optimizer = optimizer_class(base_config["learning"]["optimizer"]["options"])

        training_mse_list: List[float] = []
        testing_mse_list: List[float] = []

        print(f"=== Beta = {beta} (LOGISTIC) ===")
        for train_inputs, train_targets, test_inputs, test_targets in fold_splits:
            initial_weights = data_handler.initialize_random_weights(len(train_inputs[0]))
            model = SingleLayerPerceptronModel(activation, optimizer, initial_weights)
            model.train_model(train_inputs, train_targets)
            training_mse_list.append(model.calculate_mean_squared_error(train_inputs, train_targets))
            testing_mse_list.append(model.calculate_mean_squared_error(test_inputs, test_targets))

        train_mse = float(np.mean(training_mse_list))
        test_mse = float(np.mean(testing_mse_list))
        print(f"MSE Entrenamiento: {train_mse:.6f}")
        print(f"MSE Prueba:        {test_mse:.6f}")
        results.append((float(beta), train_mse, test_mse))

    return results


def save_results_csv(results: List[Tuple[float, float, float]], csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["beta", "train_mse", "test_mse"])
        for beta, train_mse, test_mse in results:
            writer.writerow([beta, train_mse, test_mse])


def plot_results(results: List[Tuple[float, float, float]], png_path: str) -> None:
    betas = [r[0] for r in results]
    test_mse = [r[2] for r in results]

    plt.figure(figsize=(7, 4.5))
    plt.plot(betas, test_mse, marker="o")
    plt.xscale("log")
    plt.xlabel("Beta")
    plt.ylabel("Mean Squared Error (Test)")
    plt.title("MSE vs Beta (Batch)")
    plt.grid(True, which="both", linestyle=":", linewidth=0.7)
    plt.tight_layout()

    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    plt.savefig(png_path, dpi=160)
    plt.close()


if __name__ == "__main__":
    # Barrido amplio de betas (incluye valores pequeños y grandes)
    betas = [0.1, 0.2, 0.3, 0.7, 1.0, 2.0, 4.0, 5.0]
    results = run_beta_sweep(betas)

    csv_out = os.path.join(os.path.dirname(__file__), "resources", "beta_mse_batch.csv")
    png_out = os.path.join(os.path.dirname(__file__), "resources", "beta_mse_batch.png")

    save_results_csv(results, csv_out)
    plot_results(results, png_out)

    print(f"Resultados guardados en: {csv_out}")
    print(f"Gráfico guardado en:    {png_out}")


