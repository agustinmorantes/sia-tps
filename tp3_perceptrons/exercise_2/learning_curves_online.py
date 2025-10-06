import os
import json
import csv
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

from src.exercise2.training_config import PerceptronTrainingConfig
from src.exercise2.data_handler import Exercise2DataHandler
from src.exercise2.activation_functions import ActivationFunction
from src.exercise2.gradient_optimizer import GradientDescentOptimizer
from src.exercise2.perceptron_model import SingleLayerPerceptronModel


def train_online_collect_curve(activation_type: str, beta: float, base_config: Dict, data_handler: Exercise2DataHandler) -> List[float]:
    """Entrena en modo ONLINE (actualización por muestra) y devuelve MSE de entrenamiento por época."""
    target_min = np.min(data_handler.target_values)
    target_max = np.max(data_handler.target_values)

    activation = ActivationFunction.create_activation_function(activation_type, {"beta": beta})
    activation.configure_output_normalization(target_min, target_max)

    optimizer_class = GradientDescentOptimizer.create_optimizer(base_config["learning"]["optimizer"]["type"])
    optimizer = optimizer_class(base_config["learning"]["optimizer"]["options"])

    X = data_handler.input_features
    y = data_handler.target_values
    rng = PerceptronTrainingConfig.get_instance().random_generator

    weights = data_handler.initialize_random_weights(X.shape[1])
    mse_per_epoch: List[float] = []
    max_epochs = PerceptronTrainingConfig.get_instance().max_training_epochs
    eps = PerceptronTrainingConfig.get_instance().convergence_threshold

    for _ in range(max_epochs):
        indices = rng.permutation(len(X))
        for i in indices:
            x_i = X[i]
            y_i = y[i]
            pred = activation(x_i, weights)
            output_error = y_i - activation.denormalize(pred)
            derivative = activation.derivative(x_i, weights)
            grad = output_error * derivative * x_i
            weights = np.add(weights, optimizer.calculate_weight_update(grad))

        temp_model = SingleLayerPerceptronModel(activation, optimizer, weights)
        mse = float(temp_model.calculate_mean_squared_error(X, y))
        mse_per_epoch.append(mse)
        if abs(mse) <= eps:
            break

    return mse_per_epoch


if __name__ == "__main__":
    root = os.path.dirname(__file__)

    config_path = os.path.join(root, "config", "ej2_tanh.json")
    with open(config_path, "r") as f:
        base_config = json.load(f)

    PerceptronTrainingConfig(
        epsilon=base_config["epsilon"],
        seed=base_config.get("seed", None),
        maxEpochs=base_config["maxEpochs"],
    )

    data_handler = Exercise2DataHandler(os.path.join(root, "resources", "set.csv"))

    beta_value1 = 0.2  # Logistic
    beta_value2 = 0.3  # Tanh

    curves: Dict[str, List[float]] = {}
    curves["Linear"] = train_online_collect_curve("LINEAR", 0.0, base_config, data_handler)
    curves["Logistic (β=0.2)"] = train_online_collect_curve("LOGISTIC", beta_value1, base_config, data_handler)
    curves["Tanh (β=0.3)"] = train_online_collect_curve("TANH", beta_value2, base_config, data_handler)

    csv_out = os.path.join(root, "resources", "learning_curves_online.csv")
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    max_len = max(len(v) for v in curves.values())
    names = list(curves.keys())
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + names)
        for epoch in range(max_len):
            row = [epoch + 1]
            for name in names:
                series = curves[name]
                val = series[min(epoch, len(series) - 1)]
                row.append(val)
            writer.writerow(row)

    png_out = os.path.join(root, "resources", "learning_curves_online.png")
    plt.figure(figsize=(8, 5))
    for name in names:
        plt.plot(range(1, max_len + 1), [curves[name][min(i, len(curves[name]) - 1)] for i in range(max_len)], label=name)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Training Mean Squared Error")
    plt.title("Poder de Aprendizaje (Online)")
    plt.grid(True, which="both", linestyle=":", linewidth=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_out, dpi=160)
    plt.close()

    print(f"Curvas guardadas en: {csv_out}")
    print(f"Gráfico guardado en:  {png_out}")


