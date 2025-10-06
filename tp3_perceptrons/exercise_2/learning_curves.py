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


def train_batch_collect_curve(activation_type: str, beta: float, base_config: Dict, data_handler: Exercise2DataHandler) -> List[float]:
    """Entrena en modo BATCH (acumula gradientes y actualiza por época) y devuelve MSE de entrenamiento por época."""
    target_min = np.min(data_handler.target_values)
    target_max = np.max(data_handler.target_values)

    activation = ActivationFunction.create_activation_function(activation_type, {"beta": beta})
    activation.configure_output_normalization(target_min, target_max)

    optimizer_class = GradientDescentOptimizer.create_optimizer(base_config["learning"]["optimizer"]["type"])
    optimizer = optimizer_class(base_config["learning"]["optimizer"]["options"])

    X = data_handler.input_features
    y = data_handler.target_values

    initial_weights = data_handler.initialize_random_weights(X.shape[1])
    model = SingleLayerPerceptronModel(activation, optimizer, initial_weights)
    model.train_model(X, y)

    # Convertir historial de pesos por época a curva de MSE
    mse_per_epoch: List[float] = []
    for weights in model.get_weights_per_epoch():
        temp_model = SingleLayerPerceptronModel(activation, optimizer, weights)
        mse_per_epoch.append(float(temp_model.calculate_mean_squared_error(X, y)))
    return mse_per_epoch


if __name__ == "__main__":
    root = os.path.dirname(__file__)

    # Config base (usa tanh/logistic solo por epsilon, epochs, seed, rate)
    config_path = os.path.join(root, "config", "ej2_tanh.json")
    with open(config_path, "r") as f:
        base_config = json.load(f)

    PerceptronTrainingConfig(
        epsilon=base_config["epsilon"],
        seed=base_config.get("seed", None),
        maxEpochs=base_config["maxEpochs"],
    )

    data_handler = Exercise2DataHandler(os.path.join(root, "resources", "set.csv"))

    # Modo batch, con β=0.3 para funciones no lineales
    curves: Dict[str, List[float]] = {}
    curves["Linear"] = train_batch_collect_curve("LINEAR", 0.0, base_config, data_handler)
    curves["Logistic (β=0.2)"] = train_batch_collect_curve("LOGISTIC", 0.2, base_config, data_handler)
    curves["Tanh (β=0.3)"] = train_batch_collect_curve("TANH", 0.3, base_config, data_handler)

    # Guardar CSV
    csv_out = os.path.join(root, "resources", "learning_curves_batch.csv")
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    # Alinear longitudes rellenando con el último valor
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

    # Graficar
    png_out = os.path.join(root, "resources", "learning_curves_batch.png")
    plt.figure(figsize=(8, 5))
    for name in names:
        plt.plot(range(1, max_len + 1), [curves[name][min(i, len(curves[name]) - 1)] for i in range(max_len)], label=name)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Training Mean Squared Error")
    plt.title("Poder de Aprendizaje (Batch, β=0.3)")
    plt.grid(True, which="both", linestyle=":", linewidth=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_out, dpi=160)
    plt.close()

    print(f"Curvas guardadas en: {csv_out}")
    print(f"Gráfico guardado en:  {png_out}")


