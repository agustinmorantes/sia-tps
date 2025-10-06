import os
import json
import csv
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt

from src.exercise2.training_config import PerceptronTrainingConfig
from src.exercise2.data_handler import Exercise2DataHandler
from src.exercise2.activation_functions import ActivationFunction
from src.exercise2.gradient_optimizer import GradientDescentOptimizer
from src.exercise2.perceptron_model import SingleLayerPerceptronModel


def train_one_fold_curve(X_train, y_train, X_test, y_test, beta: float, base_config) -> (List[float], List[float]):
    target_min = min(np.min(y_train), np.min(y_test))
    target_max = max(np.max(y_train), np.max(y_test))

    activation = ActivationFunction.create_activation_function("TANH", {"beta": beta})
    activation.configure_output_normalization(target_min, target_max)

    optimizer_class = GradientDescentOptimizer.create_optimizer(base_config["learning"]["optimizer"]["type"])
    optimizer = optimizer_class(base_config["learning"]["optimizer"]["options"])

    initial_weights = PerceptronTrainingConfig.get_instance().random_generator.uniform(-1, 1, X_train.shape[1])
    model = SingleLayerPerceptronModel(activation, optimizer, initial_weights)
    model.train_model(X_train, y_train)

    train_curve: List[float] = []
    test_curve: List[float] = []
    for weights in model.get_weights_per_epoch():
        tmp = SingleLayerPerceptronModel(activation, optimizer, weights)
        train_curve.append(float(tmp.calculate_mean_squared_error(X_train, y_train)))
        test_curve.append(float(tmp.calculate_mean_squared_error(X_test, y_test)))
    return train_curve, test_curve


def mean_curve_over_folds(curves: List[List[float]]) -> List[float]:
    if not curves:
        return []
    max_len = max(len(c) for c in curves)
    padded = []
    for c in curves:
        last = c[-1]
        padded.append(c + [last] * (max_len - len(c)))
    return list(np.mean(np.array(padded, dtype=float), axis=0))


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

    data = Exercise2DataHandler(os.path.join(root, "resources", "set.csv"))
    beta = 0.3

    # Diferentes valores de K a comparar
    k_values = [2, 3, 5, 10]

    # Para CSV (una fila por epoch con columnas por curva)
    csv_out = os.path.join(root, "resources", "generalization_k_tanh_batch.csv")
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)

    curves_dict: Dict[str, List[float]] = {}

    for k in k_values:
        folds = data.create_k_fold_splits(k_folds=k)
        train_curves: List[List[float]] = []
        test_curves: List[List[float]] = []
        for Xtr, ytr, Xte, yte in folds:
            tr, te = train_one_fold_curve(Xtr, ytr, Xte, yte, beta, base_config)
            train_curves.append(tr)
            test_curves.append(te)

        mean_train = mean_curve_over_folds(train_curves)
        mean_test = mean_curve_over_folds(test_curves)

        curves_dict[f"Train (k={k})"] = mean_train
        curves_dict[f"Test (k={k})"] = mean_test

    # Persistir CSV en formato ancho
    max_len = max(len(v) for v in curves_dict.values())
    names = list(curves_dict.keys())
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + names)
        for i in range(max_len):
            row = [i + 1]
            for name in names:
                series = curves_dict[name]
                val = series[min(i, len(series) - 1)]
                row.append(val)
            w.writerow(row)

    # Graficar
    png_out = os.path.join(root, "resources", "generalization_k_tanh_batch.png")
    plt.figure(figsize=(9, 5))
    for name in names:
        linestyle = "--" if name.startswith("Test") else "-"
        plt.plot(range(1, max_len + 1), [curves_dict[name][min(i, len(curves_dict[name]) - 1)] for i in range(max_len)], label=name, linestyle=linestyle)
    plt.yscale("log")
    plt.xlabel("Épocas")
    plt.ylabel("Mean Squared Error")
    plt.title("Error promedio por época para distintos valores de k (TANH β=0.3, Batch)")
    plt.grid(True, which="both", linestyle=":", linewidth=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_out, dpi=160)
    plt.close()

    print(f"Resultados guardados en: {csv_out}")
    print(f"Gráfico guardado en:    {png_out}")


