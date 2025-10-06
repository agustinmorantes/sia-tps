import os
import json
import csv
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from src.exercise2.training_config import PerceptronTrainingConfig
from src.exercise2.data_handler import Exercise2DataHandler
from src.exercise2.activation_functions import ActivationFunction
from src.exercise2.gradient_optimizer import GradientDescentOptimizer
from src.exercise2.perceptron_model import SingleLayerPerceptronModel


def train_one_fold_mse_curves(X_train, y_train, X_test, y_test, beta: float, base_config) -> (List[float], List[float]):
    target_min = min(np.min(y_train), np.min(y_test))
    target_max = max(np.max(y_train), np.max(y_test))

    activation = ActivationFunction.create_activation_function("TANH", {"beta": beta})
    activation.configure_output_normalization(target_min, target_max)

    optimizer_class = GradientDescentOptimizer.create_optimizer(base_config["learning"]["optimizer"]["type"])
    optimizer = optimizer_class(base_config["learning"]["optimizer"]["options"])

    initial_weights = PerceptronTrainingConfig.get_instance().random_generator.uniform(-1, 1, X_train.shape[1])
    model = SingleLayerPerceptronModel(activation, optimizer, initial_weights)
    model.train_model(X_train, y_train)

    train_mse_curve: List[float] = []
    test_mse_curve: List[float] = []
    for weights in model.get_weights_per_epoch():
        tmp = SingleLayerPerceptronModel(activation, optimizer, weights)
        train_mse_curve.append(float(tmp.calculate_mean_squared_error(X_train, y_train)))
        test_mse_curve.append(float(tmp.calculate_mean_squared_error(X_test, y_test)))

    return train_mse_curve, test_mse_curve


if __name__ == "__main__":
    root = os.path.dirname(__file__)

    # Config base (usa epsilon, maxEpochs, rate)
    config_path = os.path.join(root, "config", "ej2_tanh.json")
    with open(config_path, "r") as f:
        base_config = json.load(f)

    PerceptronTrainingConfig(
        epsilon=base_config["epsilon"],
        seed=base_config.get("seed", None),
        maxEpochs=base_config["maxEpochs"],
    )

    data = Exercise2DataHandler(os.path.join(root, "resources", "set.csv"))
    folds = data.create_k_fold_splits(k_folds=7)

    beta = 0.3
    train_curves: List[List[float]] = []
    test_curves: List[List[float]] = []

    for Xtr, ytr, Xte, yte in folds:
        tr, te = train_one_fold_mse_curves(Xtr, ytr, Xte, yte, beta, base_config)
        train_curves.append(tr)
        test_curves.append(te)

    # Alinear longitudes rellenando con el último valor de cada fold
    max_len = max(len(c) for c in train_curves)
    def pad_to(series: List[float], n: int) -> List[float]:
        if not series:
            return [0.0] * n
        last = series[-1]
        return series + [last] * (n - len(series))

    train_mat = np.array([pad_to(c, max_len) for c in train_curves])
    test_mat = np.array([pad_to(c, max_len) for c in test_curves])

    mean_train = train_mat.mean(axis=0)
    mean_test = test_mat.mean(axis=0)

    # Guardar CSV
    csv_out = os.path.join(root, "resources", "generalization_tanh_batch.csv")
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "trainingMSE", "testingMSE"]) 
        for i in range(max_len):
            w.writerow([i + 1, float(mean_train[i]), float(mean_test[i])])

    # Graficar
    png_out = os.path.join(root, "resources", "generalization_tanh_batch.png")
    plt.figure(figsize=(8, 5))
    epochs = range(1, max_len + 1)
    plt.plot(epochs, mean_train, label="Training Set", color="#003049")
    plt.plot(epochs, mean_test, label="Testing Set", color="#d62828", linestyle="--")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title("Poder de Generalización - TANH (β=0.3, Batch)")
    plt.grid(True, which="both", linestyle=":", linewidth=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_out, dpi=160)
    plt.close()

    print(f"Resultados guardados en: {csv_out}")
    print(f"Gráfico guardado en:    {png_out}")


