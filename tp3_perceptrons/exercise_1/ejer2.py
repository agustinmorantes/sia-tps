from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from perceptron import SimplePerceptron


@dataclass
class Dataset:
    X: List[List[float]]
    y: List[int]


def load_csv_as_classification(csv_path: Path, threshold: float | None = None) -> Tuple[Dataset, float]:
    X: List[List[float]] = []
    y_continuous: List[float] = []

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x1 = float(row["x1"])  # type: ignore[index]
            x2 = float(row["x2"])  # type: ignore[index]
            x3 = float(row["x3"])  # type: ignore[index]
            y_val = float(row["y"])  # type: ignore[index]
            X.append([x1, x2, x3])
            y_continuous.append(y_val)

    # Determine threshold if not provided (use median as suggested)
    chosen_threshold = threshold if threshold is not None else median(y_continuous)

    # Binarize: map to {-1, 1} to match perceptron
    y: List[int] = [1 if v >= chosen_threshold else -1 for v in y_continuous]

    return Dataset(X=X, y=y), chosen_threshold


def median(values: List[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def train_test_split(X: List[List[float]], y: List[int], test_size: float = 0.2, seed: int | None = 42) -> Tuple[List[List[float]], List[List[float]], List[int], List[int]]:
    assert 0.0 < test_size < 1.0
    indices = list(range(len(X))) #genero mi lista de indices
    if seed is not None:
        random.Random(seed).shuffle(indices)
    else:
        random.shuffle(indices)

    split = int((1 - test_size) * len(indices))
    train_idx, test_idx = indices[:split], indices[split:]

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]
    return X_train, X_test, y_train, y_test


def k_fold_indices(n_samples: int, k: int, seed: int | None = 42) -> List[Tuple[List[int], List[int]]]:
    assert k >= 2
    indices = list(range(n_samples))
    if seed is not None:
        random.Random(seed).shuffle(indices)
    else:
        random.shuffle(indices)

    fold_sizes = [n_samples // k + (1 if i < n_samples % k else 0) for i in range(k)]
    folds: List[List[int]] = []
    current = 0
    for size in fold_sizes:
        folds.append(indices[current: current + size])
        current += size

    result: List[Tuple[List[int], List[int]]] = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = [idx for j, fold in enumerate(folds) if j != i for idx in fold]
        result.append((train_idx, val_idx))
    return result


def evaluate_perceptron(X_train: List[List[float]], y_train: List[int], X_test: List[List[float]], y_test: List[int], learning_rate: float, max_epochs: int) -> Tuple[float, float]:
    model = SimplePerceptron(learning_rate=learning_rate, max_epochs=max_epochs)
    model.train(X_train, y_train)
    train_acc, _ = model.evaluate(X_train, y_train) #porcentaje de aciertos en entrenamiento
    test_acc, _ = model.evaluate(X_test, y_test) #porcentaje de aciertos en test
    return train_acc, test_acc


def run_holdout(dataset: Dataset, learning_rate: float, max_epochs: int, test_size: float, seed: int) -> None:
    X_train, X_test, y_train, y_test = train_test_split(dataset.X, dataset.y, test_size=test_size, seed=seed)
    train_acc, test_acc = evaluate_perceptron(X_train, y_train, X_test, y_test, learning_rate, max_epochs)
    print(f"Hold-out -> train_acc={train_acc:.3f} test_acc={test_acc:.3f}")


def run_kfold(dataset: Dataset, learning_rate: float, max_epochs: int, k: int, seed: int) -> None:
    splits = k_fold_indices(len(dataset.X), k=k, seed=seed)# devuelve donde una lista donde cada tupla indica qué índices usar como entrenamiento y validación.
    test_accs: List[float] = []
    train_accs: List[float] = []
    for fold_num, (train_idx, val_idx) in enumerate(splits, start=1): #Usa los índices de train_idx y val_idx para construir listas concretas de datos y etiquetas
        X_train = [dataset.X[i] for i in train_idx]
        y_train = [dataset.y[i] for i in train_idx]
        X_val = [dataset.X[i] for i in val_idx]
        y_val = [dataset.y[i] for i in val_idx]

        train_acc, val_acc = evaluate_perceptron(X_train, y_train, X_val, y_val, learning_rate, max_epochs)
        train_accs.append(train_acc)
        test_accs.append(val_acc)
        print(f"Fold {fold_num}/{k}: train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

    mean_train = sum(train_accs) / len(train_accs)
    mean_val = sum(test_accs) / len(test_accs)
    std_train = math.sqrt(sum((a - mean_train) ** 2 for a in train_accs) / len(train_accs))
    std_val = math.sqrt(sum((a - mean_val) ** 2 for a in test_accs) / len(test_accs))
    print(f"K-Fold mean train_acc={mean_train:.3f}±{std_train:.3f} mean val_acc={mean_val:.3f}±{std_val:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ejercicio 2 - Simple Perceptron (lineal)")
    parser.add_argument("--csv", type=str, default=str(Path(__file__).resolve().parents[1] / "TP3-ej2-conjunto.csv"), help="Ruta al CSV con columnas x1,x2,x3,y")
    parser.add_argument("--threshold", type=float, default=None, help="Umbral para binarizar y (por defecto: mediana)")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate del perceptrón")
    parser.add_argument("--epochs", type=int, default=1000, help="Máximo de épocas de entrenamiento")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proporción del conjunto de test para hold-out")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para aleatoriedad")
    parser.add_argument("--kfold", type=int, default=5, help="Si >0, usa validación cruzada K-fold con K indicado")

    args = parser.parse_args()

    csv_path = Path(args.csv)
    dataset, used_threshold = load_csv_as_classification(csv_path, threshold=args.threshold)
    print(f"Cargado {len(dataset.X)} muestras desde {csv_path.name}. Umbral usado para y: {used_threshold:.3f}")

    if args.kfold and args.kfold > 1:
        run_kfold(dataset, learning_rate=args.lr, max_epochs=args.epochs, k=args.kfold, seed=args.seed)
    else:
        run_holdout(dataset, learning_rate=args.lr, max_epochs=args.epochs, test_size=args.test_size, seed=args.seed)


if __name__ == "__main__":
    main()


