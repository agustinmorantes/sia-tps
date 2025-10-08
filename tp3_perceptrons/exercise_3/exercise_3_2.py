import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from src.multi_layer_perceptron import MultiLayerPerceptron, tanh, sigmoid

def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in s)

def main(config: dict):
    # --- Carga y preprocesamiento ---
    txt_path = os.path.join(os.path.dirname(__file__), "resources/TP3-ej3-digitos.txt")
    data = np.loadtxt(txt_path, dtype=int)
    X = data.reshape(-1, 35)

    n_digits = 10
    n_repeats = X.shape[0] // n_digits
    digit_labels = np.tile(np.arange(n_digits), n_repeats)

    activation_name = config.get("activation", "tanh")

    if activation_name == "sigmoid":
        X = X.astype(float)
        Y = np.array([1 if d % 2 else 0 for d in digit_labels]).reshape(-1, 1)
    else:
        X = X * 2 - 1
        Y = np.array([1 if d % 2 else -1 for d in digit_labels]).reshape(-1, 1)

    # --- Train/Test split (80/20) ---
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    split_idx = int(0.8 * n_samples)
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    # --- Inicialización ---
    mlp_init_params = {
        "layer_sizes": config["layer_sizes"],
        "activation": activation_name,
        "eta": config.get("eta", 0.05),
        "optimizer": config.get("optimizer", "sgd"),
        "batch_size": config.get("batch_size", None),
        "alpha": config.get("alpha", 0.9),
        "seed": config.get("seed", 123),
    }

    mlp = MultiLayerPerceptron(**mlp_init_params)

    epochs = config.get("epochs", 5000)
    epsilon = config.get("epsilon", 1e-4)

    run_name = config.get("run_name", f"{activation_name}_{mlp.optimizer}_eta{mlp.eta}")
    run_name_safe = _safe_name(str(run_name))

    print(f"\n🧠 Entrenando ({activation_name}, opt={mlp.optimizer}, η={mlp.eta})...")
    mlp.train(X_train, Y_train, epochs=epochs, epsilon=epsilon)

    # --- Evaluación ---
    preds_train = mlp.predict(X_train)
    preds_test = mlp.predict(X_test)

    if activation_name == "sigmoid":
        preds_train_bin = np.where(preds_train >= 0.5, 1, 0)
        preds_test_bin = np.where(preds_test >= 0.5, 1, 0)
    else:
        preds_train_bin = np.where(preds_train > 0, 1, -1)
        preds_test_bin = np.where(preds_test > 0, 1, -1)

    # --- Métricas ---
    def metrics_report(Y_true, Y_pred):
        acc = accuracy_score(Y_true, Y_pred)
        prec = precision_score(Y_true, Y_pred, zero_division=0)
        rec = recall_score(Y_true, Y_pred, zero_division=0)
        f1 = f1_score(Y_true, Y_pred, zero_division=0)
        cm = confusion_matrix(Y_true, Y_pred)
        tn, fp, fn, tp = cm.ravel()
        return acc, prec, rec, f1, tp, fp, fn, tn

    acc_train, prec_train, rec_train, f1_train, tp_train, fp_train, fn_train, tn_train = metrics_report(
        Y_train, preds_train_bin
    )
    acc_test, prec_test, rec_test, f1_test, tp_test, fp_test, fn_test, tn_test = metrics_report(
        Y_test, preds_test_bin
    )

    print(f"✅ Train: acc={acc_train:.3f}, prec={prec_train:.3f}, rec={rec_train:.3f}, f1={f1_train:.3f}")
    print(f"🧪 Test:  acc={acc_test:.3f}, prec={prec_test:.3f}, rec={rec_test:.3f}, f1={f1_test:.3f}")

    # --- Directorio resultados ---
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    # --- Guardar loss_history para poder graficar error vs épocas luego ---
    loss_hist = getattr(mlp, "loss_history", None)
    if loss_hist is not None:
        loss_json = [float(x) for x in loss_hist]
        with open(os.path.join(results_dir, f"loss_history_{run_name_safe}.json"), "w") as f:
            json.dump(loss_json, f)

    # --- Curva de pérdida (Error vs Épocas) ---
    if loss_hist is not None:
        plt.figure()
        plt.plot(loss_hist)
        plt.title(f"Error en función de las épocas\n({activation_name}, η={mlp.eta}, opt={mlp.optimizer})")
        plt.xlabel("Épocas")
        plt.ylabel("Error (MSE)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"error_vs_epochs_{run_name_safe}.png"), dpi=200)
        plt.close()

    # --- Matriz de confusión (con TP, FP, FN, TN) ---
    def plot_confusion(Y_true, Y_pred, split):
        cm = confusion_matrix(Y_true, Y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set(
            xticks=np.arange(2),
            yticks=np.arange(2),
            xticklabels=["Par", "Impar"],
            yticklabels=["Par", "Impar"],
            ylabel="Valor real",
            xlabel="Predicción",
            title=f"Matriz de Confusión ({split.upper()})\n{activation_name}, opt={mlp.optimizer}",
        )
        labels = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                        color=color, fontsize=12, fontweight="bold")
                ax.text(j + 0.3, i - 0.3, labels[i][j],
                        ha="right", va="top", fontsize=9, color="black")
        ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"confusion_{split}_{run_name_safe}.png"), dpi=200)
        plt.close()

    plot_confusion(Y_train, preds_train_bin, "train")
    plot_confusion(Y_test, preds_test_bin, "test")

    # --- Comparativa de métricas (Train vs Test) ---
    metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
    train_values = [acc_train, prec_train, rec_train, f1_train]
    test_values = [acc_test, prec_test, rec_test, f1_test]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, train_values, width, label="Train", alpha=0.8)
    plt.bar(x + width / 2, test_values, width, label="Test", alpha=0.8)
    plt.xticks(x, metrics, fontsize=11)
    plt.ylim(0, 1.05)
    plt.ylabel("Score", fontsize=12)
    plt.title(f"Comparativa de métricas — {activation_name}, opt={mlp.optimizer}, η={mlp.eta}")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"metrics_comparison_{run_name_safe}.png"), dpi=200)
    plt.close()

    # --- Guardar resultados (incluyendo TP, FP, FN, TN) ---
    results = {
        "run_name": str(run_name),
        "activation": str(activation_name),
        "eta": float(mlp.eta),
        "optimizer": str(mlp.optimizer),
        "epochs": int(epochs),
        "accuracy_train": float(acc_train),
        "precision_train": float(prec_train),
        "recall_train": float(rec_train),
        "f1_train": float(f1_train),
        "tp_train": int(tp_train),
        "fp_train": int(fp_train),
        "fn_train": int(fn_train),
        "tn_train": int(tn_train),
        "accuracy_test": float(acc_test),
        "precision_test": float(prec_test),
        "recall_test": float(rec_test),
        "f1_test": float(f1_test),
        "tp_test": int(tp_test),
        "fp_test": int(fp_test),
        "fn_test": int(fn_test),
        "tn_test": int(tn_test)
    }

    results_path = os.path.join(results_dir, "results_parity.json")
    with open(results_path, "a") as f:
        f.write(json.dumps(results) + "\n")

    print(f"💾 Resultados guardados en {results_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config", "exercise_3_2.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No se encontró el archivo de configuración en {config_path}")

    with open(config_path, "r") as f:
        # en modo directo este archivo tiene {"config":{...}}
        data = json.load(f)
        config = data["config"] if isinstance(data, dict) and "config" in data else data

    main(config)
