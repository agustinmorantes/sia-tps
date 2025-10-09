import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
from src.multi_layer_perceptron import MultiLayerPerceptron


def load_data():
    txt_path = os.path.join(os.path.dirname(__file__), "resources/TP3-ej3-digitos.txt")
    data = np.loadtxt(txt_path, dtype=int)
    X = data.reshape(-1, 35)
    n_digits = 10
    n_repeats = X.shape[0] // n_digits
    digit_labels = np.tile(np.arange(n_digits), n_repeats)
    Y = np.array([1 if d % 2 else -1 for d in digit_labels]).reshape(-1, 1)
    return X, Y

def split_data(X, Y, train_ratio=0.8, seed=123):
    np.random.seed(seed)
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    split_idx = int(len(X) * train_ratio)
    train_idx, test_idx = idx[:split_idx], idx[split_idx:]
    return X[train_idx], Y[train_idx], X[test_idx], Y[test_idx]

def calculate_metrics(y_true, y_pred):
    y_pred_bin = np.where(y_pred >= 0, 1, -1)
    cm = confusion_matrix(y_true, y_pred_bin)
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_true, y_pred_bin)
    prec = precision_score(y_true, y_pred_bin, zero_division=0)
    rec = recall_score(y_true, y_pred_bin, zero_division=0)
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "confusion_matrix": cm.tolist()
    }

def plot_confusion(cm, title, save_path):
    import numpy as np
    import matplotlib.pyplot as plt

    cm = np.array(cm)
    classes = ["Par (-1)", "Impar (+1)"]

    fig, ax = plt.subplots(figsize=(4.5, 4))  # 👈 un poco más de ancho para la barra
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)

    # Título y ejes
    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")

    # Anotaciones en las celdas
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    # 👇 Colorbar con padding más chico
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")  # 👈 este bbox_inches evita que la barra tape el título
    plt.close(fig)

def plot_loss_curve(loss_history, save_path):
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history, label="Error de entrenamiento (MSE)")
    plt.xlabel("Épocas")
    plt.ylabel("Error (MSE)")
    plt.title("Evolución del error de entrenamiento")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_accuracy_curve(accuracy_history, save_path):
    plt.figure(figsize=(6, 4))
    plt.plot(accuracy_history, label="Accuracy de entrenamiento", color="orange")
    plt.xlabel("Épocas")
    plt.ylabel("Accuracy")
    plt.title("Evolución del accuracy en entrenamiento")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    if len(sys.argv) != 2:
        print("Uso: python3 main.py <ruta_config.json>")
        sys.exit(1)

    config_path = sys.argv[1]
    if not os.path.exists(config_path):
        print(f"No existe el archivo {config_path}")
        sys.exit(1)

    # 👉 Obtener sufijo desde el nombre del archivo
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    suffix = config_name.replace("config_", "")

    with open(config_path) as f:
        config = json.load(f)

    eta_value = float(config.get("eta", 0.05))

    print("\n===============================")
    print(f"🚀 Ejecutando experimento con {config_name}")
    print("===============================")
    print(f"Entrenando con eta={eta_value} ...")

    X, Y = load_data()
    X_train, Y_train, X_test, Y_test = split_data(X, Y, train_ratio=0.8, seed=config.get("seed", 123))

    mlp = MultiLayerPerceptron(
        layer_sizes=config["layer_sizes"],
        activation=config.get("activation", "tanh"),
        eta=eta_value,
        optimizer=config.get("optimizer", "sgd"),
        alpha=config.get("alpha", 0.9),
        batch_size=config.get("batch_size", None),
        seed=config.get("seed", 123)
    )

    # 🔁 Entrenamiento manual para accuracy por epoch
    epochs = config["epochs"]
    epsilon = config["epsilon"]
    accuracy_history = []

    for epoch in range(epochs):
        mlp.train(X_train, Y_train, epochs=1, epsilon=epsilon)
        y_pred_epoch = mlp.predict(X_train)
        acc_epoch = accuracy_score(Y_train, np.where(y_pred_epoch >= 0, 1, -1))
        accuracy_history.append(acc_epoch)
        if mlp.loss_history[-1] < epsilon:
            print(f"Convergencia alcanzada en epoch {epoch+1}")
            break

    y_pred_train = mlp.predict(X_train)
    y_pred_test = mlp.predict(X_test)

    metrics_train = calculate_metrics(Y_train, y_pred_train)
    metrics_test = calculate_metrics(Y_test, y_pred_test)

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    # === Guardar resultados JSON ===
    result_path = os.path.join(results_dir, f"results_{suffix}.json")
    with open(result_path, "w") as f:
        json.dump({"train": metrics_train, "test": metrics_test}, f, indent=4)
    print(f"\n✅ Resultados guardados en {result_path}")

    # === Gráficos ===
    plot_confusion(metrics_train["confusion_matrix"], f"Matriz de Confusión (Train) - {suffix}",
                   os.path.join(results_dir, f"confusion_train_{suffix}.png"))
    plot_confusion(metrics_test["confusion_matrix"], f"Matriz de Confusión (Test) - {suffix}",
                   os.path.join(results_dir, f"confusion_test_{suffix}.png"))
    plot_loss_curve(mlp.loss_history,
                    os.path.join(results_dir, f"loss_curve_{suffix}.png"))
    plot_accuracy_curve(accuracy_history,
                    os.path.join(results_dir, f"accuracy_curve_{suffix}.png"))

    print(f"🖼️  Gráficos generados:")
    print(f"   • confusion_train_{suffix}.png")
    print(f"   • confusion_test_{suffix}.png")
    print(f"   • loss_curve_{suffix}.png")
    print(f"   • accuracy_curve_{suffix}.png")

    print(f"\n📊 Métricas finales (TEST):")
    print(json.dumps(metrics_test, indent=4))


if __name__ == "__main__":
    main()
