import os
import json
import matplotlib.pyplot as plt

def plot_error_vs_epochs():
    results_dir = os.path.join(os.path.dirname(__file__), "results_3_2")
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"No existe el directorio {results_dir}")

    files = [f for f in os.listdir(results_dir) if f.startswith("loss_history_") and f.endswith(".json")]
    if not files:
        print("⚠️ No se encontraron archivos loss_history_*.json. Ejecutá primero los experimentos.")
        return

    plt.figure(figsize=(8, 6))
    for fname in sorted(files):
        run_name = fname[len("loss_history_"):-len(".json")]
        with open(os.path.join(results_dir, fname), "r") as f:
            loss = json.load(f)
        plt.plot(loss, label=run_name)

    plt.xlabel("Épocas")
    plt.ylabel("Error (MSE)")
    plt.title("Error en función de las épocas (todas las corridas encontradas)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=8, ncol=1)
    plt.tight_layout()

    out_path = os.path.join(results_dir, "error_vs_epochs_all.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"✅ Gráfico generado en: {out_path}")

if __name__ == "__main__":
    plot_error_vs_epochs()
