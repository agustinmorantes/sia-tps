import os
import json
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_3_2")

experiments = [
    ("tanh_eta_0.01", 0.01),
    ("baseline_tanh_sgd", 0.05),
    ("tanh_eta_0.1", 0.1),
    ("tanh_eta_0.2", 0.2),
]

colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

plt.figure(figsize=(8, 5))

for (name, eta), color in zip(experiments, colors):
    file_path = os.path.join(RESULTS_DIR, f"loss_history_{name}.json")
    if not os.path.exists(file_path):
        print(f"⚠️ No se encontró {file_path}, se salta este experimento.")
        continue

    with open(file_path, "r") as f:
        loss = json.load(f)

    # Mostrar solo las primeras 100 épocas o menos
    epochs_to_show = min(100, len(loss))
    plt.plot(range(epochs_to_show), loss[:epochs_to_show],
             label=f"η={eta}", linewidth=2.5, color=color)

plt.title("Evolución del error en las primeras épocas\n(tanh, optimizador=SGD)")
plt.xlabel("Épocas")
plt.ylabel("Error (MSE)")
plt.yscale("log")  # Escala logarítmica para apreciar diferencias pequeñas
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(title="Learning rate", loc="upper right")
plt.tight_layout()

output_path = os.path.join(RESULTS_DIR, "error_vs_epochs_comparacion_eta_zoom.png")
plt.savefig(output_path, dpi=300)
plt.show()

print(f"✅ Gráfico combinado guardado en {output_path}")
