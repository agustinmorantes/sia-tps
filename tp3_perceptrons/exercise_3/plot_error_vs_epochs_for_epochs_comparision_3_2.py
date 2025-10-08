import matplotlib.pyplot as plt
import json
import os

results_dir = os.path.join(os.path.dirname(__file__), "results_3_2")

cases = [
    ("loss_history_tanh_sgd_epochs_1000_epsilon_1e-3.json", "1000 épocas"),
    ("loss_history_baseline_tanh_sgd.json", "5000 épocas"),
    ("loss_history_tanh_sgd_epochs_10000_epsilon_1e-5.json", "10000 épocas"),
]

plt.figure(figsize=(8,5))

for filename, label in cases:
    path = os.path.join(results_dir, filename)
    if os.path.exists(path):
        with open(path) as f:
            loss = json.load(f)
        plt.plot(loss, label=label)
    else:
        print(f"⚠️ No se encontró: {path}")

plt.xlabel("Épocas")
plt.ylabel("Error (MSE)")
plt.title("Evolución del error — efecto del número de épocas")
plt.yscale("log")  # Escala logarítmica para ver diferencias finas
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# Guardar la figura
output_path = os.path.join(results_dir, "error_vs_epochs_comparacion_epocas.png")
plt.savefig(output_path, dpi=200)
plt.show()

print(f"✅ Gráfico guardado en {output_path}")
