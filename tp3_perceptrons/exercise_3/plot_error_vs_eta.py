import os
import json
import matplotlib.pyplot as plt

def plot_error_vs_eta(optimizer_filter="sgd"):
    results_path = os.path.join(os.path.dirname(__file__), "results_3_2", "results_parity.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"No se encontró el archivo {results_path}")

    # Cargar resultados
    results = []
    with open(results_path, "r") as f:
        for line in f:
            try:
                results.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

    # Filtrar por optimizer si se pide
    if optimizer_filter:
        results = [r for r in results if r.get("optimizer") == optimizer_filter]

    # Agrupar por eta y tomar promedio de error (por si hay repetidos)
    by_eta = {}
    for r in results:
        eta = float(r["eta"])
        err_train = 1.0 - float(r["accuracy_train"])
        err_test  = 1.0 - float(r["accuracy_test"])
        by_eta.setdefault(eta, {"train": [], "test": []})
        by_eta[eta]["train"].append(err_train)
        by_eta[eta]["test"].append(err_test)

    etas = sorted(by_eta.keys())
    errors_train = [sum(by_eta[e]["train"])/len(by_eta[e]["train"]) for e in etas]
    errors_test  = [sum(by_eta[e]["test"] )/len(by_eta[e]["test"])  for e in etas]

    # Graficar
    plt.figure(figsize=(7, 5))
    plt.plot(etas, errors_train, marker="o", label="Error Train")
    plt.plot(etas, errors_test, marker="s", label="Error Test")
    plt.xscale("log")
    plt.xlabel("Tasa de aprendizaje (η)")
    plt.ylabel("Error (1 - Accuracy)")
    plt.title(f"Error en función de η — optimizer={optimizer_filter}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), "results", f"error_vs_eta_{optimizer_filter}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"✅ Gráfico generado en: {out_path}")

if __name__ == "__main__":
    plot_error_vs_eta("sgd")  # cambiá a "momentum" o "adam" si querés
