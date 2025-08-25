import os
import matplotlib.pyplot as plt
import numpy as np

def format_algorithm_name(name: str) -> str:
    name = name.strip().lower()
    if name.startswith("a_star"):
        return "A* (" + name.split("_", 2)[-1].capitalize() + ")"
    elif name.startswith("greedy"):
        return "Greedy (" + name.split("_", 1)[-1].capitalize() + ")"
    elif name == "bfs":
        return "BFS"
    elif name == "dfs":
        return "DFS"
    else:
        return name.capitalize()

def plot_results():
    results_dir = "."
    output_dir = os.path.join(results_dir, "plotMaps")
    os.makedirs(output_dir, exist_ok=True)

    result_files = [f for f in os.listdir(results_dir) if f.startswith("results_Map") and f.endswith(".txt")]

    algo_order = [
        "BFS", "DFS",
        "Greedy (Manhattan)", "Greedy (Euclidean)", "Greedy (Chebyshev)", "Greedy (Hamming)",
        "A* (Manhattan)", "A* (Euclidean)", "A* (Chebyshev)", "A* (Hamming)"
    ]

    for result_file in result_files:
        map_name = result_file.replace("results_", "").replace(".txt", "")
        raw_algorithms = []
        expanded_nodes = {}

        with open(os.path.join(results_dir, result_file), "r") as f:
            for line in f:
                algo, nodes = line.strip().split(": ")
                formatted = format_algorithm_name(algo)
                raw_algorithms.append(formatted)
                expanded_nodes[formatted] = int(nodes)

        algorithms = [a for a in algo_order if a in raw_algorithms]
        values = [expanded_nodes[a] for a in algorithms]

        colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))

        plt.figure(figsize=(12, 6))
        bars = plt.barh(algorithms, values, color=colors, edgecolor="black")

        max_val = max(values)

        # Ajustar el límite del eje X para dejar más espacio a la derecha
        plt.xlim(0, max_val * 1.1)

        # Colocar los valores con un margen extra
        for bar, value in zip(bars, values):
            xval = bar.get_width()
            plt.text(xval + max_val * 0.01, bar.get_y() + bar.get_height()/2,
                     f"{value:,}".replace(",", "."),
                     va="center", ha="left", fontsize=9)

        plt.ylabel("Algorithm", fontsize=12)
        plt.xlabel("Nodes Expanded", fontsize=12)
        plt.title(f"Nodes Expanded per Algorithm for {map_name}", fontsize=14, weight="bold")
        plt.grid(axis="x", linestyle="--", alpha=0.6)

        plt.tight_layout()
        output_path = os.path.join(output_dir, f"plot_{map_name}.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Generated horizontal plot for {map_name} in {output_path}")

if __name__ == "__main__":
    plot_results()
