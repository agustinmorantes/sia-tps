import os
import matplotlib.pyplot as plt
import numpy as np

def format_algorithm_name(name: str) -> str:
    name = name.strip().lower()
    if name.startswith("a_star"):
        # ejemplo: a_star_manhattan -> A* (Manhattan)
        return "A* (" + name.split("_", 2)[-1].capitalize() + ")"
    elif name.startswith("greedy"):
        # ejemplo: greedy_euclidean -> Greedy (Euclidean)
        return "Greedy (" + name.split("_", 1)[-1].capitalize() + ")"
    elif name == "bfs":
        return "BFS"
    elif name == "dfs":
        return "DFS"
    else:
        return name.capitalize()

def plot_results():
    results_dir = "."
    result_files = [f for f in os.listdir(results_dir) if f.startswith("results_Map") and f.endswith(".txt")]

    for result_file in result_files:
        map_name = result_file.replace("results_", "").replace(".txt", "")
        algorithms = []
        expanded_nodes = []

        with open(os.path.join(results_dir, result_file), "r") as f:
            for line in f:
                algo, nodes = line.strip().split(": ")
                algorithms.append(format_algorithm_name(algo))
                expanded_nodes.append(int(nodes))

        # Colores distintos para cada barra
        colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))

        plt.figure(figsize=(12, 6))
        bars = plt.barh(algorithms, expanded_nodes, color=colors, edgecolor="black")

        # Etiquetas de valores al final de cada barra (con . como separador de miles)
        for bar, value in zip(bars, expanded_nodes):
            xval = bar.get_width()
            plt.text(xval + max(expanded_nodes)*0.01, bar.get_y() + bar.get_height()/2,
                     f"{value:,}".replace(",", "."), 
                     va="center", ha="left", fontsize=9)

        plt.ylabel("Algorithm", fontsize=12)
        plt.xlabel("Nodes Expanded", fontsize=12)
        plt.title(f"Nodes Expanded per Algorithm for {map_name}", fontsize=14, weight="bold")
        plt.grid(axis="x", linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.savefig(f"plot_{map_name}.png", dpi=300)
        plt.close()
        print(f"Generated horizontal plot for {map_name} as plot_{map_name}.png")

if __name__ == "__main__":
    plot_results()
