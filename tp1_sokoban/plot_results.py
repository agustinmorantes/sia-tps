import os
import matplotlib.pyplot as plt
import numpy as np
import ast # Import the ast module

def format_algorithm_name(name: str) -> str:
    name = name.strip().lower()
    if name.startswith("a_star"):
        parts = name.split("_", 2)
        if len(parts) > 2:
            return "A* (" + parts[-1].capitalize() + ")"
        else:
            return "A*"
    elif name.startswith("greedy"):
        parts = name.split("_", 1)
        if len(parts) > 1:
            return "Greedy (" + parts[-1].capitalize() + ")"
        else:
            return "Greedy"
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
        
        all_metrics = {
            "solution_cost": {},
            "nodes_expanded": {},
            "border_nodes_count": {}
        }

        with open(os.path.join(results_dir, result_file), "r") as f:
            for line in f:
                parts = line.strip().split(": ", 1) # Split only on the first ": "
                if len(parts) < 2: # Skip malformed lines
                    continue
                algo_raw, stats_str = parts
                formatted_algo = format_algorithm_name(algo_raw)
                
                try:
                    stats = ast.literal_eval(stats_str)
                except (ValueError, SyntaxError) as e:
                    print(f"Error parsing stats for {algo_raw} in {result_file}: {e}")
                    continue

                if stats["nodes_expanded"] != -1: # Only plot if a solution was found
                    all_metrics["solution_cost"][formatted_algo] = stats["solution_cost"]
                    all_metrics["nodes_expanded"][formatted_algo] = stats["nodes_expanded"]
                    all_metrics["border_nodes_count"][formatted_algo] = stats["border_nodes_count"]

        # Prepare data for plotting
        algorithms = [a for a in algo_order if a in all_metrics["nodes_expanded"]]

        if not algorithms: # Skip if no data for plotting
            print(f"No valid data to plot for {map_name}")
            continue

        metric_titles = {
            "nodes_expanded": "Nodes Expanded",
            "solution_cost": "Solution Cost (Movements)",
            "border_nodes_count": "Max Border Nodes"
        }

        for metric_key, metric_title in metric_titles.items():
            values = [all_metrics[metric_key][a] for a in algorithms]
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))

            plt.figure(figsize=(12, 6))
            bars = plt.barh(algorithms, values, color=colors, edgecolor="black")

            max_val = max(values) if values else 1 # Avoid division by zero

            plt.xlim(0, max_val * 1.1)

            for bar, value in zip(bars, values):
                xval = bar.get_width()
                plt.text(xval + max_val * 0.01, bar.get_y() + bar.get_height()/2,
                         f"{value:,}".replace(",", "."),
                         va="center", ha="left", fontsize=9)

            plt.ylabel("Algorithm", fontsize=12)
            plt.xlabel(metric_title, fontsize=12)
            plt.title(f"{metric_title} per Algorithm for {map_name}", fontsize=14, weight="bold")
            plt.grid(axis="x", linestyle="--", alpha=0.6)

            plt.tight_layout()
            output_path = os.path.join(output_dir, f"plot_{map_name}_{metric_key}.png")
            plt.savefig(output_path, dpi=300)
            plt.close()
            print(f"Generated plot for {map_name} - {metric_title} in {output_path}")

if __name__ == "__main__":
    plot_results()
