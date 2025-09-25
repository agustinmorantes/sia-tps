import json
import os.path
import numpy as np
import matplotlib.pyplot as plt


def load_run_metrics(output_dir: str) -> dict:
    metrics_path = os.path.join(output_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        return None

    metrics: dict
    with open(metrics_path, 'r') as file:
        metrics = json.load(file)

    gen_metrics = metrics['gen_metrics']
    keys = gen_metrics[0].keys()
    gen_metrics = { k: np.stack([d[k] for d in gen_metrics]) for k in keys }
    metrics['gen_metrics'] = gen_metrics

    return metrics

if __name__ == "__main__":
    graph_dir = "graphs"
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir, exist_ok=True)

    config_names = [os.path.splitext(config)[0] for config in os.listdir("configs")]
    output_dirs = ["outputs"]
    all_run_metrics = {config: [load_run_metrics(os.path.join(output_dir, config)) for output_dir in output_dirs] for config in config_names}

    graphs = [
        {
            "configs": ("elite_one_point", "elite_two_points"),
            "legend": ("One Point", "Two Points"),
            "title": "Estrategia de selección 'Elite'",
            "filename": "elite_selection_strategy_crossover_fitness.png",
            "ylabel": "Fitness máximo",
            "ylim": (0, 1),
            "xlabel": "Generación",
            "y_data_key": "max_fitness",
            "x_data_key": "generation",
        },
        {
            "configs": ("elite_one_point", "elite_two_points"),
            "legend": ("One Point", "Two Points"),
            "title": "Estrategia de selección 'Elite'",
            "filename": "elite_selection_strategy_crossover_diversity.png",
            "ylabel": "Diversidad",
            "ylim": (0, 5),
            "xlabel": "Generación",
            "y_data_key": "diversity",
            "x_data_key": "generation",
        },
        {
            "configs": ("base", "two_points_base"),
            "legend": ("One Point", "Two Points"),
            "title": "Configuración Base",
            "filename": "base_config_diversity.png",
            "ylabel": "Diversidad",
            "ylim": (0, 5),
            "xlabel": "Generación",
            "y_data_key": "diversity",
            "x_data_key": "generation",
        },
        {
            "configs": ("base", "two_points_base"),
            "legend": ("One Point", "Two Points"),
            "title": "Configuración Base",
            "filename": "base_config_fitness.png",
            "ylabel": "Fitness máximo",
            "ylim": (0, 1),
            "xlabel": "Generación",
            "y_data_key": "max_fitness",
            "x_data_key": "generation",
        },
        {
            "configs": ("higher_pop_one_point", "higher_pop_two_points"),
            "legend": ("One Point", "Two Points"),
            "title": "Población de 400 individuos",
            "filename": "higher_pop_diversity.png",
            "ylabel": "Diversidad",
            "ylim": (0, 5),
            "xlabel": "Generación",
            "y_data_key": "diversity",
            "x_data_key": "generation",
        },
        {
            "configs": ("higher_pop_one_point", "higher_pop_two_points"),
            "legend": ("One Point", "Two Points"),
            "title": "Población de 400 individuos",
            "filename": "higher_pop_fitness.png",
            "ylabel": "Fitness máximo",
            "ylim": (0, 1),
            "xlabel": "Generación",
            "y_data_key": "max_fitness",
            "x_data_key": "generation",
        },
        {
            "configs": ("lower_mutation_probability_one_point", "lower_mutation_probability_two_points"),
            "legend": ("One Point", "Two Points"),
            "title": "Probabilidad de mutación 0.3",
            "filename": "lower_mutation_probability_diversity.png",
            "ylabel": "Diversidad",
            "ylim": (0, 5),
            "xlabel": "Generación",
            "y_data_key": "diversity",
            "x_data_key": "generation",
        },
        {
            "configs": ("lower_mutation_probability_one_point", "lower_mutation_probability_two_points"),
            "legend": ("One Point", "Two Points"),
            "title": "Probabilidad de mutación 0.3",
            "filename": "lower_mutation_probability_fitness.png",
            "ylabel": "Fitness máximo",
            "ylim": (0, 1),
            "xlabel": "Generación",
            "y_data_key": "max_fitness",
            "x_data_key": "generation",
        },
        {
            "configs": ("lower_recombination_probability_one_point", "lower_recombination_probability_two_points"),
            "legend": ("One Point", "Two Points"),
            "title": "Probabilidad de recombinación 0.4",
            "filename": "lower_recombination_probability_diversity.png",
            "ylabel": "Diversidad",
            "ylim": (0, 5),
            "xlabel": "Generación",
            "y_data_key": "diversity",
            "x_data_key": "generation",
        },
        {
            "configs": ("lower_recombination_probability_one_point", "lower_recombination_probability_two_points"),
            "legend": ("One Point", "Two Points"),
            "title": "Probabilidad de recombinación 0.4",
            "filename": "lower_recombination_probability_fitness.png",
            "ylabel": "Fitness máximo",
            "ylim": (0, 1),
            "xlabel": "Generación",
            "y_data_key": "max_fitness",
            "x_data_key": "generation",
        },
        {
            "configs": ("lower_selection_percentage_one_point", "lower_selection_percentage_two_points"),
            "legend": ("One Point", "Two Points"),
            "title": "Porcentaje de población para selección de padres 20%",
            "filename": "lower_selection_percentage_diversity.png",
            "ylabel": "Diversidad",
            "ylim": (0, 5),
            "xlabel": "Generación",
            "y_data_key": "diversity",
            "x_data_key": "generation",
        },
        {
            "configs": ("lower_selection_percentage_one_point", "lower_selection_percentage_two_points"),
            "legend": ("One Point", "Two Points"),
            "title": "Porcentaje de población para selección de padres 20%",
            "filename": "lower_selection_percentage_fitness.png",
            "ylabel": "Fitness máximo",
            "ylim": (0, 1),
            "xlabel": "Generación",
            "y_data_key": "max_fitness",
            "x_data_key": "generation",
        },
        {
            "configs": ("more_triangles_one_point", "more_triangles_two_points"),
            "legend": ("One Point", "Two Points"),
            "title": "60 triángulos por solución",
            "filename": "more_triangles_diversity.png",
            "ylabel": "Diversidad",
            "ylim": (0, 5),
            "xlabel": "Generación",
            "y_data_key": "diversity",
            "x_data_key": "generation",
        },
        {
            "configs": ("more_triangles_one_point", "more_triangles_two_points"),
            "legend": ("One Point", "Two Points"),
            "title": "60 triángulos por solución",
            "filename": "more_triangles_fitness.png",
            "ylabel": "Fitness máximo",
            "ylim": (0, 1),
            "xlabel": "Generación",
            "y_data_key": "max_fitness",
            "x_data_key": "generation",
        },
    ]

    # Generate all graphs
    for graph in graphs:
        plt.figure(figsize=(10, 6))

        for i, config in enumerate(graph["configs"]):
            if config in all_run_metrics and all_run_metrics[config][0] is not None:
                metrics = all_run_metrics[config][0]
                gen_metrics = metrics['gen_metrics']

                x_data = gen_metrics[graph["x_data_key"]]
                y_data = gen_metrics[graph["y_data_key"]]

                plt.plot(x_data, y_data, label=graph["legend"][i], linewidth=2)

        plt.title(graph["title"], fontsize=14, fontweight='bold')
        plt.xlabel(graph["xlabel"], fontsize=12)
        plt.ylabel(graph["ylabel"], fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        if "ylim" in graph:
            plt.ylim(graph["ylim"])

        plt.tight_layout()

        # Save the graph
        output_path = os.path.join(graph_dir, graph["filename"])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Generated graph: {graph['filename']}")

    print(f"All graphs saved to '{graph_dir}' directory")
