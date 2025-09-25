import json
import copy
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import os

from genetic_algorithm.algorithm_definition import EvolutionaryImageApproximator
from genetic_algorithm.utils.create_individuals import create_initial_population

# --- Configuración ---
CONFIG_FILE = "./configs/new_generation_criteria.json"
OUTPUT_METRICS_FILE = "./metrics/all_experiments_metrics.csv"

# --- Leer configuraciones ---
with open(CONFIG_FILE, "r") as f:
    configs = json.load(f)

# --- Crear carpeta de métricas ---
Path("./metrics").mkdir(exist_ok=True)

# --- Ejecutar experimentos ---
all_results = []

for cfg in configs:
    print(f"\n=== Running: {cfg['selection_algorithm']}, young_bias={cfg['young_bias']} ===")

    # Leer imagen de referencia
    target_image = cv2.imread(cfg["target_image_path"])
    from genetic_algorithm.fitness import ImageSimilarityEvaluator
    fitness_calculator = ImageSimilarityEvaluator(target_image).evaluate_average_pixel_difference

    # Crear población inicial
    initial_population = create_initial_population(cfg["initial_population_size"], cfg["triangles_per_solution"])

    # Crear aproximador
    approximator = EvolutionaryImageApproximator(
        similarity_evaluator=fitness_calculator,
        reference_image=target_image,
        initial_population_count=cfg["initial_population_size"],
        gen_cutoff=cfg["gen_cutoff"],
        parents_selection_percentage=cfg["parents_selection_percentage"],
        mutation_gens=cfg.get("mutation_algorithm", "single"),
        output_dir=f"outputs/{cfg.get('name', cfg['selection_algorithm'])}"
    )

    # Ejecutar evolución
    best_solution, max_fitness, generations, cutoff_reason, gen_metrics = approximator.run(
        initial_solution_set=initial_population,
        primitives_per_solution=cfg["triangles_per_solution"],
        recombination_probability=cfg["recombination_probability"],
        mutation_probability=cfg["mutation_probability"],
        mutation_delta_max_percent=cfg["mutation_delta_max_percent"],
        young_bias=cfg["young_bias"],
        selection_algorithm_name=cfg["selection_algorithm"],
        crossover_method_name=cfg.get("crossover_algorithm", "one_point"),
        mutation_algorithm_name=cfg.get("mutation_algorithm", "single"),
        fitness_cutoff=cfg["fitness_cutoff"],
        minutes_cutoff=cfg["minutes_cutoff"],
        no_change_gens_cutoff=cfg["no_change_gens_cutoff"],
        progress_saving=cfg.get("progress_saving", False),
        **{k: v for k, v in cfg.items() if k.startswith("boltzmann")}
    )

    # Guardar métricas con info de la configuración
    df = pd.DataFrame(gen_metrics)
    df["selection_algorithm"] = cfg["selection_algorithm"]
    df["young_bias"] = cfg["young_bias"]
    df["config_name"] = cfg.get("name", f"{cfg['selection_algorithm']}_yb{cfg['young_bias']}")
    all_results.append(df)

# --- Concatenar resultados ---
results_df = pd.concat(all_results, ignore_index=True)
results_df.to_csv(OUTPUT_METRICS_FILE, index=False)
print(f"\n✅ All metrics saved to {OUTPUT_METRICS_FILE}")

# --- Graficar fitness máximo ---
plt.figure(figsize=(12, 6))
for (sel, bias), group in results_df.groupby(["selection_algorithm", "young_bias"]):
    plt.plot(group["generation"], group["max_fitness"], label=f"{sel}, young_bias={bias}")

plt.xlabel("Generación")
plt.ylabel("Fitness máximo")
plt.title("Comparación de criterios de nueva generación y selección")
plt.legend()
plt.grid(True)
plt.show()
