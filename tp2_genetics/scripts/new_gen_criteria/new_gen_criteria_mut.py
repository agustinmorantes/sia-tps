import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

from genetic_algorithm.algorithm_definition import EvolutionaryImageApproximator
from genetic_algorithm.utils.create_individuals import create_initial_population
from genetic_algorithm.fitness import ImageSimilarityEvaluator

# --- Configuración ---
CONFIG_FILE = "./configs/new_gen_criteria/new_gen_criteria_mut_12.json"
OUTPUT_METRICS_FILE = "./metrics/new_gen_criteria_mut_12_1_percent.csv"
OUTPUT_DIR = "./outputs/new_gen_criteria_mut_12_1_percent"

# --- Crear carpetas ---
Path("./metrics").mkdir(exist_ok=True)
Path(OUTPUT_DIR).mkdir(exist_ok=True)
comparison_dir = Path(OUTPUT_DIR) / "comparisons"
comparison_dir.mkdir(exist_ok=True)

# --- Leer configuraciones ---
with open(CONFIG_FILE, "r") as f:
    configs = json.load(f)

all_results = []

# --- Ejecutar experimentos ---
for cfg in configs:
    print(f"\n=== Running: {cfg['selection_algorithm']}, young_bias={cfg['young_bias']} ===")

    # Leer imagen de referencia
    target_image = cv2.imread(cfg["target_image_path"])
    if target_image is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {cfg['target_image_path']}")

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
        output_dir=f"{OUTPUT_DIR}/{cfg.get('name', cfg['selection_algorithm'])}"
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

    # --- Guardar métricas con info de la configuración ---
    df = pd.DataFrame(gen_metrics)
    df["selection_algorithm"] = cfg["selection_algorithm"]
    df["young_bias"] = cfg["young_bias"]
    df["config_name"] = cfg.get("name", f"{cfg['selection_algorithm']}_yb{cfg['young_bias']}")
    # Columnas necesarias para los comparativos
    df["initial_population_size"] = cfg["initial_population_size"]
    df["triangles_per_solution"] = cfg["triangles_per_solution"]
    df["parents_selection_percentage"] = cfg["parents_selection_percentage"]
    df["mutation_probability"] = cfg["mutation_probability"]
    df["mutation_delta_max_percent"] = cfg["mutation_delta_max_percent"]

    all_results.append(df)

    # --- Graficar individual por caso ---
    case_name = df["config_name"].iloc[0]
    case_dir = Path(OUTPUT_DIR) / case_name
    case_dir.mkdir(exist_ok=True, parents=True)

    # 1. Fitness vs Generación
    plt.figure(figsize=(10, 5))
    plt.plot(df["generation"], df["max_fitness"], label="Max fitness")
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.title(f"Fitness vs Generación - {case_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(case_dir / "fitness_vs_gen.png")
    plt.close()

    # 2. Diversidad vs Generación
    if "diversity" in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df["generation"], df["diversity"], label="Diversidad")
        plt.xlabel("Generación")
        plt.ylabel("Diversidad")
        plt.title(f"Diversidad vs Generación - {case_name}")
        plt.legend()
        plt.grid(True)
        plt.savefig(case_dir / "diversity_vs_gen.png")
        plt.close()

        # 3. Diversidad vs Fitness
        plt.figure(figsize=(10, 5))
        plt.scatter(df["max_fitness"], df["diversity"], s=10, alpha=0.7)
        plt.xlabel("Fitness")
        plt.ylabel("Diversidad")
        plt.title(f"Diversidad vs Fitness - {case_name}")
        plt.grid(True)
        plt.savefig(case_dir / "diversity_vs_fitness.png")
        plt.close()

# --- Concatenar resultados ---
results_df = pd.concat(all_results, ignore_index=True)
results_df.to_csv(OUTPUT_METRICS_FILE, index=False)
print(f"\n✅ All metrics saved to {OUTPUT_METRICS_FILE}")

# --- Graficos comparativos Young Bias vs Traditional ---
for sel_algo in results_df["selection_algorithm"].unique():
    sel_df = results_df[results_df["selection_algorithm"] == sel_algo]

    # Comparar Young vs Traditional para cada grupo de parámetros
    group_cols = ["initial_population_size", "triangles_per_solution",
                  "parents_selection_percentage", "mutation_probability",
                  "mutation_delta_max_percent"]

    for param_values, group_df in sel_df.groupby(group_cols):
        plt.figure(figsize=(10, 5))
        for bias in [True, False]:
            bias_df = group_df[group_df["young_bias"] == bias]
            plt.plot(bias_df["generation"], bias_df["max_fitness"], label=f"Young Bias={bias}")
        plt.xlabel("Generación")
        plt.ylabel("Fitness")
        plt.title(f"Comparativo Fitness vs Generación - {sel_algo}")
        plt.legend()
        plt.grid(True)
        plt.savefig(comparison_dir / f"{sel_algo}_fitness_vs_gen_{param_values}.png")
        plt.close()

        if "diversity" in group_df.columns:
            plt.figure(figsize=(10, 5))
            for bias in [True, False]:
                bias_df = group_df[group_df["young_bias"] == bias]
                plt.plot(bias_df["generation"], bias_df["diversity"], label=f"Young Bias={bias}")
            plt.xlabel("Generación")
            plt.ylabel("Diversidad")
            plt.title(f"Comparativo Diversidad vs Generación - {sel_algo}")
            plt.legend()
            plt.grid(True)
            plt.savefig(comparison_dir / f"{sel_algo}_diversity_vs_gen_{param_values}.png")
            plt.close()

            plt.figure(figsize=(10, 5))
            for bias in [True, False]:
                bias_df = group_df[group_df["young_bias"] == bias]
                plt.scatter(bias_df["max_fitness"], bias_df["diversity"], s=10, alpha=0.7, label=f"Young Bias={bias}")
            plt.xlabel("Fitness")
            plt.ylabel("Diversidad")
            plt.title(f"Comparativo Diversidad vs Fitness - {sel_algo}")
            plt.legend()
            plt.grid(True)
            plt.savefig(comparison_dir / f"{sel_algo}_diversity_vs_fitness_{param_values}.png")
            plt.close()
