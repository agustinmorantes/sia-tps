import shutil
import sys

from genetic_algorithm.algorithm_definition import EvolutionaryImageApproximator
from genetic_algorithm.utils.generate_canvas import render_solution_to_image
from genetic_algorithm.fitness import ImageSimilarityEvaluator # Importación corregida
from genetic_algorithm.utils.create_individuals import create_initial_population

import cv2
import json
import os

SELECTION_OPTIONAL_PARAMS = {
    "boltzmann": ["boltzmann_temperature"],
    "boltzmann_annealing": ["boltzmann_t0", "boltzmann_tc", "boltzmann_k"],
    "tournament_deterministic": ["tournament_size"],
    "tournament_probabilistic": ["tournament_size", "tournament_threshold"]
}

if __name__ == "__main__":
    config_path = "./config.json"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"El archivo de configuración '{config_path}' no existe.")

    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    config_filename = os.path.splitext(os.path.basename(config_path))[0]
    output_dir = os.path.join("outputs", config_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    target_image_path = config.get("target_image_path")
    initial_population_size = config.get("initial_population_size")
    primitives_per_solution = config.get("triangles_per_solution")
    recombination_probability = config.get("recombination_probability")
    mutations = config.get("mutations")
    mutation_probability = config.get("mutation_probability")
    gen_cutoff = config.get("gen_cutoff")
    parents_selection_percentage = config.get("parents_selection_percentage")
    selection_algorithm_name = config.get("selection_algorithm")
    crossover_algorithm_name = config.get("crossover_algorithm")
    mutation_algorithm_name = config.get("mutation_algorithm")
    mutation_delta_max_percent = config.get("mutation_delta_max_percent")
    fitness_cutoff = config.get("fitness_cutoff", 1.0)
    minutes_cutoff = config.get("minutes_cutoff")
    no_change_gens_cutoff = config.get("no_change_gens_cutoff")
    progress_saving = bool(config.get("progress_saving", True))
    young_bias = bool(config.get("young_bias", True))

    optional_params = {}
    if selection_algorithm_name in SELECTION_OPTIONAL_PARAMS:
        for param_name in SELECTION_OPTIONAL_PARAMS[selection_algorithm_name]:
            if param_name in config:
                optional_params[param_name] = config[param_name]

    for key in config.keys():
        if key.startswith("boltzmann") or key.startswith("tournament"):
            if key not in optional_params:
                raise ValueError(f"Parámetro '{key}' no es aceptable para '{selection_algorithm_name}'")

    img_name = os.path.splitext(os.path.basename(target_image_path))[0]

    target_image = cv2.imread(target_image_path)
    fitness_calculator = ImageSimilarityEvaluator(target_image).evaluate_average_pixel_difference

    initial_solutions = create_initial_population(initial_population_size, primitives_per_solution)

    evolutionary_approximator = EvolutionaryImageApproximator(
        fitness_calculator, target_image, initial_population_size, gen_cutoff, parents_selection_percentage, mutations,
        output_dir
    )

    best_solution, fitness_value, generation_number = evolutionary_approximator.run(
        initial_solutions,
        primitives_per_solution,
        recombination_probability,
        mutation_probability,
        young_bias,
        selection_algorithm_name,
        crossover_algorithm_name,
        mutation_algorithm_name,
        mutation_delta_max_percent,
        fitness_cutoff,
        minutes_cutoff,
        no_change_gens_cutoff,
        progress_saving,
        **optional_params,
    )

    final_image = render_solution_to_image(best_solution)
    final_image.save(
        f"{output_dir}/{img_name}_p{initial_population_size}_t{primitives_per_solution}"
        f"_rec{recombination_probability}_prob_mut{mutation_probability}_r{gen_cutoff}"
        f"_mut{mutations}_sel{selection_algorithm_name}_cross{crossover_algorithm_name}"
        f"_parents{parents_selection_percentage}_fitness{fitness_value}.png"
    )

    print(f"The best result was found on generation {generation_number} with a fitness value of: {fitness_value}")