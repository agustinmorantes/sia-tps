from genetic_algorithm.algorithm_definition import EvolutionaryImageApproximator
from genetic_algorithm.utils.generate_canvas import render_solution_to_image
from genetic_algorithm.fitness import ImageSimilarityEvaluator # Importación corregida
from genetic_algorithm.utils.create_individuals import create_initial_population

import cv2
import json
import os

if __name__ == "__main__":

    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    with open("config.json", 'r') as config_file:
        config = json.load(config_file)
        target_image_path = config.get("target_image_path")
        initial_population_size = config.get("initial_population_size")
        primitives_per_solution = config.get("triangles_per_solution")
        recombination_probability = config.get("recombination_probability")
        mutations = config.get("mutations")
        mutation_probability = config.get("mutation_probability")
        rounds = config.get("rounds")
        parents_selection_percentage = config.get("parents_selection_percentage")
        selection_algorithm_name = config.get("selection_algorithm")
        crossover_algorithm_name = config.get("crossover_algorithm")

    img_name = os.path.splitext(os.path.basename(target_image_path))[0]

    target_image = cv2.imread(target_image_path)
    fitness_calculator = ImageSimilarityEvaluator(target_image).evaluate_average_pixel_difference # Instancia de la clase renombrada y método corregido

    initial_solutions = create_initial_population(initial_population_size, primitives_per_solution)

    evolutionary_approximator = EvolutionaryImageApproximator(fitness_calculator, target_image, initial_population_size, rounds, parents_selection_percentage, mutations)
    best_solution, fitness_value, generation_number = evolutionary_approximator.run(initial_solutions, primitives_per_solution, recombination_probability, mutation_probability, selection_algorithm_name, crossover_algorithm_name)

    final_image = render_solution_to_image(best_solution)
    final_image.save(f"outputs/{img_name}_p{initial_population_size}_t{primitives_per_solution}_rec{recombination_probability}_prob_mut{mutation_probability}_r{rounds}_mut{mutations}_sel{selection_algorithm_name}_cross{crossover_algorithm_name}_parents{parents_selection_percentage}_fitness{fitness_value}.png")
    print(f"The best result was found on generation {generation_number} with a fitness value of: {fitness_value}")