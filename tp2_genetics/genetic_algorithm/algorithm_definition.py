import copy
import os
import time
import shutil
from typing import List, Tuple

from genetic_algorithm.crossover import OnePointCrossover, TwoPointsCrossover, CrossoverStrategy
from genetic_algorithm.models.individual_solution import IndividualSolution
from genetic_algorithm.mutation import get_mutation_strategy
from genetic_algorithm.selection_algorithms.selection_strategies import get_selection_strategy
from genetic_algorithm.utils.generate_canvas import render_solution_to_image
from genetic_algorithm.utils.random_seed_manager import central_random_generator as random_generator


class EvolutionaryImageApproximator:
    """Implementa el algoritmo genético para aproximar una imagen con formas geométricas."""
    def __init__(self, similarity_evaluator, reference_image, initial_population_count=50, gen_cutoff=200, parents_selection_percentage=0.25, mutation_gens="single", output_dir="outputs"):
        self.current_population: List[IndividualSolution] = []
        self.initial_population_count = initial_population_count
        self.best_solution_found: IndividualSolution | None = None
        self.similarity_evaluator = similarity_evaluator
        self.gen_cutoff = gen_cutoff
        self.max_fitness = 0
        self.generation_number = 0
        self.best_solution_gen = 0
        self.reference_image = reference_image
        self.parents_selection_percentage = parents_selection_percentage
        self.mutation_gens = mutation_gens
        self.fitness_cache = {}
        self.output_dir = output_dir
        self.progress_dir = os.path.join(output_dir, "progress")

        if os.path.exists(self.progress_dir):
            shutil.rmtree(self.progress_dir)
        os.makedirs(self.progress_dir, exist_ok=True)


    def calculate_population_fitness(self, population: List[IndividualSolution]) -> Tuple[List[float], float, IndividualSolution]:
        fitness_values = []
        best_fitness_value = 0.0
        best_solution: IndividualSolution | None = None

        for individual in population:
            individual_hash = hash(individual)
            if individual_hash in self.fitness_cache:
                value = self.fitness_cache[individual_hash]
            else:
                value = self.similarity_evaluator(individual)
                self.fitness_cache[individual_hash] = value
            
            fitness_values.append(value)

            if value > best_fitness_value:
                best_fitness_value = value
                best_solution = individual

        return fitness_values, best_fitness_value, best_solution

    def save_solution_image(self, solution: IndividualSolution, generation: int):
        final_image = render_solution_to_image(solution)
        final_image.save(
            os.path.join(self.progress_dir, f"gen_{generation}.png"),
        )

    def run(
            self, 
            initial_solution_set: List[IndividualSolution], 
            primitives_per_solution: int = 50, 
            recombination_probability: float = 1.0,
            mutation_probability: float = 0.0,
            young_bias: bool = True,
            selection_algorithm_name: str ="elite",
            crossover_method_name: str ="one_point",
            mutation_algorithm_name: str ="limited_multi",
            mutation_delta_max_percent: float = 0.2,
            fitness_cutoff: float = 1.0,
            minutes_cutoff: float = 120,
            no_change_gens_cutoff: int = 500,
            progress_saving: bool = True,
            **selection_params
        ) -> Tuple[IndividualSolution, float, int, str, list[dict]]:

        gen_metrics = []

        self.current_population = initial_solution_set

        k_size = int(self.parents_selection_percentage * self.initial_population_count)
        
        selection_strategy = get_selection_strategy(selection_algorithm_name, selection_params)
        mutation_strategy = get_mutation_strategy(mutation_algorithm_name, mutation_probability, mutation_delta_max_percent)

        # Las clases OnePointCrossover y TwoPointsCrossover reciben primitives_per_solution en su init
        crossover_strategy: CrossoverStrategy # Instanciar directamente la estrategia
        if crossover_method_name == "one_point":
            crossover_strategy = OnePointCrossover(primitives_per_solution)
        elif crossover_method_name == "two_points":
            crossover_strategy = TwoPointsCrossover(primitives_per_solution)
        else:
            raise ValueError(f"Método de cruce desconocido: {crossover_method_name}")

        start_time = time.time()
        cutoff_reason = "gen_cutoff"
        try:
            while self.generation_number <= self.gen_cutoff:
                fitness_values, generation_max_fitness, best_current_solution = self.calculate_population_fitness(self.current_population)

                gen_metrics.append({
                    "generation": self.generation_number,
                    "max_fitness": generation_max_fitness,
                    "average_fitness": sum(fitness_values) / len(fitness_values),
                    "min_fitness": min(fitness_values),
                    "population_size": len(self.current_population),
                    "time_elapsed": time.time() - start_time,
                })

                if generation_max_fitness > self.max_fitness:
                    self.max_fitness: float = generation_max_fitness
                    self.best_solution_found: IndividualSolution = copy.deepcopy(best_current_solution)
                    self.best_solution_gen = self.generation_number

                    if self.max_fitness >= fitness_cutoff:
                        print("Fitness cutoff reached. Stopping the algorithm.")
                        cutoff_reason = "fitness_cutoff"
                        break

                    if progress_saving:
                        self.save_solution_image(self.best_solution_found, self.generation_number)

                if self.generation_number - self.best_solution_gen >= no_change_gens_cutoff:
                    print(f"No improvement in fitness for {no_change_gens_cutoff} generations. Stopping the algorithm.")
                    cutoff_reason = "no_change_gens_cutoff"
                    break

                elapsed = time.time() - start_time
                mins, secs = divmod(int(elapsed), 60)

                if mins >= minutes_cutoff:
                    print(f"Time cutoff of {minutes_cutoff} minutes reached. Stopping the algorithm.")
                    cutoff_reason = "time_cutoff"
                    break

                print(f"Generation: {self.generation_number}, Max fitness: {self.max_fitness}, Time: {mins:02d}:{secs:02d}")

                selected_parents = selection_strategy.select(k_size, self.current_population, self.fitness_cache, self.generation_number)

                offspring = []
                shuffled_parents = selected_parents.copy()
                random_generator.shuffle(shuffled_parents)
                for i in range(int(len(shuffled_parents)/2)):
                    parent_a = i * 2
                    parent_b = i * 2 + 1
                    random_recombination_check = random_generator.random()
                    if random_recombination_check < recombination_probability:
                        current_parent_pair = [shuffled_parents[parent_a], shuffled_parents[parent_b]]

                        new_offspring_pair = crossover_strategy.cross(current_parent_pair) # Uso de la estrategia de cruce

                        for child in new_offspring_pair:
                            offspring.append(child)
                    else:
                        offspring.append(shuffled_parents[parent_a])
                        offspring.append(shuffled_parents[parent_b])

                mutation_strategy.apply_mutations_to_population(offspring)

                self.calculate_population_fitness(offspring) # Calculo el fitness de los nuevos individuos
                if young_bias: # Sesgo joven
                    n_size = self.initial_population_count
                    if k_size > n_size:
                        self.current_population = selection_strategy.select(n_size, offspring, self.fitness_cache, self.generation_number)
                    else:
                        self.current_population = selection_strategy.select(n_size - k_size, self.current_population, self.fitness_cache, self.generation_number)
                        selected_offspring = selection_strategy.select(k_size, offspring, self.fitness_cache, self.generation_number)
                        self.current_population.extend(selected_offspring)
                else: # Tradicional
                    self.current_population.extend(offspring)
                    self.current_population = selection_strategy.select(self.initial_population_count, self.current_population, self.fitness_cache, self.generation_number)

                self.generation_number += 1
        except KeyboardInterrupt:
                print("Execution interrupted by the user. Saving the best solution found so far...")

        print(self.max_fitness)
        print(self.similarity_evaluator(self.best_solution_found))
        return self.best_solution_found, self.max_fitness, self.generation_number, cutoff_reason, gen_metrics
