from genetic_algorithm.utils.create_individuals import create_initial_population
from genetic_algorithm.selection_algorithms.selection_strategies import EliteSelection, RouletteSelection, UniversalSelection, SelectionStrategy, TournamentDeterministicSelection, TournamentProbabilisticSelection, RankingSelection, BoltzmannSelection, BoltzmannAnnealingSelection
from genetic_algorithm.crossover import Crossover, OnePointCrossover, TwoPointsCrossover, CrossoverStrategy
from genetic_algorithm.mutation import GeneModifier
from genetic_algorithm.utils.random_seed_manager import central_random_generator as random_generator # Importación del generador centralizado
from genetic_algorithm.fitness import ImageSimilarityEvaluator
from genetic_algorithm.models.individual_solution import IndividualSolution
import numpy as np
import time
import copy
from typing import List, Tuple

class EvolutionaryImageApproximator:
    """Implementa el algoritmo genético para aproximar una imagen con formas geométricas."""
    def __init__(self, similarity_evaluator, reference_image, initial_population_count=50, rounds=200, parents_selection_percentage=0.25, mutation_gens="single"):
        self.current_population: List[IndividualSolution] = [] # Anotación de tipo
        self.initial_population_count = initial_population_count # Renombrado
        self.best_solution_found: IndividualSolution = None # Renombrado y anotación de tipo
        self.similarity_evaluator = similarity_evaluator # Renombrado
        self.rounds = rounds
        self.max_fitness = 0
        self.generation_number = 0
        self.reference_image = reference_image # Renombrado
        self.parents_selection_percentage = parents_selection_percentage
        self.mutation_gens = mutation_gens
        self.fitness_cache = {}

    def calculate_population_fitness(self, population: List[IndividualSolution]) -> Tuple[List[float], float, IndividualSolution]: # Anotaciones de tipo
        fitness_values = []
        best_fitness_value = 0.0 # Inicializar como float
        best_solution: IndividualSolution = None # Renombrado y anotación de tipo

        for individual in population:
            individual_hash = hash(individual)
            if individual_hash in self.fitness_cache:
                value = self.fitness_cache[individual_hash]
            else:
                value = self.similarity_evaluator(individual) # Llamada a la función corregida
                self.fitness_cache[individual_hash] = value
            
            fitness_values.append(value)

            if value > best_fitness_value:
                best_fitness_value = value
                best_solution = individual # Renombrado

        return fitness_values, best_fitness_value, best_solution

    def run(
            self, 
            initial_solution_set: List[IndividualSolution], 
            primitives_per_solution: int = 50, 
            recombination_probability: float = 1.0,
            mutation_probability: float = 0.0,
            selection_algorithm_name: str ="elite",
            crossover_method_name: str ="one_point",
            **selection_params
        ) -> Tuple[IndividualSolution, float, int]: # Anotaciones de tipo

        self.current_population = initial_solution_set # Renombrado

        size = int(self.parents_selection_percentage * self.initial_population_count) # Renombrado
        
        # Selección del método de selección
        selection_strategy: SelectionStrategy
        if selection_algorithm_name == "elite":
            selection_strategy = EliteSelection(size)
        elif selection_algorithm_name == "roulette":
            selection_strategy = RouletteSelection(size)
        elif selection_algorithm_name == "universal":
            selection_strategy = UniversalSelection(size)
        elif selection_algorithm_name == "tournament_deterministic":
            tournament_size = selection_params.get("tournament_size", None)
            if tournament_size is not None:
                selection_strategy = TournamentDeterministicSelection(size, tournament_size=tournament_size)
            else:
                selection_strategy = TournamentDeterministicSelection(size)
        elif selection_algorithm_name == "tournament_probabilistic":
            tournament_size = selection_params.get("tournament_size", None)
            tournament_threshold = selection_params.get("tournament_threshold", None)
            if tournament_size is not None and tournament_threshold is not None:
                selection_strategy = TournamentProbabilisticSelection(
                    size, tournament_size=tournament_size, threshold=tournament_threshold
                )
            else:
                selection_strategy = TournamentProbabilisticSelection(size)
        elif selection_algorithm_name == "ranking":
            selection_strategy = RankingSelection(size)
        elif selection_algorithm_name == "boltzmann":
            boltzmann_temperature = selection_params.get("boltzmann_temperature", None)
            if boltzmann_temperature is not None:
                selection_strategy = BoltzmannSelection(size, temperature=boltzmann_temperature)
            else:
                selection_strategy = BoltzmannSelection(size)
        elif selection_algorithm_name == "boltzmann_annealing":
            boltzmann_t0 = selection_params.get("boltzmann_t0", None)
            boltzmann_tc = selection_params.get("boltzmann_tc", None)
            boltzmann_k = selection_params.get("boltzmann_k", None)
            if boltzmann_t0 is not None and boltzmann_tc is not None and boltzmann_k is not None:
                selection_strategy = BoltzmannAnnealingSelection(size, t0=boltzmann_t0, tc=boltzmann_tc, k=boltzmann_k)
            else:
                selection_strategy = BoltzmannAnnealingSelection(size)
        else:
            raise ValueError(f"Método de selección desconocido: {selection_algorithm_name}")
        
        # Las clases OnePointCrossover y TwoPointsCrossover reciben primitives_per_solution en su init
        crossover_strategy: CrossoverStrategy # Instanciar directamente la estrategia
        if crossover_method_name == "one_point":
            crossover_strategy = OnePointCrossover(primitives_per_solution)
        elif crossover_method_name == "two_points":
            crossover_strategy = TwoPointsCrossover(primitives_per_solution)
        else:
            raise ValueError(f"Método de cruce desconocido: {crossover_method_name}")

        gene_modifier = GeneModifier(mutation_probability, primitives_per_solution) # Instancia de GeneModifier

        start_time = time.time()
        while self.generation_number <= self.rounds and self.max_fitness < 1.0:

            fitness_values, generation_max_fitness, best_current_solution = self.calculate_population_fitness(self.current_population) # Renombrado

            if generation_max_fitness > self.max_fitness:
                self.max_fitness = generation_max_fitness 
                self.best_solution_found = copy.deepcopy(best_current_solution) # Renombrado

            elapsed = time.time() - start_time
            mins, secs = divmod(int(elapsed), 60)
            print(f"Generation: {self.generation_number}, Max fitness: {self.max_fitness}, Time: {mins:02d}:{secs:02d}")
            
            selected_parents = selection_strategy.select(self.current_population, self.fitness_cache) # Renombrado

            offspring = [] # Renombrado de children a offspring
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

            gene_modifier.apply_mutations(offspring) # Llamada al nuevo método
       
            self.current_population.extend(offspring) # Renombrado
            self.calculate_population_fitness(self.current_population) # Renombrado
            # Usar EliteSelection para la selección de la próxima generación desde la población combinada
            next_generation_selector = EliteSelection(self.initial_population_count) # Usar EliteSelection como estrategia por defecto
            self.current_population = next_generation_selector.select(self.current_population, self.fitness_cache) # Renombrado
            self.generation_number += 1

        print(self.max_fitness)
        print(self.similarity_evaluator(self.best_solution_found)) # Renombrado
        return self.best_solution_found, self.max_fitness, self.generation_number