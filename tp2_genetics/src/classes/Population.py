import random
from .Individual import Individual
from .Triangle import Triangle
from typing import List

class Population: # Representa una población de individuos.
    def __init__(self, individuals: List[Individual]):
        self.individuals = individuals
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []

    def size(self) -> int:
        return len(self.individuals)
    
    def get_best_individual(self) -> Individual:
        return max(self.individuals, key=lambda x: x.fitness)

    def get_worst_individual(self) -> Individual:
        return min(self.individuals, key=lambda x: x.fitness)
    
    def get_avg_fitness(self) -> float:
        if not self.individuals:
            return 0.0
        return sum(individual.fitness for individual in self.individuals) / len(self.individuals)
    
    def update_fitness_history(self):
        self.best_fitness_history.append(self.get_best_individual().fitness)
        self.avg_fitness_history.append(self.get_avg_fitness())

    def increment_generation(self):
        self.generation += 1

    def sort_by_fitness(self):
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)
    
    @classmethod
    def random_population(cls, population_size: int, num_triangles: int, width: int, height: int) -> "Population":
        individuals = [Individual.random_individual(num_triangles, width, height) for _ in range(population_size)]
        return cls(individuals)