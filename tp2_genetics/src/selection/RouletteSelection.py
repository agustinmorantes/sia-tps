import random
from typing import List
from .Selection import Selection
from ..classes.Individual import Individual

class RouletteSelection(Selection):
    def select(self, population: List[Individual], num_selected: int) -> List[Individual]:
        total_fitness = sum(individual.fitness for individual in population)
        
        if total_fitness == 0:
            return random.choices(population, k=num_selected)
        
        selection_probs = [individual.fitness / total_fitness for individual in population]
        selected_individuals = []
        
        for _ in range(num_selected):
            random_value = random.random()
            current_prob = 0
            for i in range(len(population)):
                current_prob += selection_probs[i]
                if current_prob > random_value:
                    selected_individuals.append(population[i])
                    break
        return selected_individuals