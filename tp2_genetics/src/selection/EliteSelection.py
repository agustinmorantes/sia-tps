from .Selection import Selection
from typing import List
from ..classes.Individual import Individual

class EliteSelection(Selection):
    def select(self, population: List[Individual], num_selected: int) -> List[Individual]:
        sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)
        return sorted_population[:num_selected]