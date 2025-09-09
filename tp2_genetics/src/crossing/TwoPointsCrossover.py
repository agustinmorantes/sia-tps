import random
from typing import Tuple
from .Crossing import Crossing
from ..classes.Individual import Individual

class TwoPointsCrossover(Crossing):
    def cross(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        genome_size = parent1.get_genome_size()
        
        if genome_size <= 2:
            return parent1.copy(), parent2.copy()
        
        crossover_point1 = random.randint(1, genome_size - 1)
        crossover_point2 = random.randint(crossover_point1, genome_size - 1)

        child1_triangles = (parent1.triangles[:crossover_point1] + 
                            parent2.triangles[crossover_point1:crossover_point2] + 
                            parent1.triangles[crossover_point2:])
        child2_triangles = (parent2.triangles[:crossover_point1] + 
                            parent1.triangles[crossover_point1:crossover_point2] + 
                            parent2.triangles[crossover_point2:])

        child1 = Individual(child1_triangles)
        child2 = Individual(child2_triangles)

        return child1, child2
