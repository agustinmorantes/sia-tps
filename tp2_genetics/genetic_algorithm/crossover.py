from genetic_algorithm.models.individual_solution import IndividualSolution, GeometricPrimitive
from genetic_algorithm.utils.random_seed_manager import central_random_generator as random_generator # Importación del generador centralizado
from typing import List
from abc import ABC, abstractmethod

class CrossoverStrategy(ABC):
    def __init__(self, triangles_amount):
        self.triangles_amount = triangles_amount

    @abstractmethod
    def cross(self, parents: List[IndividualSolution]) -> List[IndividualSolution]:
        pass

class OnePointCrossover(CrossoverStrategy):
    def cross(self, parents: List[IndividualSolution]) -> List[IndividualSolution]:
        childs = []
        cross_point = random_generator.randint(0, len(parents[0].chromosome) - 1)

        p1,p2 = parents[0], parents[1]

        if len(p1.chromosome) != len(p2.chromosome):
            raise ValueError("Los individuos deben tener la misma cantidad de genes")
        
        child1_genes = p1.chromosome[:cross_point] + p2.chromosome[cross_point:]
        child2_genes = p2.chromosome[:cross_point] + p1.chromosome[cross_point:]

        childs.append(Crossover.build_individual_solution_from_genes(child1_genes))
        childs.append(Crossover.build_individual_solution_from_genes(child2_genes))

        return childs

class TwoPointsCrossover(CrossoverStrategy):
    def cross(self, parents: List[IndividualSolution]) -> List[IndividualSolution]:
        childs = []
        genome_size = len(parents[0].chromosome)
        
        if genome_size <= 2:
            return parents[0], parents[1]
        
        crossover_point1 = random_generator.randint(1, genome_size - 1)
        crossover_point2 = random_generator.randint(crossover_point1, genome_size - 1)

        p1,p2 = parents[0], parents[1]

        child1_genes = (p1.chromosome[:crossover_point1] +
                            p2.chromosome[crossover_point1:crossover_point2] +
                            p1.chromosome[crossover_point2:])
        child2_genes = (p2.chromosome[:crossover_point1] +
                            p1.chromosome[crossover_point1:crossover_point2] +
                            p2.chromosome[crossover_point2:])

        childs.append(Crossover.build_individual_solution_from_genes(child1_genes))
        childs.append(Crossover.build_individual_solution_from_genes(child2_genes))

        return childs

class Crossover:
    def __init__(self, triangles_amount):
        self.triangles_amount = triangles_amount

    @staticmethod
    def build_individual_solution_from_genes(chromosoma: List) -> IndividualSolution:
        solution = IndividualSolution()
        index = 0
        while index < len(chromosoma):
            r = chromosoma[index].value
            g = chromosoma[index+1].value
            b = chromosoma[index+2].value
            vertex1 = chromosoma[index+3].value
            vertex2 = chromosoma[index+4].value
            vertex3 = chromosoma[index+5].value
            primitive = GeometricPrimitive([vertex1, vertex2, vertex3], (r, g, b))
            solution.add_primitive(primitive)
            index += 6
        return solution