from abc import abstractmethod, ABC

from genetic_algorithm.models.individual_solution import IndividualSolution
from genetic_algorithm.utils.random_seed_manager import central_random_generator as random_generator # Importación del generador centralizado
from typing import List

class MutationStrategy(ABC):
    def __init__(self, probability: float, mutation_percent: float = 0.2):
        self.probability = probability
        self.mutation_percent = mutation_percent

    @abstractmethod
    def apply_mutations(self, solution: IndividualSolution):
        pass

    def apply_mutations_to_population(self, population: List[IndividualSolution]):
        for individual in population:
            self.apply_mutations(individual)

class GeneMutationStrategy(MutationStrategy): #Mutacion de un solo gen 
    def apply_mutations(self, solution: IndividualSolution):
        random_mutation_chance = random_generator.random()
        if random_mutation_chance < self.probability:
            gene_to_modify_index = random_generator.randint(0, len(solution.chromosome)-1)
            attribute_to_modify = solution.chromosome[gene_to_modify_index]
            attribute_to_modify.mutate(max_percent=self.mutation_percent)
            solution.update_primitive_from_gene(attribute_to_modify, gene_to_modify_index)

class LimitedMultigeneMutationStrategy(MutationStrategy): #Mutacion limitada de multiples genes 
    def __init__(self, probability: float, mutation_percent: float = 0.2, max_gene_percent_to_modify: float = 0.05):
        super().__init__(probability, mutation_percent)
        self.max_gene_percent_to_modify = max_gene_percent_to_modify #Maximo porcentaje de genes a modificar

    def apply_mutations(self, solution: IndividualSolution): 
        if random_generator.random() >= self.probability:#Decidimos si mutamos el individuo
            return

        # Mezclo los índices
        indices = list(range(len(solution.chromosome)))
        random_generator.shuffle(indices)#Mezclamos los indices de todos los genes de los cromosomas

        # Cantidad aleatoria de genes para modificar
        n = random_generator.randint(1, int(len(solution.chromosome) * self.max_gene_percent_to_modify))

        # Tomo los primeros n indices
        selected_indices = indices[:n]

        for index in selected_indices:
            attribute_to_modify = solution.chromosome[index]
            attribute_to_modify.mutate(max_percent=self.mutation_percent)#Mutamos el gen
            solution.update_primitive_from_gene(attribute_to_modify, index)

def get_mutation_strategy(strategy_name: str, probability: float, mutation_percent: float = 0.2) -> MutationStrategy:
    if strategy_name == "single":
        return GeneMutationStrategy(probability, mutation_percent)
    elif strategy_name == "limited_multi":
        return LimitedMultigeneMutationStrategy(probability, mutation_percent)
    else:
        raise ValueError(f"Estrategia de mutación desconocida: {strategy_name}")
