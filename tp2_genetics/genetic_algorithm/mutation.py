from genetic_algorithm.models.individual_solution import IndividualSolution
from genetic_algorithm.utils.random_seed_manager import central_random_generator as random_generator # Importación del generador centralizado
from typing import List

class GeneModifier:
    """Gestiona la aplicación de mutaciones a los genes de las soluciones individuales."""
    def __init__(self, modification_rate: float, shapes_per_solution: int):
        self.modification_rate = modification_rate # Renombrado
        self.shapes_per_solution = shapes_per_solution # Renombrado

    def apply_mutations(self, target_population: List[IndividualSolution]): # Renombrado de método y parámetro
        for current_solution in target_population: # Renombrado de variable de bucle
            random_mutation_chance = random_generator.random() # Renombrado
            if random_mutation_chance < self.modification_rate:
                gene_to_modify_index = random_generator.randint(0, len(current_solution.chromosome)-1) # Renombrado
                attribute_to_modify = current_solution.chromosome[gene_to_modify_index] # Renombrado
                attribute_to_modify.mutate(percent=0.2)  # Porcentaje de mutación (change_magnitude)
                current_solution.update_primitive_from_gene(attribute_to_modify, gene_to_modify_index)