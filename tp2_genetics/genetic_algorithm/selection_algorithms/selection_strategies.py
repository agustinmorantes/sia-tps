import heapq as hp
from abc import ABC, abstractmethod
from typing import List
from genetic_algorithm.models.individual_solution import IndividualSolution
from genetic_algorithm.utils.random_seed_manager import central_random_generator as random_generator # Importación del generador centralizado

class SelectionStrategy(ABC):
    def __init__(self, size):
        self.size = size

    @abstractmethod
    def select(self, population: List[IndividualSolution], fitness_cache: dict) -> List[IndividualSolution]:
        pass

class EliteSelection(SelectionStrategy):
    def select(self, population: List[IndividualSolution], fitness_cache: dict) -> List[IndividualSolution]:
        fitness_priority_queue = []
        counter = 0
        for individual in population:
            counter += 1
            individual_hash = hash(str(individual))
            hp.heappush(fitness_priority_queue, (-fitness_cache[individual_hash], counter, individual))
        
        best_individuals = []
        for _ in range(self.size):
            if fitness_priority_queue:
                _, _, individual = hp.heappop(fitness_priority_queue)
                best_individuals.append(individual)
            else:
                break
        return best_individuals

class RouletteSelection(SelectionStrategy):
    def select(self, population: List[IndividualSolution], fitness_cache: dict) -> List[IndividualSolution]:
        total_fitness = sum(fitness_cache[hash(str(individual))] for individual in population)
        
        if total_fitness == 0:
            return random_generator.choices(population, k=self.size)
        
        fitness_values_list = []
        for ind in population:
            individual_hash = hash(str(ind))
            fitness_values_list.append((fitness_cache[individual_hash], ind))
        
        selection_probs = [item[0] / total_fitness for item in fitness_values_list]
        selected_individuals = []
        
        for _ in range(self.size):
            random_value = random_generator.random()
            current_prob = 0
            for i in range(len(population)):
                current_prob += selection_probs[i]
                if current_prob > random_value:
                    selected_individuals.append(fitness_values_list[i][1])
                    break
        return selected_individuals

class UniversalSelection(SelectionStrategy):
    def select(self, population: List[IndividualSolution], fitness_cache: dict) -> List[IndividualSolution]:
        total_fitness = sum(fitness_cache[hash(str(individual))] for individual in population)
        
        if total_fitness == 0:
            return random_generator.choices(population, k=self.size)
        
        fitness_values_list = []
        for ind in population:
            individual_hash = hash(str(ind))
            fitness_values_list.append((fitness_cache[individual_hash], ind))

        selection_probs = [item[0] / total_fitness for item in fitness_values_list]
        selected_individuals = []

        start_point = random_generator.random()
        
        for k in range(self.size):
            pointer = (start_point + k) / self.size
            current_prob_sum = 0
            for i in range(len(population)):
                current_prob_sum += selection_probs[i]
                if current_prob_sum >= pointer:
                    selected_individuals.append(fitness_values_list[i][1])
                    break
        
        return selected_individuals

class RankingSelection(SelectionStrategy):
    def select(self, population: List[IndividualSolution], fitness_cache: dict) -> List[IndividualSolution]:

        sorted_population = sorted(population, key=lambda ind: fitness_cache[hash(str(ind))], reverse=True)
        ranks = list(range(1, len(sorted_population)+1))

        pseudo_fit_values = []
        for rank in ranks:
            pseudo_fit_values.append(((len(sorted_population) - rank) / len(population), sorted_population[rank - 1]))

        total_pseudo_fit = sum(p[0] for p in pseudo_fit_values)
        
        pseudo_fit_values_rel = []
        for pseudo_fit, ind in pseudo_fit_values:
            pseudo_fit_values_rel.append(((pseudo_fit / total_pseudo_fit), ind))
        
        selected_individuals = []
        for _ in range(self.size):
            random_value = random_generator.random()
            current_prob = 0
            for prob, ind in pseudo_fit_values_rel:
                current_prob += prob
                if current_prob > random_value:
                    selected_individuals.append(ind)
                    break

        return selected_individuals
        

class TournamentSelection(SelectionStrategy):
    def __init__(self, size, tournament_size=2):
        super().__init__(size)
        self.tournament_size = tournament_size

    def select(self, population: List[IndividualSolution], fitness_cache: dict) -> List[IndividualSolution]:
        pass

class TournamentDeterministicSelection(TournamentSelection):
    def select(self, population: List[IndividualSolution], fitness_cache: dict) -> List[IndividualSolution]:
        selected_individuals = []

        for _ in range(self.size):
            tournament = random_generator.sample(population, self.tournament_size)
            best_individual = max(tournament, key=lambda ind: fitness_cache[hash(str(ind))])
            selected_individuals.append(best_individual)
        
        return selected_individuals
    
    def select_without_repetition(self, population: List[IndividualSolution], fitness_cache: dict) -> List[IndividualSolution]:
        selected_individuals = []
        
        available_population = population.copy()
        for _ in range(self.size):
            if len(available_population) < self.tournament_size:
                break
            tournament = random_generator.sample(available_population, self.tournament_size)
            best_individual = max(tournament, key=lambda ind: fitness_cache[hash(str(ind))])
            selected_individuals.append(best_individual)
            available_population.remove(best_individual)
        
        return selected_individuals

        
class TournamentProbabilisticSelection(TournamentSelection):
    def __init__(self, size, tournament_size=2, threshold=0.5):
        super().__init__(size)
        self.tournament_size = 2
        self.threshold = threshold
            
    def select(self, population: List[IndividualSolution], fitness_cache: dict) -> List[IndividualSolution]:
        selected_individuals = [] 

        for _ in range(self.size):
            tournament = random_generator.sample(population, self.tournament_size)
            r = random_generator.random()
            if r < self.threshold:
                selected_individual = min(tournament, key=lambda ind: fitness_cache[hash(str(ind))])
            else:
                selected_individual = max(tournament, key=lambda ind: fitness_cache[hash(str(ind))])
            
            selected_individuals.append(selected_individual)
        
        return selected_individuals