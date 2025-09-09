from abc import ABC, abstractmethod
from typing import Tuple
from ..classes.Individual import Individual

class Crossing(ABC):
    @abstractmethod
    def cross(self, parent1: Individual, parent2: Individual) -> Tuple(Individual, Individual):
        pass
