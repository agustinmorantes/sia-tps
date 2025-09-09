from abc import ABC, abstractmethod
from typing import List
from ..classes.Individual import Individual

class Selection(ABC):
    @abstractmethod
    def select(self, population: List[Individual], num_selected: int) -> List[Individual]:
        pass