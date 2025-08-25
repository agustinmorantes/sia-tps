from .Heuristic import Heuristic
from ..State import State


class HammingHeuristic(Heuristic): #Cantidad de cajas fuera de lugar 
    def __init__(self):
        super().__init__("Hamming", self.hamming)

    @staticmethod
    def hamming(state: State) -> float:
        return sum(1 for box in state.boxes if box not in state.goals)
