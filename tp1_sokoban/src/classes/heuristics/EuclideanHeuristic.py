from ..State import State
from .Heuristic import Heuristic

inf = float('inf')

class EuclideanHeuristic(Heuristic): # distancia recta,como si midieras con una regla,no respeta el movimiento real de Sokoban
    def __init__(self):
        super().__init__("Euclidean", self.euclidean)

    @staticmethod
    def euclidean(state: State) -> float:
        total_dist = 0

        for box in state.boxes:
            min_dist = inf
            for goal in state.goals:
                dist = ((box.x - goal.x) ** 2 + (box.y - goal.y) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
            total_dist += min_dist

        return total_dist
