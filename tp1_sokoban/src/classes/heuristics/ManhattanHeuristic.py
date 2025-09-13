from ..State import State
from .Heuristic import Heuristic

inf = float('inf')

class ManhattanHeuristic(Heuristic): #Esta distancia representa el número mínimo de pasos ortogonales (arriba, abajo, izquierda, derecha) para llegar a la meta.
    def __init__(self):
        super().__init__("Manhattan", self.heuristic)

    @staticmethod
    def heuristic(state: State) -> float:
        total_dist = 0

        for box in state.boxes:
            min_dist = inf
            for goal in state.goals:
                dist = abs(box.x - goal.x) + abs(box.y - goal.y)
                if dist < min_dist:
                    min_dist = dist
            total_dist += min_dist

        return total_dist