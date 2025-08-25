from .Heuristic import Heuristic

inf = float('inf')

class ChebyshevHeuristic(Heuristic):  #Esto mide la cantidad mínima de pasos si pudieras moverte en 8 direcciones (vertical, horizontal y diagonal)
    def __init__(self):
        super().__init__("Chebyshev", self.chebyshev)

    @staticmethod
    def chebyshev(state) -> float:
        total_dist = 0

        for box in state.boxes:
            min_dist = inf
            for goal in state.goals:
                dist = max(abs(box.x - goal.x), abs(box.y - goal.y))
                if dist < min_dist:
                    min_dist = dist
            total_dist += min_dist

        return total_dist