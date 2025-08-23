from sokoban_map_viewer.classes.heuristics.Heuristic import Heuristic


class HammingHeuristic(Heuristic):
    def __init__(self):
        super().__init__("Hamming", self.hamming)

    @staticmethod
    def hamming(state) -> float:
        return sum(1 for box in state.boxes if box not in state.goals)
