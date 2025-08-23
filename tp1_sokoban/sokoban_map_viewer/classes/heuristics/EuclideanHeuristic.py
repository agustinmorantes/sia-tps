from sokoban_map_viewer.classes.State import State
from sokoban_map_viewer.classes.heuristics.Heuristic import Heuristic


class EuclideanHeuristic(Heuristic):
    def __init__(self):
        super().__init__("Euclidean", self.euclidean)

    @staticmethod
    def euclidean(state: State) -> float:
        total_dist = 0

        for box in state.boxes:
            min_dist = float('inf')
            for goal in state.goals:
                dist = ((box.x - goal.x) ** 2 + (box.y - goal.y) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
            total_dist += min_dist

        return total_dist
