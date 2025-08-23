from sokoban_map_viewer.classes.State import State
from sokoban_map_viewer.classes.heuristics.Heuristic import Heuristic


class ManhattanHeuristic(Heuristic):
    def __init__(self):
        super().__init__("Manhattan", self.heuristic)

    @staticmethod
    def heuristic(state: State) -> float:
        total_dist = 0

        for box in state.boxes:
            min_dist = float('inf')
            for goal in state.goals:
                dist = abs(box.x - goal.x) + abs(box.y - goal.y)
                if dist < min_dist:
                    min_dist = dist
            total_dist += min_dist

        return total_dist
