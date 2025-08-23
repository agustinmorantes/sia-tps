from sokoban_map_viewer.classes.Point import Point
from sokoban_map_viewer.classes.State import State
from sokoban_map_viewer.classes.heuristics.Heuristic import Heuristic


class ManhattanImprovedHeuristic(Heuristic):
    def __init__(self):
        super().__init__("Manhattan Improved", self.manhattan_improved)

    @staticmethod
    def manhattan_improved(state: State) -> float:
        # Deadlock detection
        for box in state.boxes:
            if box not in state.goals:
                x, y = box.x, box.y
                walls = state.walls
                if (Point(x-1, y) in walls and Point(x, y-1) in walls) or \
                    (Point(x+1, y) in walls and Point(x, y-1) in walls) or \
                    (Point(x-1, y) in walls and Point(x, y+1) in walls) or \
                    (Point(x+1, y) in walls and Point(x, y+1) in walls):
                        return float('inf') # Deadlock, no solution possible

        total_distance = 0
        for box in state.boxes:
            min_distance = float('inf')

            for goal in state.goals:
                distance = abs(box.x - goal.x) + abs(box.y - goal.y)
                if distance < min_distance:
                    min_distance = distance

            total_distance += min_distance
        return total_distance