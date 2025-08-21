from Point import Point
from Direction import Direction

class State:
    def __init__(self, player: Point, boxes: set[Point], walls: set[Point], goals: set[Point], deadlocks: set[Point]):
        self.player = player         # Player position
        self.boxes = boxes           # Boxes positions
        self.walls = walls           # Walls positions
        self.goals = goals           # Goals positions
        self.deadlocks = deadlocks   # Deadlocks positions

    def __eq__(self, other):
        return self.player == other.player and self.boxes == other.boxes    # Walls, goals and deadlocks not relevant for equality
    
    def __hash__(self):
        return hash((tuple(self.boxes), self.player))   # Same as before
    
    def is_solved(self):
        return self.boxes.issubset(self.goals)
    
    def can_move(self, direction):  # TODO
       return True
    
    def move(self, direction):  # TODO
        return True
    
    def get_children(self):
        children = []
        for direction in Direction:
            if self.can_move(direction):
                children.append(self.move(direction))
        return children