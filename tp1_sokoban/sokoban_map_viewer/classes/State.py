from sokoban_map_viewer.classes.Direction import Direction
from sokoban_map_viewer.classes.Point import Point


class State:
    def __init__(self, player: Point, boxes: set[Point], walls: set[Point], goals: set[Point], deadlocks: set[Point]):
        self.player: Point = player         # Player position
        self.boxes: set[Point] = boxes           # Boxes positions
        self.walls: set[Point] = walls           # Walls positions
        self.goals: set[Point] = goals           # Goals positions
        self.deadlocks: set[Point] = deadlocks   # Deadlocks positions

    def __eq__(self, other):
        return self.player == other.player and self.boxes == other.boxes 
    
    def __hash__(self):
        return hash((tuple(self.boxes), self.player))
    
    def is_solved(self):
        return self.boxes.issubset(self.goals)
    
    def can_move(self, direction: Direction) -> bool:
        new_player = self.player.move(direction)

        if new_player in self.walls:
            return False

        if new_player in self.boxes:
            new_box = new_player.move(direction)
            if new_box in self.walls or new_box in self.boxes:
                return False
            if new_box in self.deadlocks and new_box not in self.goals:
                return False

        return True

    
    def move(self, direction):  # TODO
        new_player = self.player.move(direction)
        if new_player in self.walls:
            return None
        
        if new_player in self.boxes:
            new_box = new_player.move(direction)
            if new_box in self.walls or new_box in self.boxes:
                return None
            new_boxes = self.boxes.copy()
            new_boxes.remove(new_player)
            new_boxes.add(new_box)
            if new_box in self.deadlocks and new_box not in self.goals:
                return False
            return State(new_player, new_boxes, self.walls, self.goals, self.deadlocks)
        # Caso sin empujar caja
        return State(new_player, self.boxes.copy(), self.walls, self.goals, self.deadlocks)

    
    def get_children(self):
        children = []
        for direction in [Direction.LEFT, Direction.UP, Direction.RIGHT, Direction.DOWN]:
            if self.can_move(direction):
                children.append(self.move(direction))
        return children

    def __str__(self):
        max_row = max(p.x for p in self.walls | self.goals | self.boxes | {self.player}) + 1
        max_col = max(p.y for p in self.walls | self.goals | self.boxes | {self.player}) + 1

        result = ""
        for r in range(max_row):
            for c in range(max_col):
                p = Point(r, c)
                if p == self.player:
                    result += "@"
                elif p in self.boxes:
                    if p in self.goals:
                        result += "*"  # Caja sobre goal
                    else:
                        result += "$"
                elif p in self.goals:
                    result += "."
                elif p in self.walls:
                    result += "#"
                else:
                    result += " "
            result += "\n"
        return result
