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
    
    def can_move(self, direction: Direction) -> bool:
        new_player = self.player.move(direction)

        # 1️⃣ No se puede mover a una pared
        if new_player in self.walls:
            return False

        # 2️⃣ Si hay caja en la dirección
        if new_player in self.boxes:
            new_box = new_player.move(direction)
            
            # La caja no puede chocar con pared o con otra caja
            if new_box in self.walls or new_box in self.boxes:
                return False
            
            # La caja no puede ir a deadlock (si no es un goal)
            if new_box in self.deadlocks and new_box not in self.goals:
                return False

        # 3️⃣ Si no hay obstáculos, se puede mover
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