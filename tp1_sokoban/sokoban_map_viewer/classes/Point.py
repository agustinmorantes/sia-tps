class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __str__(self):
        return f"Point({self.x}, {self.y})"
    
    def __repr__(self):
        return f"Point({self.x}, {self.y})"
    
    def move(self, direction):
        return Point(self.x + direction.value[0], self.y + direction.value[1])
