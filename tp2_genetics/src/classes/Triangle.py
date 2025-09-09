import numpy as np
from typing import Tuple, List
import random

class Triangle: # Representa un triángulo con tres vértices y un color RGBA.
    def __init(self, x1: float, y1: float, x2: float, y2: float, x3: float, y3: float,
               r :int, g :int, b :int, alpha :float):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.x3 = x3
        self.y3 = y3
        self.r = r
        self.g = g
        self.b = b
        self.alpha = alpha

    def copy(self) -> "Triangle":
        return Triangle(self.x1, self.y1, self.x2, self.y2, self.x3, self.y3, 
                        self.r, self.g, self.b, self.alpha)

    def to_dict(self) -> dict:
        return {
            'vertices': [(self.x1, self.y1), (self.x2, self.y2), (self.x3, self.y3)],
            'color': (self.r, self.g, self.b, self.alpha)
        }

    def mutate_vertex(self, width: int, height: int, mutation_rate: float = 0.1):
        vertex = random.choice([1, 2, 3])
        if vertex == 1:
            self.x1 = max(0, min(width, self.x1 + random.gauss(0, width * mutation_rate)))
            self.y1 = max(0, min(height, self.y1 + random.gauss(0, height * mutation_rate)))
        elif vertex == 2:
            self.x2 = max(0, min(width, self.x2 + random.gauss(0, width * mutation_rate)))
            self.y2 = max(0, min(height, self.y2 + random.gauss(0, height * mutation_rate)))
        elif vertex == 3:
            self.x3 = max(0, min(width, self.x3 + random.gauss(0, width * mutation_rate)))
            self.y3 = max(0, min(height, self.y3 + random.gauss(0, height * mutation_rate)))

    def mutate_color(self, mutation_rate: float = 0.1):
        color_mutation = random.gauss(0, 255 * mutation_rate)
        self.r = max(0, min(255, self.r + color_mutation))
        self.g = max(0, min(255, self.g + color_mutation))
        self.b = max(0, min(255, self.b + color_mutation))

        alpha_mutation = random.gauss(0, mutation_rate)
        self.alpha = max(0.0, min(1.0, self.alpha + alpha_mutation))

    @classmethod
    def random_triangle(cls, width: int, height: int) -> "Triangle":
        x1 = random.uniform(0, width)
        y1 = random.uniform(0, height)
        x2 = random.uniform(0, width)
        y2 = random.uniform(0, height)
        x3 = random.uniform(0, width)
        y3 = random.uniform(0, height)

        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        alpha = random.uniform(0.1, 1.0)

        return cls(x1, y1, x2, y2, x3, y3, r, g, b, alpha)