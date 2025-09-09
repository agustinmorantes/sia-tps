import random
from typing import List
from .Triangle import Triangle

class Individual: # Representa un individuo de la población genética. Es una lista de triángulos que aproxima la imagen.
    def __init__(self, triangles: List[Triangle]):
        self.triangles = triangles
        self.fitness = 0.0  # No se va a establecer acá, el fitness de cada individuo es la MSE entre la imagen objetivo y la imagen aproximada por los triángulos.
        self.age = 0

    def copy(self) -> "Individual":
        return Individual([triangle.copy() for triangle in self.triangles])
    
    def to_dict(self) -> dict:
        return {
            'triangles': [triangle.to_dict() for triangle in self.triangles],
            'fitness': self.fitness,
            'age': self.age
        }

    def increment_age(self):
        self.age += 1

    def get_genome_size(self) -> int:
        return len(self.triangles)

    @classmethod
    def random_individual(cls, num_triangles: int, width: int, height: int) -> "Individual":
        triangles = [Triangle.random_triangle(width, height) for _ in range(num_triangles)] 
        return cls(triangles)