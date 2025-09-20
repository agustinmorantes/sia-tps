import math
import random
from pydoc import render_doc

import numpy as np

from genetic_algorithm.utils.random_seed_manager import central_random_generator as random_generator # Importación del generador centralizado

class GeneticAttribute:
    """Clase base para atributos genéticos que pueden ser mutados."""
    def __init__(self, value, min_val, max_val):
        # Validación inicial de valores (se mantiene la lógica original)
        if isinstance(value, int):
            if not (min_val <= value <= max_val):
                raise ValueError(f"Valor entero '{value}' fuera del rango permitido [{min_val}, {max_val}]")
        elif isinstance(value, tuple):
            if not (min_val[0] <= value[0] <= max_val[0] and min_val[1] <= value[1] <= max_val[1]):
                raise ValueError(f"Valor de tupla '{value}' fuera del rango permitido X:[{min_val[0]}, {max_val[0]}], Y:[{min_val[1]}, {max_val[1]}]")
        else:
            pass

        self.value = value
        self.min_val = min_val
        self.max_val = max_val

    def mutate(self, percent=0.2):
        """Método genérico de mutación. Las subclases deben sobrescribirlo para una lógica específica."""
        if isinstance(self.value, int):
            delta = round((self.max_val - self.min_val) * percent)
            new_value = random_generator.randint(self.value - delta, self.value + delta)
            self.value = max(self.min_val, min(self.max_val, new_value))

class TriangleColorAttribute(GeneticAttribute):
    """Representa un color (R, G, B) de un primitivo geométrico."""

    def __init__(self, value: tuple[float,float,float]):
        super().__init__(value, (0.0,0.0,0.0), (255.0, 255.0, 255.0))

    @staticmethod
    def random_three_vector() -> tuple[float,float,float]:
        """
        Generates a random 3D unit vector (direction) with a uniform spherical distribution
        Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
        :return:
        """
        phi = random_generator.uniform(0, math.pi * 2)
        costheta = random_generator.uniform(-1.0,1.0)

        theta = np.arccos( costheta )
        x = np.sin( theta) * np.cos( phi )
        y = np.sin( theta) * np.sin( phi )
        z = np.cos( theta )
        return x,y,z

    def mutate(self, max_percent=0.2):
        direction = self.random_three_vector()

        # Delta en el rango [0, 1] con distribución sesgada hacia 0
        delta_magnitude = (random_generator.uniform(0,1) ** 4) * 128
        if random_generator.random() < 0.5: # 50% de probabilidad de invertir el delta
            delta_magnitude = -delta_magnitude

        r_dir, g_dir, b_dir = direction
        new_r = max(0.0, min(255.0, self.value[0] + delta_magnitude * r_dir))
        new_g = max(0.0, min(255.0, self.value[1] + delta_magnitude * g_dir))
        new_b = max(0.0, min(255.0, self.value[2] + delta_magnitude * b_dir))

        self.value = (int(new_r), int(new_g), int(new_b))

class TrianglePositionAttribute(GeneticAttribute):
    bounds_extension = 0.2

    """Representa las posiciones de los vértices de un triángulo."""
    def __init__(self, value: tuple[tuple[float,float],tuple[float,float],tuple[float,float]], max_coords):
        # Permitir que la coordenada este fuera de los límites, pero no demasiado
        min_coords = [0, 0]
        min_coords[0] = 0 - max_coords[0] * self.bounds_extension
        min_coords[1] = 0 - max_coords[1] * self.bounds_extension

        max_coords = list(max_coords)
        max_coords[0] = max_coords[0] + max_coords[0] * self.bounds_extension
        max_coords[1] = max_coords[1] + max_coords[1] * self.bounds_extension

        super().__init__(value, min_coords, max_coords)

    @staticmethod
    def random_two_vector() -> tuple[float,float]:
        """
        Generates a random 2D unit vector (direction) with a uniform spherical distribution
        :return:
        """
        theta = random_generator.uniform(0, math.pi * 2)

        x = np.cos(theta)
        y = np.sin(theta)
        return x,y

    def mutate(self, max_percent=0.2):
        direction = self.random_two_vector()
        magnitude = (random_generator.uniform(0,1) ** 4) * (self.max_val[1] / 2)
        delta = (direction[0] * magnitude, direction[1] * magnitude)

        num_vertices_to_modify = random_generator.randint(1,3) # Entre 1 y 3 vertices a modificar
        indices = [0,1,2]
        random_generator.shuffle(indices)
        selected_indices = indices[:num_vertices_to_modify]

        for index in selected_indices:
            vertex = self.value[index]
            new_x = max(self.min_val[0], min(self.max_val[0], vertex[0] + delta[0]))
            new_y = max(self.min_val[1], min(self.max_val[1], vertex[1] + delta[1]))

            self.value[index] = (new_x, new_y)

