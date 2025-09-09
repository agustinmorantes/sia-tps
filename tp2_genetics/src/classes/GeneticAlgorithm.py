import numpy as np
from PIL import Image, ImageDraw
import random
from typing import List, Tuple, Dict, Any
import time
from .Individual import Individual

class GeneticAlgorithm:
    def __init__(self, image_path: str, num_triangles :int, population_size :int = 100, 
                crossover_rate: float = 0.8, mutation_rate: float = 0.1, 
                max_generations: int = 1000, convergence_threshold: float = 0.001,
                convergence_generations: int = 50):

        # Imagen objetivo
        self.original_image = Image.open(image_path).convert("RGB")
        self.width, self.height = self.original_image.size
        self.num_triangles = num_triangles

        # Parámetros del algoritmo genético
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.convergence_threshold = convergence_threshold
        self.convergence_generations = convergence_generations

        # Metricas
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.convergence_count = 0
        self.last_best_fitness = float('inf')

        # Convertir imagen a array para comparar más rápido
        self.original_image_array = np.array(self.original_image)

    def calculate_fitness(self, individual: Individual) -> float:
        # Generar imagen del individuo y luego el array 
        generated_image = self._render_image(individual)
        generated_image_array = np.array(generated_image)

        # Calcular diferencia pixel por pixel
        diff = np.sqrt(np.sum((self.original_image_array - generated_image_array) ** 2, axis=2))
        mse = np.mean(diff ** 2)

        # Convertir MSE a fitness
        max_possible_error = np.sqrt(3 * 255 ** 2)
        fitness = max_possible_error - np.sqrt(mse)

        return max(0, fitness)

    def _render_image(self, individual: Individual) -> Image.Image:
        # Crear imagen en blanco
        image = Image.new('RGB', (self.width, self.height), 'white')
        draw = ImageDraw.Draw(image, 'RGBA')

        # Dibujar triángulos
        for triangle in individual.triangles:
            vertices = [(triangle.x1, triangle.y1), 
                        (triangle.x2, triangle.y2), 
                        (triangle.x3, triangle.y3)]

            color = (int(triangle.r), int(triangle.g), int(triangle.b), int(triangle.alpha * 255))

            draw.polygon(vertices, fill=color)

        return image.convert('RGB')
