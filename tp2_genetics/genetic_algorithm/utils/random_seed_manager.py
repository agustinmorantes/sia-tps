import random

# Crea una única instancia del generador de números aleatorios
central_random_generator = random.Random()

# Siembra el generador de forma centralizada para reproducibilidad
central_random_generator.seed(43)
