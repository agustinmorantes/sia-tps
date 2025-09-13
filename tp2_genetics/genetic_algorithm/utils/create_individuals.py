import random
from genetic_algorithm.models.individual_solution import GeometricPrimitive, IndividualSolution
from genetic_algorithm.utils.random_seed_manager import central_random_generator as random_generator # Importación del generador centralizado

def create_initial_population(num_solutions, num_primitives, width=500, height=500):
    
    solutions = []
    
    for _ in range(num_solutions):
        solution = IndividualSolution()

        for _ in range(num_primitives):
            # Generar el primitivo con 3 vértices random y un color random
            vertices = generate_random_vertices(width, height)
            color = (random_generator.randint(0, 255), random_generator.randint(0, 255), random_generator.randint(0, 255))

            primitive = GeometricPrimitive(vertices, color)
            solution.add_primitive(primitive)
                    
        solutions.append(solution)

    return solutions

def generate_random_vertices(width, height):
    # Generar el primer vértice random
    cx = random_generator.randint(0, width - 1)
    cy = random_generator.randint(0, height - 1)

    vertices = [(cx, cy)]

    # Para los dos vértices restantes, elijo un desplazamiento aleatorio (a partir del primer vértice)
    for _ in range(2):
        x = cx + random_generator.randint(-width, width)
        y = cy + random_generator.randint(-height, height)

        # Ajustar el vértice a los límites del canvas si se sale
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))

        vertices.append((x, y))

    return vertices