from genetic_algorithm.utils.random_seed_manager import central_random_generator as random_generator # Importación del generador centralizado

class GeneticAttribute:
    """Clase base para atributos genéticos que pueden ser mutados."""
    def __init__(self, name, value, min_val, max_val):
        # Validación inicial de valores (se mantiene la lógica original)
        if isinstance(value, int):
            if not (min_val <= value <= max_val):
                raise ValueError(f"Valor entero '{value}' fuera del rango permitido [{min_val}, {max_val}]")
        elif isinstance(value, tuple):
            if not (min_val[0] <= value[0] <= max_val[0] and min_val[1] <= value[1] <= max_val[1]):
                raise ValueError(f"Valor de tupla '{value}' fuera del rango permitido X:[{min_val[0]}, {max_val[0]}], Y:[{min_val[1]}, {max_val[1]}]")
        else:
            pass

        self.name = name
        self.value = value
        self.min_val = min_val
        self.max_val = max_val

    def mutate(self, percent=0.2):
        """Método genérico de mutación. Las subclases deben sobrescribirlo para una lógica específica."""
        if isinstance(self.value, int):
            delta = round((self.max_val - self.min_val) * percent)
            new_value = random_generator.randint(self.value - delta, self.value + delta)
            self.value = max(self.min_val, min(self.max_val, new_value))

class RGBComponentAttribute(GeneticAttribute):
    """Representa un componente de color (R, G, B) de un primitivo geométrico."""
    COLOR_SENSITIVITY = {
        'Red': 1.0,
        'Green': 0.8,
        'Blue': 1.2
    }
        
    def __init__(self, name, value):
        super().__init__(name, value, 0, 255)

    def mutate(self, percent=0.2):
        delta = round((self.max_val - self.min_val) * percent * self.COLOR_SENSITIVITY[self.name])
        new_value = random_generator.randint(self.value - delta, self.value + delta)
        self.value = max(self.min_val, min(self.max_val, new_value))

class VertexCoordinateAttribute(GeneticAttribute):
    """Representa las coordenadas (x, y) de un vértice de un primitivo geométrico."""
    def __init__(self, name, value, max_coords):
        super().__init__(name, value, (0,0), max_coords)

    def mutate(self, percent=0.2):
        if random_generator.random() < 0.5:
            self._mutate_coordinate(0, percent)
        else:
            self._mutate_coordinate(1, percent)

    def _mutate_coordinate(self, coord_index, percent):
        current_coord = self.value[coord_index]
        min_coord = self.min_val[coord_index]
        max_coord = self.max_val[coord_index]

        delta = (max_coord - min_coord) * percent
        new_coord = random_generator.uniform(current_coord - delta, current_coord + delta)
        new_coord = max(min_coord, min(max_coord, new_coord))
        
        if coord_index == 0:
            self.value = (new_coord, self.value[1])
        else:
            self.value = (self.value[0], new_coord)
