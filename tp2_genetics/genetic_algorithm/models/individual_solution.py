from genetic_algorithm.models.genetic_attribute import TriangleColorAttribute, TrianglePositionAttribute


class GeometricPrimitive:
    def __init__(self, vertices, color):
        # vertices: lista de 3 tuplas (x, y) que definen los puntos del primitivo
        # color: tupla (R, G, B) para el color uniforme del primitivo

        self.vertices = vertices
        self.color = color
        self.genes: list[TriangleColorAttribute | TrianglePositionAttribute] = []
        self._set_primitive_genes()

    def _set_primitive_genes(self):
        # Cada primitivo se define por 1 gen color y 1 gen posición (con 2 coordenadas cada uno)
        self.genes.append(TriangleColorAttribute(self.color))
        self.genes.append(TrianglePositionAttribute(self.vertices, (499,499)))

    def update_from_gene(self, gene):
        if isinstance(gene, TriangleColorAttribute):
            # Actualiza el componente de color basado en el gen
            self.color = gene.value
        elif isinstance(gene, TrianglePositionAttribute):
            # Actualiza las coordenadas de un vértice basado en el gen
            self.vertices = gene.value

class IndividualSolution:
    id_counter = 0

    def __init__(self):
        self.primitives: list[GeometricPrimitive] = [] # Lista de GeometricPrimitives
        self.chromosome: list[TriangleColorAttribute | TrianglePositionAttribute] = [] # Lista extendida de todos los genes de los primitivos

        self.id = IndividualSolution.id_counter
        IndividualSolution.id_counter += 1

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, IndividualSolution):
            return NotImplemented

        return self.id == other.id

    def add_primitive(self, primitive: GeometricPrimitive):
        self.primitives.append(primitive)
        self.chromosome.extend(primitive.genes)

    def update_primitive_from_gene(self, gene, gene_position):
        # Asumiendo 2 genes por GeometricPrimitive (1 color, 1 posición)
        primitive_index = int(gene_position / 2)
        self.primitives[primitive_index].update_from_gene(gene)
