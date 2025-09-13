from .genetic_attribute import RGBComponentAttribute, VertexCoordinateAttribute # Nombres de genes actualizados (importación corregida)

class GeometricPrimitive:
    def __init__(self, vertices, color):
        # vertices: lista de 3 tuplas (x, y) que definen los puntos del primitivo
        # color: tupla (R, G, B) para el color uniforme del primitivo
        self.vertices = vertices
        self.color = color
        self.genes = []
        self._set_primitive_genes()

    def _set_primitive_genes(self):
        # Cada primitivo se define por 3 componentes de color y 3 vértices (con 2 coordenadas cada uno)
        color_names = ["Red", "Green", "Blue"]
        for i in range(3):
            self.genes.append(RGBComponentAttribute(color_names[i], self.color[i]))
        
        vertex_names = ["Vertex1", "Vertex2", "Vertex3"]
        for i in range(3):
            # Asumiendo dimensiones máximas (499,499) como en el original
            self.genes.append(VertexCoordinateAttribute(vertex_names[i], self.vertices[i], (499,499)))

    def update_from_gene(self, gene):
        if isinstance(gene, RGBComponentAttribute):
            # Actualiza el componente de color basado en el gen
            if gene.name == "Red":
                self.color = (gene.value, self.color[1], self.color[2])
            elif gene.name == "Green":
                self.color = (self.color[0], gene.value, self.color[2])
            elif gene.name == "Blue":
                self.color = (self.color[0], self.color[1], gene.value)
        elif isinstance(gene, VertexCoordinateAttribute):
            # Actualiza las coordenadas de un vértice basado en el gen
            if gene.name == "Vertex1": 
                self.vertices[0] = gene.value
            elif gene.name == "Vertex2": 
                self.vertices[1] = gene.value
            elif gene.name == "Vertex3":  
                self.vertices[2] = gene.value

class IndividualSolution:
    def __init__(self):
        self.primitives = [] # Lista de GeometricPrimitives
        self.chromosome = [] # Lista extendida de todos los genes de los primitivos

    def add_primitive(self, primitive: GeometricPrimitive):
        self.primitives.append(primitive)
        self.chromosome.extend(primitive.genes)

    def update_primitive_from_gene(self, gene, gene_position):
        # Asumiendo 6 genes por GeometricPrimitive (3 color, 3 posición)
        primitive_index = int(gene_position / 6)
        self.primitives[primitive_index].update_from_gene(gene)
