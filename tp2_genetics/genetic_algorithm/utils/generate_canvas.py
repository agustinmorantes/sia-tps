from PIL import Image, ImageDraw
from .create_individuals import create_initial_population # Nombre de función actualizado
from genetic_algorithm.models.individual_solution import IndividualSolution, GeometricPrimitive # Nuevas importaciones (corregida)

def render_solution_to_image(solution: IndividualSolution, width=500, height=500):
    image = Image.new('RGBA', (width, height), (255, 255, 255, 255)) # Canvas blanco inicial

    for primitive in solution.primitives:
        draw = ImageDraw.Draw(image, 'RGBA')

        vertices = primitive.vertices
        r, g, b = primitive.color
        alpha = int(0.8 * 255) # Opacidad fija por ahora

        draw.polygon(vertices, fill=(r, g, b, alpha))

    # Convertir a RGB para la salida final si es necesario (sin canal alfa)
    image_rgb = image.convert('RGB')
    return image_rgb

if __name__ == "__main__":
    # Ejemplo de uso con los nuevos nombres
    solutions = create_initial_population(1, 50) 
    generated_image = render_solution_to_image(solutions[0])
    generated_image.show()
    generated_image.save("output.png")
