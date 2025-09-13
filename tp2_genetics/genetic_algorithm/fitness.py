import cv2
import numpy as np
# from skimage.metrics import structural_similarity as ssim # Eliminado porque fitness_ssim se borrará
from .utils.generate_canvas import render_solution_to_image
from genetic_algorithm.models.individual_solution import IndividualSolution # Para anotación de tipo

class ImageSimilarityEvaluator:
    """Clase para evaluar la similitud entre la imagen objetivo y una solución individual."""
    def __init__(self, target_image_data):
        self.target_image_data = target_image_data # Renombrado de target_image

    def evaluate_average_pixel_difference(self, solution_individual: IndividualSolution) -> float:
        """Evalúa la similitud basándose en la diferencia absoluta promedio de píxeles."""
        target_image_cv = self.target_image_data
        rendered_solution_pil = render_solution_to_image(solution_individual)
        rendered_solution_np = np.array(rendered_solution_pil)
        rendered_solution_cv = cv2.cvtColor(rendered_solution_np, cv2.COLOR_RGB2BGR)
        rendered_solution_cv = cv2.resize(rendered_solution_cv, (self.target_image_data.shape[1], target_image_cv.shape[0]))
        
        pixel_difference_matrix = cv2.absdiff(target_image_cv, rendered_solution_cv)
        mean_pixel_difference = np.mean(pixel_difference_matrix)

        linear_similarity_score = 1 - (mean_pixel_difference / 255)
        final_fitness_value = linear_similarity_score ** 2 # Aplicar penalización no lineal

        return final_fitness_value


