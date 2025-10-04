"""
Maneja actualizaciones de pesos usando gradiente descendente
"""

import numpy as np


class GradientDescentOptimizer:
    """Optimizador de Gradiente Descendente para entrenamiento de perceptrón"""
    
    def __init__(self, options=None):
        self.learning_rate = options.get("rate", 0.01)

    def calculate_weight_update(self, gradient, **kwargs):
        """Calcular actualización de pesos usando gradiente descendente"""
        return self.learning_rate * gradient

    @staticmethod
    def create_optimizer(optimizer_type):
        """Método factory para crear optimizadores"""
        if optimizer_type == "GRADIENT_DESCENT":
            return GradientDescentOptimizer
        else:
            raise ValueError("Solo se soporta GRADIENT_DESCENT")
