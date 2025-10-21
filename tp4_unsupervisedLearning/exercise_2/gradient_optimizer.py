import numpy as np
#Maneja actualizaciones de pesos usando gradiente descendente

class GradientDescentOptimizer: #Optimizador de Gradiente Descendente para entrenamiento de perceptrón
   
    
    def __init__(self, options=None):
        self.learning_rate = options.get("rate", 0.01)

    def calculate_weight_update(self, gradient, **kwargs):
        return self.learning_rate * gradient

    @staticmethod
    def create_optimizer(optimizer_type):
        if optimizer_type == "GRADIENT_DESCENT":
            return GradientDescentOptimizer
        else:
            raise ValueError("Solo se soporta GRADIENT_DESCENT")


class OjaRuleOptimizer:
    """
    Optimizador que implementa la regla de Oja para encontrar la primera componente principal
    Regla: Δw = η(Ox_i^n - O^2w_i^n)
    """
    
    def __init__(self, options=None):
        self.learning_rate = options.get("rate", 0.01)
    
    def calculate_weight_update(self, input_vector, weights, **kwargs):
        """
        Calcula la actualización de pesos usando la regla de Oja
        """
        # O = salida de la neurona (producto punto w·x)
        output = np.dot(weights, input_vector)
        
        # Δw = η(Ox_i^n - O^2w_i^n)
        weight_update = self.learning_rate * (output * input_vector - output**2 * weights)
        
        return weight_update
    
    @staticmethod
    def create_optimizer(optimizer_type):
        if optimizer_type == "OJA_RULE":
            return OjaRuleOptimizer
        else:
            raise ValueError("Solo se soporta OJA_RULE para este optimizador")