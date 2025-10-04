"""
Modelo de Perceptrón
"""

import numpy as np
from .training_config import PerceptronTrainingConfig
from .activation_functions import ActivationFunction
from .gradient_optimizer import GradientDescentOptimizer


class SingleLayerPerceptronModel:
    
    
    def __init__(self, activation_function: ActivationFunction, optimizer: GradientDescentOptimizer, initial_weights):
        self.activation_function = activation_function
        self.optimizer = optimizer
        self.weights = initial_weights
        self.weight_updates = np.zeros(len(initial_weights))
        self.weight_count = len(initial_weights)
        self.weight_history = [initial_weights]
        self.weights_per_epoch = []
        self.training_config = PerceptronTrainingConfig.get_instance()

    def train_model(self, input_data, target_outputs):
        """Entrenar el modelo de perceptrón"""
        for epoch in range(self.training_config.max_training_epochs):
            for input_vector, target_output in zip(input_data, target_outputs):
                predicted_output = self.activation_function(input_vector, self.weights)
                error_gradient = self._calculate_error_gradient(input_vector, target_output, predicted_output)
                weight_update = self.optimizer.calculate_weight_update(error_gradient)
                self._accumulate_weight_updates(weight_update)

            self._apply_weight_updates()
            if self._check_convergence(input_data, target_outputs):
                self._save_epoch_weights()
                return
            self._save_epoch_weights()
        print("Entrenamiento completado sin convergencia.")

    def calculate_mean_squared_error(self, input_data, target_outputs):
        """Calcular Error Cuadrático Medio"""
        total_error = 0
        for input_vector, target_output in zip(input_data, target_outputs):
            predicted_output = self.activation_function(input_vector, self.weights)
            denormalized_output = self.activation_function.denormalize(predicted_output)
            total_error += np.power(target_output - denormalized_output, 2)
        return total_error / len(target_outputs)

    def predict(self, input_vector):
        """Hacer predicción para una entrada única"""
        predicted_output = self.activation_function(input_vector, self.weights)
        return self.activation_function.denormalize(predicted_output)

    def _calculate_error_gradient(self, input_vector, target_output, predicted_output):
        """Calcular gradiente de error para actualizaciones de pesos"""
        output_error = target_output - self.activation_function.denormalize(predicted_output)
        derivative = self.activation_function.derivative(input_vector, self.weights)
        return output_error * derivative * input_vector

    def _accumulate_weight_updates(self, weight_update):
        """Acumular actualizaciones de pesos para aprendizaje por lotes"""
        self.weight_updates = np.add(self.weight_updates, weight_update)

    def _apply_weight_updates(self):
        """Aplicar actualizaciones de pesos acumuladas"""
        self.weights = np.add(self.weights, self.weight_updates)
        self.weight_history.append(np.copy(self.weights))
        self.weight_updates = np.zeros(self.weight_count)
        
    def _save_epoch_weights(self):
        """Guardar pesos para la época actual"""
        self.weights_per_epoch.append(np.copy(self.weights))

    def _check_convergence(self, input_data, target_outputs):
        """Verificar si el modelo ha convergido"""
        return np.abs(self.calculate_mean_squared_error(input_data, target_outputs)) <= self.training_config.convergence_threshold

    def get_weights_per_epoch(self):
        """Obtener historial de pesos para todas las épocas"""
        return self.weights_per_epoch
