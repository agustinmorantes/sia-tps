
# Modelo de Perceptrón de una sola capa 


import numpy as np
from training_config import PerceptronTrainingConfig
from activation_functions import ActivationFunction
from gradient_optimizer import GradientDescentOptimizer, OjaRuleOptimizer


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
        for epoch in range(self.training_config.max_training_epochs):#itero tantas veces como max_training_epochs
            for input_vector, target_output in zip(input_data, target_outputs):
                predicted_output = self.activation_function(input_vector, self.weights) # call de activation_functions 
                
                if isinstance(self.optimizer, OjaRuleOptimizer):
                    # Para Oja, pasamos input_vector y weights directamente
                    weight_update = self.optimizer.calculate_weight_update(input_vector, self.weights)
                else:
                    # Comportamiento original para gradiente descendente
                    error_gradient = self._calculate_error_gradient(input_vector, target_output, predicted_output)# vector de 4 elementos (gradiente de cada peso)
                    weight_update = self.optimizer.calculate_weight_update(error_gradient) #en gradient_optimizer se calcula el weight_update=tasa de aprendizaje * error_gradient 
                    #este  weight_update me va diciendo lo que deberia variar cada peso para que el error sea menor
                
                self._accumulate_weight_updates(weight_update) #voy sumando todos los weight_update para luego hacer la actualizacion de pesos(modelo batch)

            self._apply_weight_updates() #Como es Batch Actualiza los pesos ahora  y reinicia el acumulador , self.weights = self.weights + self.weight_updates

            if self._check_convergence(input_data, target_outputs):
                self._save_epoch_weights()
                return
            self._save_epoch_weights()
        print("Entrenamiento completado sin convergencia.")

    def train_oja_pca(self, input_data, max_epochs=1000, convergence_threshold=1e-6):
        """
        Entrenamiento específico para PCA usando la regla de Oja
        """
        if not isinstance(self.optimizer, OjaRuleOptimizer):
            raise ValueError("Este método requiere un optimizador de tipo OjaRuleOptimizer")
        
        previous_weights = np.copy(self.weights)
        
        for epoch in range(max_epochs):
            # Mezclar los datos para cada época
            indices = np.random.permutation(len(input_data))
            
            for idx in indices:
                input_vector = input_data[idx]
                weight_update = self.optimizer.calculate_weight_update(input_vector, self.weights)
                self._accumulate_weight_updates(weight_update)
            
            self._apply_weight_updates()
            self._save_epoch_weights()
            
            # Verificar convergencia basada en el cambio de pesos
            weight_change = np.linalg.norm(self.weights - previous_weights)
            if weight_change < convergence_threshold:
                print(f"Convergencia alcanzada en época {epoch + 1}")
                return
            
            previous_weights = np.copy(self.weights)
            
            if (epoch + 1) % 100 == 0:
                print(f"Época {epoch + 1}: Cambio en pesos = {weight_change:.6f}")
        
        print("Entrenamiento completado sin convergencia.")

    def calculate_mean_squared_error(self, input_data, target_outputs):#Calcular Error Cuadrático Medio = (1/n) × Σ(target_i - predicted_i)²
        total_error = 0                                                  #n: Número de muestras , target_i: Valor real de la muestra i , predicted_i: Valor predicho para la muestra i
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
        output_error = target_output - self.activation_function.denormalize(predicted_output) #calcula que tan grande fue el error
        derivative = self.activation_function.derivative(input_vector, self.weights)     #derivate de activation_fun
        return output_error * derivative * input_vector  #devuelve un vector con el gradiente de cada peso de bias,x1,x2,x3   osea (0,g1,g2,g3)

    def _accumulate_weight_updates(self, weight_update):         #Acumular actualizaciones de pesos para aprendizaje por lotes suma los pesos de bias,x1,x2,x3   
        self.weight_updates = np.add(self.weight_updates, weight_update)

    def _apply_weight_updates(self):
        """Aplicar actualizaciones de pesos acumuladas""" #suma los pesos de bias,x1,x2,x3   
        self.weights = np.add(self.weights, self.weight_updates)
        self.weight_history.append(np.copy(self.weights))
        self.weight_updates = np.zeros(self.weight_count)
        
    def _save_epoch_weights(self):#guarda los pesos de bias,x1,x2,x3   
        self.weights_per_epoch.append(np.copy(self.weights))

    def _check_convergence(self, input_data, target_outputs): #verifica si el error es menor que el threshold
        
        return np.abs(self.calculate_mean_squared_error(input_data, target_outputs)) <= self.training_config.convergence_threshold

    def get_weights_per_epoch(self):# Obtener historial de pesos para todas las épocas
        return self.weights_per_epoch
