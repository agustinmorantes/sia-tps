"""
Comparación entre perceptrones lineales y no lineales
"""

import numpy as np
from .data_handler import Exercise2DataHandler
from .activation_functions import ActivationFunction
from .gradient_optimizer import GradientDescentOptimizer
from .perceptron_model import SingleLayerPerceptronModel


class PerceptronComparison:
    
    
    def __init__(self, dataset_path, config):
        self.data_handler = Exercise2DataHandler(dataset_path)
        self.config = config

    def train_perceptron_type(self, activation_type, k_folds=7):
        """Entrenar perceptrón con función de activación especificada"""
        print(f"=== PERCEPTRÓN {activation_type} ===")
        
        # Crear función de activación
        activation_function = ActivationFunction.create_activation_function(
            activation_type, 
            self.config["perceptron"]["architecture"][0]["activationFunction"]["options"]
        )
        
        # Configurar normalización para funciones no lineales
        if activation_type != "LINEAR":
            target_min = np.min(self.data_handler.target_values)
            target_max = np.max(self.data_handler.target_values)
            activation_function.configure_output_normalization(target_min, target_max)
        
        # Crear optimizador
        optimizer_class = GradientDescentOptimizer.create_optimizer(
            self.config["learning"]["optimizer"]["type"]
        )
        optimizer = optimizer_class(self.config["learning"]["optimizer"]["options"])

        # Validación Cruzada K-Fold
        training_mse_list = []
        testing_mse_list = []
        
        fold_splits = self.data_handler.create_k_fold_splits(k_folds)
        for train_inputs, train_targets, test_inputs, test_targets in fold_splits:
            # Inicializar pesos
            initial_weights = self.data_handler.initialize_random_weights(len(train_inputs[0]))
            
            # Crear y entrenar modelo
            perceptron = SingleLayerPerceptronModel(activation_function, optimizer, initial_weights)
            perceptron.train_model(train_inputs, train_targets)
            
            # Calcular MSE
            training_mse = perceptron.calculate_mean_squared_error(train_inputs, train_targets)
            testing_mse = perceptron.calculate_mean_squared_error(test_inputs, test_targets)
            
            training_mse_list.append(training_mse)
            testing_mse_list.append(testing_mse)

        # Calcular estadísticas
        mean_training_mse = np.mean(training_mse_list)
        std_training_mse = np.std(training_mse_list)
        mean_testing_mse = np.mean(testing_mse_list)
        std_testing_mse = np.std(testing_mse_list)

        print(f"MSE Entrenamiento: {mean_training_mse:.6f} ± {std_training_mse:.6f}")
        print(f"MSE Prueba: {mean_testing_mse:.6f} ± {std_testing_mse:.6f}")
        
        return mean_training_mse, mean_testing_mse

    def compare_perceptrons(self):
        """Comparar perceptrones lineales vs no lineales"""
        print("Ejercicio 2: Comparación de Perceptrón Lineal vs No Lineal")
        print("Dataset: TP3-ej2-conjunto.csv")
        print("Métricas: Error Cuadrático Medio (MSE)")
        print("Validación: Validación Cruzada K-Fold")
        print("=" * 60)
        
        # Entrenar Perceptrón Lineal (más rápido con k=3)
        linear_train_mse, linear_test_mse = self.train_perceptron_type("LINEAR", k_folds=7)
        
        # Entrenar Perceptrón No Lineal (más confiable con k=7)
        non_linear_type = self.config["perceptron"]["architecture"][0]["activationFunction"]["type"]
        non_linear_train_mse, non_linear_test_mse = self.train_perceptron_type(non_linear_type, k_folds=7)
        
        # Comparar resultados
        self._display_comparison_results(
            linear_train_mse, linear_test_mse,
            non_linear_train_mse, non_linear_test_mse
        )

    def _display_comparison_results(self, linear_train_mse, linear_test_mse, 
                                  non_linear_train_mse, non_linear_test_mse):
        """Mostrar resultados de comparación"""
        print("\n=== RESULTADOS DE COMPARACIÓN ===")
        print(f"Perceptrón Lineal - MSE Entrenamiento: {linear_train_mse:.6f}, MSE Prueba: {linear_test_mse:.6f}")
        print(f"Perceptrón No Lineal - MSE Entrenamiento: {non_linear_train_mse:.6f}, MSE Prueba: {non_linear_test_mse:.6f}")
        
        # Determinar cuál funciona mejor
        if non_linear_test_mse < linear_test_mse:
            improvement = linear_test_mse - non_linear_test_mse
            print(f"\n ¡El Perceptrón No Lineal funciona mejor!")
            print(f"   MSE menor: {non_linear_test_mse:.6f} vs {linear_test_mse:.6f}")
            print(f"   Mejora: {improvement:.6f} ({improvement/linear_test_mse*100:.2f}%)")
            print("   El Perceptrón No Lineal tiene mejor capacidad de generalización")
        else:
            improvement = non_linear_test_mse - linear_test_mse
            print(f"\n ¡El Perceptrón Lineal funciona mejor!")
            print(f"   MSE menor: {linear_test_mse:.6f} vs {non_linear_test_mse:.6f}")
            print(f"   Mejora: {improvement:.6f} ({improvement/non_linear_test_mse*100:.2f}%)")
            print("   El Perceptrón Lineal tiene mejor capacidad de generalización")
