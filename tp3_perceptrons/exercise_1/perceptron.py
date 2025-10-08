from typing import List, Tuple
import random
import math

class SimplePerceptron:
    
    def __init__(self, learning_rate: float = 0.1, max_epochs: int = 1000):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = None
        self.training_history = []
    
    def step_activation(self, x: float) -> int:
        return 1 if x >= 0 else -1
    
    def predict(self, inputs: List[float]) -> int:
        if self.weights is None:
            raise ValueError("El perceptrón no ha sido entrenado")
        
        # Calcular la suma ponderada
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        
        # Aplicar función de activación
        return self.step_activation(weighted_sum)
    
    def train(self, X: List[List[float]], y: List[int]) -> bool:

        # Inicializar pesos y bias aleatoriamente
        n_features = len(X[0])
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(n_features)]
        self.bias = random.uniform(-0.5, 0.5)
        
        print(f"Inicializando perceptrón con:")
        print(f"Pesos iniciales: {[f'{w:.3f}' for w in self.weights]}")
        print(f"Bias inicial: {self.bias:.3f}\n")
        
        for epoch in range(self.max_epochs):
            errors = 0
            
            # Guardar pesos ANTES de procesar la época
            weights_before = self.weights.copy()
            bias_before = self.bias
            
            for i, (inputs, target) in enumerate(zip(X, y)):
                # Calcular predicción
                prediction = self.predict(inputs)
                
                # Calcular error
                error = target - prediction
                
                if error != 0:
                    errors += 1
                    # Actualizar pesos y bias
                    for j in range(len(self.weights)):
                        self.weights[j] += self.learning_rate * error * inputs[j]
                    self.bias += self.learning_rate * error
            
            # Guardar historial de entrenamiento con pesos ANTES de actualizar
            self.training_history.append({
                'epoch': epoch,
                'errors': errors,
                'weights': weights_before,
                'bias': bias_before
            })
            
            # Si no hay errores, el perceptrón convergió
            if errors == 0:
                print(f"Convergencia alcanzada en {epoch} épocas")
                print(f"Pesos finales: {[f'{w:.3f}' for w in self.weights]}")
                print(f"Bias final: {self.bias:.3f}")
                return True
        
        print(f"No se alcanzó convergencia después de {self.max_epochs} épocas")
        return False
    
    def evaluate(self, X: List[List[float]], y: List[int]) -> Tuple[float, List[int]]:
    
        predictions = []
        correct = 0
        
        for inputs, target in zip(X, y):
            prediction = self.predict(inputs)
            predictions.append(prediction)
            if prediction == target:
                correct += 1
        
        accuracy = correct / len(y)
        return accuracy, predictions

