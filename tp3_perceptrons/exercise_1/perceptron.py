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
            
            # Guardar historial de entrenamiento
            self.training_history.append({
                'epoch': epoch + 1,
                'errors': errors,
                'weights': self.weights.copy(),
                'bias': self.bias
            })
            
            # Si no hay errores, el perceptrón convergió
            if errors == 0:
                print(f"Convergencia alcanzada en {epoch + 1} épocas")
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

class LinearPerceptron:
    
    def __init__(self, learning_rate: float = 0.001, max_epochs: int = 1000):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = None
        self.training_history = []
    
    def linear_activation(self, x: float) -> float:
        return x
    
    def predict(self, inputs: List[float]) -> float:
        if self.weights is None:
            raise ValueError("El perceptrón no ha sido entrenado")
        
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return self.linear_activation(weighted_sum)
    
    def train(self, X: List[List[float]], y: List[float]) -> bool:
        n_features = len(X[0])
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(n_features)]
        self.bias = random.uniform(-0.5, 0.5)
        
        print(f"Inicializando perceptrón lineal con:")
        print(f"Pesos iniciales: {[f'{w:.3f}' for w in self.weights]}")
        print(f"Bias inicial: {self.bias:.3f}\n")
        
        # Guardar historial de entrenamiento para graficar o analizar
        self.training_history = []
        
        for epoch in range(self.max_epochs):
            total_error = 0.0
            for i, (inputs, target) in enumerate(zip(X, y)):
                prediction = self.predict(inputs)
                error = target - prediction
                total_error += 0.5 * (error**2) # Usamos el error cuadrático 

                # Actualizar pesos y bias 
                for j in range(len(self.weights)):
                    self.weights[j] += self.learning_rate * error * inputs[j]
                self.bias += self.learning_rate * error
            
            mse = total_error / len(X)
            self.training_history.append({'epoch': epoch + 1, 'mse': mse, 'weights': self.weights.copy(), 'bias': self.bias})

          
            if mse < 0.001 and epoch > 100: # Un umbral de error bajo y un mínimo de épocas
                 print(f"Convergencia 'aproximada' alcanzada en {epoch + 1} épocas (MSE: {mse:.6f})")
                 print(f"Pesos finales: {[f'{w:.3f}' for w in self.weights]}")
                 print(f"Bias final: {self.bias:.3f}")
                 return True
            
        print(f"Entrenamiento finalizado después de {self.max_epochs} épocas (MSE final: {mse:.6f})")
        return False # No hubo una convergencia 'perfecta' a un error muy bajo
    
    def evaluate(self, X: List[List[float]], y: List[float]) -> Tuple[float, List[float]]:
        predictions = []
        for inputs in X:
            predictions.append(self.predict(inputs))
        
        # Para evaluar la precisión de un perceptrón lineal en una tarea de clasificación binaria,
        # necesitamos binarizar sus salidas. Usaremos 0 como umbral.
        binary_predictions = [1 if p >= 0 else -1 for p in predictions]
        
        correct = sum(1 for pred, target in zip(binary_predictions, y) if pred == target)
        accuracy = correct / len(y)
        return accuracy, predictions


class NonLinearPerceptron:
    
    def __init__(self, learning_rate: float = 0.1, max_epochs: int = 1000, beta: float = 1.0):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.beta = beta 
        self.weights = None
        self.bias = None
        self.training_history = []

    # Función de activación sigmoidea 
    def sigmoid_activation(self, x: float) -> float:
        return 1 / (1 + math.exp(-self.beta * x))

    # Derivada de la función de activación sigmoidea
    def sigmoid_derivative(self, x: float) -> float:
        return self.beta * x * (1 - x)
    
    def predict_raw(self, inputs: List[float]) -> float:
        if self.weights is None:
            raise ValueError("El perceptrón no ha sido entrenado")
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return self.sigmoid_activation(weighted_sum)

    def predict(self, inputs: List[float]) -> int:
        # Para clasificación binaria, binarizamos la salida continua de la sigmoide
        return 1 if self.predict_raw(inputs) >= 0.5 else -1
    
    def train(self, X: List[List[float]], y: List[int]) -> bool:
        n_features = len(X[0])
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(n_features)]
        self.bias = random.uniform(-0.5, 0.5)
        
        print(f"Inicializando perceptrón no lineal (Sigmoide) con:")
        print(f"Pesos iniciales: {[f'{w:.3f}' for w in self.weights]}")
        print(f"Bias inicial: {self.bias:.3f}\n")
        
        self.training_history = []
        
        # Mapear las salidas y a (0, 1) para la función logística si son -1, 1
        y_mapped = [(1 if val == 1 else 0) for val in y]

        for epoch in range(self.max_epochs):
            total_error = 0.0
            for i, (inputs, target) in enumerate(zip(X, y_mapped)):
                # La salida de la sigmoide es continua entre 0 y 1
                raw_prediction = self.predict_raw(inputs) # Esto es la 'O' 
                error = target - raw_prediction # Error = target_mapeado - salida_sigmoide
                total_error += 0.5 * (error**2)

                # Regla de actualización de pesos para perceptrón no lineal 
                # El error se multiplica por la derivada de la función de activación en este punto
                for j in range(len(self.weights)):
                    self.weights[j] += self.learning_rate * error * self.sigmoid_derivative(raw_prediction) * inputs[j]
                self.bias += self.learning_rate * error * self.sigmoid_derivative(raw_prediction)
            
            mse = total_error / len(X)
            self.training_history.append({'epoch': epoch + 1, 'mse': mse, 'weights': self.weights.copy(), 'bias': self.bias})

            if mse < 0.001 and epoch > 100: # Convergencia por MSE bajo
                 print(f"Convergencia 'aproximada' alcanzada en {epoch + 1} épocas (MSE: {mse:.6f})")
                 print(f"Pesos finales: {[f'{w:.3f}' for w in self.weights]}")
                 print(f"Bias final: {self.bias:.3f}")
                 return True
            
        print(f"Entrenamiento finalizado después de {self.max_epochs} épocas (MSE final: {mse:.6f})")
        return False
    
    def evaluate(self, X: List[List[float]], y: List[int]) -> Tuple[float, List[int]]:
        predictions = []
        for inputs in X:
            predictions.append(self.predict(inputs))
        
        correct = sum(1 for pred, target in zip(predictions, y) if pred == target)
        accuracy = correct / len(y)
        return accuracy, predictions