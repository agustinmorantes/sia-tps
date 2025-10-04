import numpy as np


class ActivationFunction:

    
    def __init__(self, options=None):
        self.options = options or {}

    def configure_output_normalization(self, min_val, max_val):
        """Configurar normalización para funciones no lineales"""
        self.min_output = min_val
        self.max_output = max_val
    
    @staticmethod
    def create_activation_function(function_type, options=None):
 
        if function_type == "LINEAR":
            return LinearActivationFunction(options)
        elif function_type == "LOGISTIC":
            return LogisticActivationFunction(options)
        elif function_type == "TANH":
            return TanhActivationFunction(options)
        else:
            raise ValueError("Solo se soportan LINEAR, LOGISTIC y TANH")


class LinearActivationFunction(ActivationFunction):
    """Función de activación lineal: f(x) = w·x"""
    
    def __call__(self, input_vector, weights):
        return np.dot(weights, input_vector)

    def derivative(self, input_vector, weights):
        return 1
    
    def normalize(self, value):
        return value
    
    def denormalize(self, value):
        return value


class LogisticActivationFunction(ActivationFunction):
    """Función de activación logística: f(x) = 1/(1 + e^(-2β(w·x)))"""
    
    def __call__(self, input_vector, weights):
        beta = self.options.get("beta", 1.0)
        return 1 / (1 + np.exp(-2 * beta * (np.dot(weights, input_vector))))

    def derivative(self, input_vector, weights):
        beta = self.options.get("beta", 1.0)
        output = self.__call__(input_vector, weights)
        return 2 * beta * output * (1 - output)
    
    def normalize(self, value):
        return np.interp(value, [self.min_output, self.max_output], [0, 1])
    
    def denormalize(self, value):
        return np.interp(value, [0, 1], [self.min_output, self.max_output])


class TanhActivationFunction(ActivationFunction):
    """Función de activación tangente hiperbólica: f(x) = tanh(β(w·x))"""
    
    def __call__(self, input_vector, weights):
        beta = self.options.get("beta", 1.0)
        return np.tanh(beta * (np.dot(weights, input_vector)))

    def derivative(self, input_vector, weights):
        beta = self.options.get("beta", 1.0)
        output = self.__call__(input_vector, weights)
        return beta * (1 - np.power(output, 2))
    
    def normalize(self, value):
        return np.interp(value, [self.min_output, self.max_output], [-1, 1])
    
    def denormalize(self, value):
        return np.interp(value, [-1, 1], [self.min_output, self.max_output])
