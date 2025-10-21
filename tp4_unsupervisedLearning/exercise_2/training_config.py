"""
Maneja todos los parámetros de entrenamiento y gestión de configuración
"""

import numpy as np


class PerceptronTrainingConfig:
   
    
    def __init__(self, **kwargs):
        self.convergence_threshold = kwargs.get("epsilon", 1e-5)
        self.random_seed = kwargs.get("seed", None)
        self.random_generator = np.random.default_rng(self.random_seed)
        self.max_training_epochs = kwargs.get("maxEpochs", 1000)
        self.bias_value = 1

    @staticmethod
    def get_instance():
        if PerceptronTrainingConfig._instance is None:
            raise ValueError("La configuración de entrenamiento no ha sido inicializada")
        return PerceptronTrainingConfig._instance

  
    _instance = None

    def __new__(cls, **kwargs):
        if cls._instance is None:
            cls._instance = super(PerceptronTrainingConfig, cls).__new__(cls)
            cls._instance._initialize(**kwargs)
        return cls._instance

    def _initialize(self, **kwargs):
        self.convergence_threshold = kwargs.get("epsilon", 1e-5)
        self.random_seed = kwargs.get("seed", None)
        self.random_generator = np.random.default_rng(self.random_seed)
        self.max_training_epochs = kwargs.get("maxEpochs", 1000)
        self.bias_value = 1
