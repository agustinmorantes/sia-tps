"""
Carga de datasets, preprocesamiento y validación cruzada K-Fold
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from .training_config import PerceptronTrainingConfig


class Exercise2DataHandler:
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.input_features = None
        self.target_values = None
        self._load_dataset()

    def _load_dataset(self):
        """Cargar dataset desde archivo CSV"""
        dataset = pd.read_csv(self.dataset_path)
        self.target_values = dataset["y"].to_numpy()        # Me guardo los valores de  y 
        features = dataset.drop(columns=["y"]).to_numpy()
        bias = PerceptronTrainingConfig.get_instance().bias_value
        bias_column = np.full((features.shape[0], 1), bias)
        self.input_features = np.concatenate([bias_column, features], axis=1) #devuelvo bias,x1,x2,x3

    def get_dataset_info(self):
        """Obtener información básica sobre el dataset"""
        return {
            "total_samples": len(self.input_features),
            "feature_count": self.input_features.shape[1],
            "target_range": (np.min(self.target_values), np.max(self.target_values))
        }

    def create_k_fold_splits(self, k_folds=7):              #K-1 partes para entrenar,restante para validar,repitoproceso K veces, rotando qué parte se usa para test
        kf = KFold(n_splits=k_folds)                        #Crea divisiones de validación cruzada K-Fold
        fold_splits = []
        
        for train_indices, test_indices in kf.split(self.input_features, self.target_values):
            train_inputs = self.input_features[train_indices]
            train_targets = self.target_values[train_indices]
            test_inputs = self.input_features[test_indices]
            test_targets = self.target_values[test_indices]
            fold_splits.append((train_inputs, train_targets, test_inputs, test_targets)) # me devuelve una lista con k elementos 
        
        return fold_splits

    def initialize_random_weights(self, feature_count, use_zeros=False): #Inicializar pesos aleatorios para el perceptrón

        if use_zeros:
            return np.zeros(feature_count)
        
        random_gen = PerceptronTrainingConfig.get_instance().random_generator
        return random_gen.uniform(-1, 1, feature_count) #devuelve vector de valores aleatorios entre -1 y 1.
