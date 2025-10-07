import numpy as np
import sys
import os

# Agregar el directorio src al path para importar MultiLayerPerceptron
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from multi_layer_perceptron import MultiLayerPerceptron
from mnist_loader import load_mnist, preprocess_mnist, create_validation_set

class MNISTClassifier:
    def __init__(self, layer_sizes=[784, 128, 64, 10], activation="tanh", 
                 optimizer="adam", eta=0.001, batch_size=32, seed=42):
        """
        Clasificador MNIST usando perceptrón multicapa
        
        Args:
            layer_sizes: Arquitectura de la red [input, hidden1, hidden2, output]
            activation: Función de activación ("tanh" o "sigmoid")
            optimizer: Optimizador ("sgd", "momentum", "adam")
            eta: Learning rate
            batch_size: Tamaño del batch (None=batch completo, 1=online, >1=minibatch)
            seed: Semilla para reproducibilidad
        """
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.optimizer = optimizer
        self.eta = eta
        self.batch_size = batch_size
        self.seed = seed
        
        # Inicializar el modelo
        self.model = None
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def create_model(self):
        """Crea el modelo de perceptrón multicapa"""
        self.model = MultiLayerPerceptron(
            layer_sizes=self.layer_sizes,
            activation=self.activation,
            eta=self.eta,
            optimizer=self.optimizer,
            batch_size=self.batch_size,
            seed=self.seed
        )
        print(f"Modelo creado: {self.layer_sizes}")
        print(f"Parámetros totales: {self._count_parameters()}")
    
    def _count_parameters(self):
        """Cuenta el número total de parámetros"""
        total = 0
        for i in range(len(self.layer_sizes) - 1):
            # Pesos
            total += self.layer_sizes[i] * self.layer_sizes[i + 1]
            # Bias
            total += self.layer_sizes[i + 1]
        return total
    
    def load_data(self, validation_split=0.2, sample_size=None):
        """Carga y preprocesa los datos MNIST"""
        print("Cargando datos MNIST...")
        
        # Cargar datos
        train_images, train_labels, test_images, test_labels = load_mnist()
        
        # Si se especifica sample_size, usar solo una muestra
        if sample_size is not None:
            print(f"Usando muestra de {sample_size} datos para entrenamiento")
            indices = np.random.choice(train_images.shape[0], sample_size, replace=False)
            train_images = train_images[indices]
            train_labels = train_labels[indices]
        
        # Preprocesar
        X_train, y_train, X_test, y_test, train_labels_orig, test_labels_orig = preprocess_mnist(
            train_images, train_labels, test_images, test_labels
        )
        
        # Crear conjunto de validación
        X_train, y_train, X_val, y_val = create_validation_set(
            X_train, y_train, validation_split
        )
        
        self.data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'train_labels_orig': train_labels_orig,
            'test_labels_orig': test_labels_orig
        }
        
        print(f"Datos preparados:")
        print(f"  Train: {X_train.shape}")
        print(f"  Validation: {X_val.shape}")
        print(f"  Test: {X_test.shape}")
        
        return self.data
    
    def train(self, epochs=50, verbose=True):
        """Entrena el modelo"""
        if self.model is None:
            self.create_model()
        
        print(f"Entrenando modelo por {epochs} épocas...")
        
        for epoch in range(epochs):
            # Entrenar una época
            self.model.train(
                self.data['X_train'], 
                self.data['y_train'], 
                epochs=1, 
                epsilon=1e-10  # Muy pequeño para no parar prematuramente
            )
            
            # Evaluar en train y validation
            train_pred = self.model.predict(self.data['X_train'])
            val_pred = self.model.predict(self.data['X_val'])
            
            train_loss = self._calculate_loss(self.data['y_train'], train_pred)
            val_loss = self._calculate_loss(self.data['y_val'], val_pred)
            
            train_acc = self._calculate_accuracy(self.data['y_train'], train_pred)
            val_acc = self._calculate_accuracy(self.data['y_val'], val_pred)
            
            # Guardar historial
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)
            
            if verbose and epoch % 5 == 0:
                print(f"Época {epoch:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                      f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        
        print("Entrenamiento completado")
    
    def _calculate_loss(self, y_true, y_pred):
        """Calcula la pérdida MSE"""
        return 0.5 * np.mean((y_true - y_pred) ** 2)
    
    def _calculate_accuracy(self, y_true, y_pred):
        """Calcula la precisión"""
        # Convertir one-hot a clases
        y_true_classes = np.argmax(y_true, axis=1)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        return np.mean(y_true_classes == y_pred_classes)
    
    def evaluate(self, X_test=None, y_test=None):
        """Evalúa el modelo en el conjunto de test"""
        if X_test is None:
            X_test = self.data['X_test']
            y_test = self.data['y_test']
        
        print("Evaluando modelo...")
        
        # Predicciones
        y_pred = self.model.predict(X_test)
        
        # Métricas
        test_loss = self._calculate_loss(y_test, y_pred)
        test_acc = self._calculate_accuracy(y_test, y_pred)
        
        # Clasificación por clase
        y_test_classes = np.argmax(y_test, axis=1)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Matriz de confusión
        confusion_matrix = self._calculate_confusion_matrix(y_test_classes, y_pred_classes)
        
        # Precision, Recall, F1 por clase
        precision, recall, f1 = self._calculate_class_metrics(y_test_classes, y_pred_classes)
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'confusion_matrix': confusion_matrix,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        print(f"Resultados de evaluación:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Precision promedio: {np.mean(precision):.4f}")
        print(f"  Recall promedio: {np.mean(recall):.4f}")
        print(f"  F1-score promedio: {np.mean(f1):.4f}")
        
        return results
    
    def _calculate_confusion_matrix(self, y_true, y_pred):
        """Calcula la matriz de confusión"""
        n_classes = 10
        cm = np.zeros((n_classes, n_classes), dtype=int)
        
        for i in range(len(y_true)):
            cm[y_true[i], y_pred[i]] += 1
        
        return cm
    
    def _calculate_class_metrics(self, y_true, y_pred):
        """Calcula precision, recall y F1 por clase"""
        n_classes = 10
        precision = np.zeros(n_classes)
        recall = np.zeros(n_classes)
        f1 = np.zeros(n_classes)
        
        for i in range(n_classes):
            tp = np.sum((y_true == i) & (y_pred == i))
            fp = np.sum((y_true != i) & (y_pred == i))
            fn = np.sum((y_true == i) & (y_pred != i))
            
            precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
        
        return precision, recall, f1
    
    def predict(self, X):
        """Hace predicciones"""
        return self.model.predict(X)
    
    def get_training_history(self):
        """Retorna el historial de entrenamiento"""
        return self.training_history
