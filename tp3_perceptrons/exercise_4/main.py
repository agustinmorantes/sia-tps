import numpy as np
import matplotlib.pyplot as plt
import os
import time
import json
import sys
from mnist_classifier import MNISTClassifier
from visualization import (
    plot_training_history,
    plot_confusion_matrix,
    plot_class_metrics,
    plot_sample_predictions
)

def load_config(config_path="exercise_4/config/config.json"):
    """Carga configuración desde JSON"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de configuración: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: El archivo JSON no es válido: {e}")
        sys.exit(1)

def run_experiment_from_config(config, config_name):
    """Ejecuta un experimento basado en la configuración JSON"""
    
    exp_config = config['config']
    
    print(f"Configuración cargada desde {config_name}:")
    print(f"  Arquitectura: {exp_config['layer_sizes']}")
    print(f"  Activación: {exp_config['activation']}")
    print(f"  Optimizador: {exp_config['optimizer']}")
    print(f"  Learning rate: {exp_config['eta']}")
    print(f"  Batch size: {exp_config['batch_size']}")
    print(f"  Épocas: {exp_config['epochs']}")
    print(f"  Muestra: {exp_config['sample_size']} datos")
    
    # Crear carpeta resources si no existe
    os.makedirs('resources', exist_ok=True)

    # Crear clasificador
    classifier = MNISTClassifier(
        layer_sizes=exp_config['layer_sizes'],
        activation=exp_config['activation'],
        optimizer=exp_config['optimizer'],
        eta=exp_config['eta'],
        batch_size=exp_config['batch_size'],
        seed=exp_config['seed']
    )
    
    # Cargar datos
    print("\nCargando datos MNIST...")
    classifier.load_data(sample_size=exp_config['sample_size'])
    
    # Entrenar
    print(f"\nEntrenando modelo...")
    start_time = time.time()
    classifier.train(epochs=exp_config['epochs'], verbose=exp_config['verbose'])
    training_time = time.time() - start_time
    
    # Evaluar
    print(f"\nEvaluando modelo...")
    results = classifier.evaluate()
    
    # Prefijo de nombre para guardar resultados
    name_prefix = os.path.splitext(os.path.basename(config_name))[0]  # ej: config_shallow
    
    # Guardar gráficos
    print(f"\nGenerando visualizaciones para {name_prefix}...")
    plot_training_history(
        classifier.get_training_history(),
        f'resources/training_history_{name_prefix}.png'
    )
    plot_confusion_matrix(
        results['confusion_matrix'],
        f'resources/confusion_matrix_{name_prefix}.png'
    )
    plot_class_metrics(
        results['precision'], results['recall'], results['f1_score'],
        f'resources/class_metrics_{name_prefix}.png'
    )
    
    # Predicciones de ejemplo
    X_test = classifier.data['X_test']
    y_test = classifier.data['y_test']
    y_pred = classifier.predict(X_test)
    plot_sample_predictions(
        X_test, y_test, y_pred, n_samples=10,
        save_path=f'resources/sample_predictions_{name_prefix}.png'
    )
    
    # Resumen
    print("\n" + "="*80)
    print(f"RESULTADOS FINALES - {name_prefix.upper()}")
    print("="*80)
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"Tiempo de entrenamiento: {training_time:.2f}s")
    print(f"Precision promedio: {np.mean(results['precision']):.4f}")
    print(f"Recall promedio: {np.mean(results['recall']):.4f}")
    print(f"F1-score promedio: {np.mean(results['f1_score']):.4f}")
    print(f"\nResultados y gráficos guardados en: resources/")
    
    return results

def main():
    """Función principal que ejecuta un experimento desde configuración JSON"""
    # Determinar ruta del archivo config
    config_path = "config/config.json"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    print(f"Cargando configuración desde: {config_path}")
    
    # Cargar configuración y nombre base
    config = load_config(config_path)
    config_name = os.path.basename(config_path)  # ej: config_shallow.json
    
    try:
        results = run_experiment_from_config(config, config_name)
    except Exception as e:
        print(f"\nError durante la ejecución: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
