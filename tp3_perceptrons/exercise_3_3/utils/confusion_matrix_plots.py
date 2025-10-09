import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def load_result(name):
    """Carga un archivo de resultados por nombre de experimento"""
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    filepath = os.path.join(results_dir, f"exercise_3_3_{name}.json")
    with open(filepath, 'r') as f:
        return json.load(f)

def create_confusion_matrix(data):
    """Crea la matriz de confusión a partir de los datos usando todas las corridas"""
    digit_labels = data['digit_labels']

    # Agregar predicciones de todas las corridas
    all_true_labels = []
    all_predictions = []

    for run in data['individual_runs']:
        predicted_digits = run['predicted_digits']
        # Repetir las etiquetas verdaderas para cada corrida
        all_true_labels.extend(digit_labels)
        all_predictions.extend(predicted_digits)

    # Crear matriz de confusión con todas las predicciones
    cm = confusion_matrix(all_true_labels, all_predictions, labels=list(range(10)))
    return cm

def plot_confusion_matrix(cm, title, ax):
    """Grafica una sola matriz de confusión"""
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                cbar_kws={'label': 'Cantidad'})
    ax.set_xlabel('Predicho', fontsize=10, fontweight='bold')
    ax.set_ylabel('Verdadero', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')

def plot_confusion_matrix_comparison(configs, main_title, filename):
    """Grafica matrices de confusión para múltiples configuraciones"""
    n_configs = len(configs)
    cols = 2
    rows = (n_configs + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(14, 6*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for idx, (name, config_title) in enumerate(configs):
        try:
            data = load_result(name)
            cm = create_confusion_matrix(data)
            accuracy = data['aggregate_statistics']['mean_accuracy']
            noise = data['noise_scale']

            title = f"{config_title}\nRuido: {noise:.1f}, Accuracy: {accuracy*100:.1f}%"
            plot_confusion_matrix(cm, title, axes[idx])
        except FileNotFoundError:
            axes[idx].text(0.5, 0.5, f'Datos no encontrados:\n{name}',
                          ha='center', va='center', fontsize=12)
            axes[idx].set_title(config_title, fontsize=11, fontweight='bold')

    # Oculta subplots extra
    for idx in range(len(configs), len(axes)):
        axes[idx].axis('off')

    fig.suptitle(main_title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.99))

    graphs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "graphs")
    os.makedirs(graphs_dir, exist_ok=True)
    output_path = os.path.join(graphs_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {filename}")

def generate_all_confusion_matrices():
    """Genera todos los gráficos de matrices de confusión para distintas comparaciones de hiperparámetros"""

    # 1. Comparación de niveles de ruido
    plot_confusion_matrix_comparison(
        [
            ('baseline', 'Base (Sin Ruido)'),
            ('noise_0.4', 'Ruido Moderado (0.4)'),
            ('noise_0.8', 'Ruido Alto (0.8)'),
            ('noise_1.2', 'Ruido Muy Alto (1.2)')
        ],
        'Matrices de Confusión: Impacto de los Niveles de Ruido',
        'confusion_matrix_noise_levels.png'
    )

    # 2. Comparación de función de activación
    plot_confusion_matrix_comparison(
        [
            ('eta_0.05_no_noise', 'Sigmoide (Sin Ruido)'),
            ('tanh_no_noise', 'Tanh (Sin Ruido)'),
            ('eta_0.05_high_noise', 'Sigmoide (Ruido Alto)'),
            ('tanh_high_noise', 'Tanh (Ruido Alto)')
        ],
        'Matrices de Confusión: Comparación de Función de Activación',
        'confusion_matrix_activation.png'
    )

    # 3. Comparación de tasa de aprendizaje
    plot_confusion_matrix_comparison(
        [
            ('eta_0.01_no_noise', 'η=0.01 (Sin Ruido)'),
            ('eta_0.05_no_noise', 'η=0.05 (Sin Ruido)'),
            ('eta_0.1_no_noise', 'η=0.1 (Sin Ruido)'),
            ('eta_0.01_high_noise', 'η=0.01 (Ruido Alto)'),
            ('eta_0.05_high_noise', 'η=0.05 (Ruido Alto)'),
            ('eta_0.1_high_noise', 'η=0.1 (Ruido Alto)')
        ],
        'Matrices de Confusión: Impacto de la Tasa de Aprendizaje',
        'confusion_matrix_learning_rate.png'
    )

    # 4. Comparación de optimizador
    plot_confusion_matrix_comparison(
        [
            ('eta_0.05_no_noise', 'SGD (Sin Ruido)'),
            ('momentum_no_noise', 'Momentum (Sin Ruido)'),
            ('adam_no_noise', 'Adam (Sin Ruido)'),
            ('eta_0.05_high_noise', 'SGD (Ruido Alto)'),
            ('momentum_high_noise', 'Momentum (Ruido Alto)'),
            ('adam_high_noise', 'Adam (Ruido Alto)')
        ],
        'Matrices de Confusión: Comparación de Optimizador',
        'confusion_matrix_optimizer.png'
    )

    # 5. Comparación de tamaño de batch
    plot_confusion_matrix_comparison(
        [
            ('eta_0.05_no_noise', 'Online (bs=1, Sin Ruido)'),
            ('minibatch_2_no_noise', 'Mini-batch (bs=2, Sin Ruido)'),
            ('minibatch_5_no_noise', 'Mini-batch (bs=5, Sin Ruido)'),
            ('batch_no_noise', 'Batch (Sin Ruido)'),
            ('eta_0.05_high_noise', 'Online (bs=1, Ruido Alto)'),
            ('minibatch_2_high_noise', 'Mini-batch (bs=2, Ruido Alto)'),
            ('minibatch_5_high_noise', 'Mini-batch (bs=5, Ruido Alto)'),
            ('batch_high_noise', 'Batch (Ruido Alto)')
        ],
        'Matrices de Confusión: Impacto del Tamaño de Batch',
        'confusion_matrix_batch_size.png'
    )

    # 6. Comparación de arquitectura
    plot_confusion_matrix_comparison(
        [
            ('eta_0.05_no_noise', 'Estándar (35-50-10, Sin Ruido)'),
            ('big_middle_layer_no_noise', 'Más Ancha (35-100-10, Sin Ruido)'),
            ('deep_no_noise', 'Más Profunda (35-50-50-10, Sin Ruido)'),
            ('eta_0.05_high_noise', 'Estándar (35-50-10, Ruido Alto)'),
            ('big_middle_layer_high_noise', 'Más Ancha (35-100-10, Ruido Alto)'),
            ('deep_high_noise', 'Más Profunda (35-50-50-10, Ruido Alto)')
        ],
        'Matrices de Confusión: Comparación de Arquitectura',
        'confusion_matrix_architecture.png'
    )

    # 7. Comparación de alpha de momentum
    plot_confusion_matrix_comparison(
        [
            ('alpha_0.5_no_noise', 'α=0.5 (Sin Ruido)'),
            ('momentum_no_noise', 'α=0.9 (Sin Ruido)'),
            ('alpha_0.5_high_noise', 'α=0.5 (Ruido Alto)'),
            ('momentum_high_noise', 'α=0.9 (Ruido Alto)')
        ],
        'Matrices de Confusión: Comparación de Alpha de Momentum',
        'confusion_matrix_momentum_alpha.png'
    )

    # 8. Mejor vs peor desempeño
    plot_confusion_matrix_comparison(
        [
            ('baseline', 'Mejor: Base (Sin Ruido)'),
            ('noise_2.0', 'Peor: Ruido Extremo (2.0)')
        ],
        'Matrices de Confusión: Mejor vs Peor Caso',
        'confusion_matrix_best_vs_worst.png'
    )

if __name__ == "__main__":
    print("Generando gráficos de matrices de confusión...")
    generate_all_confusion_matrices()
    print("\n¡Todos los gráficos de matrices de confusión generados exitosamente!")
