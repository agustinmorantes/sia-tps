import json
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import sys

def load_result(experiment_name, results_dir):
    """Carga un archivo de resultados individual"""
    filepath = os.path.join(results_dir, f"exercise_3_3_{experiment_name}.json")
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def plot_noise_robustness_ranking(results_dir, graphs_dir):
    """Clasifica configuraciones por su robustez al ruido (menor caída de accuracy)"""

    print("Generando ranking de robustez al ruido...", flush=True)

    # Define configuration pairs (no noise, high noise)
    config_pairs = [
        ('baseline', 'noise_0.8', 'Base (Sigmoide, SGD, η=0.05)'),
        ('tanh_no_noise', 'tanh_high_noise', 'Activación Tanh'),
        ('eta_0.01_no_noise', 'eta_0.01_high_noise', 'Tasa de aprendizaje η=0.01'),
        ('eta_0.05_no_noise', 'eta_0.05_high_noise', 'Tasa de aprendizaje η=0.05'),
        ('eta_0.1_no_noise', 'eta_0.1_high_noise', 'Tasa de aprendizaje η=0.1'),
        ('momentum_no_noise', 'momentum_high_noise', 'Optimizador Momentum'),
        ('adam_no_noise', 'adam_high_noise', 'Optimizador Adam'),
        ('minibatch_2_no_noise', 'minibatch_2_high_noise', 'Mini-batch (tamaño=2)'),
        ('minibatch_5_no_noise', 'minibatch_5_high_noise', 'Mini-batch (tamaño=5)'),
        ('batch_no_noise', 'batch_high_noise', 'Batch completo'),
        ('alpha_0.5_no_noise', 'alpha_0.5_high_noise', 'Momentum α=0.5'),
        ('big_middle_layer_no_noise', 'big_middle_layer_high_noise', 'Red ancha (100)'),
        ('deep_no_noise', 'deep_high_noise', 'Red profunda (2 capas)'),
    ]

    robustness_data = []

    for no_noise_exp, high_noise_exp, label in config_pairs:
        no_noise = load_result(no_noise_exp, results_dir)
        high_noise = load_result(high_noise_exp, results_dir)

        if no_noise and high_noise:
            no_noise_acc = no_noise['aggregate_statistics']['mean_accuracy'] * 100
            high_noise_acc = high_noise['aggregate_statistics']['mean_accuracy'] * 100
            degradation = no_noise_acc - high_noise_acc
            retention = (high_noise_acc / no_noise_acc * 100) if no_noise_acc > 0 else 0

            robustness_data.append({
                'label': label,
                'no_noise_acc': no_noise_acc,
                'high_noise_acc': high_noise_acc,
                'degradation': degradation,
                'retention': retention
            })

    # Ordenar por retención (mayor es mejor)
    robustness_data.sort(key=lambda x: x['retention'], reverse=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Gráfico 1: Degradación de accuracy
    labels = [d['label'] for d in robustness_data]
    degradations = [d['degradation'] for d in robustness_data]

    colors = ['#27ae60' if d < 20 else '#f39c12' if d < 40 else '#e74c3c' for d in degradations]

    y_pos = np.arange(len(labels))
    bars = ax1.barh(y_pos, degradations, color=colors, alpha=0.8)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel('Degradación de accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Robustez al ruido: Caída de accuracy\n(Menor es mejor)',
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    # Agregar etiquetas de valores
    for i, (bar, val) in enumerate(zip(bars, degradations)):
        ax1.text(val + 1, i, f'{val:.1f}%', va='center', fontsize=8)

    # Gráfico 2: Porcentaje de retención de accuracy
    retentions = [d['retention'] for d in robustness_data]
    colors2 = ['#27ae60' if r > 80 else '#f39c12' if r > 60 else '#e74c3c' for r in retentions]

    bars2 = ax2.barh(y_pos, retentions, color=colors2, alpha=0.8)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel('Retención de accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Robustez al ruido: Retención de accuracy\n(Mayor es mejor)',
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.axvline(x=100, color='green', linestyle='--', alpha=0.5, linewidth=2)

    # Agregar etiquetas de valores
    for i, (bar, val) in enumerate(zip(bars2, retentions)):
        ax2.text(val - 5, i, f'{val:.1f}%', va='center', ha='right', fontsize=8, color='white', fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(graphs_dir, "noise_robustness_ranking.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Ranking de robustez al ruido guardado en: {output_path}", flush=True)
    plt.close()

    # Imprimir tabla resumen
    print("\n" + "="*80, flush=True)
    print("RANKING DE ROBUSTEZ AL RUIDO (de más a menos robusto)", flush=True)
    print("="*80, flush=True)
    print(f"{'Rank':<6} {'Configuración':<40} {'Sin ruido':<12} {'Ruido alto':<12} {'Caída':<10} {'Retención':<10}", flush=True)
    print("-"*80, flush=True)
    for rank, data in enumerate(robustness_data, 1):
        print(f"{rank:<6} {data['label']:<40} {data['no_noise_acc']:>10.1f}% {data['high_noise_acc']:>10.1f}% "
              f"{data['degradation']:>8.1f}% {data['retention']:>8.1f}%", flush=True)
    print("="*80 + "\n", flush=True)

def plot_noise_sensitivity_curves(results_dir, graphs_dir):
    """Grafica accuracy vs escala de ruido para la configuración base"""

    print("Generando curvas de sensibilidad al ruido...", flush=True)

    noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    noise_scales = []
    mean_accuracies = []
    std_accuracies = []
    min_accuracies = []
    max_accuracies = []

    for noise in noise_levels:
        if noise == 0.0:
            exp_name = 'baseline'
        else:
            exp_name = f'noise_{noise}'

        data = load_result(exp_name, results_dir)
        if data:
            stats = data['aggregate_statistics']
            noise_scales.append(noise)
            mean_accuracies.append(stats['mean_accuracy'] * 100)
            std_accuracies.append(stats['std_accuracy'] * 100)
            min_accuracies.append(stats['min_accuracy'] * 100)
            max_accuracies.append(stats['max_accuracy'] * 100)

    # Gráfico 1: Accuracy promedio con desviación estándar
    ax1.errorbar(noise_scales, mean_accuracies, yerr=std_accuracies,
                 marker='o', markersize=8, capsize=5, linewidth=2.5,
                 color='#2E86AB', ecolor='#A23B72', label='Media ± Desv. Est.')

    ax1.fill_between(noise_scales,
                     [m - s for m, s in zip(mean_accuracies, std_accuracies)],
                     [m + s for m, s in zip(mean_accuracies, std_accuracies)],
                     alpha=0.2, color='#2E86AB')

    ax1.set_xlabel('Escala de ruido', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Configuración base: Accuracy vs ruido\n(Sigmoide, SGD, η=0.05, Online)',
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])
    ax1.legend(fontsize=10)

    # Gráfico 2: Accuracy mín, media, máx
    ax2.plot(noise_scales, max_accuracies, marker='^', markersize=7,
             linewidth=2, label='Máximo', color='#27ae60', alpha=0.7)
    ax2.plot(noise_scales, mean_accuracies, marker='o', markersize=8,
             linewidth=2.5, label='Media', color='#2E86AB')
    ax2.plot(noise_scales, min_accuracies, marker='v', markersize=7,
             linewidth=2, label='Mínimo', color='#e74c3c', alpha=0.7)

    ax2.fill_between(noise_scales, min_accuracies, max_accuracies,
                     alpha=0.15, color='gray', label='Rango')

    ax2.set_xlabel('Escala de ruido', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Rango de accuracy en 100 corridas', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])
    ax2.legend(fontsize=10)

    plt.tight_layout()
    output_path = os.path.join(graphs_dir, "noise_sensitivity_detailed.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Curvas de sensibilidad al ruido guardadas en: {output_path}", flush=True)
    plt.close()

def plot_hyperparameter_stability(results_dir, graphs_dir):
    """Compara desviación estándar entre diferentes hiperparámetros"""

    print("Generando comparación de estabilidad de hiperparámetros...", flush=True)

    config_pairs = [
        ('baseline', 'noise_0.8', 'Base'),
        ('tanh_no_noise', 'tanh_high_noise', 'Tanh'),
        ('eta_0.01_no_noise', 'eta_0.01_high_noise', 'η=0.01'),
        ('eta_0.1_no_noise', 'eta_0.1_high_noise', 'η=0.1'),
        ('momentum_no_noise', 'momentum_high_noise', 'Momentum'),
        ('adam_no_noise', 'adam_high_noise', 'Adam'),
        ('minibatch_5_no_noise', 'minibatch_5_high_noise', 'Mini-batch'),
        ('batch_no_noise', 'batch_high_noise', 'Batch completo'),
        ('big_middle_layer_no_noise', 'big_middle_layer_high_noise', 'Red ancha'),
        ('deep_no_noise', 'deep_high_noise', 'Red profunda'),
    ]

    fig, ax = plt.subplots(figsize=(12, 7))

    labels = []
    no_noise_stds = []
    high_noise_stds = []

    for no_noise_exp, high_noise_exp, label in config_pairs:
        no_noise = load_result(no_noise_exp, results_dir)
        high_noise = load_result(high_noise_exp, results_dir)

        if no_noise and high_noise:
            labels.append(label)
            no_noise_stds.append(no_noise['aggregate_statistics']['std_accuracy'] * 100)
            high_noise_stds.append(high_noise['aggregate_statistics']['std_accuracy'] * 100)

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, no_noise_stds, width, label='Sin ruido',
                   color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, high_noise_stds, width, label='Ruido alto (0.8)',
                   color='#e74c3c', alpha=0.8)

    ax.set_xlabel('Configuración', fontsize=12, fontweight='bold')
    ax.set_ylabel('Desviación estándar (%)', fontsize=12, fontweight='bold')
    ax.set_title('Comparación de estabilidad de predicción\n(Menor desv. est. = predicciones más consistentes)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = os.path.join(graphs_dir, "hyperparameter_stability.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparación de estabilidad de hiperparámetros guardada en: {output_path}", flush=True)
    plt.close()

def generate_all_noise_analysis():
    """Genera todos los gráficos de análisis de robustez al ruido"""
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    graphs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    print("\n" + "="*60, flush=True)
    print("Generando gráficos de análisis de robustez al ruido", flush=True)
    print("="*60 + "\n", flush=True)

    sys.stdout.flush()

    plot_noise_robustness_ranking(results_dir, graphs_dir)
    plot_noise_sensitivity_curves(results_dir, graphs_dir)
    plot_hyperparameter_stability(results_dir, graphs_dir)

    print("\n" + "="*60, flush=True)
    print("¡Gráficos de análisis de ruido generados exitosamente!", flush=True)
    print(f"Directorio de salida: {graphs_dir}", flush=True)
    print("="*60 + "\n", flush=True)

if __name__ == "__main__":
    generate_all_noise_analysis()
