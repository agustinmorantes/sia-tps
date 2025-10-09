import json
import os
import matplotlib.pyplot as plt
import numpy as np

def load_result(experiment_name, results_dir):
    """Load a single result file"""
    filepath = os.path.join(results_dir, f"exercise_3_3_{experiment_name}.json")
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def plot_activation_comparison(results_dir, graphs_dir):
    """Compare sigmoid vs tanh activation functions"""
    experiments = [
        ('baseline', 'Sigmoid (no noise)', 'sigmoid'),
        ('noise_0.8', 'Sigmoid (noise=0.8)', 'sigmoid'),
        ('tanh_no_noise', 'Tanh (no noise)', 'tanh'),
        ('tanh_high_noise', 'Tanh (noise=0.8)', 'tanh'),
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Data for plotting
    sigmoid_data = []
    tanh_data = []
    
    for exp_name, label, activation in experiments:
        data = load_result(exp_name, results_dir)
        if data:
            stats = data['aggregate_statistics']
            mean_acc = stats['mean_accuracy'] * 100
            std_acc = stats['std_accuracy'] * 100
            noise = data['noise_scale']
            
            if activation == 'sigmoid':
                sigmoid_data.append((noise, mean_acc, std_acc, label))
            else:
                tanh_data.append((noise, mean_acc, std_acc, label))
    
    # Plot 1: Grouped bar chart
    x = np.arange(2)  # Two conditions: no noise, high noise
    width = 0.35
    
    sigmoid_means = [sigmoid_data[0][1], sigmoid_data[1][1]]
    sigmoid_stds = [sigmoid_data[0][2], sigmoid_data[1][2]]
    tanh_means = [tanh_data[0][1], tanh_data[1][1]]
    tanh_stds = [tanh_data[0][2], tanh_data[1][2]]
    
    bars1 = ax1.bar(x - width/2, sigmoid_means, width, yerr=sigmoid_stds, 
                    label='Sigmoide', capsize=5, color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, tanh_means, width, yerr=tanh_stds,
                    label='Tanh', capsize=5, color='#e74c3c', alpha=0.8)
    
    ax1.set_xlabel('Condición de ruido', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Comparación de función de activación', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Sin ruido', 'Ruido alto (0.8)'])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 105])
    
    # Plot 2: Accuracy distribution (box plot)
    sigmoid_no_noise = load_result('baseline', results_dir)['aggregate_statistics']['all_accuracies']
    sigmoid_high_noise = load_result('noise_0.8', results_dir)['aggregate_statistics']['all_accuracies']
    tanh_no_noise = load_result('tanh_no_noise', results_dir)['aggregate_statistics']['all_accuracies']
    tanh_high_noise = load_result('tanh_high_noise', results_dir)['aggregate_statistics']['all_accuracies']
    
    box_data = [
        [acc * 100 for acc in sigmoid_no_noise],
        [acc * 100 for acc in sigmoid_high_noise],
        [acc * 100 for acc in tanh_no_noise],
        [acc * 100 for acc in tanh_high_noise]
    ]
    
    bp = ax2.boxplot(box_data, labels=['Sigmoide\n(sin ruido)', 'Sigmoide\n(ruido=0.8)',
                                        'Tanh\n(sin ruido)', 'Tanh\n(ruido=0.8)'],
                     patch_artist=True, showmeans=True)
    
    colors = ['#3498db', '#3498db', '#e74c3c', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Distribución de accuracy por activación', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 105])
    
    plt.tight_layout()
    output_path = os.path.join(graphs_dir, "activation_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparación de activación guardada en: {output_path}")
    plt.close()

def plot_learning_rate_comparison(results_dir, graphs_dir):
    """Compare different learning rates"""
    learning_rates = [0.01, 0.05, 0.1]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    no_noise_data = []
    high_noise_data = []
    
    for eta in learning_rates:
        no_noise = load_result(f'eta_{eta}_no_noise', results_dir)
        high_noise = load_result(f'eta_{eta}_high_noise', results_dir)
        
        if no_noise:
            stats = no_noise['aggregate_statistics']
            no_noise_data.append((eta, stats['mean_accuracy'] * 100, stats['std_accuracy'] * 100))
        
        if high_noise:
            stats = high_noise['aggregate_statistics']
            high_noise_data.append((eta, stats['mean_accuracy'] * 100, stats['std_accuracy'] * 100))
    
    # Plot lines with error bars
    etas_no = [d[0] for d in no_noise_data]
    means_no = [d[1] for d in no_noise_data]
    stds_no = [d[2] for d in no_noise_data]
    
    etas_high = [d[0] for d in high_noise_data]
    means_high = [d[1] for d in high_noise_data]
    stds_high = [d[2] for d in high_noise_data]
    
    ax.errorbar(etas_no, means_no, yerr=stds_no, marker='o', markersize=10, 
                capsize=5, linewidth=2, label='Sin ruido', color='#27ae60')
    ax.errorbar(etas_high, means_high, yerr=stds_high, marker='s', markersize=10,
                capsize=5, linewidth=2, label='Ruido alto (0.8)', color='#e67e22')

    ax.set_xlabel('Tasa de aprendizaje (η)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Impacto de la tasa de aprendizaje en accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    ax.set_xscale('log')
    
    plt.tight_layout()
    output_path = os.path.join(graphs_dir, "learning_rate_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparación de tasa de aprendizaje guardada en: {output_path}")
    plt.close()

def plot_optimizer_comparison(results_dir, graphs_dir):
    """Compare different optimizers"""
    optimizers = ['sgd', 'momentum', 'adam']
    optimizer_labels = ['SGD', 'Momentum', 'Adam']
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    x = np.arange(len(optimizers))
    width = 0.35
    
    no_noise_means = []
    no_noise_stds = []
    high_noise_means = []
    high_noise_stds = []
    
    for opt in optimizers:
        # For SGD, use baseline experiments
        if opt == 'sgd':
            no_noise = load_result('baseline', results_dir)
            high_noise = load_result('noise_0.8', results_dir)
        else:
            no_noise = load_result(f'{opt}_no_noise', results_dir)
            high_noise = load_result(f'{opt}_high_noise', results_dir)
        
        if no_noise:
            stats = no_noise['aggregate_statistics']
            no_noise_means.append(stats['mean_accuracy'] * 100)
            no_noise_stds.append(stats['std_accuracy'] * 100)
        
        if high_noise:
            stats = high_noise['aggregate_statistics']
            high_noise_means.append(stats['mean_accuracy'] * 100)
            high_noise_stds.append(stats['std_accuracy'] * 100)
    
    bars1 = ax.bar(x - width/2, no_noise_means, width, yerr=no_noise_stds,
                   label='Sin ruido', capsize=5, color='#9b59b6', alpha=0.8)
    bars2 = ax.bar(x + width/2, high_noise_means, width, yerr=high_noise_stds,
                   label='Ruido alto (0.8)', capsize=5, color='#e74c3c', alpha=0.8)

    ax.set_xlabel('Optimizador', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Comparación de optimizadores', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(optimizer_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    output_path = os.path.join(graphs_dir, "optimizer_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparación de optimizadores guardada en: {output_path}")
    plt.close()

def plot_batch_size_comparison(results_dir, graphs_dir):
    """Compare different batch sizes"""
    batch_configs = [
        ('baseline', 'Online (batch=1)', 1),
        ('minibatch_2_no_noise', 'Mini-batch (2)', 2),
        ('minibatch_5_no_noise', 'Mini-batch (5)', 5),
        ('batch_no_noise', 'Full Batch', 'full'),
    ]
    
    batch_configs_high = [
        ('noise_0.8', 'Online (batch=1)', 1),
        ('minibatch_2_high_noise', 'Mini-batch (2)', 2),
        ('minibatch_5_high_noise', 'Mini-batch (5)', 5),
        ('batch_high_noise', 'Full Batch', 'full'),
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # No noise
    labels = []
    means = []
    stds = []
    
    for exp_name, label, _ in batch_configs:
        data = load_result(exp_name, results_dir)
        if data:
            labels.append(label)
            stats = data['aggregate_statistics']
            means.append(stats['mean_accuracy'] * 100)
            stds.append(stats['std_accuracy'] * 100)
    
    x = np.arange(len(labels))
    bars = ax1.bar(x, means, yerr=stds, capsize=5, color='#16a085', alpha=0.8)
    ax1.set_xlabel('Configuración de batch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Impacto del tamaño de batch - Sin ruido', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 105])
    
    # High noise
    labels_high = []
    means_high = []
    stds_high = []
    
    for exp_name, label, _ in batch_configs_high:
        data = load_result(exp_name, results_dir)
        if data:
            labels_high.append(label)
            stats = data['aggregate_statistics']
            means_high.append(stats['mean_accuracy'] * 100)
            stds_high.append(stats['std_accuracy'] * 100)
    
    x_high = np.arange(len(labels_high))
    bars = ax2.bar(x_high, means_high, yerr=stds_high, capsize=5, color='#c0392b', alpha=0.8)
    ax2.set_xlabel('Configuración de batch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Impacto del tamaño de batch - Ruido alto (0.8)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_high)
    ax2.set_xticklabels(labels_high, rotation=15, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 105])
    
    plt.tight_layout()
    output_path = os.path.join(graphs_dir, "batch_size_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparación de tamaño de batch guardada en: {output_path}")
    plt.close()

def plot_architecture_comparison(results_dir, graphs_dir):
    """Compare different network architectures"""
    architectures = [
        ('baseline', '[35, 50, 10]', 'Baseline'),
        ('big_middle_layer_no_noise', '[35, 100, 10]', 'Wide'),
        ('deep_no_noise', '[35, 50, 50, 10]', 'Deep'),
    ]
    
    architectures_high = [
        ('noise_0.8', '[35, 50, 10]', 'Baseline'),
        ('big_middle_layer_high_noise', '[35, 100, 10]', 'Wide'),
        ('deep_high_noise', '[35, 50, 50, 10]', 'Deep'),
    ]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(architectures))
    width = 0.35
    
    no_noise_means = []
    no_noise_stds = []
    high_noise_means = []
    high_noise_stds = []
    labels = []
    
    for (exp_no, arch, label), (exp_high, _, _) in zip(architectures, architectures_high):
        labels.append(f'{label}\n{arch}')
        
        data_no = load_result(exp_no, results_dir)
        data_high = load_result(exp_high, results_dir)
        
        if data_no:
            stats = data_no['aggregate_statistics']
            no_noise_means.append(stats['mean_accuracy'] * 100)
            no_noise_stds.append(stats['std_accuracy'] * 100)
        
        if data_high:
            stats = data_high['aggregate_statistics']
            high_noise_means.append(stats['mean_accuracy'] * 100)
            high_noise_stds.append(stats['std_accuracy'] * 100)
    
    bars1 = ax.bar(x - width/2, no_noise_means, width, yerr=no_noise_stds,
                   label='No Noise', capsize=5, color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, high_noise_means, width, yerr=high_noise_stds,
                   label='High Noise (0.8)', capsize=5, color='#e67e22', alpha=0.8)
    
    ax.set_xlabel('Arquitectura de red', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Comparación de arquitecturas de red', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    output_path = os.path.join(graphs_dir, "architecture_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparación de arquitecturas guardada en: {output_path}")
    plt.close()

def plot_momentum_alpha_comparison(results_dir, graphs_dir):
    """Compare different momentum coefficients"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Using momentum optimizer with different alpha values
    configs = [
        ('momentum_no_noise', 'α=0.9 (no noise)', 0.9, False),
        ('momentum_high_noise', 'α=0.9 (noise=0.8)', 0.9, True),
        ('alpha_0.5_no_noise', 'α=0.5 (no noise)', 0.5, False),
        ('alpha_0.5_high_noise', 'α=0.5 (noise=0.8)', 0.5, True),
    ]
    
    x_pos = [0, 1, 3, 4]  # Group by alpha value
    colors = ['#3498db', '#e74c3c', '#3498db', '#e74c3c']
    
    means = []
    stds = []
    labels = []
    
    for exp_name, label, alpha, is_noisy in configs:
        data = load_result(exp_name, results_dir)
        if data:
            stats = data['aggregate_statistics']
            means.append(stats['mean_accuracy'] * 100)
            stds.append(stats['std_accuracy'] * 100)
            labels.append(label)
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    
    # Add legend manually
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#3498db', alpha=0.8, label='Sin ruido'),
                      Patch(facecolor='#e74c3c', alpha=0.8, label='Ruido alto (0.8)')]
    ax.legend(handles=legend_elements, fontsize=11)
    
    ax.set_xlabel('Coeficiente de momentum (α)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Impacto del coeficiente de momentum', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    output_path = os.path.join(graphs_dir, "momentum_alpha_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparación de coeficiente de momentum guardada en: {output_path}")
    plt.close()

def plot_comprehensive_summary(results_dir, graphs_dir):
    """Create a comprehensive summary heatmap of all hyperparameters"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Define all experiments with their hyperparameters
    experiments = [
        # Baseline
        ('baseline', 'Baseline (sigmoid, η=0.05, SGD, online)', 0.0),
        ('noise_0.8', 'Baseline (sigmoid, η=0.05, SGD, online)', 0.8),
        
        # Activation
        ('tanh_no_noise', 'Tanh activation', 0.0),
        ('tanh_high_noise', 'Tanh activation', 0.8),
        
        # Learning rates
        ('eta_0.01_no_noise', 'Low learning rate (η=0.01)', 0.0),
        ('eta_0.01_high_noise', 'Low learning rate (η=0.01)', 0.8),
        ('eta_0.1_no_noise', 'High learning rate (η=0.1)', 0.0),
        ('eta_0.1_high_noise', 'High learning rate (η=0.1)', 0.8),
        
        # Optimizers
        ('momentum_no_noise', 'Momentum optimizer', 0.0),
        ('momentum_high_noise', 'Momentum optimizer', 0.8),
        ('adam_no_noise', 'Adam optimizer', 0.0),
        ('adam_high_noise', 'Adam optimizer', 0.8),
        
        # Batch sizes
        ('minibatch_5_no_noise', 'Mini-batch (size=5)', 0.0),
        ('minibatch_5_high_noise', 'Mini-batch (size=5)', 0.8),
        ('batch_no_noise', 'Full batch', 0.0),
        ('batch_high_noise', 'Full batch', 0.8),
        
        # Architecture
        ('big_middle_layer_no_noise', 'Wide network (100 hidden)', 0.0),
        ('big_middle_layer_high_noise', 'Wide network (100 hidden)', 0.8),
        ('deep_no_noise', 'Deep network (2 hidden layers)', 0.0),
        ('deep_high_noise', 'Deep network (2 hidden layers)', 0.8),
    ]
    
    # Collect data
    data_matrix = []
    row_labels = []
    
    current_config = None
    for exp_name, config_label, noise in experiments:
        if config_label != current_config:
            current_config = config_label
            data = load_result(exp_name, results_dir)
            if data:
                stats = data['aggregate_statistics']
                no_noise_acc = stats['mean_accuracy'] * 100
                
                # Find corresponding high noise experiment
                high_noise_exp = None
                for e, c, n in experiments:
                    if c == config_label and n > 0:
                        high_noise_exp = e
                        break
                
                if high_noise_exp:
                    high_data = load_result(high_noise_exp, results_dir)
                    if high_data:
                        high_noise_acc = high_data['aggregate_statistics']['mean_accuracy'] * 100
                        degradation = no_noise_acc - high_noise_acc
                        
                        data_matrix.append([no_noise_acc, high_noise_acc, degradation])
                        row_labels.append(config_label)
    
    # Create heatmap
    data_array = np.array(data_matrix)
    im = ax.imshow(data_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    # Set ticks and labels
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Sin ruido\nAccuracy (%)', 'Ruido alto (0.8)\nAccuracy (%)', 'Degradación\n(%)'],
                       fontsize=10, fontweight='bold')
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    
    # Add text annotations
    for i in range(len(row_labels)):
        for j in range(3):
            text = ax.text(j, i, f'{data_array[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=8, fontweight='bold')
    
    ax.set_title('Resumen de resultados por hiperparámetro',
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Valor', rotation=270, labelpad=20, fontsize=11, fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(graphs_dir, "comprehensive_summary_heatmap.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Heatmap de resumen integral guardado en: {output_path}")
    plt.close()

def generate_all_graphs():
    """Generate all comparison graphs"""
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    graphs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "graphs")
    os.makedirs(graphs_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("Generando gráficos de comparación de hiperparámetros")
    print("="*60 + "\n")
    
    plot_activation_comparison(results_dir, graphs_dir)
    plot_learning_rate_comparison(results_dir, graphs_dir)
    plot_optimizer_comparison(results_dir, graphs_dir)
    plot_batch_size_comparison(results_dir, graphs_dir)
    plot_architecture_comparison(results_dir, graphs_dir)
    plot_momentum_alpha_comparison(results_dir, graphs_dir)
    plot_comprehensive_summary(results_dir, graphs_dir)
    
    print("\n" + "="*60)
    print("¡Todos los gráficos generados exitosamente!")
    print(f"Directorio de salida: {graphs_dir}")
    print("="*60 + "\n")

if __name__ == "__main__":
    generate_all_graphs()
