import json
import os.path
import numpy as np
import matplotlib.pyplot as plt


def load_run_metrics(output_dir: str) -> dict:
    metrics_path = os.path.join(output_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        return None

    metrics: dict
    with open(metrics_path, 'r') as file:
        metrics = json.load(file)

    gen_metrics = metrics['gen_metrics']
    keys = gen_metrics[0].keys()
    gen_metrics = { k: np.stack([d[k] for d in gen_metrics]) for k in keys }
    metrics['gen_metrics'] = gen_metrics

    return metrics

if __name__ == "__main__":
    graph_dir = "graphs"
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir, exist_ok=True)

    config_names = [os.path.splitext(config)[0] for config in os.listdir("configs")]
    output_dirs = ["outputs1", "outputs2", "outputs3"]
    all_run_metrics = {config: [load_run_metrics(os.path.join(output_dir, config)) for output_dir in output_dirs] for config in config_names}

    selection_method_configs = [
        "boltzmann_annealing",
        "boltzmann_high_temp",
        "boltzmann_low_temp",
        "elite",
        "ranking",
        "roulette",
        "tournament_deterministic",
        "tournament_probabilistic_high_threshold",
        "tournament_probabilistic_low_threshold",
        "universal",
    ]

    # === Plot: Fitness Evolution Across Selection Methods ===
    # -------------------------------------------------------
    plt.figure(figsize=(12, 6))

    # Define colors for better distinction
    colors = plt.cm.tab10(np.linspace(0, 1, len(selection_method_configs)))

    graph_selection_methods = selection_method_configs.copy()
    graph_selection_methods.remove("tournament_probabilistic_high_threshold")
    for i, config in enumerate(graph_selection_methods):
        # Average across all runs for smoother visualization
        all_fitness = []
        all_generations = []

        for run_idx in range(len(all_run_metrics[config])):
            fitness = np.array(all_run_metrics[config][run_idx]['gen_metrics']['max_fitness'])
            gen = np.array(all_run_metrics[config][run_idx]['gen_metrics']['generation'])
            all_fitness.append(fitness)
            all_generations.append(gen)

        # Calculate mean and std across runs
        max_len = min(len(f) for f in all_fitness)
        fitness_array = np.array([f[:max_len] for f in all_fitness])
        gen_array = all_generations[0][:max_len]  # Assuming generations are the same across runs

        mean_fitness = np.mean(fitness_array, axis=0)
        std_fitness = np.std(fitness_array, axis=0)

        # Plot mean line with error bands
        plt.plot(gen_array, mean_fitness, label=config.replace('_', ' ').title(),
                color=colors[i], linewidth=2)
        plt.fill_between(gen_array, mean_fitness - std_fitness, mean_fitness + std_fitness,
                        color=colors[i], alpha=0.2)

    plt.xlabel('Generation', fontsize=12, fontweight='bold')
    plt.ylabel('Maximum Fitness', fontsize=12, fontweight='bold')
    plt.title('Evolution of Maximum Fitness Across Different Selection Methods',
              fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(graph_dir, 'selection_methods_fitness_evolution.png'),
                dpi=300, bbox_inches='tight')
    # === End Plot: Fitness Evolution ===

    # === Plot: Dual Axis Fitness & Change Rate (Convergence) ===
    # ----------------------------------------------------------
    plt.figure(figsize=(15, 10))

    # Create subplots for each selection method
    n_cols = 3
    n_rows = (len(selection_method_configs) + n_cols - 1) // n_cols

    for i, config in enumerate(selection_method_configs):
        plt.subplot(n_rows, n_cols, i + 1)

        # Get data from first run for individual plots
        fitness = np.array(all_run_metrics[config][0]['gen_metrics']['max_fitness'])
        gen = np.array(all_run_metrics[config][0]['gen_metrics']['generation'])

        # Primary axis - Fitness
        ax1 = plt.gca()
        line1 = ax1.plot(gen, fitness, 'b-', linewidth=2, label='Max Fitness')
        ax1.set_xlabel('Generation', fontsize=10)
        ax1.set_ylabel('Max Fitness', color='b', fontsize=10)
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, alpha=0.3)

        # Secondary axis - Generation progress rate (fitness improvement rate)
        ax2 = ax1.twinx()
        fitness_diff = np.diff(fitness, prepend=fitness[0])
        line2 = ax2.plot(gen, fitness_diff, 'r--', alpha=0.7, linewidth=1.5, label='Fitness Change Rate')
        ax2.set_ylabel('Fitness Change Rate', color='r', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='r')

        # Title and legend
        plt.title(config.replace('_', ' ').title(), fontsize=11, fontweight='bold')

        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, 'selection_methods_dual_axis_comparison.png'),
                dpi=600, bbox_inches='tight')
    # === End Plot: Dual Axis Fitness & Change Rate ===

    # === Plot: Final Fitness vs Runtime (Dual Axis Bar Chart) ===
    # -----------------------------------------------------------
    plt.figure(figsize=(16, 10))

    # Calculate statistics for each selection method
    fitness_means = []
    fitness_stds = []
    runtime_means = []
    runtime_stds = []
    config_labels = []

    for config in selection_method_configs:
        # Get final fitness and runtime data from all runs for this config
        final_fitness_values = []
        runtimes = []

        for run_metrics in all_run_metrics[config]:
            if run_metrics:
                final_fitness_values.append(run_metrics['fitness'])
                if 'runtime_seconds' in run_metrics:
                    runtimes.append(run_metrics['runtime_seconds'])

        if final_fitness_values and runtimes:
            fitness_means.append(np.mean(final_fitness_values))
            fitness_stds.append(np.std(final_fitness_values))
            runtime_means.append(np.mean(runtimes))
            runtime_stds.append(np.std(runtimes))
            config_labels.append(config.replace('_', ' ').title())

    # Sort by final fitness (descending)
    sort_indices = np.argsort(fitness_means)[::-1]
    fitness_means = [fitness_means[i] for i in sort_indices]
    fitness_stds = [fitness_stds[i] for i in sort_indices]
    runtime_means = [runtime_means[i] for i in sort_indices]
    runtime_stds = [runtime_stds[i] for i in sort_indices]
    config_labels = [config_labels[i] for i in sort_indices]

    # Create the dual-axis plot
    x_pos = np.arange(len(config_labels))

    # Primary axis - Final Fitness (bars)
    ax1 = plt.gca()
    bars1 = ax1.bar(x_pos - 0.2, fitness_means, width=0.4, yerr=fitness_stds,
                    capsize=3, alpha=0.8, color='steelblue',
                    label='Final Fitness', edgecolor='darkblue', linewidth=1)

    ax1.set_xlabel('Selection Method', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Final Fitness', fontsize=12, fontweight='bold', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.grid(True, alpha=0.3, axis='y')

    # Secondary axis - Runtime (bars)
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x_pos + 0.2, runtime_means, width=0.4, yerr=runtime_stds,
                    capsize=3, alpha=0.8, color='coral',
                    label='Runtime (s)', edgecolor='darkred', linewidth=1)

    ax2.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold', color='coral')
    ax2.tick_params(axis='y', labelcolor='coral')

    # Customize the plot
    plt.title('Selection Methods: Final Fitness vs Runtime Comparison',
              fontsize=16, fontweight='bold', pad=20)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(config_labels, rotation=45, ha='right')

    # Add value labels on bars
    for i, (fitness_val, fitness_std) in enumerate(zip(fitness_means, fitness_stds)):
        ax1.text(i - 0.2, fitness_val + fitness_std + max(fitness_means) * 0.01,
                f'{fitness_val:.3f}', ha='center', va='bottom', fontweight='bold',
                color='steelblue', fontsize=9)

    for i, (runtime_val, runtime_std) in enumerate(zip(runtime_means, runtime_stds)):
        ax2.text(i + 0.2, runtime_val + runtime_std + max(runtime_means) * 0.01,
                f'{runtime_val:.0f}s', ha='center', va='bottom', fontweight='bold',
                color='coral', fontsize=9)

    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, 'selection_methods_fitness_vs_runtime.png'),
                dpi=300, bbox_inches='tight')
    # === End Plot: Final Fitness vs Runtime ===

    # === Plot: Fitness vs Runtime Trade-off (Scatter) ===
    # ---------------------------------------------------
    plt.figure(figsize=(14, 8))

    # Remove 'tournament probabilistic high threshold' from tradeoff graph data
    exclude_label = 'Tournament Probabilistic High Threshold'
    filtered_indices = [i for i, label in enumerate(config_labels) if label != exclude_label]
    filtered_fitness_means = [fitness_means[i] for i in filtered_indices]
    filtered_fitness_stds = [fitness_stds[i] for i in filtered_indices]
    filtered_runtime_means = [runtime_means[i] for i in filtered_indices]
    filtered_runtime_stds = [runtime_stds[i] for i in filtered_indices]
    filtered_config_labels = [config_labels[i] for i in filtered_indices]

    # Create scatter plot
    colors = plt.cm.Set3(np.linspace(0, 1, len(filtered_config_labels)))
    scatter = plt.scatter(filtered_runtime_means, filtered_fitness_means,
                         s=200, alpha=0.7, c=colors,
                         edgecolors='black', linewidth=2)

    # Add error bars
    plt.errorbar(filtered_runtime_means, filtered_fitness_means,
                xerr=filtered_runtime_stds, yerr=filtered_fitness_stds,
                fmt='none', color='black', alpha=0.5, capsize=3)

    # Add labels for each point
    for i, label in enumerate(filtered_config_labels):
        plt.annotate(label, (filtered_runtime_means[i], filtered_fitness_means[i]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.xlabel('Runtime (seconds)', fontsize=12, fontweight='bold')
    plt.ylabel('Final Fitness', fontsize=12, fontweight='bold')
    plt.title('Selection Methods: Fitness vs Runtime Trade-off Analysis',
              fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)

    # Add quadrant lines to help identify optimal methods
    median_runtime = np.median(filtered_runtime_means)
    median_fitness = np.median(filtered_fitness_means)
    plt.axvline(x=median_runtime, color='red', linestyle='--', alpha=0.5, label='Median Runtime')
    plt.axhline(y=median_fitness, color='red', linestyle='--', alpha=0.5, label='Median Fitness')

    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, 'selection_methods_tradeoff_analysis.png'),
                dpi=300, bbox_inches='tight')
    # === End Plot: Fitness vs Runtime Trade-off ===


    new_gen_methods = [
        "traditional",
        "young_bias",
    ]

    # === Plot: Fitness Evolution Across New Generation Methods ===
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 7))
    colors = plt.cm.Set1(np.linspace(0, 1, len(new_gen_methods)))
    for i, config in enumerate(new_gen_methods):
        all_fitness = []
        all_generations = []
        for run_metrics in all_run_metrics[config]:
            if run_metrics:
                fitness = np.array(run_metrics['gen_metrics']['max_fitness'])
                gen = np.array(run_metrics['gen_metrics']['generation'])
                all_fitness.append(fitness)
                all_generations.append(gen)
        if all_fitness:
            max_len = min(len(f) for f in all_fitness)
            fitness_array = np.array([f[:max_len] for f in all_fitness])
            gen_array = all_generations[0][:max_len]
            mean_fitness = np.mean(fitness_array, axis=0)
            std_fitness = np.std(fitness_array, axis=0)
            plt.plot(gen_array, mean_fitness, label=config.replace('_', ' ').title(), color=colors[i], linewidth=2)
            plt.fill_between(gen_array, mean_fitness - std_fitness, mean_fitness + std_fitness, color=colors[i], alpha=0.2)
    plt.xlabel('Generation', fontsize=12, fontweight='bold')
    plt.ylabel('Maximum Fitness', fontsize=12, fontweight='bold')
    plt.title('Fitness Evolution: New Generation Methods', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, 'new_generation_methods_fitness_evolution.png'), dpi=300, bbox_inches='tight')
    # === End Plot: Fitness Evolution Across New Generation Methods ===


    crossover_methods = [
        "one_point",
        "two_points",
    ]

    # === Plot: Fitness Evolution Across Crossover Methods ===
    # -------------------------------------------------------
    plt.figure(figsize=(10, 7))
    colors = plt.cm.Set2(np.linspace(0, 1, len(crossover_methods)))
    for i, config in enumerate(crossover_methods):
        all_fitness = []
        all_generations = []
        for run_metrics in all_run_metrics[config]:
            if run_metrics:
                fitness = np.array(run_metrics['gen_metrics']['max_fitness'])
                gen = np.array(run_metrics['gen_metrics']['generation'])
                all_fitness.append(fitness)
                all_generations.append(gen)
        if all_fitness:
            max_len = min(len(f) for f in all_fitness)
            fitness_array = np.array([f[:max_len] for f in all_fitness])
            gen_array = all_generations[0][:max_len]
            mean_fitness = np.mean(fitness_array, axis=0)
            std_fitness = np.std(fitness_array, axis=0)
            plt.plot(gen_array, mean_fitness, label=config.replace('_', ' ').title(), color=colors[i], linewidth=2)
            plt.fill_between(gen_array, mean_fitness - std_fitness, mean_fitness + std_fitness, color=colors[i], alpha=0.2)
    plt.xlabel('Generation', fontsize=12, fontweight='bold')
    plt.ylabel('Maximum Fitness', fontsize=12, fontweight='bold')
    plt.title('Fitness Evolution: Crossover Methods', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, 'crossover_methods_fitness_evolution.png'), dpi=300, bbox_inches='tight')
    # === End Plot: Fitness Evolution Across Crossover Methods ===


    mutation_methods = [
        "single_gene",
        "limited_multigene",
    ]

    # === Plot: Fitness Evolution Across Mutation Methods ===
    # ------------------------------------------------------
    plt.figure(figsize=(12, 7))
    colors = plt.cm.Paired(np.linspace(0, 1, len(mutation_methods)))
    for i, config in enumerate(mutation_methods):
        all_fitness = []
        all_generations = []
        for run_metrics in all_run_metrics[config]:
            if run_metrics:
                fitness = np.array(run_metrics['gen_metrics']['max_fitness'])
                gen = np.array(run_metrics['gen_metrics']['generation'])
                all_fitness.append(fitness)
                all_generations.append(gen)
        if all_fitness:
            max_len = min(len(f) for f in all_fitness)
            fitness_array = np.array([f[:max_len] for f in all_fitness])
            gen_array = all_generations[0][:max_len]
            mean_fitness = np.mean(fitness_array, axis=0)
            std_fitness = np.std(fitness_array, axis=0)
            plt.plot(gen_array, mean_fitness, label=config.replace('_', ' ').title(), color=colors[i], linewidth=2)
            plt.fill_between(gen_array, mean_fitness - std_fitness, mean_fitness + std_fitness, color=colors[i], alpha=0.2)
    plt.xlabel('Generation', fontsize=12, fontweight='bold')
    plt.ylabel('Maximum Fitness', fontsize=12, fontweight='bold')
    plt.title('Fitness Evolution: Mutation Methods', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, 'mutation_methods_fitness_evolution.png'), dpi=300, bbox_inches='tight')
    # === End Plot: Fitness Evolution Across Mutation Methods ===


    print(f"All graphs saved to '{graph_dir}' directory!")
