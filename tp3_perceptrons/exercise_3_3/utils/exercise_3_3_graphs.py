import json
import os
import matplotlib.pyplot as plt

def plot_accuracy_vs_noise():
    """Plot accuracy vs noise scale with standard deviation error bars"""

    # Directory containing the results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

    # List all result files for exercise_3_3
    result_files = [f for f in os.listdir(results_dir) if f.startswith("exercise_3_3_noise") and f.endswith(".json")]

    # Sort files to ensure consistent ordering
    result_files.sort()

    noise_scales = []
    mean_accuracies = []
    std_accuracies = []

    # Load data from each file
    for filename in result_files:
        filepath = os.path.join(results_dir, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)

        noise_scale = data['noise_scale']
        aggregate_stats = data['aggregate_statistics']
        mean_accuracy = aggregate_stats['mean_accuracy']
        std_accuracy = aggregate_stats['std_accuracy']

        noise_scales.append(noise_scale)
        mean_accuracies.append(mean_accuracy * 100)  # Convert to percentage
        std_accuracies.append(std_accuracy * 100)    # Convert to percentage

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot with error bars
    ax.errorbar(noise_scales, mean_accuracies, yerr=std_accuracies,
                marker='o', markersize=8, capsize=5, capthick=2,
                linewidth=2, elinewidth=2, color='#2E86AB',
                ecolor='#A23B72', label='Accuracy promedio ± desvío std')

    # Formatting
    ax.set_xlabel('Ruido', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy vs Ruido\n(100 runs por nivel de ruido)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)

    # Set y-axis limits for better visualization
    ax.set_ylim([0, 105])

    # Add horizontal line at 100% for reference
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, linewidth=1, label='Perfect Accuracy')

    plt.tight_layout()

    # Save the plot
    graphs_dir = os.path.join(os.path.dirname(results_dir), "graphs")
    os.makedirs(graphs_dir, exist_ok=True)
    output_path = os.path.join(graphs_dir, "accuracy_vs_noise.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    # Show the plot
    plt.show()

    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 60)
    print(f"{'Noise Scale':<15} {'Mean Accuracy':<20} {'Std Dev':<15}")
    print("-" * 60)
    for noise, mean, std in zip(noise_scales, mean_accuracies, std_accuracies):
        print(f"{noise:<15.1f} {mean:<20.2f}% {std:<15.2f}%")

if __name__ == "__main__":
    plot_accuracy_vs_noise()
