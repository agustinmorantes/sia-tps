import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

class HopfieldVisualizer:
    def __init__(self, metrics_dir="metrics"):
        self.metrics_dir = Path(metrics_dir)
        self.output_dir = Path("output_graphs")
        self.output_dir.mkdir(exist_ok=True)

    def load_metrics(self, filename):
        """Load metrics from JSON file"""
        filepath = self.metrics_dir / filename
        with open(filepath, 'r') as f:
            return json.load(f)

    def reshape_to_grid(self, pattern):
        """Reshape a 25-element pattern to 5x5 grid"""
        return np.array(pattern).reshape(5, 5)

    def plot_pattern(self, ax, pattern, title):
        """Plot a single pattern as a 5x5 grid"""
        grid = self.reshape_to_grid(pattern)
        ax.imshow(grid, cmap='binary', vmin=-1, vmax=1, interpolation='nearest')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        # Add grid lines
        for i in range(6):
            ax.axhline(i-0.5, color='gray', linewidth=0.5)
            ax.axvline(i-0.5, color='gray', linewidth=0.5)

    def plot_crosstalk_heatmap(self, ax, crosstalk, title):
        """Plot crosstalk as a 5x5 heatmap"""
        grid = self.reshape_to_grid(crosstalk)
        sns.heatmap(grid, ax=ax, cmap='RdBu_r', center=0,
                   annot=True, fmt='.2f', cbar=True,
                   square=True, linewidths=0.5,
                   vmin=-1, vmax=1)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')

    def plot_energy_convergence(self, ax, energies, title):
        """Plot energy convergence over iterations"""
        iterations = range(len(energies))
        ax.plot(iterations, energies, 'o-', linewidth=2, markersize=8, color='#2E86AB')
        ax.set_xlabel('Iteration', fontsize=10)
        ax.set_ylabel('Energy', fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(iterations)

    def plot_state_evolution(self, states, pattern_name, noise_level):
        """Plot evolution of states through iterations"""
        n_states = len(states)
        cols = min(5, n_states)
        rows = (n_states + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
        fig.suptitle(f'State Evolution: {pattern_name.upper()} ({noise_level})',
                    fontsize=14, fontweight='bold')

        if rows == 1:
            axes = axes.reshape(1, -1)

        for idx, state in enumerate(states):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[0, col]

            self.plot_pattern(ax, state, f'Iteration {idx}')

        # Hide unused subplots
        for idx in range(n_states, rows * cols):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[0, col]
            ax.axis('off')

        plt.tight_layout()
        output_file = self.output_dir / f'state_evolution_{pattern_name}_{noise_level}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

    def plot_comprehensive_metrics(self, pattern_name, noise_level):
        """Create comprehensive visualization for a single run"""
        metrics = self.load_metrics(f"{pattern_name}_{noise_level}.json")

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # Title
        fig.suptitle(f'Hopfield Network Analysis: Pattern {pattern_name.upper()} ({noise_level.replace("_", " ").title()})',
                    fontsize=16, fontweight='bold')

        # Row 1: Query Pattern, Final State, Crosstalk Heatmap, Energy
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_pattern(ax1, metrics['query_pattern'], 'Query Pattern (Input)')

        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_pattern(ax2, metrics['states'][-1], 'Final State (Output)')

        ax3 = fig.add_subplot(gs[0, 2])
        self.plot_crosstalk_heatmap(ax3, metrics['crosstalk'], 'Crosstalk')

        ax4 = fig.add_subplot(gs[0, 3])
        self.plot_energy_convergence(ax4, metrics['energies'], 'Energy Convergence')

        # Row 2: Key states evolution (first, intermediate, and final)
        total_states = len(metrics['states'])

        # Select up to 4 representative states
        if total_states <= 4:
            state_indices = list(range(total_states))
        else:
            # Always show first and last, plus evenly spaced intermediate states
            state_indices = [0]  # First state
            # Add 2 intermediate states
            mid1 = total_states // 3
            mid2 = (2 * total_states) // 3
            state_indices.extend([mid1, mid2])
            state_indices.append(total_states - 1)  # Final state

        for plot_idx, state_idx in enumerate(state_indices):
            ax = fig.add_subplot(gs[1, plot_idx])
            self.plot_pattern(ax, metrics['states'][state_idx], f'State at Iteration {state_idx}')

        # Row 3: Additional metrics and analysis
        ax_info = fig.add_subplot(gs[2, :2])
        ax_info.axis('off')

        # Calculate additional metrics
        final_energy = metrics['energies'][-1]
        energy_change = metrics['energies'][-1] - metrics['energies'][0]
        converged_at = len(metrics['energies']) - 1

        # Calculate pattern match (similarity between query and final)
        query = np.array(metrics['query_pattern'])
        final = np.array(metrics['states'][-1])
        pattern_match = np.sum(query == final) / len(query) * 100

        info_text = f"""
        Metrics Summary:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        • Total Iterations: {metrics['num_iterations']}
        • Converged at: Iteration {converged_at}
        • Initial Energy: {metrics['energies'][0]:.4f}
        • Final Energy: {final_energy:.4f}
        • Energy Change: {energy_change:.4f}
        • Pattern Match: {pattern_match:.1f}%
        • Mean Crosstalk: {np.mean(metrics['crosstalk']):.4f}
        • Std Crosstalk: {np.std(metrics['crosstalk']):.4f}
        """

        ax_info.text(0.1, 0.5, info_text, fontsize=11, family='monospace',
                    verticalalignment='center')

        # Plot crosstalk distribution
        ax_dist = fig.add_subplot(gs[2, 2:])
        crosstalk_values = metrics['crosstalk']
        ax_dist.hist(crosstalk_values, bins=15, color='#A23B72', alpha=0.7, edgecolor='black')
        ax_dist.set_xlabel('Crosstalk Value', fontsize=10)
        ax_dist.set_ylabel('Frequency', fontsize=10)
        ax_dist.set_title('Crosstalk Distribution', fontsize=10, fontweight='bold')
        ax_dist.grid(True, alpha=0.3)
        ax_dist.axvline(np.mean(crosstalk_values), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(crosstalk_values):.2f}')
        ax_dist.legend()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            plt.tight_layout(rect=(0, 0, 1, 0.98))
        output_file = self.output_dir / f'comprehensive_{pattern_name}_{noise_level}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

    def plot_noise_comparison(self, pattern_name):
        """Compare low noise vs high noise for the same pattern"""
        try:
            low_metrics = self.load_metrics(f"{pattern_name}_low_noise.json")
            high_metrics = self.load_metrics(f"{pattern_name}_high_noise.json")
        except FileNotFoundError:
            print(f"Metrics files not found for pattern {pattern_name}")
            return

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'Noise Comparison for Pattern {pattern_name.upper()}',
                    fontsize=16, fontweight='bold')

        # Low noise row
        self.plot_pattern(axes[0, 0], low_metrics['query_pattern'], 'Low Noise: Query')
        self.plot_pattern(axes[0, 1], low_metrics['states'][-1], 'Low Noise: Final State')
        self.plot_crosstalk_heatmap(axes[0, 2], low_metrics['crosstalk'], 'Low Noise: Crosstalk')
        self.plot_energy_convergence(axes[0, 3], low_metrics['energies'], 'Low Noise: Energy')

        # High noise row
        self.plot_pattern(axes[1, 0], high_metrics['query_pattern'], 'High Noise: Query')
        self.plot_pattern(axes[1, 1], high_metrics['states'][-1], 'High Noise: Final State')
        self.plot_crosstalk_heatmap(axes[1, 2], high_metrics['crosstalk'], 'High Noise: Crosstalk')
        self.plot_energy_convergence(axes[1, 3], high_metrics['energies'], 'High Noise: Energy')

        plt.tight_layout()
        output_file = self.output_dir / f'noise_comparison_{pattern_name}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

    def plot_all_patterns_summary(self):
        """Create a summary comparing all patterns"""
        patterns = ['a', 'j', 't', 'x']
        noise_levels = ['low_noise', 'high_noise']

        fig, axes = plt.subplots(len(patterns), 2, figsize=(10, 12))
        fig.suptitle('Summary: Final Energy by Pattern and Noise Level',
                    fontsize=16, fontweight='bold')

        for idx, pattern in enumerate(patterns):
            for noise_idx, noise in enumerate(noise_levels):
                try:
                    metrics = self.load_metrics(f"{pattern}_{noise}.json")
                    ax = axes[idx, noise_idx]

                    iterations = range(len(metrics['energies']))
                    ax.plot(iterations, metrics['energies'], 'o-', linewidth=2, markersize=8)
                    ax.set_title(f'{pattern.upper()} - {noise.replace("_", " ").title()}',
                               fontsize=10, fontweight='bold')
                    ax.set_xlabel('Iteration', fontsize=9)
                    ax.set_ylabel('Energy', fontsize=9)
                    ax.grid(True, alpha=0.3)
                    ax.set_xticks(iterations)

                    # Add final energy value as text
                    final_energy = metrics['energies'][-1]
                    ax.text(0.95, 0.95, f'Final: {final_energy:.2f}',
                           transform=ax.transAxes,
                           verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                           fontsize=9)
                except FileNotFoundError:
                    axes[idx, noise_idx].text(0.5, 0.5, 'No data',
                                             ha='center', va='center')
                    axes[idx, noise_idx].set_title(f'{pattern.upper()} - {noise}')

        plt.tight_layout()
        output_file = self.output_dir / 'all_patterns_energy_summary.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

    def plot_crosstalk_comparison(self):
        """Compare crosstalk statistics across all runs"""
        patterns = ['a', 'j', 't', 'x']
        noise_levels = ['low_noise', 'high_noise']

        data = []
        labels = []

        for pattern in patterns:
            for noise in noise_levels:
                try:
                    metrics = self.load_metrics(f"{pattern}_{noise}.json")
                    data.append(metrics['crosstalk'])
                    labels.append(f"{pattern.upper()}\n{noise.replace('_', ' ')}")
                except FileNotFoundError:
                    continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Crosstalk Analysis Across All Runs', fontsize=16, fontweight='bold')

        # Box plot
        bp = ax1.boxplot(data, tick_labels=labels, patch_artist=True)
        for patch, i in zip(bp['boxes'], range(len(bp['boxes']))):
            if i % 2 == 0:
                patch.set_facecolor('#A8DADC')
            else:
                patch.set_facecolor('#F1FAEE')
        ax1.set_ylabel('Crosstalk Value', fontsize=11)
        ax1.set_title('Crosstalk Distribution by Run', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Mean comparison
        means = [np.mean(d) for d in data]
        colors = ['#457B9D' if i % 2 == 0 else '#E63946' for i in range(len(means))]
        bars = ax2.bar(range(len(means)), means, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.set_ylabel('Mean Crosstalk', fontsize=11)
        ax2.set_title('Mean Crosstalk by Run', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)

        # Add value labels on bars
        for i, (bar, mean) in enumerate(zip(bars, means)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mean:.2f}',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=8)

        plt.tight_layout()
        output_file = self.output_dir / 'crosstalk_comparison.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

    def generate_all_visualizations(self):
        """Generate all visualizations for all metrics files"""
        patterns = ['a', 'j', 't', 'x']
        noise_levels = ['low_noise', 'high_noise']

        print("="*60)
        print("GENERATING HOPFIELD NETWORK VISUALIZATIONS")
        print("="*60)

        # Individual comprehensive plots for each run
        print("\n1. Generating comprehensive plots for each run...")
        for pattern in patterns:
            for noise in noise_levels:
                try:
                    print(f"   Processing {pattern}_{noise}...")
                    self.plot_comprehensive_metrics(pattern, noise)
                    self.plot_state_evolution(
                        self.load_metrics(f"{pattern}_{noise}.json")['states'],
                        pattern, noise
                    )
                except FileNotFoundError:
                    print(f"   Skipping {pattern}_{noise} (file not found)")

        # Noise comparisons
        print("\n2. Generating noise comparison plots...")
        for pattern in patterns:
            print(f"   Comparing noise levels for pattern {pattern}...")
            self.plot_noise_comparison(pattern)

        # Summary plots
        print("\n3. Generating summary plots...")
        self.plot_all_patterns_summary()
        self.plot_crosstalk_comparison()

        print("\n" + "="*60)
        print(f"ALL VISUALIZATIONS SAVED TO: {self.output_dir.absolute()}")
        print("="*60)


if __name__ == "__main__":
    visualizer = HopfieldVisualizer()
    visualizer.generate_all_visualizations()

