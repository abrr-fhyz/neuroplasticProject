import numpy as numpy_lib
import matplotlib.pyplot as plt
from scipy import stats
from models.NNModel import NeuralNetwork
from models.NPModel import NPNeuralNetwork

def compute_confidence_interval(data, confidence=0.95):
    data_array = numpy_lib.array(data)
    mean = numpy_lib.mean(data_array, axis=0)
    sem = stats.sem(data_array, axis=0)
    ci = sem * stats.t.ppf((1 + confidence) / 2, len(data_array) - 1)
    return mean, mean - ci, mean + ci

def show_comparison_stats_with_ci(all_acc_1, all_acc_2, all_acc_3, all_acc_4, all_acc_5,
                                   all_lss_1, all_lss_2, all_lss_3, all_lss_4, all_lss_5,
                                   idn, dataset, confidence=0.95,
                                   label_1='Pruning/genesis', label_2='Adaptive LR', 
                                   label_3='Hebbian', label_4='Full NPNN', label_5='Standard NN'):
    """
    Create publication-quality comparison of 5 models with clean presentation
    """
    
    # Define color scheme for 5 models
    colors_5 = {
        'model1': '#FF0000',  # Red
        'model2': '#00FF00',  # Green
        'model3': "#FF6600",  # Orange
        'model4': '#000000',  # Black
        'model5': '#0000FF',  # Blue (Standard NN)
    }
    
    def compute_confidence_interval(data, confidence):
        """Helper to compute CI"""
        data_array = numpy_lib.array(data)
        mean = numpy_lib.mean(data_array, axis=0)
        std = numpy_lib.std(data_array, axis=0)
        n = data_array.shape[0]
        
        # Calculate CI
        from scipy import stats
        ci = stats.t.ppf((1 + confidence) / 2, n - 1) * std / numpy_lib.sqrt(n)
        
        lower = mean - ci
        upper = mean + ci
        
        return mean, lower, upper
    
    # Compute statistics for each model
    acc_1_mean, acc_1_lower, acc_1_upper = compute_confidence_interval(all_acc_1, confidence)
    acc_2_mean, acc_2_lower, acc_2_upper = compute_confidence_interval(all_acc_2, confidence)
    acc_3_mean, acc_3_lower, acc_3_upper = compute_confidence_interval(all_acc_3, confidence)
    acc_4_mean, acc_4_lower, acc_4_upper = compute_confidence_interval(all_acc_4, confidence)
    acc_5_mean, acc_5_lower, acc_5_upper = compute_confidence_interval(all_acc_5, confidence)
    
    lss_1_mean, lss_1_lower, lss_1_upper = compute_confidence_interval(all_lss_1, confidence)
    lss_2_mean, lss_2_lower, lss_2_upper = compute_confidence_interval(all_lss_2, confidence)
    lss_3_mean, lss_3_lower, lss_3_upper = compute_confidence_interval(all_lss_3, confidence)
    lss_4_mean, lss_4_lower, lss_4_upper = compute_confidence_interval(all_lss_4, confidence)
    lss_5_mean, lss_5_lower, lss_5_upper = compute_confidence_interval(all_lss_5, confidence)
    
    # Convert to percentages
    acc_1_mean_percent = acc_1_mean * 100
    acc_1_lower_percent = acc_1_lower * 100
    acc_1_upper_percent = acc_1_upper * 100
    
    acc_2_mean_percent = acc_2_mean * 100
    acc_2_lower_percent = acc_2_lower * 100
    acc_2_upper_percent = acc_2_upper * 100
    
    acc_3_mean_percent = acc_3_mean * 100
    acc_3_lower_percent = acc_3_lower * 100
    acc_3_upper_percent = acc_3_upper * 100
    
    acc_4_mean_percent = acc_4_mean * 100
    acc_4_lower_percent = acc_4_lower * 100
    acc_4_upper_percent = acc_4_upper * 100
    
    acc_5_mean_percent = acc_5_mean * 100
    acc_5_lower_percent = acc_5_lower * 100
    acc_5_upper_percent = acc_5_upper * 100
    
    epochs = range(1, len(acc_1_mean) + 1)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11), sharex=True)
    
    # Plot accuracy with confidence intervals
    # Ablation components: dotted lines
    ax1.plot(epochs, acc_1_mean_percent, label=f'{label_1}', 
             color=colors_5['model1'], linewidth=2.5, linestyle='--', zorder=3)
    ax1.fill_between(epochs, acc_1_lower_percent, acc_1_upper_percent, 
                      color=colors_5['model1'], alpha=0.2, zorder=2)
    
    ax1.plot(epochs, acc_2_mean_percent, label=f'{label_2}', 
             color=colors_5['model2'], linewidth=2.5, linestyle='--', zorder=3)
    ax1.fill_between(epochs, acc_2_lower_percent, acc_2_upper_percent, 
                      color=colors_5['model2'], alpha=0.2, zorder=2)
    
    ax1.plot(epochs, acc_3_mean_percent, label=f'{label_3}', 
             color=colors_5['model3'], linewidth=2.5, linestyle='--', zorder=3)
    ax1.fill_between(epochs, acc_3_lower_percent, acc_3_upper_percent, 
                      color=colors_5['model3'], alpha=0.2, zorder=2)
    
    # Full NPNN and Baseline: solid lines
    ax1.plot(epochs, acc_4_mean_percent, label=f'{label_4}', 
             color=colors_5['model4'], linewidth=2.5, linestyle='-', zorder=3)
    ax1.fill_between(epochs, acc_4_lower_percent, acc_4_upper_percent, 
                      color=colors_5['model4'], alpha=0.2, zorder=2)
    
    ax1.plot(epochs, acc_5_mean_percent, label=f'{label_5}', 
             color=colors_5['model5'], linewidth=2.5, linestyle='-', zorder=3)
    ax1.fill_between(epochs, acc_5_lower_percent, acc_5_upper_percent, 
                      color=colors_5['model5'], alpha=0.2, zorder=2)
    
    # Add final value markers (no text annotations to avoid clutter)
    final_epoch = epochs[-1]
    ax1.plot(final_epoch, acc_1_mean_percent[-1], 'o', color=colors_5['model1'], 
             markersize=10, markeredgewidth=2, markeredgecolor='white', zorder=4)
    ax1.plot(final_epoch, acc_2_mean_percent[-1], 'o', color=colors_5['model2'], 
             markersize=10, markeredgewidth=2, markeredgecolor='white', zorder=4)
    ax1.plot(final_epoch, acc_3_mean_percent[-1], 'o', color=colors_5['model3'], 
             markersize=10, markeredgewidth=2, markeredgecolor='white', zorder=4)
    ax1.plot(final_epoch, acc_4_mean_percent[-1], 'o', color=colors_5['model4'], 
             markersize=10, markeredgewidth=2, markeredgecolor='white', zorder=4)
    ax1.plot(final_epoch, acc_5_mean_percent[-1], 'o', color=colors_5['model5'], 
             markersize=10, markeredgewidth=2, markeredgecolor='white', zorder=4)
    
    ax1.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    
    # Enhanced legend with final values
    legend_labels = [
        f'{label_1} (Final: {acc_1_mean_percent[-1]:.2f}%)',
        f'{label_2} (Final: {acc_2_mean_percent[-1]:.2f}%)',
        f'{label_3} (Final: {acc_3_mean_percent[-1]:.2f}%)',
        f'{label_4} (Final: {acc_4_mean_percent[-1]:.2f}%)',
        f'{label_5} (Final: {acc_5_mean_percent[-1]:.2f}%)'
    ]
    
    # Get handles and update labels
    handles, _ = ax1.get_legend_handles_labels()
    ax1.legend(handles, legend_labels, loc='lower right', fontsize=11, 
               frameon=True, shadow=True, framealpha=0.95)
    
    ax1.set_title(f"Ablation Study: Accuracy Comparison (n={len(all_acc_1)} runs, {int(confidence*100)}% CI)", 
                  fontsize=15, fontweight='bold', pad=15)
    ax1.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot loss with confidence intervals
    # Ablation components: dotted lines
    ax2.plot(epochs, lss_1_mean, label=f'{label_1}', 
             color=colors_5['model1'], linewidth=2.5, linestyle='--', zorder=3)
    ax2.fill_between(epochs, lss_1_lower, lss_1_upper, 
                     color=colors_5['model1'], alpha=0.2, zorder=2)
    
    ax2.plot(epochs, lss_2_mean, label=f'{label_2}', 
             color=colors_5['model2'], linewidth=2.5, linestyle='--', zorder=3)
    ax2.fill_between(epochs, lss_2_lower, lss_2_upper, 
                     color=colors_5['model2'], alpha=0.2, zorder=2)
    
    ax2.plot(epochs, lss_3_mean, label=f'{label_3}', 
             color=colors_5['model3'], linewidth=2.5, linestyle='--', zorder=3)
    ax2.fill_between(epochs, lss_3_lower, lss_3_upper, 
                     color=colors_5['model3'], alpha=0.2, zorder=2)
    
    # Full NPNN and Baseline: solid lines
    ax2.plot(epochs, lss_4_mean, label=f'{label_4}', 
             color=colors_5['model4'], linewidth=2.5, linestyle='-', zorder=3)
    ax2.fill_between(epochs, lss_4_lower, lss_4_upper, 
                     color=colors_5['model4'], alpha=0.2, zorder=2)
    
    ax2.plot(epochs, lss_5_mean, label=f'{label_5}', 
             color=colors_5['model5'], linewidth=2.5, linestyle='-', zorder=3)
    ax2.fill_between(epochs, lss_5_lower, lss_5_upper, 
                     color=colors_5['model5'], alpha=0.2, zorder=2)
    
    # Add final value markers
    ax2.plot(final_epoch, lss_1_mean[-1], 'o', color=colors_5['model1'], 
             markersize=10, markeredgewidth=2, markeredgecolor='white', zorder=4)
    ax2.plot(final_epoch, lss_2_mean[-1], 'o', color=colors_5['model2'], 
             markersize=10, markeredgewidth=2, markeredgecolor='white', zorder=4)
    ax2.plot(final_epoch, lss_3_mean[-1], 'o', color=colors_5['model3'], 
             markersize=10, markeredgewidth=2, markeredgecolor='white', zorder=4)
    ax2.plot(final_epoch, lss_4_mean[-1], 'o', color=colors_5['model4'], 
             markersize=10, markeredgewidth=2, markeredgecolor='white', zorder=4)
    ax2.plot(final_epoch, lss_5_mean[-1], 'o', color=colors_5['model5'], 
             markersize=10, markeredgewidth=2, markeredgecolor='white', zorder=4)
    
    ax2.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=14, fontweight='bold')
    
    # Enhanced legend with final values for loss
    legend_labels_loss = [
        f'{label_1} (Final: {lss_1_mean[-1]:.4f})',
        f'{label_2} (Final: {lss_2_mean[-1]:.4f})',
        f'{label_3} (Final: {lss_3_mean[-1]:.4f})',
        f'{label_4} (Final: {lss_4_mean[-1]:.4f})',
        f'{label_5} (Final: {lss_5_mean[-1]:.4f})'
    ]
    
    handles, _ = ax2.get_legend_handles_labels()
    ax2.legend(handles, legend_labels_loss, loc='upper right', fontsize=11, 
               frameon=True, shadow=True, framealpha=0.95)
    
    ax2.set_title(f"Ablation Study: Loss Comparison ({dataset} Dataset)", 
                  fontsize=15, fontweight='bold', pad=15)
    ax2.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'Images/comparison_stats_ablation_{dataset}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Print summary statistics
    print(f"\n{'='*70}")
    print(f"{idn} - Ablation Study Summary (n={len(all_acc_1)} runs, {dataset})")
    print(f"{'='*70}")
    print(f"\nFinal Accuracy (%):")
    print(f"  {label_1}: {acc_1_mean_percent[-1]:.2f} ± {(acc_1_upper_percent[-1]-acc_1_mean_percent[-1]):.2f}")
    print(f"  {label_2}: {acc_2_mean_percent[-1]:.2f} ± {(acc_2_upper_percent[-1]-acc_2_mean_percent[-1]):.2f}")
    print(f"  {label_3}: {acc_3_mean_percent[-1]:.2f} ± {(acc_3_upper_percent[-1]-acc_3_mean_percent[-1]):.2f}")
    print(f"  {label_4}: {acc_4_mean_percent[-1]:.2f} ± {(acc_4_upper_percent[-1]-acc_4_mean_percent[-1]):.2f}")
    print(f"  {label_5}: {acc_5_mean_percent[-1]:.2f} ± {(acc_5_upper_percent[-1]-acc_5_mean_percent[-1]):.2f}")
    print(f"\nFinal Loss:")
    print(f"  {label_1}: {lss_1_mean[-1]:.5f} ± {(lss_1_upper[-1]-lss_1_mean[-1]):.5f}")
    print(f"  {label_2}: {lss_2_mean[-1]:.5f} ± {(lss_2_upper[-1]-lss_2_mean[-1]):.5f}")
    print(f"  {label_3}: {lss_3_mean[-1]:.5f} ± {(lss_3_upper[-1]-lss_3_mean[-1]):.5f}")
    print(f"  {label_4}: {lss_4_mean[-1]:.5f} ± {(lss_4_upper[-1]-lss_4_mean[-1]):.5f}")
    print(f"  {label_5}: {lss_5_mean[-1]:.5f} ± {(lss_5_upper[-1]-lss_5_mean[-1]):.5f}")
    
    # Show relative improvements
    print(f"\nImprovement of {label_4} over {label_5}:")
    acc_improvement = acc_4_mean_percent[-1] - acc_5_mean_percent[-1]
    loss_improvement = lss_5_mean[-1] - lss_4_mean[-1]
    print(f"  Accuracy: {acc_improvement:+.2f}%")
    print(f"  Loss: {loss_improvement:+.5f}")
    print(f"{'='*70}\n")

def batch_analysis(arch_idx, dataset, num_runs=5, confidence=0.95):
    all_acc_1, all_acc_2, all_acc_3, all_acc_4, all_acc_5 = [], [], [], [], []
    all_lss_1, all_lss_2, all_lss_3, all_lss_4, all_lss_5 = [], [], [], [], []
    
    for run_idx in range(num_runs):
        print(f"  Loading run {run_idx}...", end=' ')
        
        # Initialize models
        model_1 = NPNeuralNetwork(arch[arch_idx])
        model_2 = NPNeuralNetwork(arch[arch_idx])
        model_3 = NPNeuralNetwork(arch[arch_idx])
        model_4 = NPNeuralNetwork(arch[arch_idx])
        model_5 = NeuralNetwork(arch[arch_idx])
        
        # Load models
        model_1.load_model(1000 + arch_idx * 100 + run_idx)
        model_2.load_model(2000 + arch_idx * 100 + run_idx)
        model_3.load_model(3000 + arch_idx * 100 + run_idx)
        model_4.load_model(train_idx * 10 + run_idx)
        model_5.load_model(train_idx * 10 + run_idx)
        
        # Get stats
        acc_1, lss_1 = model_1.get_stats()
        acc_2, lss_2 = model_2.get_stats()
        acc_3, lss_3 = model_3.get_stats()
        acc_4, lss_4 = model_4.get_stats()
        acc_5, lss_5 = model_5.get_stats()
        
        # Append to lists
        all_acc_1.append(acc_1)
        all_acc_2.append(acc_2)
        all_acc_3.append(acc_3)
        all_acc_4.append(acc_4)
        all_acc_5.append(acc_5)
        
        all_lss_1.append(lss_1)
        all_lss_2.append(lss_2)
        all_lss_3.append(lss_3)
        all_lss_4.append(lss_4)
        all_lss_5.append(lss_5)
        
        print("✓")
    
    print(f"All runs loaded. Generating visualization...\n")
    
    # Generate comparison plot with confidence intervals
    show_comparison_stats_with_ci(
        all_acc_1, all_acc_2, all_acc_3, all_acc_4, all_acc_5,
        all_lss_1, all_lss_2, all_lss_3, all_lss_4, all_lss_5,
        arch_idx, dataset, confidence
    )

# Architecture definitions
arch = [
    [784, 256, 128, 10],       # Architecture 0: MNIST-like
    [3072, 512, 256, 128, 10],  # Architecture 1: CIFAR-like
    [3072, 512, 256, 128, 100]
]

ds = [
    "MNIST",
    "CIFAR-10",
    "CIFAR-100"
]

idx = 1
train_idx = 2

def main():
    batch_analysis(idx, ds[idx], num_runs=5, confidence=0.95)

if __name__ == "__main__":
    main()