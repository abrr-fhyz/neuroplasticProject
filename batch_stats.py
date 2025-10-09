import numpy as np
from tensorflow.keras.utils import to_categorical
from scipy import stats
import matplotlib.pyplot as plt
from Stats import evaluate_model
from models.NNModel import NeuralNetwork
from models.NPModel import NPNeuralNetwork
from tensorflow.keras.datasets import (
    mnist, fashion_mnist, cifar10
) 

def process_and_save_results_statistical(idn, models_1_list, models_2_list, X_test, y_test, y_test_orig):
    """
    Process multiple model runs and perform statistical analysis
    
    Args:
        idn: Identifier for the experiment
        models_1_list: List of 10 Standard NN models
        models_2_list: List of 10 NP NN models
        X_test, y_test, y_test_orig: Test data
    """
    
    print(f"\nEvaluating {len(models_1_list)} model runs for IDN {idn}...")
    
    # Collect stats from all runs
    all_acc_1, all_lss_1 = [], []
    all_acc_2, all_lss_2 = [], []
    
    std_results_list = []
    np_results_list = []
    
    for i, (model_1, model_2) in enumerate(zip(models_1_list, models_2_list)):
        acc_1, lss_1 = model_1.get_stats()
        acc_2, lss_2 = model_2.get_stats()
        
        all_acc_1.append(acc_1)
        all_lss_1.append(lss_1)
        all_acc_2.append(acc_2)
        all_lss_2.append(lss_2)
        
        # Evaluate each model
        std_results = evaluate_model(model_1, X_test, y_test, y_test_orig, f"Standard NN Run {i+1}")
        np_results = evaluate_model(model_2, X_test, y_test, y_test_orig, f"NP NN Run {i+1}")
        
        std_results_list.append(std_results)
        np_results_list.append(np_results)
    
    # Perform statistical analysis
    show_statistical_comparison(all_acc_1, all_acc_2, all_lss_1, all_lss_2, idn)
    
    # Perform significance tests
    perform_significance_tests(std_results_list, np_results_list, idn)
    
    return std_results_list, np_results_list

def show_statistical_comparison(all_acc_1, all_acc_2, all_lss_1, all_lss_2, idn, 
                                label_1='Standard NN', label_2='NP NN'):
    """
    Create visualization with mean, std, and confidence intervals
    """
    # Convert to numpy arrays for easier manipulation
    all_acc_1 = np.array(all_acc_1)  # Shape: (n_runs, n_epochs)
    all_acc_2 = np.array(all_acc_2)
    all_lss_1 = np.array(all_lss_1)
    all_lss_2 = np.array(all_lss_2)
    
    # Calculate statistics across runs
    mean_acc_1 = np.mean(all_acc_1, axis=0) * 100
    std_acc_1 = np.std(all_acc_1, axis=0) * 100
    mean_acc_2 = np.mean(all_acc_2, axis=0) * 100
    std_acc_2 = np.std(all_acc_2, axis=0) * 100
    
    mean_lss_1 = np.mean(all_lss_1, axis=0)
    std_lss_1 = np.std(all_lss_1, axis=0)
    mean_lss_2 = np.mean(all_lss_2, axis=0)
    std_lss_2 = np.std(all_lss_2, axis=0)
    
    # Calculate 95% confidence intervals
    n_runs = all_acc_1.shape[0]
    ci_acc_1 = 1.96 * std_acc_1 / np.sqrt(n_runs)
    ci_acc_2 = 1.96 * std_acc_2 / np.sqrt(n_runs)
    ci_lss_1 = 1.96 * std_lss_1 / np.sqrt(n_runs)
    ci_lss_2 = 1.96 * std_lss_2 / np.sqrt(n_runs)
    
    epochs = range(1, len(mean_acc_1) + 1)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot accuracy with confidence intervals
    ax1.plot(epochs, mean_acc_1, label=f'{label_1} (mean)', color='blue', linewidth=2)
    ax1.fill_between(epochs, mean_acc_1 - ci_acc_1, mean_acc_1 + ci_acc_1, 
                     color='blue', alpha=0.2, label=f'{label_1} 95% CI')
    
    ax1.plot(epochs, mean_acc_2, label=f'{label_2} (mean)', color='green', linewidth=2)
    ax1.fill_between(epochs, mean_acc_2 - ci_acc_2, mean_acc_2 + ci_acc_2, 
                     color='green', alpha=0.2, label=f'{label_2} 95% CI')
    
    # Annotate final values
    ax1.plot(epochs[-1], mean_acc_1[-1], 'o', color='blue', markersize=8)
    ax1.plot(epochs[-1], mean_acc_2[-1], 'o', color='green', markersize=8)
    
    ax1.annotate(f'{mean_acc_1[-1]:.2f}% ± {std_acc_1[-1]:.2f}%', 
                 (epochs[-1], mean_acc_1[-1]),
                 textcoords="offset points", xytext=(-40, 15), ha='center',
                 fontsize=9, color='blue', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='blue'))
    ax1.annotate(f'{mean_acc_2[-1]:.2f}% ± {std_acc_2[-1]:.2f}%', 
                 (epochs[-1], mean_acc_2[-1]),
                 textcoords="offset points", xytext=(40, 15), ha='center',
                 fontsize=9, color='green', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='green'))
    
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.legend(loc='best', fontsize=9)
    ax1.set_title(f"Model Comparison: Accuracy (n={n_runs} runs)", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Plot loss with confidence intervals
    ax2.plot(epochs, mean_lss_1, label=f'{label_1} (mean)', color='red', linewidth=2)
    ax2.fill_between(epochs, mean_lss_1 - ci_lss_1, mean_lss_1 + ci_lss_1, 
                     color='red', alpha=0.2, label=f'{label_1} 95% CI')
    
    ax2.plot(epochs, mean_lss_2, label=f'{label_2} (mean)', color='orange', linewidth=2)
    ax2.fill_between(epochs, mean_lss_2 - ci_lss_2, mean_lss_2 + ci_lss_2, 
                     color='orange', alpha=0.2, label=f'{label_2} 95% CI')
    
    # Annotate final values
    ax2.plot(epochs[-1], mean_lss_1[-1], 'o', color='red', markersize=8)
    ax2.plot(epochs[-1], mean_lss_2[-1], 'o', color='orange', markersize=8)
    
    ax2.annotate(f'{mean_lss_1[-1]:.5f} ± {std_lss_1[-1]:.5f}', 
                 (epochs[-1], mean_lss_1[-1]),
                 textcoords="offset points", xytext=(-40, -15), ha='center',
                 fontsize=9, color='red', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='red'))
    ax2.annotate(f'{mean_lss_2[-1]:.5f} ± {std_lss_2[-1]:.5f}', 
                 (epochs[-1], mean_lss_2[-1]),
                 textcoords="offset points", xytext=(40, -15), ha='center',
                 fontsize=9, color='orange', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='orange'))
    
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.legend(loc='best', fontsize=9)
    ax2.set_title(f"Model Comparison: Loss (n={n_runs} runs)", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(f'Images/statistical_comparison_{idn}.png', dpi=200, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"Statistical Summary for IDN {idn}")
    print(f"{'='*60}")
    print(f"\nFinal Accuracy (%):")
    print(f"  {label_1}: {mean_acc_1[-1]:.2f} ± {std_acc_1[-1]:.2f}")
    print(f"  {label_2}: {mean_acc_2[-1]:.2f} ± {std_acc_2[-1]:.2f}")
    print(f"\nFinal Loss:")
    print(f"  {label_1}: {mean_lss_1[-1]:.5f} ± {std_lss_1[-1]:.5f}")
    print(f"  {label_2}: {mean_lss_2[-1]:.5f} ± {std_lss_2[-1]:.5f}")
    print(f"{'='*60}\n")

def perform_significance_tests(std_results_list, np_results_list, idn):
    """
    Perform statistical significance tests on model results
    """
    # Extract final metrics from all runs
    # Assuming each result dict has keys like 'accuracy', 'f1_score', etc.
    metrics_to_test = ['accuracy', 'precision', 'recall', 'f1', 'inference_time']
    
    print(f"\n{'='*60}")
    print(f"Statistical Significance Tests for IDN {idn}")
    print(f"{'='*60}\n")
    
    for metric in metrics_to_test:
        try:
            std_values = [r[metric] for r in std_results_list]
            np_values = [r[metric] for r in np_results_list]
            
            # Paired t-test (since models trained on same data splits)
            t_stat, p_value = stats.ttest_rel(std_values, np_values)
            
            # Effect size (Cohen's d for paired samples)
            diff = np.array(std_values) - np.array(np_values)
            cohens_d = np.mean(diff) / np.std(diff, ddof=1)
            
            mean_std = np.mean(std_values)
            mean_np = np.mean(np_values)
            std_std = np.std(std_values, ddof=1)
            std_np = np.std(np_values, ddof=1)
            
            print(f"{metric.upper()}:")
            print(f"  Standard NN: {mean_std:.4f} ± {std_std:.4f}")
            print(f"  NP NN:       {mean_np:.4f} ± {std_np:.4f}")
            print(f"  Difference:  {mean_np - mean_std:.4f}")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value:     {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
            print(f"  Cohen's d:   {cohens_d:.4f} ({'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'} effect)")
            print()
            
        except KeyError:
            print(f"  {metric}: Not found in results")
            continue
    
    print(f"{'='*60}\n")
    print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns not significant")
    print(f"{'='*60}\n")

def create_box_plots(std_results_list, np_results_list, idn):
    """
    Create box plots comparing distributions of metrics
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        try:
            std_values = [r[metric] * 100 for r in std_results_list]  # Convert to percentage
            np_values = [r[metric] * 100 for r in np_results_list]
            
            bp = axes[i].boxplot([std_values, np_values], 
                                 labels=['Standard NN', 'NP NN'],
                                 patch_artist=True,
                                 showmeans=True)
            
            # Color the boxes
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('lightgreen')
            
            axes[i].set_ylabel(f'{metric.capitalize()} (%)')
            axes[i].set_title(f'{metric.capitalize()} Distribution')
            axes[i].grid(True, linestyle='--', alpha=0.6, axis='y')
            
        except KeyError:
            axes[i].text(0.5, 0.5, f'{metric} not available', 
                        ha='center', va='center', transform=axes[i].transAxes)
    
    plt.suptitle(f'Model Performance Distributions (IDN {idn})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'Images/boxplot_comparison_{idn}.png', dpi=200, bbox_inches='tight')
    plt.show()
    plt.close()

def get_test_data():
    (_, _), (X_test, y_test) = fashion_mnist.load_data()
    X_test = X_test.reshape(-1, 784) / 255.0
    y_test_orig = y_test.flatten() if y_test.ndim > 1 else y_test
    y_test = to_categorical(y_test, 10)
    
    return X_test, y_test, y_test_orig

def get_cifar_data():
    (_, _), (X_test, y_test) = cifar10.load_data()
    X_test = X_test.reshape(-1, 3072) / 255.0
    y_test_orig = y_test.flatten() if y_test.ndim > 1 else y_test
    y_test = to_categorical(y_test, 10)
    
    return X_test, y_test, y_test_orig

def main():
    arch = [ 
        [784, 256, 128, 10],
        [3072, 512, 256, 128, 10]
    ]
    models_1_list = []
    models_2_list = []
    experiment_name = "CIFAR10"
    k = 1
    for i in range(20, 25):
        nn = NeuralNetwork(arch[k])
        nn.load_model(i)
        models_1_list.append(nn)
        np = NPNeuralNetwork(arch[k])
        np.load_model(i)
        models_2_list.append(np)

    X_test, y_test, y_test_orig = get_cifar_data()

    std_results, np_results = process_and_save_results_statistical(
        idn=experiment_name,
        models_1_list=models_1_list,
        models_2_list=models_2_list,
        X_test = X_test,
        y_test = y_test,
        y_test_orig = y_test_orig
    )

    # Optional: Create box plots
    create_box_plots(std_results, np_results, experiment_name)

if __name__ == "__main__":
    main()