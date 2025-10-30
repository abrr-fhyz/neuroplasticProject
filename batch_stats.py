import numpy as numpy_lib
import pandas as pd
from tensorflow.keras.utils import to_categorical
from scipy import stats
import matplotlib.pyplot as plt
from Stats import evaluate_model, to_numpy
from models.NNModel import NeuralNetwork
from models.NPModel import NPNeuralNetwork
from sklearn.decomposition import PCA
import seaborn as sns
from tensorflow.keras.datasets import (
    mnist, fashion_mnist, cifar10
)

from main import load_data_CIFAR10_stan, load_data_CIFAR100_stan, load_data_MNIST, load_data_fashion, load_data_CIFAR10

def plot_batch_confusion_matrix(all_std_results, all_np_results, dataset_name, figsize=(20, 8)):
    """Plot averaged confusion matrices across multiple runs"""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    std_cms = numpy_lib.array([r['confusion_matrix'] for r in all_std_results])
    np_cms = numpy_lib.array([r['confusion_matrix'] for r in all_np_results])
    
    std_cm_mean = numpy_lib.mean(std_cms, axis=0)
    np_cm_mean = numpy_lib.mean(np_cms, axis=0)
    
    sns.heatmap(std_cm_mean, annot=True, fmt='.1f', cmap='Blues', ax=axes[0])
    axes[0].set_title(f"Confusion Matrix - Standard NN (Mean of {len(all_std_results)} runs)")
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    
    sns.heatmap(np_cm_mean, annot=True, fmt='.1f', cmap='Greens', ax=axes[1])
    axes[1].set_title(f"Confusion Matrix - NP NN (Mean of {len(all_np_results)} runs)")
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig(f'Images/batch_confusion_matrix_{dataset_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Batch confusion matrix saved for {dataset_name}")

def plot_batch_roc_curves(all_std_results, all_np_results, dataset_name, figsize=(18, 8)):
    """Plot ROC curves with mean and std bands"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    std_tprs = []
    np_tprs = []
    mean_fpr = numpy_lib.linspace(0, 1, 100)
    
    for result in all_std_results:
        interp_tpr = numpy_lib.interp(mean_fpr, result['fpr']['micro'], result['tpr']['micro'])
        std_tprs.append(interp_tpr)
    
    for result in all_np_results:
        interp_tpr = numpy_lib.interp(mean_fpr, result['fpr']['micro'], result['tpr']['micro'])
        np_tprs.append(interp_tpr)
    
    std_tprs = numpy_lib.array(std_tprs)
    np_tprs = numpy_lib.array(np_tprs)
    
    std_mean_tpr = numpy_lib.mean(std_tprs, axis=0)
    std_std_tpr = numpy_lib.std(std_tprs, axis=0)
    np_mean_tpr = numpy_lib.mean(np_tprs, axis=0)
    np_std_tpr = numpy_lib.std(np_tprs, axis=0)
    
    std_mean_auc = numpy_lib.mean([r['roc_auc']['micro'] for r in all_std_results])
    np_mean_auc = numpy_lib.mean([r['roc_auc']['micro'] for r in all_np_results])
    
    ax1.plot(mean_fpr, std_mean_tpr, 'b-', label=f"Standard NN (AUC = {std_mean_auc:.3f})", linewidth=2)
    ax1.fill_between(mean_fpr, std_mean_tpr - std_std_tpr, std_mean_tpr + std_std_tpr, 
                     alpha=0.2, color='b', label='±1 std')
    
    ax1.plot(mean_fpr, np_mean_tpr, 'g-', label=f"NP NN (AUC = {np_mean_auc:.3f})", linewidth=2)
    ax1.fill_between(mean_fpr, np_mean_tpr - np_std_tpr, np_mean_tpr + np_std_tpr, 
                     alpha=0.2, color='g', label='±1 std')
    
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'Micro-Average ROC Curves (n={len(all_std_results)} runs)')
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    n_classes = all_std_results[0]['n_classes']
    selected_classes = list(range(min(3, n_classes)))
    
    for cls in selected_classes:
        std_cls_tprs = []
        for result in all_std_results:
            interp_tpr = numpy_lib.interp(mean_fpr, result['fpr'][cls], result['tpr'][cls])
            std_cls_tprs.append(interp_tpr)
        std_cls_tprs = numpy_lib.array(std_cls_tprs)
        std_cls_mean = numpy_lib.mean(std_cls_tprs, axis=0)
        std_cls_auc = numpy_lib.mean([r['roc_auc'][cls] for r in all_std_results])
        
        np_cls_tprs = []
        for result in all_np_results:
            interp_tpr = numpy_lib.interp(mean_fpr, result['fpr'][cls], result['tpr'][cls])
            np_cls_tprs.append(interp_tpr)
        np_cls_tprs = numpy_lib.array(np_cls_tprs)
        np_cls_mean = numpy_lib.mean(np_cls_tprs, axis=0)
        np_cls_auc = numpy_lib.mean([r['roc_auc'][cls] for r in all_np_results])
        
        ax2.plot(mean_fpr, std_cls_mean, '--', label=f"Std Class {cls} (AUC={std_cls_auc:.3f})")
        ax2.plot(mean_fpr, np_cls_mean, '-', label=f"NP Class {cls} (AUC={np_cls_auc:.3f})")
    
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Per-Class ROC Curves (Mean)')
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(f'Images/batch_roc_curves_{dataset_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Batch ROC curves saved for {dataset_name}")

def plot_batch_pca_visualization(X_test, y_test_orig, all_std_results, all_np_results, dataset_name, figsize=(16, 7)):
    """Plot PCA showing prediction consistency across runs"""
    X_test_np = to_numpy(X_test)
    y_test_orig_np = to_numpy(y_test_orig)
    
    pca = PCA(n_components=2)
    X_test_pca = pca.fit_transform(X_test_np)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    std_correct_counts = numpy_lib.zeros(len(y_test_orig_np))
    np_correct_counts = numpy_lib.zeros(len(y_test_orig_np))
    
    for result in all_std_results:
        std_correct_counts += (result['predictions'] == y_test_orig_np).astype(int)
    
    for result in all_np_results:
        np_correct_counts += (result['predictions'] == y_test_orig_np).astype(int)
    
    std_correct_pct = std_correct_counts / len(all_std_results)
    np_correct_pct = np_correct_counts / len(all_np_results)
    
    scatter1 = axes[0].scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=std_correct_pct,
                               cmap='RdYlGn', alpha=0.6, s=20)
    axes[0].set_title(f"Standard NN - Prediction Consistency (n={len(all_std_results)} runs)")
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')
    cbar1 = plt.colorbar(scatter1, ax=axes[0])
    cbar1.set_label('Fraction Correct')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    scatter2 = axes[1].scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=np_correct_pct,
                               cmap='RdYlGn', alpha=0.6, s=20)
    axes[1].set_title(f"NP NN - Prediction Consistency (n={len(all_np_results)} runs)")
    axes[1].set_xlabel('Principal Component 1')
    axes[1].set_ylabel('Principal Component 2')
    cbar2 = plt.colorbar(scatter2, ax=axes[1])
    cbar2.set_label('Fraction Correct')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(f'Images/batch_pca_visualization_{dataset_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Batch PCA visualization saved for {dataset_name}")

def process_and_save_results_statistical(idn, models_1_list, models_2_list, X_test, y_test, y_test_orig, n_classes):
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
        std_results = evaluate_model(model_1, X_test, y_test, y_test_orig, f"Standard NN Run {i+1}", n_classes)
        np_results = evaluate_model(model_2, X_test, y_test, y_test_orig, f"NP NN Run {i+1}", n_classes)
        
        std_results_list.append(std_results)
        np_results_list.append(np_results)
    
    # Perform statistical analysis
    show_statistical_comparison(all_acc_1, all_acc_2, all_lss_1, all_lss_2, idn)
    
    # Perform significance tests
    perform_significance_tests(std_results_list, np_results_list, idn)
    
    return std_results_list, np_results_list

def show_statistical_comparison(all_acc_1, all_acc_2, all_lss_1, all_lss_2, idn, label_1='Standard NN', label_2='NP NN'):
    """
    Create visualization with mean, std, and confidence intervals
    """
    # Convert to numpy arrays for easier manipulation
    all_acc_1 = numpy_lib.array(all_acc_1)  # Shape: (n_runs, n_epochs)
    all_acc_2 = numpy_lib.array(all_acc_2)
    all_lss_1 = numpy_lib.array(all_lss_1)
    all_lss_2 = numpy_lib.array(all_lss_2)
    
    # Calculate statistics across runs
    mean_acc_1 = numpy_lib.mean(all_acc_1, axis=0) * 100
    std_acc_1 = numpy_lib.std(all_acc_1, axis=0) * 100
    mean_acc_2 = numpy_lib.mean(all_acc_2, axis=0) * 100
    std_acc_2 = numpy_lib.std(all_acc_2, axis=0) * 100
    
    mean_lss_1 = numpy_lib.mean(all_lss_1, axis=0)
    std_lss_1 = numpy_lib.std(all_lss_1, axis=0)
    mean_lss_2 = numpy_lib.mean(all_lss_2, axis=0)
    std_lss_2 = numpy_lib.std(all_lss_2, axis=0)
    
    # Calculate 95% confidence intervals
    n_runs = all_acc_1.shape[0]
    ci_acc_1 = 1.96 * std_acc_1 / numpy_lib.sqrt(n_runs)
    ci_acc_2 = 1.96 * std_acc_2 / numpy_lib.sqrt(n_runs)
    ci_lss_1 = 1.96 * std_lss_1 / numpy_lib.sqrt(n_runs)
    ci_lss_2 = 1.96 * std_lss_2 / numpy_lib.sqrt(n_runs)
    
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
    print(f"Statistical Significance Tests for {idn}")
    print(f"{'='*60}\n")
    
    for metric in metrics_to_test:
        try:
            std_values = [r[metric] for r in std_results_list]
            np_values = [r[metric] for r in np_results_list]
            
            # Paired t-test (since models trained on same data splits)
            t_stat, p_value = stats.ttest_rel(std_values, np_values)
            
            # Effect size (Cohen's d for paired samples)
            diff = numpy_lib.array(std_values) - numpy_lib.array(np_values)
            cohens_d = numpy_lib.mean(diff) / numpy_lib.std(diff, ddof=1)
            
            mean_std = numpy_lib.mean(std_values)
            mean_np = numpy_lib.mean(np_values)
            std_std = numpy_lib.std(std_values, ddof=1)
            std_np = numpy_lib.std(np_values, ddof=1)
            
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
    
    plt.suptitle(f'Model Performance Distributions ({idn})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'Images/boxplot_comparison_{idn}.png', dpi=200, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_epoch_convergence(all_acc_std, all_acc_np, dataset_name, threshold_start=0.80, threshold_end=0.99, num_thresholds=5, figsize=(12, 6)):
    """Plot convergence with custom threshold range"""
    all_acc_std_np = [[to_numpy(a) if hasattr(a, 'get') else a for a in acc_list] 
                      for acc_list in all_acc_std]
    all_acc_np_np = [[to_numpy(a) if hasattr(a, 'get') else a for a in acc_list] 
                     for acc_list in all_acc_np]
    
    thresholds = numpy_lib.linspace(threshold_start, threshold_end, num_thresholds)
    
    std_epochs_all = []
    np_epochs_all = []
    
    for threshold in thresholds:
        std_epochs_for_threshold = []
        np_epochs_for_threshold = []
        
        for acc_list in all_acc_std_np:
            acc_arr = numpy_lib.array(acc_list)
            epoch = numpy_lib.argmax(acc_arr >= threshold) if numpy_lib.any(acc_arr >= threshold) else len(acc_arr)
            std_epochs_for_threshold.append(epoch + 1)
        
        for acc_list in all_acc_np_np:
            acc_arr = numpy_lib.array(acc_list)
            epoch = numpy_lib.argmax(acc_arr >= threshold) if numpy_lib.any(acc_arr >= threshold) else len(acc_arr)
            np_epochs_for_threshold.append(epoch + 1)
        
        std_epochs_all.append(std_epochs_for_threshold)
        np_epochs_all.append(np_epochs_for_threshold)
    
    std_means = [numpy_lib.mean(epochs) for epochs in std_epochs_all]
    std_stds = [numpy_lib.std(epochs) for epochs in std_epochs_all]
    np_means = [numpy_lib.mean(epochs) for epochs in np_epochs_all]
    np_stds = [numpy_lib.std(epochs) for epochs in np_epochs_all]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = numpy_lib.arange(len(thresholds))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, std_means, width, yerr=std_stds, 
                   label='Standard NN', capsize=5, color='#1f77b4')
    bars2 = ax.bar(x + width/2, np_means, width, yerr=np_stds, 
                   label='NP NN', capsize=5, color='#2ca02c')
    
    for i, (std_m, np_m) in enumerate(zip(std_means, np_means)):
        diff = std_m - np_m
        color = 'green' if diff > 0 else 'red'
        y_pos = max(std_m + std_stds[i], np_m + np_stds[i]) + 2
        ax.annotate(f"{diff:+.1f}", xy=(i, y_pos), ha='center', va='bottom',
                   color=color, weight='bold', fontsize=10)
    
    ax.set_xlabel('Accuracy Threshold', fontsize=12)
    ax.set_ylabel('Epochs to Reach Threshold', fontsize=12)
    ax.set_title(f'Convergence Speed - {dataset_name} (n={len(all_acc_std)} runs)', 
                fontsize=14, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t*100:.1f}%" for t in thresholds])
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'Images/batch_convergence_{dataset_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Batch convergence plot saved for {dataset_name}")
    
    convergence_df = pd.DataFrame({
        'Accuracy Threshold': [f"{t*100:.1f}%" for t in thresholds],
        'Standard NN Mean': [f"{m:.1f}±{s:.1f}" for m, s in zip(std_means, std_stds)],
        'NP NN Mean': [f"{m:.1f}±{s:.1f}" for m, s in zip(np_means, np_stds)],
        'Difference (Std - NP)': [f"{std_m - np_m:+.1f}" for std_m, np_m in zip(std_means, np_means)]
    })
    
    return convergence_df

def main():
    arch = [ 
        [784, 256, 128, 10],
        [3072, 512, 256, 128, 10],
        [3072, 512, 256, 128, 100]
    ]
    models_1_list = []
    models_2_list = []
    nn_acc = []
    np_acc = []
    experiment_name = "CIFAR10_Scaled"
    k = 1
    n_c = 10
    for i in range(30, 40):
        nn = NeuralNetwork(arch[k])
        nn.load_model(i)
        nn_acc.append(nn.acc_stat)
        models_1_list.append(nn)
        np = NPNeuralNetwork(arch[k])
        np.load_model(i)
        np_acc.append(np.acc_stat)
        models_2_list.append(np)

    _, _, X_test, y_test, y_test_orig = load_data_CIFAR10_stan()

    std_results, np_results = process_and_save_results_statistical(
        idn=experiment_name,
        models_1_list=models_1_list,
        models_2_list=models_2_list,
        X_test = X_test,
        y_test = y_test,
        y_test_orig = y_test_orig,
        n_classes=n_c
    )

    create_box_plots(std_results, np_results, experiment_name)
    plot_batch_confusion_matrix(std_results, np_results, experiment_name)
    plot_batch_pca_visualization(X_test, y_test_orig, std_results, np_results, experiment_name)
    plot_epoch_convergence(nn_acc, np_acc, experiment_name, threshold_start=0.35, threshold_end=0.85, num_thresholds=5)


if __name__ == "__main__":
    main()