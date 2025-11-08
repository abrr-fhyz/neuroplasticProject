import cupy as np
import numpy as numpy_lib
import matplotlib.pyplot as plt
import time
import json
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score, 
    r2_score, roc_curve, auc, precision_recall_curve, average_precision_score,
    classification_report
)
import pandas as pd

def to_numpy(arr):
    """Convert CuPy array to NumPy array if needed."""
    if hasattr(arr, 'get'):
        return arr.get()
    return arr

def show_comparison_stats(acc_1, acc_2, lss_1, lss_2, idn, label_1='Standard NN', label_2='NP NN'):
    # Convert CuPy lists to numpy for plotting
    acc_1 = [to_numpy(a) if hasattr(a, 'get') else a for a in acc_1]
    acc_2 = [to_numpy(a) if hasattr(a, 'get') else a for a in acc_2]
    lss_1 = [to_numpy(l) if hasattr(l, 'get') else l for l in lss_1]
    lss_2 = [to_numpy(l) if hasattr(l, 'get') else l for l in lss_2]
    
    epochs_1 = range(1, len(acc_1) + 1)
    epochs_2 = range(1, len(acc_2) + 1)
    acc_1_percent = [a * 100 for a in acc_1]
    acc_2_percent = [a * 100 for a in acc_2]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(epochs_1, acc_1_percent, label=f'{label_1} Accuracy', color='blue')
    ax1.plot(epochs_2, acc_2_percent, label=f'{label_2} Accuracy', color='green')
    ax1.plot(epochs_1[-1], acc_1_percent[-1], 'o', color='blue')
    ax1.plot(epochs_2[-1], acc_2_percent[-1], 'o', color='green')
    ax1.annotate(f'{acc_1_percent[-1]:.2f}%', (epochs_1[-1], acc_1_percent[-1]),
                 textcoords="offset points", xytext=(-30,10), ha='center',
                 fontsize=8, color='blue',
                 arrowprops=dict(arrowstyle='->', color='blue'))
    ax1.annotate(f'{acc_2_percent[-1]:.2f}%', (epochs_2[-1], acc_2_percent[-1]),
                 textcoords="offset points", xytext=(30,10), ha='center',
                 fontsize=8, color='green',
                 arrowprops=dict(arrowstyle='->', color='green'))
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.set_title("Model Comparison: Accuracy")
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(epochs_1, lss_1, label=f'{label_1} Loss', color='red')
    ax2.plot(epochs_2, lss_2, label=f'{label_2} Loss', color='orange')
    ax2.plot(epochs_1[-1], lss_1[-1], 'o', color='red')
    ax2.plot(epochs_2[-1], lss_2[-1], 'o', color='orange')
    ax2.annotate(f'{lss_1[-1]:.5f}', (epochs_1[-1], lss_1[-1]),
                 textcoords="offset points", xytext=(-30,-10), ha='center',
                 fontsize=8, color='red',
                 arrowprops=dict(arrowstyle='->', color='red'))
    ax2.annotate(f'{lss_2[-1]:.5f}', (epochs_2[-1], lss_2[-1]),
                 textcoords="offset points", xytext=(30,-10), ha='center',
                 fontsize=8, color='orange',
                 arrowprops=dict(arrowstyle='->', color='orange'))
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.set_title("Model Comparison: Loss")
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(f'Images/comparison_stats_{idn}.png', dpi=150, bbox_inches='tight')
    plt.close()

def evaluate_model(model, X_test, y_test, y_test_orig, model_name, n_classes = 10):
    start_time = time.time()
    preds = model.predict(X_test)
    inference_time = (time.time() - start_time) / len(X_test)
    
    # Convert CuPy arrays to NumPy for sklearn/numpy operations
    preds_np = to_numpy(preds)
    y_test_np = to_numpy(y_test)
    y_test_orig_np = to_numpy(y_test_orig)
    
    # Use NumPy for argmax to avoid dtype parameter issue with CuPy
    predicted_classes = numpy_lib.argmax(preds_np, axis=1)
    true_classes = numpy_lib.argmax(y_test_np, axis=1)
    
    accuracy = numpy_lib.mean(predicted_classes == true_classes)
    precision = precision_score(true_classes, predicted_classes, average='macro')
    recall = recall_score(true_classes, predicted_classes, average='macro')
    f1 = f1_score(true_classes, predicted_classes, average='macro')
    r2 = r2_score(y_test_np, preds_np)
    cm = confusion_matrix(true_classes, predicted_classes)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve((true_classes == i).astype(int), 
                                      preds_np[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    fpr["micro"], tpr["micro"], _ = roc_curve(numpy_lib.eye(n_classes)[true_classes].ravel(), 
                                            preds_np.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    precision_curve = dict()
    recall_curve = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision_curve[i], recall_curve[i], _ = precision_recall_curve(
            (true_classes == i).astype(int), preds_np[:, i]
        )
        average_precision[i] = average_precision_score(
            (true_classes == i).astype(int), preds_np[:, i]
        )
    
    class_report = classification_report(true_classes, predicted_classes, output_dict=True)
    
    class_metrics = {}
    for i in range(n_classes):
        class_metrics[i] = {
            'precision': class_report[str(i)]['precision'],
            'recall': class_report[str(i)]['recall'],
            'f1-score': class_report[str(i)]['f1-score'],
            'roc_auc': roc_auc[i],
            'avg_precision': average_precision[i]
        }
    
    return {
        'name': model_name,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'r2': float(r2),
        'inference_time': float(inference_time * 1000),
        'confusion_matrix': cm,
        'predictions': predicted_classes,
        'probabilities': preds_np,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'class_metrics': class_metrics
    }

def save_summary_table(results, idn):
    summary_text = ""
    summary_text += "===== MODEL PERFORMANCE SUMMARY =====\n\n"
    
    metrics_table = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Inference Time (ms)'],
        results[0]['name']: [
            f"{results[0]['accuracy']*100:.2f}%",
            f"{results[0]['precision']:.4f}",
            f"{results[0]['recall']:.4f}",
            f"{results[0]['f1']:.4f}",
            f"{results[0]['inference_time']:.4f}"
        ],
        results[1]['name']: [
            f"{results[1]['accuracy']*100:.2f}%",
            f"{results[1]['precision']:.4f}",
            f"{results[1]['recall']:.4f}",
            f"{results[1]['f1']:.4f}",
            f"{results[1]['inference_time']:.4f}"
        ],
        'Difference': [
            f"{(results[1]['accuracy'] - results[0]['accuracy'])*100:+.2f}%",
            f"{results[1]['precision'] - results[0]['precision']:+.4f}",
            f"{results[1]['recall'] - results[0]['recall']:+.4f}",
            f"{results[1]['f1'] - results[0]['f1']:+.4f}",
            f"{results[1]['inference_time'] - results[0]['inference_time']:+.4f}"
        ]
    })
    
    summary_text += metrics_table.to_string(index=False) + "\n\n"

    auc_table = pd.DataFrame({
        'Class': list(range(10)) + ['Micro-Average'],
        results[0]['name']: [results[0]['roc_auc'][i] for i in range(10)] + [results[0]['roc_auc']['micro']],
        results[1]['name']: [results[1]['roc_auc'][i] for i in range(10)] + [results[1]['roc_auc']['micro']],
        'Difference': [results[1]['roc_auc'][i] - results[0]['roc_auc'][i] for i in range(10)] + 
                     [results[1]['roc_auc']['micro'] - results[0]['roc_auc']['micro']]
    })
    
    summary_text += "===== ROC AUC SUMMARY =====\n\n"
    summary_text += auc_table.to_string(index=False)
    
    with open(f'logs/summary_{idn}.txt', 'w') as f:
        f.write(summary_text)

def save_metrics_json(results, idn):
    metrics_data = {
        'idn': idn,
        'standard_nn': {
            'accuracy': float(results[0]['accuracy']),
            'precision': float(results[0]['precision']),
            'recall': float(results[0]['recall']),
            'f1': float(results[0]['f1']),
            'inference_time': float(results[0]['inference_time'])
        },
        'np_nn': {
            'accuracy': float(results[1]['accuracy']),
            'precision': float(results[1]['precision']),
            'recall': float(results[1]['recall']),
            'f1': float(results[1]['f1']),
            'inference_time': float(results[1]['inference_time'])
        }
    }
    
    try:
        with open('data.json', 'r') as f:
            all_data = json.load(f)
    except FileNotFoundError:
        all_data = []
    
    all_data.append(metrics_data)
    
    with open('data.json', 'w') as f:
        json.dump(all_data, f, indent=2)

def plot_final_metrics():
    with open('data.json', 'r') as f:
        all_data = json.load(f)

    datasets = {
        'MNIST': [d for d in all_data if 0 <= d['idn'] <= 9],
        'Fashion MNIST': [d for d in all_data if 10 <= d['idn'] <= 19], 
        'CIFAR10': [d for d in all_data if 20 <= d['idn'] <= 29],
        'CIFAR10 (Scaled)': [d for d in all_data if 30 <= d['idn'] <= 39],
        'CIFAR10 (Plateau)': [d for d in all_data if 40 <= d['idn'] <= 49],
        'CIFAR100 (Scaled)': [d for d in all_data if 50 <= d['idn'] <= 59]
    }

    metrics = ['accuracy', 'precision', 'recall', 'f1', 'inference_time']

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        dataset_names = []
        std_means = []
        std_stds = []
        np_means = []
        np_stds = []

        for dataset_name, data in datasets.items():
            if data:
                std_values = [d['standard_nn'][metric] for d in data]
                np_values = [d['np_nn'][metric] for d in data]
                
                dataset_names.append(dataset_name)
                std_means.append(numpy_lib.median(std_values))
                std_stds.append(numpy_lib.subtract(*numpy_lib.percentile(std_values, [75, 25])) / 2)
                np_means.append(numpy_lib.median(np_values))
                np_stds.append(numpy_lib.subtract(*numpy_lib.percentile(np_values, [75, 25])) / 2)

        x = numpy_lib.arange(len(dataset_names))
        width = 0.35

        bars1 = ax.bar(x - width/2, std_means, width, yerr=std_stds, label='Standard NN',
                    capsize=5, color='#1f77b4', edgecolor='black')
        bars2 = ax.bar(x + width/2, np_means, width, yerr=np_stds, label='NP NN',
                    capsize=5, color='#ff7f0e', edgecolor='black')

        ax.set_xlabel('Dataset', fontsize=14, weight='bold')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=14, weight='bold')
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison Across Datasets', fontsize=16, weight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_names, fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height + 0.005),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, weight='bold')

        for i, (std_mean, np_mean) in enumerate(zip(std_means, np_means)):
            diff = np_mean - std_mean
            color = 'green' if diff > 0 else 'red'
            ax.annotate(f'{diff:+.4f}', xy=(i, max(std_mean, np_mean) + max(std_stds[i], np_stds[i]) + 0.01),
                        ha='center', va='bottom', color=color, weight='bold', fontsize=12)

        ax.set_ylim(0, max(max(std_means), max(np_means)) + max(std_stds + np_stds) + 0.05)

        plt.tight_layout()
        plt.savefig(f'Images/ANALYSIS_{metric.upper()}.png', dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()

    print("✅ Main metric comparison plots saved.")

    for dataset_name, data in datasets.items():
        if not data:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        
        std_acc = [d['standard_nn']['accuracy'] for d in data]
        np_acc = [d['np_nn']['accuracy'] for d in data]
        indices = numpy_lib.arange(1, len(data) + 1)

        ax.plot(indices, std_acc, label='Standard NN', marker='o', linestyle='-', linewidth=2, color='#1f77b4')
        ax.plot(indices, np_acc, label='NP NN', marker='s', linestyle='--', linewidth=2, color='#ff7f0e')

        ax.set_xlabel('Test Iteration', fontsize=13)
        ax.set_ylabel('Accuracy', fontsize=13)
        ax.set_title(f'Accuracy Across Iterations - {dataset_name}', fontsize=15, weight='bold')
        ax.set_xticks(indices)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)

        for i, (s, n) in enumerate(zip(std_acc, np_acc)):
            ax.annotate(f'{s:.4f}', (indices[i], s + 0.015), ha='center', fontsize=9, color='#1f77b4')
            ax.annotate(f'{n:.4f}', (indices[i], n - 0.04), ha='center', fontsize=9, color='#ff7f0e')

        plt.tight_layout()
        plt.savefig(f'Images/ITER_ACCURACY_{dataset_name.replace(" ", "_").upper()}.png', dpi=150)
        plt.show()
        plt.close()

    print("✅ Per-iteration accuracy plots saved.")

def process_and_save_results(idn, model_1, model_2, X_test, y_test, y_test_orig):
    
    acc_1, lss_1 = model_1.get_stats()
    acc_2, lss_2 = model_2.get_stats()

    print(f"\nEvaluating models for IDN {idn}...")
    n_classes = 10
    if idn >= 50:
        n_classes = 100
    std_results = evaluate_model(model_1, X_test, y_test, y_test_orig, "Standard NN", n_classes)
    np_results = evaluate_model(model_2, X_test, y_test, y_test_orig, "NP NN", n_classes)
    results = [std_results, np_results]

    show_comparison_stats(acc_1, acc_2, lss_1, lss_2, idn)
    #plot_summary_metrics(results, idn)

    save_summary_table(results, idn)
    save_metrics_json(results, idn)
    
    print(f"Results saved for IDN {idn}")