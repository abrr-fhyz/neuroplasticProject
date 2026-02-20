import cupy as cp
import numpy as np
from models.CNN import ConvPoolModule
from models.NNGPU import NeuralNetwork
from models.NPGPU import NPNeuralNetwork
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from scipy import stats
import pandas as pd
import time

from tensorflow.keras.datasets import cifar10

_, (X_test_np, y_test_np) = cifar10.load_data()

# Preprocess
X_test = (X_test_np.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)  # (N, 3, 32, 32)
y_test = y_test_np.flatten()  # (N,) class labels

X_test_gpu = cp.asarray(X_test)
y_test_gpu = cp.asarray(y_test)

print(f"Test set: {X_test.shape}, Labels: {y_test.shape}\n")

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def evaluate_model(cpm, fc_model, X_gpu, y_true, batch_size=256):
    all_preds = []
    batch_times = []
    
    Xb_warmup = X_gpu[:batch_size]
    flat_warmup = cpm.forward(Xb_warmup)
    _ = fc_model.predict(flat_warmup)
    cp.cuda.Stream.null.synchronize()
    
    for start in range(0, X_gpu.shape[0], batch_size):
        Xb = X_gpu[start : start + batch_size]
        
        # Time the forward pass
        cp.cuda.Stream.null.synchronize()  # Wait for previous ops
        t_start = time.perf_counter()
        
        flat = cpm.forward(Xb)
        y_pred = fc_model.predict(flat)
        pred_cls = cp.argmax(y_pred, axis=1)
        
        cp.cuda.Stream.null.synchronize()  # Wait for completion
        t_end = time.perf_counter()
        
        batch_times.append((t_end - t_start) * 1000)  # Convert to ms
        all_preds.append(cp.asnumpy(pred_cls))
    
    y_pred_all = np.concatenate(all_preds)
    
    # Compute metrics (macro-averaged)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_all, average='macro', zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred_all)
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = \
        precision_recall_fscore_support(y_true, y_pred_all, average=None, zero_division=0)
    
    # Timing statistics
    avg_batch_time = np.mean(batch_times)
    total_samples = X_gpu.shape[0]
    avg_time_per_sample = avg_batch_time / batch_size
    throughput = 1000.0 / avg_time_per_sample  # samples per second
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'support': support,
        'predictions': y_pred_all,
        'avg_batch_time_ms': avg_batch_time,
        'avg_time_per_sample_ms': avg_time_per_sample,
        'throughput_samples_per_sec': throughput
    }

print("=" * 80)
print("  Evaluating Standard NN-CNN Models (10 iterations)")
print("=" * 80)

nn_results = []

for i in range(10):
    print(f"\nEvaluating NN model {i}...")
    
    # Load ConvPool
    cpm = ConvPoolModule([
        {"in_channels":  3, "out_channels":  32, "kernel_size": 3, "pad": 1, "pool_size": 2},
        {"in_channels": 32, "out_channels":  64, "kernel_size": 3, "pad": 1, "pool_size": 2},
        {"in_channels": 64, "out_channels": 128, "kernel_size": 3, "pad": 1, "pool_size": 2},
    ])
    cpm.load(f"artifacts/conv_nn_{i}.npz")
    
    # Load FC
    flat_size = cpm.output_size(32, 32)
    fc = NeuralNetwork([flat_size, 256, 128, 10], gpu=True)
    fc.load_model(f"nn_{i}")
    
    # Evaluate
    metrics = evaluate_model(cpm, fc, X_test_gpu, y_test)
    nn_results.append(metrics)
    
    print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"  Precision: {metrics['precision']*100:.2f}%")
    print(f"  Recall:    {metrics['recall']*100:.2f}%")
    print(f"  F1 Score:  {metrics['f1']*100:.2f}%")
    print(f"  Inference: {metrics['avg_time_per_sample_ms']:.3f} ms/sample ({metrics['throughput_samples_per_sec']:.0f} samples/sec)")

print("\n" + "=" * 80)
print("  Evaluating NP-CNN Models (10 iterations)")
print("=" * 80)

np_results = []

for i in range(10):
    print(f"\nEvaluating NP model {i}...")
    
    # Load ConvPool
    cpm = ConvPoolModule([
        {"in_channels":  3, "out_channels":  32, "kernel_size": 3, "pad": 1, "pool_size": 2},
        {"in_channels": 32, "out_channels":  64, "kernel_size": 3, "pad": 1, "pool_size": 2},
        {"in_channels": 64, "out_channels": 128, "kernel_size": 3, "pad": 1, "pool_size": 2},
    ])
    cpm.load(f"artifacts/conv_np_{i}.npz")
    
    # Load FC
    flat_size = cpm.output_size(32, 32)
    fc = NPNeuralNetwork([flat_size, 256, 128, 10], gpu=True)
    fc.load_model(f"np_{i}")
    
    # Evaluate
    metrics = evaluate_model(cpm, fc, X_test_gpu, y_test)
    np_results.append(metrics)
    
    print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"  Precision: {metrics['precision']*100:.2f}%")
    print(f"  Recall:    {metrics['recall']*100:.2f}%")
    print(f"  F1 Score:  {metrics['f1']*100:.2f}%")
    print(f"  Inference: {metrics['avg_time_per_sample_ms']:.3f} ms/sample ({metrics['throughput_samples_per_sec']:.0f} samples/sec)")

# ──────────────────────────────────────────────
#  4. Aggregate Statistics
# ──────────────────────────────────────────────

def compute_stats(results, metric_name):
    """Extract a specific metric across all runs."""
    values = [r[metric_name] for r in results]
    return {
        'mean': np.mean(values),
        'std': np.std(values, ddof=1),
        'min': np.min(values),
        'max': np.max(values),
        'values': values
    }

metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'avg_time_per_sample_ms', 'throughput_samples_per_sec']

nn_stats = {m: compute_stats(nn_results, m) for m in metrics_to_compare}
np_stats = {m: compute_stats(np_results, m) for m in metrics_to_compare}

# ──────────────────────────────────────────────
#  5. Statistical Tests
# ──────────────────────────────────────────────

def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group2) - np.mean(group1)) / pooled_std

statistical_results = {}

for metric in metrics_to_compare:
    nn_vals = nn_stats[metric]['values']
    np_vals = np_stats[metric]['values']
    
    # Paired t-test (since same test set)
    t_stat, p_value = stats.ttest_rel(np_vals, nn_vals)
    
    # Cohen's d
    effect_size = cohens_d(nn_vals, np_vals)
    
    statistical_results[metric] = {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': effect_size
    }

# ──────────────────────────────────────────────
#  6. Per-Class Analysis (average across 10 runs)
# ──────────────────────────────────────────────

nn_per_class = {
    'precision': np.mean([r['precision_per_class'] for r in nn_results], axis=0),
    'recall': np.mean([r['recall_per_class'] for r in nn_results], axis=0),
    'f1': np.mean([r['f1_per_class'] for r in nn_results], axis=0),
}

np_per_class = {
    'precision': np.mean([r['precision_per_class'] for r in np_results], axis=0),
    'recall': np.mean([r['recall_per_class'] for r in np_results], axis=0),
    'f1': np.mean([r['f1_per_class'] for r in np_results], axis=0),
}

# ──────────────────────────────────────────────
#  7. Display Results
# ──────────────────────────────────────────────

print("\n" + "=" * 80)
print("  SUMMARY STATISTICS (10 runs each)")
print("=" * 80)

# Metrics table
metrics_display = ['accuracy', 'precision', 'recall', 'f1']
summary_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'NN Mean': [nn_stats[m]['mean']*100 for m in metrics_display],
    'NN Std': [nn_stats[m]['std']*100 for m in metrics_display],
    'NP Mean': [np_stats[m]['mean']*100 for m in metrics_display],
    'NP Std': [np_stats[m]['std']*100 for m in metrics_display],
    'Δ (abs)': [(np_stats[m]['mean'] - nn_stats[m]['mean'])*100 for m in metrics_display],
    'p-value': [statistical_results[m]['p_value'] for m in metrics_display],
    "Cohen's d": [statistical_results[m]['cohens_d'] for m in metrics_display],
})

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.precision', 4)

print("\n", summary_df.to_string(index=False))

# Timing table
print("\n" + "-" * 80)
print("  INFERENCE TIMING")
print("-" * 80)

timing_df = pd.DataFrame({
    'Metric': ['Time per sample (ms)', 'Throughput (samples/sec)'],
    'NN Mean': [
        nn_stats['avg_time_per_sample_ms']['mean'],
        nn_stats['throughput_samples_per_sec']['mean']
    ],
    'NN Std': [
        nn_stats['avg_time_per_sample_ms']['std'],
        nn_stats['throughput_samples_per_sec']['std']
    ],
    'NP Mean': [
        np_stats['avg_time_per_sample_ms']['mean'],
        np_stats['throughput_samples_per_sec']['mean']
    ],
    'NP Std': [
        np_stats['avg_time_per_sample_ms']['std'],
        np_stats['throughput_samples_per_sec']['std']
    ],
    'Δ (abs)': [
        np_stats['avg_time_per_sample_ms']['mean'] - nn_stats['avg_time_per_sample_ms']['mean'],
        np_stats['throughput_samples_per_sec']['mean'] - nn_stats['throughput_samples_per_sec']['mean']
    ],
    'p-value': [
        statistical_results['avg_time_per_sample_ms']['p_value'],
        statistical_results['throughput_samples_per_sec']['p_value']
    ],
    "Cohen's d": [
        statistical_results['avg_time_per_sample_ms']['cohens_d'],
        statistical_results['throughput_samples_per_sec']['cohens_d']
    ],
})

print("\n", timing_df.to_string(index=False))

print("\n" + "=" * 80)
print("  STATISTICAL INTERPRETATION (Performance Metrics)")
print("=" * 80)

perf_metrics = ['accuracy', 'precision', 'recall', 'f1']

for metric in perf_metrics:
    p = statistical_results[metric]['p_value']
    d = statistical_results[metric]['cohens_d']
    
    # Significance
    if p < 0.001:
        sig_level = "***"
        sig_text = "highly significant"
    elif p < 0.01:
        sig_level = "**"
        sig_text = "very significant"
    elif p < 0.05:
        sig_level = "*"
        sig_text = "significant"
    else:
        sig_level = "ns"
        sig_text = "not significant"
    
    # Effect size interpretation
    abs_d = abs(d)
    if abs_d < 0.2:
        effect_text = "negligible"
    elif abs_d < 0.5:
        effect_text = "small"
    elif abs_d < 0.8:
        effect_text = "medium"
    else:
        effect_text = "large"
    
    direction = "favors NP-CNN" if d > 0 else "favors NN-CNN"
    
    print(f"\n{metric.upper()}:")
    print(f"  p-value = {p:.6f} {sig_level} ({sig_text})")
    print(f"  Cohen's d = {d:+.4f} ({effect_text} effect size, {direction})")

print("\n" + "=" * 80)
print("  STATISTICAL INTERPRETATION (Inference Timing)")
print("=" * 80)

timing_metrics = ['avg_time_per_sample_ms', 'throughput_samples_per_sec']

for metric in timing_metrics:
    p = statistical_results[metric]['p_value']
    d = statistical_results[metric]['cohens_d']
    
    # Significance
    if p < 0.001:
        sig_level = "***"
        sig_text = "highly significant"
    elif p < 0.01:
        sig_level = "**"
        sig_text = "very significant"
    elif p < 0.05:
        sig_level = "*"
        sig_text = "significant"
    else:
        sig_level = "ns"
        sig_text = "not significant"
    
    # Effect size interpretation
    abs_d = abs(d)
    if abs_d < 0.2:
        effect_text = "negligible"
    elif abs_d < 0.5:
        effect_text = "small"
    elif abs_d < 0.8:
        effect_text = "medium"
    else:
        effect_text = "large"
    
    # For timing, positive d means NP is slower (worse), negative means faster (better)
    if metric == 'avg_time_per_sample_ms':
        direction = "NP slower" if d > 0 else "NP faster"
        display_name = "TIME PER SAMPLE"
    else:
        direction = "NP faster" if d > 0 else "NP slower"
        display_name = "THROUGHPUT"
    
    print(f"\n{display_name}:")
    print(f"  p-value = {p:.6f} {sig_level} ({sig_text})")
    print(f"  Cohen's d = {d:+.4f} ({effect_text} effect size, {direction})")

print("\n" + "=" * 80)
print("  PER-CLASS METRICS (averaged across 10 runs)")
print("=" * 80)

per_class_df = pd.DataFrame({
    'Class': CLASSES,
    'NN Precision': nn_per_class['precision'] * 100,
    'NP Precision': np_per_class['precision'] * 100,
    'NN Recall': nn_per_class['recall'] * 100,
    'NP Recall': np_per_class['recall'] * 100,
    'NN F1': nn_per_class['f1'] * 100,
    'NP F1': np_per_class['f1'] * 100,
})

print("\n", per_class_df.to_string(index=False))

# Identify classes with biggest improvement
f1_improvement = np_per_class['f1'] - nn_per_class['f1']
best_improved = np.argsort(f1_improvement)[-3:][::-1]
worst_improved = np.argsort(f1_improvement)[:3]

print("\n" + "-" * 80)
print("Top 3 classes with most F1 improvement (NP vs NN):")
for idx in best_improved:
    print(f"  {CLASSES[idx]:12s}: +{f1_improvement[idx]*100:.2f}%")

print("\nBottom 3 classes (least/negative improvement):")
for idx in worst_improved:
    print(f"  {CLASSES[idx]:12s}: {f1_improvement[idx]*100:+.2f}%")

# ──────────────────────────────────────────────
#  8. Save Results to CSV
# ──────────────────────────────────────────────

summary_df.to_csv("data/model_comparison_summary.csv", index=False)
timing_df.to_csv("data/model_comparison_timing.csv", index=False)
per_class_df.to_csv("data/model_comparison_per_class.csv", index=False)

print("\n" + "=" * 80)
print("Results saved to:")
print("  - data/model_comparison_summary.csv")
print("  - data/model_comparison_timing.csv")
print("  - data/model_comparison_per_class.csv")
print("=" * 80)