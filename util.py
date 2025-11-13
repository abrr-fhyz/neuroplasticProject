import numpy as np
from models.NPGPU import NPNeuralNetwork

def compute_convergence_metrics(all_acc_std, all_acc_np, all_loss_std, all_loss_np,
                                threshold_start=0.80, threshold_end=0.99, num_thresholds=5,
                                loss_threshold_start=0.5, loss_threshold_end=0.05, num_loss_thresholds=5):
    """
    Compute comprehensive convergence metrics comparing Standard NN vs NP NN.
    
    Args:
        all_acc_std: List of accuracy lists for Standard NN (n_runs x n_epochs)
        all_acc_np: List of accuracy lists for NP NN (n_runs x n_epochs)
        all_loss_std: List of loss lists for Standard NN (n_runs x n_epochs)
        all_loss_np: List of loss lists for NP NN (n_runs x n_epochs)
        threshold_start: Starting accuracy threshold (default: 0.80)
        threshold_end: Ending accuracy threshold (default: 0.99)
        num_thresholds: Number of accuracy thresholds to test (default: 5)
        loss_threshold_start: Starting loss threshold (default: 0.5)
        loss_threshold_end: Ending loss threshold (default: 0.05)
        num_loss_thresholds: Number of loss thresholds to test (default: 5)
    
    Returns:
        dict: Dictionary containing 8 key metrics:
            - avg_accuracy_speedup: Average relative speedup (%) across all accuracy thresholds
            - avg_loss_reduction: Average relative reduction (%) across all loss thresholds
            - auac_standard: Area Under Accuracy Curve for Standard NN
            - auac_npnn: Area Under Accuracy Curve for NP NN
            - auac_difference: Difference between NPNN and Standard (NPNN - Standard)
            - aulc_standard: Area Under Loss Curve for Standard NN
            - aulc_npnn: Area Under Loss Curve for NP NN
            - aulc_difference: Difference between Standard and NPNN (Standard - NPNN)
    """
    
    def to_numpy(x):
        """Helper to convert to numpy"""
        if hasattr(x, 'get'):
            return x.get()
        return x
    
    # Convert all inputs to numpy arrays
    all_acc_std_np = [[to_numpy(a) for a in acc_list] for acc_list in all_acc_std]
    all_acc_np_np = [[to_numpy(a) for a in acc_list] for acc_list in all_acc_np]
    all_loss_std_np = [[to_numpy(l) for l in loss_list] for loss_list in all_loss_std]
    all_loss_np_np = [[to_numpy(l) for l in loss_list] for loss_list in all_loss_np]
    
    # ========================================================================
    # 1. ACCURACY SPEEDUP ANALYSIS
    # ========================================================================
    accuracy_thresholds = np.linspace(threshold_start, threshold_end, num_thresholds)
    
    speedups = []
    for threshold in accuracy_thresholds:
        std_epochs_for_threshold = []
        np_epochs_for_threshold = []
        
        # Find epochs to reach threshold for Standard NN
        for acc_list in all_acc_std_np:
            acc_arr = np.array(acc_list)
            epoch = np.argmax(acc_arr >= threshold) if np.any(acc_arr >= threshold) else len(acc_arr)
            std_epochs_for_threshold.append(epoch + 1)
        
        # Find epochs to reach threshold for NP NN
        for acc_list in all_acc_np_np:
            acc_arr = np.array(acc_list)
            epoch = np.argmax(acc_arr >= threshold) if np.any(acc_arr >= threshold) else len(acc_arr)
            np_epochs_for_threshold.append(epoch + 1)
        
        # Compute average speedup for this threshold (as percentage)
        mean_std_epochs = np.mean(std_epochs_for_threshold)
        mean_np_epochs = np.mean(np_epochs_for_threshold)
        speedup = abs((mean_np_epochs - mean_std_epochs)) / mean_std_epochs * 100
        speedups.append(speedup)
    
    avg_accuracy_speedup = np.mean(speedups)
    
    # ========================================================================
    # 2. LOSS REDUCTION ANALYSIS
    # ========================================================================
    loss_thresholds = np.linspace(loss_threshold_start, loss_threshold_end, num_loss_thresholds)
    
    reductions = []
    for threshold in loss_thresholds:
        std_epochs_for_threshold = []
        np_epochs_for_threshold = []
        
        # Find epochs to reach threshold for Standard NN (loss goes down, so <= threshold)
        for loss_list in all_loss_std_np:
            loss_arr = np.array(loss_list)
            epoch = np.argmax(loss_arr <= threshold) if np.any(loss_arr <= threshold) else len(loss_arr)
            std_epochs_for_threshold.append(epoch + 1)
        
        # Find epochs to reach threshold for NP NN
        for loss_list in all_loss_np_np:
            loss_arr = np.array(loss_list)
            epoch = np.argmax(loss_arr <= threshold) if np.any(loss_arr <= threshold) else len(loss_arr)
            np_epochs_for_threshold.append(epoch + 1)
        
        # Compute average reduction for this threshold (as percentage)
        mean_std_epochs = np.mean(std_epochs_for_threshold)
        mean_np_epochs = np.mean(np_epochs_for_threshold)
        reduction = abs((mean_np_epochs - mean_std_epochs)) / mean_std_epochs * 100
        reductions.append(reduction)
    
    avg_loss_reduction = np.mean(reductions)
    
    # ========================================================================
    # 3. AUAC (Area Under Accuracy Curve)
    # ========================================================================
    # Compute mean accuracy curves
    std_acc_array = np.array([[to_numpy(a) for a in acc_list] for acc_list in all_acc_std_np])
    np_acc_array = np.array([[to_numpy(a) for a in acc_list] for acc_list in all_acc_np_np])
    
    mean_acc_std = np.mean(std_acc_array, axis=0)
    mean_acc_np = np.mean(np_acc_array, axis=0)
    
    # Compute AUAC using trapezoidal rule
    # AUAC represents the integral of accuracy over epochs
    n_epochs = len(mean_acc_std)
    auac_standard = np.trapz(mean_acc_std, dx=1)
    auac_npnn = np.trapz(mean_acc_np, dx=1)
    auac_difference = auac_npnn - auac_standard
    
    # ========================================================================
    # 4. AULC (Area Under Loss Curve)
    # ========================================================================
    # Compute mean loss curves
    std_loss_array = np.array([[to_numpy(l) for l in loss_list] for loss_list in all_loss_std_np])
    np_loss_array = np.array([[to_numpy(l) for l in loss_list] for loss_list in all_loss_np_np])
    
    mean_loss_std = np.mean(std_loss_array, axis=0)
    mean_loss_np = np.mean(np_loss_array, axis=0)
    
    # Compute AULC using trapezoidal rule
    # For loss, lower is better, so a smaller AULC is better
    aulc_standard = np.trapz(mean_loss_std, dx=1)
    aulc_npnn = np.trapz(mean_loss_np, dx=1)
    aulc_difference = aulc_standard - aulc_npnn  # Positive means NPNN has lower loss (better)
    
    # ========================================================================
    # RETURN ALL 8 METRICS
    # ========================================================================
    metrics = {
        'avg_accuracy_speedup': avg_accuracy_speedup,
        'avg_loss_reduction': avg_loss_reduction,
        'auac_standard': auac_standard,
        'auac_npnn': auac_npnn,
        'auac_difference': auac_difference,
        'aulc_standard': aulc_standard,
        'aulc_npnn': aulc_npnn,
        'aulc_difference': aulc_difference
    }
    
    print_convergence_metrics(metrics)

def print_convergence_metrics(metrics):
    """
    Pretty print the convergence metrics.
    
    Args:
        metrics: Dictionary returned by compute_convergence_metrics
    """
    print("\n" + "="*70)
    print("CONVERGENCE METRICS SUMMARY")
    print("="*70)
    
    print("\n1. ACCURACY CONVERGENCE SPEEDUP")
    print("-" * 70)
    print(f"   Average Speedup (%):               {metrics['avg_accuracy_speedup']:+.2f}%")
    print(f"   → NPNN reaches accuracy milestones {abs(metrics['avg_accuracy_speedup']):.2f}%")
    print(f"     {'faster' if metrics['avg_accuracy_speedup'] < 0 else 'slower'} than Standard NN on average")
    
    print("\n2. LOSS CONVERGENCE REDUCTION")
    print("-" * 70)
    print(f"   Average Reduction (%):             {metrics['avg_loss_reduction']:+.2f}%")
    print(f"   → NPNN reaches loss milestones {abs(metrics['avg_loss_reduction']):.2f}%")
    print(f"     {'faster' if metrics['avg_loss_reduction'] < 0 else 'slower'} than Standard NN on average")
    
    print("\n3. AREA UNDER ACCURACY CURVE (AUAC)")
    print("-" * 70)
    print(f"   Standard NN AUAC:                  {metrics['auac_standard']:.4f}")
    print(f"   NPNN AUAC:                         {metrics['auac_npnn']:.4f}")
    print(f"   Difference (NPNN - Standard):      {metrics['auac_difference']:+.4f}")
    print(f"   → NPNN has {abs(metrics['auac_difference']):.4f} {'higher' if metrics['auac_difference'] > 0 else 'lower'} cumulative accuracy")
    
    print("\n4. AREA UNDER LOSS CURVE (AULC)")
    print("-" * 70)
    print(f"   Standard NN AULC:                  {metrics['aulc_standard']:.4f}")
    print(f"   NPNN AULC:                         {metrics['aulc_npnn']:.4f}")
    print(f"   Difference (Standard - NPNN):      {metrics['aulc_difference']:+.4f}")
    print(f"   → NPNN has {abs(metrics['aulc_difference']):.4f} {'lower' if metrics['aulc_difference'] > 0 else 'higher'} cumulative loss")
    
    print("\n" + "="*70)
    print("THE 8 KEY NUMBERS:")
    print("="*70)
    print(f"1. Average Accuracy Speedup:       {metrics['avg_accuracy_speedup']:+.2f}%")
    print(f"2. Average Loss Reduction:         {metrics['avg_loss_reduction']:+.2f}%")
    print(f"3. AUAC Standard:                  {metrics['auac_standard']:.4f}")
    print(f"4. AUAC NPNN:                      {metrics['auac_npnn']:.4f}")
    print(f"5. AUAC Difference:                {metrics['auac_difference']:+.4f}")
    print(f"6. AULC Standard:                  {metrics['aulc_standard']:.4f}")
    print(f"7. AULC NPNN:                      {metrics['aulc_npnn']:.4f}")
    print(f"8. AULC Difference:                {metrics['aulc_difference']:+.4f}")
    print("="*70 + "\n")

def pruning_stats():
    arch = [784, 10, 10, 10] #placeholder
    for i in range(0, 60):
        model = NPNeuralNetwork(arch)
        model.load_model(i)
        
        total_connections = 0
        total_pruned = 0
        if i == 0:
            print("\n####################################################### MNIST Stats:")
        elif i == 10:
            print("\n####################################################### Fashion-MNIST Stats:")
        elif i == 20:
            print("\n####################################################### CIFAR-10 Stats:")
        elif i == 50:
            print("\n####################################################### CIFAR-100 Stats:")
        print(f"\n=== Model {i} ===")
        for layer_idx, mask in enumerate(model.masks):
            layer_total = mask.size
            layer_active = np.sum(mask)
            layer_pruned = layer_total - layer_active
            
            total_connections += layer_total
            total_pruned += layer_pruned
            if i < 30 or (i >= 50 and i < 60): 
                print(f"Layer {layer_idx}: {int(layer_pruned)}/{layer_total} pruned ({(layer_pruned/layer_total)*100:.2f}%)")
        if i < 30 or (i >= 50 and i < 60):
            print(f"Total: {int(total_pruned)}/{total_connections} pruned ({(total_pruned/total_connections)*100:.2f}%)")