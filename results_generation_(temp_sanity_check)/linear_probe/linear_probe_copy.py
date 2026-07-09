import click
import csv
import numpy as np
import torch
import os
import json

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def generate_log_train_sizes(min_size=50, max_size=16000, num_points=20):
    """Generate logarithmically spaced training set sizes."""
    log_min = np.log10(min_size)
    log_max = np.log10(max_size)
    log_sizes = np.linspace(log_min, log_max, num_points)
    sizes = np.round(10**log_sizes).astype(int)
    # Remove duplicates and ensure we have the exact min and max
    sizes = np.unique(sizes)
    if min_size not in sizes:
        sizes = np.append([min_size], sizes)
    if max_size not in sizes:
        sizes = np.append(sizes, [max_size])
    return np.sort(sizes)


def accuracy_score(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true, y_pred, labels=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    label_to_idx = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[label_to_idx[true_label], label_to_idx[pred_label]] += 1
    return cm


def classification_report(y_true, y_pred, target_names=None, output_dict=False):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    if target_names is None:
        target_names = [str(label) for label in labels]

    rows = {}
    total_support = 0
    weighted_precision = weighted_recall = weighted_f1 = 0.0
    macro_precision = macro_recall = macro_f1 = 0.0

    for label, name in zip(labels, target_names):
        tp = int(np.sum((y_true == label) & (y_pred == label)))
        fp = int(np.sum((y_true != label) & (y_pred == label)))
        fn = int(np.sum((y_true == label) & (y_pred != label)))
        support = int(np.sum(y_true == label))

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)

        rows[name] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support,
        }
        total_support += support
        weighted_precision += precision * support
        weighted_recall += recall * support
        weighted_f1 += f1 * support
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

    n_classes = max(1, len(labels))
    accuracy = accuracy_score(y_true, y_pred)
    rows["accuracy"] = accuracy
    rows["macro avg"] = {
        "precision": macro_precision / n_classes,
        "recall": macro_recall / n_classes,
        "f1-score": macro_f1 / n_classes,
        "support": total_support,
    }
    rows["weighted avg"] = {
        "precision": weighted_precision / max(1, total_support),
        "recall": weighted_recall / max(1, total_support),
        "f1-score": weighted_f1 / max(1, total_support),
        "support": total_support,
    }

    if output_dict:
        return rows

    lines = [f"{'':>14} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}"]
    for name in target_names:
        row = rows[name]
        lines.append(
            f"{name:>14} {row['precision']:10.4f} {row['recall']:10.4f} "
            f"{row['f1-score']:10.4f} {row['support']:10.0f}"
        )
    lines.append(f"\n{'accuracy':>14} {'':>10} {'':>10} {accuracy:10.4f} {total_support:10.0f}")
    for name in ["macro avg", "weighted avg"]:
        row = rows[name]
        lines.append(
            f"{name:>14} {row['precision']:10.4f} {row['recall']:10.4f} "
            f"{row['f1-score']:10.4f} {row['support']:10.0f}"
        )
    return "\n".join(lines)


def stratified_split(X, y, test_size=0.2, random_state=0):
    rng = np.random.default_rng(random_state)
    train_indices, test_indices = [], []
    for cls in np.unique(y):
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)
        n_test = max(1, int(round(len(cls_indices) * test_size)))
        test_indices.extend(cls_indices[:n_test])
        train_indices.extend(cls_indices[n_test:])
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def stratified_subset(X, y, train_size, random_state=0):
    train_size = int(train_size)
    classes, counts = np.unique(y, return_counts=True)
    if train_size < len(classes):
        raise ValueError(f"train_size={train_size} is smaller than number of classes={len(classes)}")
    if train_size >= len(y):
        return X, y

    rng = np.random.default_rng(random_state)
    proportions = counts / counts.sum()
    desired = np.floor(proportions * train_size).astype(int)
    desired = np.maximum(desired, 1)
    desired = np.minimum(desired, counts)

    while desired.sum() < train_size:
        room = counts - desired
        if room.max() <= 0:
            break
        add_to = int(np.argmax(room))
        desired[add_to] += 1
    while desired.sum() > train_size:
        removable = np.where(desired > 1)[0]
        remove_from = removable[np.argmax(desired[removable])]
        desired[remove_from] -= 1

    selected = []
    for cls, n_cls in zip(classes, desired):
        cls_indices = np.where(y == cls)[0]
        selected.extend(rng.choice(cls_indices, size=int(n_cls), replace=False))
    selected = np.array(selected)
    rng.shuffle(selected)
    return X[selected], y[selected]


def standardize_train_test(X_train, X_test):
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std, {"mean": mean, "std": std}


def fit_linear_probe(X_train, y_train, regularization, random_state=0):
    torch.manual_seed(int(random_state))
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    model = torch.nn.Linear(X_t.shape[1], int(np.max(y_train)) + 1)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=100,
        line_search_fn="strong_wolfe",
    )
    l2_weight = 1.0 / max(float(regularization), 1e-8)

    def closure():
        optimizer.zero_grad()
        logits = model(X_t)
        loss = torch.nn.functional.cross_entropy(logits, y_t)
        l2 = sum(torch.sum(param * param) for param in model.parameters())
        loss = loss + 0.5 * l2_weight * l2 / max(1, X_t.shape[0])
        loss.backward()
        return loss

    optimizer.step(closure)
    return model


def predict_linear_probe(model, X):
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        return model(X_t).argmax(dim=1).cpu().numpy()


def create_learning_curve_visualizations(results, results_folder):
        """
        Generate and save the learning curve metrics plot.
        """
        if plt is None:
            print("matplotlib is not installed; skipping learning_curve.png")
            return
        plt.figure(figsize=(10, 6))
        sizes = results['train_sizes']
        
        # Plot training and testing accuracies with standard deviation error bars
        plt.errorbar(sizes, results['train_accuracies_mean'], yerr=results['train_accuracies_std'], 
                    label='Train Accuracy', fmt='-o', capsize=5, color='#4CAF50')
        plt.errorbar(sizes, results['test_accuracies_mean'], yerr=results['test_accuracies_std'], 
                    label='Test Accuracy', fmt='-s', capsize=5, color='#2196F3')
        
        plt.xscale('log')
        plt.xlabel('Training Set Size (Log Scale)', fontsize=11)
        plt.ylabel('Accuracy', fontsize=11)
        plt.title('Linear Probe Learning Curve', fontsize=14, fontweight='bold', pad=15)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend(fontsize=10)
        
        plt.savefig(f'{results_folder}/learning_curve.png', dpi=300, bbox_inches='tight')
        plt.close()


def create_final_model_visualizations(X_test, y_test, y_pred, y_prob, class_names, results_folder, model):
    """
    Generate and save a confusion matrix for the final trained model.
    """
    cm = confusion_matrix(y_test, y_pred)
    np.savetxt(
        f'{results_folder}/final_confusion_matrix.csv',
        cm,
        delimiter=',',
        fmt='%d',
    )

    report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

    headers = ['label', 'precision', 'recall', 'f1-score', 'support']
    table_rows = []
    for label, metrics in report_dict.items():
        if isinstance(metrics, dict):
            table_rows.append([
                label,
                f"{metrics.get('precision', 0.0):.4f}",
                f"{metrics.get('recall', 0.0):.4f}",
                f"{metrics.get('f1-score', 0.0):.4f}",
                f"{metrics.get('support', 0.0):.0f}",
            ])
        else:
            table_rows.append([label, '', '', f"{float(metrics):.4f}", ''])

    with open(f'{results_folder}/classification_report.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(table_rows)

    if plt is None:
        print("matplotlib is not installed; saved CSV metrics and skipped final plot PNGs")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='Actual Class',
        xlabel='Predicted Class',
        title='Final Model Confusion Matrix',
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    plt.savefig(f'{results_folder}/final_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save as PNG
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    table = ax.table(
        cellText=[row[1:] for row in table_rows],
        rowLabels=[row[0] for row in table_rows],
        colLabels=headers[1:],
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.title('Classification Report', fontsize=14, fontweight='bold', pad=15)
    plt.savefig(f'{results_folder}/classification_report.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_table(results, results_folder):
    """Create a detailed performance table."""
    if plt is None:
        print("matplotlib is not installed; skipping performance_table.png")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    headers = ['Training Size', 'Train Acc (Mean)', 'Train Acc (Std)', 
               'Test Acc (Mean)', 'Test Acc (Std)', 'Overfitting Gap']
    
    table_data = []
    for i, size in enumerate(results['train_sizes']):
        gap = results['train_accuracies_mean'][i] - results['test_accuracies_mean'][i]
        row = [
            f"{size:,}",
            f"{results['train_accuracies_mean'][i]:.4f}",
            f"{results['train_accuracies_std'][i]:.4f}",
            f"{results['test_accuracies_mean'][i]:.4f}",
            f"{results['test_accuracies_std'][i]:.4f}",
            f"{gap:.4f}"
        ]
        table_data.append(row)
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Detailed Learning Curve Results', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(f'{results_folder}/performance_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
@click.command()
@click.option('--name', default='O16', type=click.STRING, help='The name/profile identifier for the run (e.g. O16, Mg22, C16)')
@click.option('--test-size', default=0.2, type=click.FLOAT, help='Fraction of data to use for testing')
@click.option('--seed', default=None, type=click.INT, help='Random seed for reproducibility')
@click.option('--regularization', default=1.0, type=click.FLOAT, help='Regularization strength for logistic regression')
@click.option('--min-train-size', default=50, type=click.INT, help='Minimum training set size')
@click.option('--max-train-size', default=16000, type=click.INT, help='Maximum training set size')
@click.option('--num-size-points', default=20, type=click.INT, help='Number of training sizes to test')
@click.option('--cv-folds', default=3, type=click.INT, help='Number of cross-validation folds for each size')
@click.argument('features-file', type=click.Path(exists=True))
@click.argument('labels-file', type=click.Path(exists=True))
def linear_probe_evaluation(name, test_size, seed, regularization, min_train_size, 
                            max_train_size, num_size_points, cv_folds, features_file, labels_file):
    """
    Perform linear probe evaluation with learning curve analysis using pre-extracted
    flat NumPy feature embeddings and corresponding target labels.
    """

    print("Loading features and labels...")

    global_features = np.load(features_file) #Expected shape: (N,D)
    combined_track_labels = np.load(labels_file) #Expected shape: (N,)
    unique_classes = np.unique(combined_track_labels)
    if len(unique_classes) < 2:
        raise ValueError(
            "Linear probing requires at least two classes. "
            "If labels are all -1, rerun extract_latents.py in split mode."
        )
    label_to_index = {label: index for index, label in enumerate(unique_classes)}
    combined_track_labels = np.array(
        [label_to_index[label] for label in combined_track_labels],
        dtype=np.int64,
    )

    print(f"Features shape: {global_features.shape}")
    print(f"Labels shape: {combined_track_labels.shape}")

    # Ensure we have a valid integer base seed for fold-math, even if seed=None
    base_seed = seed if seed is not None else np.random.randint(0, 100000)
    print(f"Using random seed baseline: {base_seed}")
    
    # 1. Define a global master results directory at the project root
    master_results_dir = "./linear_probe_results"
    
    # 2. Build a unique sub-folder path for this specific run using the --name parameter
    results_folder = os.path.join(master_results_dir, f"{name}_learning_curve")
    
    # 3. Create the folder tree automatically (exist_ok=True prevents crashes if re-run)
    os.makedirs(results_folder, exist_ok=True)
    
    print(f"Target results directory established: {results_folder}")
    
    # Split data into train and test sets (fixed test set for consistent evaluation)
    X_train_full, X_test, y_train_full, y_test = stratified_split(
        global_features, combined_track_labels,
        test_size=test_size,
        random_state=base_seed,
    )
    
    print(f"Full training set size: {X_train_full.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Check if we have enough data for the maximum training size
    max_possible_size = min(max_train_size, X_train_full.shape[0])
    if max_possible_size < max_train_size:
        print(f"Warning: Requested max training size ({max_train_size}) exceeds available data.")
        print(f"Using maximum available size: {max_possible_size}")
        max_train_size = max_possible_size
    
    # Generate logarithmic training sizes
    train_sizes = generate_log_train_sizes(min_train_size, max_train_size, num_size_points)
    print(f"Training sizes to test: {train_sizes}")
    
    # Standardize features (fit on full training set)
    print("Standardizing features...")
    X_train_full_scaled, X_test_scaled, scaler = standardize_train_test(X_train_full, X_test)
    
    # Initialize results storage
    learning_curve_results = {
        'train_sizes': train_sizes.tolist(),
        'train_accuracies_mean': [],
        'train_accuracies_std': [],
        'test_accuracies_mean': [],
        'test_accuracies_std': [],
        'detailed_results': []
    }
    
    print(f"\nStarting learning curve analysis with {len(train_sizes)} training sizes...")
    print("=" * 60)

    
    # For each training size
    for i, train_size in enumerate(train_sizes):
        print(f"\nProgress: {i+1}/{len(train_sizes)} - Training size: {train_size}")
        
        train_accs = []
        test_accs = []
        
        # Perform multiple runs with different random subsets for robustness
        for fold in range(cv_folds):
            # Randomly sample training data of the specified size
            current_seed = base_seed + fold
            if train_size >= len(X_train_full_scaled):
                # Use all available training data
                X_train_subset = X_train_full_scaled
                y_train_subset = y_train_full
            else:
                
                X_train_subset, y_train_subset = stratified_subset(
                    X_train_full_scaled,
                    y_train_full,
                    train_size=train_size,
                    random_state=current_seed,
                )
                
            
            # Train linear probe
            linear_probe = fit_linear_probe(
                X_train_subset,
                y_train_subset,
                regularization=regularization,
                random_state=current_seed,
            )
            
            # Evaluate
            train_pred = predict_linear_probe(linear_probe, X_train_subset)
            test_pred = predict_linear_probe(linear_probe, X_test_scaled)
            
            train_acc = accuracy_score(y_train_subset, train_pred)
            test_acc = accuracy_score(y_test, test_pred)
            
            train_accs.append(train_acc)
            test_accs.append(test_acc)
        
        # Calculate statistics
        train_acc_mean = np.mean(train_accs)
        train_acc_std = np.std(train_accs)
        test_acc_mean = np.mean(test_accs)
        test_acc_std = np.std(test_accs)
        
        # Store results
        learning_curve_results['train_accuracies_mean'].append(train_acc_mean)
        learning_curve_results['train_accuracies_std'].append(train_acc_std)
        learning_curve_results['test_accuracies_mean'].append(test_acc_mean)
        learning_curve_results['test_accuracies_std'].append(test_acc_std)
        
        # Store detailed results
        detailed_result = {
            'train_size': int(train_size),
            'train_accuracy_mean': float(train_acc_mean),
            'train_accuracy_std': float(train_acc_std),
            'test_accuracy_mean': float(test_acc_mean),
            'test_accuracy_std': float(test_acc_std),
            'train_accuracies': [float(acc) for acc in train_accs],
            'test_accuracies': [float(acc) for acc in test_accs]
        }
        learning_curve_results['detailed_results'].append(detailed_result)
        
        print(f"  Train Acc: {train_acc_mean:.4f} ± {train_acc_std:.4f}")
        print(f"  Test Acc:  {test_acc_mean:.4f} ± {test_acc_std:.4f}")
    
    print("\n" + "=" * 60)
    print("Learning curve analysis completed!")
    
    # Save detailed results
    full_results = {
        'dataset_info': {
            'features_source': features_file,
            'labels_source': labels_file,
            'total_samples': len(global_features),
            'feature_dim': global_features.shape[1],
        },
        'experiment_config': {
            'test_size': test_size,
            'regularization': regularization,
            'seed': seed,
            'cv_folds': cv_folds,
            'min_train_size': min_train_size,
            'max_train_size': max_train_size,
            'num_size_points': num_size_points
        },
        'learning_curve_results': learning_curve_results
    }
    
    with open(f'{results_folder}/learning_curve_results.json', 'w') as f:
        json.dump(full_results, f, indent=4)
    
    # Create visualizations
    create_learning_curve_visualizations(learning_curve_results, results_folder)
    create_performance_table(learning_curve_results, results_folder)
    # Train final model with full training data for additional analysis
    print("\nTraining final model with full training data...")
    final_linear_probe = fit_linear_probe(
        X_train_full_scaled,
        y_train_full,
        regularization=regularization,
        random_state=base_seed,
    )
    
    # Final evaluation
    y_train_final_pred = predict_linear_probe(final_linear_probe, X_train_full_scaled)
    y_test_final_pred = predict_linear_probe(final_linear_probe, X_test_scaled)
    y_test_final_prob = None
    
    final_train_acc = accuracy_score(y_train_full, y_train_final_pred)
    final_test_acc = accuracy_score(y_test, y_test_final_pred)
    
    print(f"Final model - Train Acc: {final_train_acc:.4f}, Test Acc: {final_test_acc:.4f}")
    
    # Create additional visualizations for final model
    class_names = [f'{int(cls)}-track' for cls in unique_classes]
    create_final_model_visualizations(
        X_test_scaled, y_test, y_test_final_pred, y_test_final_prob,
        class_names, results_folder, final_linear_probe
    )
    
    print(f"\nAll results saved to: {results_folder}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_test_final_pred, target_names=class_names))
    
    return {
        'learning_curve_results': learning_curve_results,
        'final_model': final_linear_probe,
        'scaler': scaler,
        'final_train_accuracy': final_train_acc,
        'final_test_accuracy': final_test_acc
    }



    

if __name__ == '__main__':
    linear_probe_evaluation()
