import click
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json


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


def create_learning_curve_visualizations(results, results_folder):
        """
        Generate and save the learning curve metrics plot.
        """
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
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.ylabel('Actual Class', fontsize=11)
    plt.xlabel('Predicted Class', fontsize=11)
    plt.title('Final Model Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    
    plt.savefig(f'{results_folder}/final_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    df = pd.DataFrame(report_dict).transpose().round(4)

    # Save as CSV
    df.to_csv(f'{results_folder}/classification_report.csv')

    # Save as PNG
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    table = ax.table(cellText=df.values, rowLabels=df.index, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.title('Classification Report', fontsize=14, fontweight='bold', pad=15)
    plt.savefig(f'{results_folder}/classification_report.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_table(results, results_folder):
    """Create a detailed performance table."""
    
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

    global_features = np.load(features_file) #Expected shape: (N,1024)
    combined_track_labels = np.load(labels_file) #Expected shape: (N,)
    unique_classes = np.unique(combined_track_labels)

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
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        global_features, combined_track_labels, 
        test_size=test_size, random_state=base_seed, 
        stratify=combined_track_labels
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
    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)
    
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
                
                X_train_subset, _, y_train_subset, _ = train_test_split(
                X_train_full_scaled, y_train_full,
                train_size=train_size,
                stratify=y_train_full,
                random_state=current_seed
                )                   
                
            
            # Train linear probe
            linear_probe = LogisticRegression(
                C=regularization, 
                random_state=current_seed,
                max_iter=5000
            )
            linear_probe.fit(X_train_subset, y_train_subset)
            
            # Evaluate
            train_pred = linear_probe.predict(X_train_subset)
            test_pred = linear_probe.predict(X_test_scaled)
            
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
    final_linear_probe = LogisticRegression(
        C=regularization, 
        random_state=base_seed,
        max_iter=5000
    )
    final_linear_probe.fit(X_train_full_scaled, y_train_full)
    
    # Final evaluation
    y_train_final_pred = final_linear_probe.predict(X_train_full_scaled)
    y_test_final_pred = final_linear_probe.predict(X_test_scaled)
    y_test_final_prob = final_linear_probe.predict_proba(X_test_scaled)
    
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