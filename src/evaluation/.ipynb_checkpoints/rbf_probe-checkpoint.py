# Adapted from ATTPC Latent Repo
#
# Evaluates pre-extracted embeddings using an RBF-kernel support-vector
# classifier and generates learning curves across multiple training-set sizes.

import click
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def generate_log_train_sizes(min_size=50, max_size=16000, num_points=20):
    """Generate logarithmically spaced training-set sizes."""
    log_min = np.log10(min_size)
    log_max = np.log10(max_size)

    log_sizes = np.linspace(log_min, log_max, num_points)
    sizes = np.round(10**log_sizes).astype(int)

    # Remove duplicate sizes caused by rounding.
    sizes = np.unique(sizes)

    # Ensure the exact requested endpoints are present.
    if min_size not in sizes:
        sizes = np.append([min_size], sizes)

    if max_size not in sizes:
        sizes = np.append(sizes, [max_size])

    return np.sort(sizes)


def parse_gamma(gamma):
    """
    Convert the command-line gamma value into a form accepted by SVC.

    Valid inputs:
        scale
        auto
        any positive floating-point value
    """
    gamma_lower = gamma.lower()

    if gamma_lower in {"scale", "auto"}:
        return gamma_lower

    try:
        gamma_value = float(gamma)
    except ValueError as exc:
        raise click.BadParameter(
            "--gamma must be 'scale', 'auto', or a positive number."
        ) from exc

    if gamma_value <= 0:
        raise click.BadParameter(
            "A numeric --gamma value must be greater than zero."
        )

    return gamma_value


def create_rbf_svc(c_value, gamma_value, cache_size, seed):
    """Construct an RBF-kernel support-vector classifier."""
    return SVC(
        C=c_value,
        kernel="rbf",
        gamma=gamma_value,
        cache_size=cache_size,
        probability=False,
        random_state=seed,
    )


def create_learning_curve_visualizations(results, results_folder):
    """Generate and save the learning-curve metrics plot."""
    plt.figure(figsize=(10, 6))

    sizes = results["train_sizes"]

    plt.errorbar(
        sizes,
        results["train_accuracies_mean"],
        yerr=results["train_accuracies_std"],
        label="Train Accuracy",
        fmt="-o",
        capsize=5,
        color="#4CAF50",
    )

    plt.errorbar(
        sizes,
        results["test_accuracies_mean"],
        yerr=results["test_accuracies_std"],
        label="Test Accuracy",
        fmt="-s",
        capsize=5,
        color="#2196F3",
    )

    plt.xscale("log")
    plt.xlabel("Training Set Size (Log Scale)", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)

    plt.title(
        "RBF SVC Learning Curve",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    plt.grid(
        True,
        which="both",
        linestyle="--",
        alpha=0.5,
    )

    plt.legend(fontsize=10)

    plt.savefig(
        os.path.join(results_folder, "learning_curve.png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def create_final_model_visualizations(
    y_test,
    y_pred,
    classes,
    class_names,
    results_folder,
):
    """
    Generate and save the confusion matrix and classification report for the
    final RBF SVC model.
    """
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=classes,
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.ylabel("Actual Class", fontsize=11)
    plt.xlabel("Predicted Class", fontsize=11)

    plt.title(
        "Final RBF SVC Confusion Matrix",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    plt.savefig(
        os.path.join(
            results_folder,
            "final_confusion_matrix.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=classes,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report_dict).transpose().round(4)

    # Save classification report as CSV.
    report_df.to_csv(
        os.path.join(
            results_folder,
            "classification_report.csv",
        )
    )

    # Save classification report as PNG.
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    table = ax.table(
        cellText=report_df.values,
        rowLabels=report_df.index,
        colLabels=report_df.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.title(
        "Classification Report",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    plt.savefig(
        os.path.join(
            results_folder,
            "classification_report.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def create_performance_table(results, results_folder):
    """Create and save a detailed learning-curve performance table."""
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.axis("tight")
    ax.axis("off")

    headers = [
        "Training Size",
        "Train Acc (Mean)",
        "Train Acc (Std)",
        "Test Acc (Mean)",
        "Test Acc (Std)",
        "Overfitting Gap",
    ]

    table_data = []

    for i, size in enumerate(results["train_sizes"]):
        gap = (
            results["train_accuracies_mean"][i]
            - results["test_accuracies_mean"][i]
        )

        row = [
            f"{size:,}",
            f"{results['train_accuracies_mean'][i]:.4f}",
            f"{results['train_accuracies_std'][i]:.4f}",
            f"{results['test_accuracies_mean'][i]:.4f}",
            f"{results['test_accuracies_std'][i]:.4f}",
            f"{gap:.4f}",
        ]

        table_data.append(row)

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Style the header.
    for column_index in range(len(headers)):
        table[(0, column_index)].set_facecolor("#4CAF50")
        table[(0, column_index)].set_text_props(
            weight="bold",
            color="white",
        )

    # Add alternating row shading.
    for row_index in range(1, len(table_data) + 1):
        for column_index in range(len(headers)):
            if row_index % 2 == 0:
                table[(row_index, column_index)].set_facecolor(
                    "#f0f0f0"
                )

    plt.title(
        "Detailed Learning Curve Results",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    plt.savefig(
        os.path.join(
            results_folder,
            "performance_table.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


@click.command()
@click.option(
    "--name",
    default="O16",
    type=click.STRING,
    show_default=True,
    help=(
        "Name or profile identifier for the run, such as "
        "O16_simclr_best."
    ),
)
@click.option(
    "--test-size",
    default=0.2,
    type=click.FloatRange(
        min=0.0,
        max=1.0,
        min_open=True,
        max_open=True,
    ),
    show_default=True,
    help="Fraction of the dataset reserved for testing.",
)
@click.option(
    "--seed",
    default=None,
    type=click.INT,
    help="Random seed for reproducibility.",
)
@click.option(
    "--c",
    "c_value",
    default=1.0,
    type=click.FloatRange(
        min=0.0,
        min_open=True,
    ),
    show_default=True,
    help=(
        "SVC regularization parameter. Larger values penalize "
        "training errors more strongly."
    ),
)
@click.option(
    "--gamma",
    default="scale",
    type=click.STRING,
    show_default=True,
    help=(
        "RBF-kernel coefficient. Use 'scale', 'auto', "
        "or a positive number."
    ),
)
@click.option(
    "--cache-size",
    default=4096.0,
    type=click.FloatRange(
        min=0.0,
        min_open=True,
    ),
    show_default=True,
    help="Maximum SVC kernel-cache size in megabytes.",
)
@click.option(
    "--min-train-size",
    default=50,
    type=click.IntRange(min=1),
    show_default=True,
    help="Minimum training-set size.",
)
@click.option(
    "--max-train-size",
    default=16000,
    type=click.IntRange(min=1),
    show_default=True,
    help="Maximum training-set size.",
)
@click.option(
    "--num-size-points",
    default=20,
    type=click.IntRange(min=2),
    show_default=True,
    help="Number of logarithmically spaced training sizes.",
)
@click.option(
    "--cv-folds",
    default=3,
    type=click.IntRange(min=1),
    show_default=True,
    help=(
        "Number of independently sampled training subsets "
        "evaluated at each size."
    ),
)
@click.argument(
    "features-file",
    type=click.Path(
        exists=True,
        dir_okay=False,
    ),
)
@click.argument(
    "labels-file",
    type=click.Path(
        exists=True,
        dir_okay=False,
    ),
)
def rbf_probe_evaluation(
    name,
    test_size,
    seed,
    c_value,
    gamma,
    cache_size,
    min_train_size,
    max_train_size,
    num_size_points,
    cv_folds,
    features_file,
    labels_file,
):
    """
    Perform RBF SVC evaluation with learning-curve analysis using
    pre-extracted NumPy feature embeddings and corresponding target labels.

    FEATURES_FILE should have shape:

        (number_of_samples, feature_dimension)

    LABELS_FILE should have shape:

        (number_of_samples,)
    """
    print("Loading features and labels...")

    global_features = np.load(features_file)
    combined_track_labels = np.load(labels_file)

    global_features = np.asarray(global_features)
    combined_track_labels = np.asarray(
        combined_track_labels
    ).reshape(-1)

    print(f"Features shape: {global_features.shape}")
    print(f"Labels shape: {combined_track_labels.shape}")

    # Validate input arrays before beginning potentially expensive SVC fits.
    if global_features.ndim != 2:
        raise ValueError(
            "Features must be a two-dimensional array with shape "
            "(number_of_samples, feature_dimension)."
        )

    if global_features.shape[0] != combined_track_labels.shape[0]:
        raise ValueError(
            "Features and labels contain different numbers of samples: "
            f"{global_features.shape[0]} feature rows and "
            f"{combined_track_labels.shape[0]} labels."
        )

    if not np.isfinite(global_features).all():
        raise ValueError(
            "The feature array contains NaN or infinite values."
        )

    unique_classes, class_counts = np.unique(
        combined_track_labels,
        return_counts=True,
    )

    if len(unique_classes) < 2:
        raise ValueError(
            "RBF SVC classification requires at least two classes."
        )

    gamma_value = parse_gamma(gamma)

    # Ensure there is always a concrete integer seed.
    base_seed = (
        seed
        if seed is not None
        else int(
            np.random.default_rng().integers(
                low=0,
                high=100000,
            )
        )
    )

    print(f"Using random seed baseline: {base_seed}")

    # Preserve the original output directory structure for compatibility
    # with existing scripts and comparisons.
    master_results_dir = "./linear_probe_results"

    results_folder = os.path.join(
        master_results_dir,
        f"{name}_learning_curve",
    )

    os.makedirs(
        results_folder,
        exist_ok=True,
    )

    print(
        f"Target results directory established: "
        f"{results_folder}"
    )

    # Create one fixed train/test split. The test set remains unchanged
    # across all learning-curve sizes and folds.
    (
        X_train_full,
        X_test,
        y_train_full,
        y_test,
    ) = train_test_split(
        global_features,
        combined_track_labels,
        test_size=test_size,
        random_state=base_seed,
        stratify=combined_track_labels,
    )

    print(f"Full training set size: {X_train_full.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    max_possible_size = min(
        max_train_size,
        X_train_full.shape[0],
    )

    if max_possible_size < max_train_size:
        print(
            f"Warning: Requested max training size "
            f"({max_train_size}) exceeds available data."
        )
        print(
            f"Using maximum available size: "
            f"{max_possible_size}"
        )

        max_train_size = max_possible_size

    number_of_classes = len(
        np.unique(y_train_full)
    )

    if min_train_size < number_of_classes:
        print(
            f"Warning: Requested minimum training size "
            f"({min_train_size}) is smaller than the number "
            f"of classes ({number_of_classes})."
        )

        min_train_size = number_of_classes

        print(
            f"Using minimum training size: "
            f"{min_train_size}"
        )

    if min_train_size > max_train_size:
        raise ValueError(
            f"Minimum training size ({min_train_size}) exceeds "
            f"the maximum usable training size "
            f"({max_train_size})."
        )

    train_sizes = generate_log_train_sizes(
        min_train_size,
        max_train_size,
        num_size_points,
    )

    print(f"Training sizes to test: {train_sizes}")
    print("Standardizing features...")

    learning_curve_results = {
        "train_sizes": train_sizes.tolist(),
        "train_accuracies_mean": [],
        "train_accuracies_std": [],
        "test_accuracies_mean": [],
        "test_accuracies_std": [],
        "detailed_results": [],
    }

    print(
        f"\nStarting learning curve analysis with "
        f"{len(train_sizes)} training sizes..."
    )

    print("=" * 60)

    # Evaluate each requested training-set size.
    for i, train_size in enumerate(train_sizes):
        print(
            f"\nProgress: {i + 1}/{len(train_sizes)} "
            f"- Training size: {train_size}"
        )

        train_accs = []
        test_accs = []
        support_vector_counts = []
        support_vectors_per_class = []

        # Run multiple independently sampled subsets at each size.
        for fold in range(cv_folds):
            current_seed = base_seed + fold

            if train_size >= len(X_train_full):
                X_train_subset = X_train_full
                y_train_subset = y_train_full
            else:
                (
                    X_train_subset,
                    _,
                    y_train_subset,
                    _,
                ) = train_test_split(
                    X_train_full,
                    y_train_full,
                    train_size=int(train_size),
                    stratify=y_train_full,
                    random_state=current_seed,
                )

            # Fit preprocessing only on the current training subset.
            #
            # This avoids allowing a small learning-curve subset to use
            # feature means and variances calculated from training examples
            # outside that subset.
            subset_scaler = StandardScaler()

            X_train_subset_scaled = subset_scaler.fit_transform(
                X_train_subset
            )

            X_test_subset_scaled = subset_scaler.transform(
                X_test
            )

            rbf_probe = create_rbf_svc(
                c_value=c_value,
                gamma_value=gamma_value,
                cache_size=cache_size,
                seed=current_seed,
            )

            rbf_probe.fit(
                X_train_subset_scaled,
                y_train_subset,
            )

            train_pred = rbf_probe.predict(
                X_train_subset_scaled
            )

            test_pred = rbf_probe.predict(
                X_test_subset_scaled
            )

            train_acc = accuracy_score(
                y_train_subset,
                train_pred,
            )

            test_acc = accuracy_score(
                y_test,
                test_pred,
            )

            train_accs.append(train_acc)
            test_accs.append(test_acc)

            support_vector_counts.append(
                int(rbf_probe.n_support_.sum())
            )

            support_vectors_per_class.append(
                rbf_probe.n_support_.astype(int).tolist()
            )

        train_acc_mean = float(
            np.mean(train_accs)
        )

        train_acc_std = float(
            np.std(train_accs)
        )

        test_acc_mean = float(
            np.mean(test_accs)
        )

        test_acc_std = float(
            np.std(test_accs)
        )

        learning_curve_results[
            "train_accuracies_mean"
        ].append(train_acc_mean)

        learning_curve_results[
            "train_accuracies_std"
        ].append(train_acc_std)

        learning_curve_results[
            "test_accuracies_mean"
        ].append(test_acc_mean)

        learning_curve_results[
            "test_accuracies_std"
        ].append(test_acc_std)

        detailed_result = {
            "train_size": int(train_size),
            "train_accuracy_mean": train_acc_mean,
            "train_accuracy_std": train_acc_std,
            "test_accuracy_mean": test_acc_mean,
            "test_accuracy_std": test_acc_std,
            "train_accuracies": [
                float(accuracy)
                for accuracy in train_accs
            ],
            "test_accuracies": [
                float(accuracy)
                for accuracy in test_accs
            ],
            "support_vector_counts": support_vector_counts,
            "support_vector_count_mean": float(
                np.mean(support_vector_counts)
            ),
            "support_vector_count_std": float(
                np.std(support_vector_counts)
            ),
            "support_vectors_per_class": (
                support_vectors_per_class
            ),
        }

        learning_curve_results[
            "detailed_results"
        ].append(detailed_result)

        # Preserve the original console-output format.
        print(
            f"  Train Acc: "
            f"{train_acc_mean:.4f} ± {train_acc_std:.4f}"
        )

        print(
            f"  Test Acc:  "
            f"{test_acc_mean:.4f} ± {test_acc_std:.4f}"
        )

    print("\n" + "=" * 60)
    print("Learning curve analysis completed!")

    full_results = {
        "dataset_info": {
            "features_source": os.path.abspath(
                features_file
            ),
            "labels_source": os.path.abspath(
                labels_file
            ),
            "total_samples": int(
                len(global_features)
            ),
            "feature_dim": int(
                global_features.shape[1]
            ),
            "classes": unique_classes.tolist(),
            "class_counts": class_counts.tolist(),
        },
        "experiment_config": {
            "classifier": "sklearn.svm.SVC",
            "kernel": "rbf",
            "test_size": test_size,
            "C": c_value,
            "gamma": gamma_value,
            "cache_size_mb": cache_size,
            "requested_seed": seed,
            "base_seed": base_seed,
            "cv_folds": cv_folds,
            "min_train_size": int(
                min_train_size
            ),
            "max_train_size": int(
                max_train_size
            ),
            "num_size_points": num_size_points,
            "feature_scaling": (
                "StandardScaler fitted independently "
                "on each learning-curve subset"
            ),
        },
        "learning_curve_results": (
            learning_curve_results
        ),
    }

    with open(
        os.path.join(
            results_folder,
            "learning_curve_results.json",
        ),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            full_results,
            file,
            indent=4,
        )

    create_learning_curve_visualizations(
        learning_curve_results,
        results_folder,
    )

    create_performance_table(
        learning_curve_results,
        results_folder,
    )

    # Train one final RBF SVC using all available training samples.
    print(
        "\nTraining final model with full training data..."
    )

    final_scaler = StandardScaler()

    X_train_full_scaled = final_scaler.fit_transform(
        X_train_full
    )

    X_test_scaled = final_scaler.transform(
        X_test
    )

    final_rbf_probe = create_rbf_svc(
        c_value=c_value,
        gamma_value=gamma_value,
        cache_size=cache_size,
        seed=base_seed,
    )

    final_rbf_probe.fit(
        X_train_full_scaled,
        y_train_full,
    )

    y_train_final_pred = final_rbf_probe.predict(
        X_train_full_scaled
    )

    y_test_final_pred = final_rbf_probe.predict(
        X_test_scaled
    )

    final_train_acc = accuracy_score(
        y_train_full,
        y_train_final_pred,
    )

    final_test_acc = accuracy_score(
        y_test,
        y_test_final_pred,
    )

    print(
        f"Final model - Train Acc: "
        f"{final_train_acc:.4f}, "
        f"Test Acc: {final_test_acc:.4f}"
    )

    final_classes = final_rbf_probe.classes_

    class_names = [
        f"{class_label}-track"
        for class_label in final_classes
    ]

    create_final_model_visualizations(
        y_test=y_test,
        y_pred=y_test_final_pred,
        classes=final_classes,
        class_names=class_names,
        results_folder=results_folder,
    )

    final_model_results = {
        "classifier": "sklearn.svm.SVC",
        "kernel": "rbf",
        "C": c_value,
        "gamma": gamma_value,
        "final_train_accuracy": float(
            final_train_acc
        ),
        "final_test_accuracy": float(
            final_test_acc
        ),
        "number_of_support_vectors": int(
            final_rbf_probe.n_support_.sum()
        ),
        "support_vectors_per_class": (
            final_rbf_probe.n_support_
            .astype(int)
            .tolist()
        ),
        "classes": final_classes.tolist(),
    }

    with open(
        os.path.join(
            results_folder,
            "final_model_results.json",
        ),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            final_model_results,
            file,
            indent=4,
        )

    print(
        f"\nAll results saved to: "
        f"{results_folder}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_test_final_pred,
            labels=final_classes,
            target_names=class_names,
            zero_division=0,
        )
    )

    return {
        "learning_curve_results": (
            learning_curve_results
        ),
        "final_model": final_rbf_probe,
        "scaler": final_scaler,
        "final_train_accuracy": (
            final_train_acc
        ),
        "final_test_accuracy": (
            final_test_acc
        ),
    }


if __name__ == "__main__":
    rbf_probe_evaluation()