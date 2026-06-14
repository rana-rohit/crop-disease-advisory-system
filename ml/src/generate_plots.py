"""
Generate evaluation plots for project report.
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path("ml") / "outputs"
METRICS_DIR = OUTPUT_DIR / "metrics"
PLOTS_DIR = OUTPUT_DIR / "plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def clean_class_name(class_name: str) -> str:
    """
    Convert dataset class names into report-friendly labels.
    """

    return (
        class_name
        .replace("___", " : ")
        .replace("_", " ")
    )

def load_json(filename: str):
    """
    Load JSON file.
    """

    path = METRICS_DIR / filename

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def plot_accuracy(history: dict):
    """
    Plot training and validation accuracy.
    """

    epochs = range(
        1,
        len(history["accuracy"]) + 1,
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        history["accuracy"],
        marker="o",
        linewidth=2,
        label="Training Accuracy",
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        marker="s",
        linewidth=2,
        label="Validation Accuracy",
    )

    plt.title(
        "Training vs Validation Accuracy",
        fontsize=16,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "accuracy_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_loss(history: dict):
    """
    Plot training and validation loss.
    """

    epochs = range(
        1,
        len(history["loss"]) + 1,
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        history["loss"],
        marker="o",
        linewidth=2,
        label="Training Loss",
    )

    plt.plot(
        epochs,
        history["val_loss"],
        marker="s",
        linewidth=2,
        label="Validation Loss",
    )

    plt.title(
        "Training vs Validation Loss",
        fontsize=16,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "loss_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_confusion_matrix():
    """
    Plot normalized confusion matrix.
    """

    confusion_data = load_json(
        "confusion_matrix.json"
    )

    matrix = np.array(
        confusion_data["matrix"]
    )

    matrix = matrix.astype(float)

    row_sums = matrix.sum(
        axis=1,
        keepdims=True,
    )

    matrix = np.divide(
        matrix,
        row_sums,
        where=row_sums != 0,
    )
  
    plt.figure(figsize=(16, 14))

    heatmap = plt.imshow(
        matrix,
        cmap="Greens",
        vmin=0,
        vmax=1,
    )

    cbar = plt.colorbar(heatmap)
    cbar.set_label(
        "Classification Accuracy",
        rotation=90,
    )

    plt.title(
        "Normalized Confusion Matrix",
        fontsize=18,
    )

    plt.xlabel(
        "Predicted Class",
        fontsize=12,
    )

    plt.ylabel(
        "True Class",
        fontsize=12,
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_best_classes():
    """
    Top 10 classes by F1-score.
    """

    report = load_json(
        "classification_report.json"
    )

    rows = []

    for class_name, metrics in report.items():

        if (
            isinstance(metrics, dict)
            and "f1-score" in metrics
        ):
            rows.append(
                [
                    clean_class_name(
                        class_name
                    ),
                    metrics["f1-score"],
                ]
            )

    dataframe = pd.DataFrame(
        rows,
        columns=["Class", "F1"],
    )

    dataframe = dataframe.sort_values(
        "F1",
        ascending=False,
    ).head(10)

    plt.figure(figsize=(12, 8))

    plt.barh(
        dataframe["Class"],
        dataframe["F1"],
    )

    plt.title(
        "Top 10 Performing Disease Classes",
        fontsize=16,
    )

    plt.xlabel("F1 Score")

    plt.xlim(
        0.95,
        1.01,
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "top10_classes.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_worst_classes():
    """
    Worst performing classes by F1-score.
    """

    report = load_json(
        "classification_report.json"
    )

    rows = []

    for class_name, metrics in report.items():

        if (
            isinstance(metrics, dict)
            and "f1-score" in metrics
        ):
            rows.append(
                [
                    clean_class_name(
                        class_name
                    ),
                    metrics["f1-score"],
                ]
            )

    dataframe = pd.DataFrame(
        rows,
        columns=["Class", "F1"],
    )

    dataframe = dataframe.sort_values(
        "F1",
        ascending=True,
    ).head(10)

    plt.figure(figsize=(12, 8))

    plt.barh(
        dataframe["Class"],
        dataframe["F1"],
    )

    plt.title(
        "Most Challenging Disease Classes",
        fontsize=16,
    )

    plt.xlabel("F1 Score")

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "bottom10_classes.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

def plot_performance_summary():
    """
    Overall model performance.
    """

    report = load_json(
        "classification_report.json"
    )

    evaluation = load_json(
        "evaluation_metrics.json"
    )

    metrics = {
        "Accuracy":
            evaluation["compile_metrics"],

        "Macro F1":
            report["macro avg"]["f1-score"],

        "Weighted F1":
            report["weighted avg"]["f1-score"],
    }

    plt.figure(figsize=(8, 5))

    plt.bar(
        list(metrics.keys()),
        list(metrics.values()),
    )

    plt.ylim(
        0.85,
        1.0,
    )

    plt.ylabel("Score")

    plt.title(
        "Overall Model Performance",
        fontsize=16,
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "performance_summary.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

def main():
    """
    Generate all report plots.
    """

    history = load_json(
        "training_history.json"
    )

    plot_accuracy(history)

    plot_loss(history)

    plot_confusion_matrix()

    plot_best_classes()

    plot_worst_classes()
    
    plot_performance_summary()

    print("\nPlots generated successfully.\n")

    print(
        f"Saved to: {PLOTS_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()