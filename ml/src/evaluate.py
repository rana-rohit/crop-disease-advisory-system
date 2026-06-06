"""Evaluate a trained crop disease classifier and save metrics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

try:
    from .data_loader import DEFAULT_BATCH_SIZE, DEFAULT_IMAGE_SIZE, load_datasets
    from .preprocessing import prepare_validation_dataset
except ImportError:  # Allows `python ml/src/evaluate.py` in Colab or VS Code.
    from data_loader import DEFAULT_BATCH_SIZE, DEFAULT_IMAGE_SIZE, load_datasets  # type: ignore[no-redef]
    from preprocessing import prepare_validation_dataset  # type: ignore[no-redef]


# Evaluation defaults keep generated files under the ML output tree.
DEFAULT_OUTPUT_DIR = Path("ml") / "outputs"
DEFAULT_MODEL_PATH = DEFAULT_OUTPUT_DIR / "models" / "crop_disease_mobilenetv2.keras"
DEFAULT_REPORT_FILENAME = "classification_report.json"
DEFAULT_CONFUSION_MATRIX_FILENAME = "confusion_matrix.json"
DEFAULT_EVALUATION_FILENAME = "evaluation_metrics.json"


def save_json(data: object, output_path: Path) -> None:
    """Write JSON metrics to disk."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def collect_predictions(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect true and predicted class indexes from a validation dataset."""

    true_labels: list[np.ndarray] = []
    predicted_labels: list[np.ndarray] = []

    for images, labels in dataset:
        probabilities = model.predict(images, verbose=0)
        true_labels.append(np.argmax(labels.numpy(), axis=1))
        predicted_labels.append(np.argmax(probabilities, axis=1))

    return np.concatenate(true_labels), np.concatenate(predicted_labels)


def evaluate_model(args: argparse.Namespace) -> dict[str, Path | dict[str, float]]:
    """Load the model, evaluate validation data, and save detailed reports."""

    dataset_bundle = load_datasets(
        dataset_root=args.data_dir,
        image_size=tuple(args.image_size),
        batch_size=args.batch_size,
    )
    validation_dataset = prepare_validation_dataset(
        dataset_bundle.validation,
        cache=args.cache_dataset,
    )

    model = tf.keras.models.load_model(args.model_path)
    metric_values = model.evaluate(validation_dataset, verbose=1)
    metric_names = model.metrics_names
    evaluation_metrics = {
        metric_name: float(metric_value)
        for metric_name, metric_value in zip(metric_names, metric_values)
    }

    y_true, y_pred = collect_predictions(model, validation_dataset)
    report = classification_report(
        y_true,
        y_pred,
        target_names=dataset_bundle.class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred).tolist()

    output_dir = Path(args.output_dir).expanduser().resolve()
    evaluation_path = output_dir / "metrics" / args.evaluation_filename
    report_path = output_dir / "metrics" / args.report_filename
    matrix_path = output_dir / "metrics" / args.confusion_matrix_filename

    save_json(evaluation_metrics, evaluation_path)
    save_json(report, report_path)
    save_json(
        {
            "class_names": dataset_bundle.class_names,
            "matrix": matrix,
        },
        matrix_path,
    )

    return {
        "evaluation_path": evaluation_path,
        "report_path": report_path,
        "confusion_matrix_path": matrix_path,
        "metrics": evaluation_metrics,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""

    parser = argparse.ArgumentParser(
        description="Evaluate a trained crop disease MobileNetV2 model."
    )
    parser.add_argument(
        "--data-dir",
        default=os.getenv("PLANTVILLAGE_DATA_DIR"),
        required=os.getenv("PLANTVILLAGE_DATA_DIR") is None,
        help="Dataset root containing train/ and val/ folders.",
    )
    parser.add_argument(
        "--model-path",
        default=os.getenv("CROP_DISEASE_MODEL_PATH", str(DEFAULT_MODEL_PATH)),
        help="Path to the trained Keras model.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("ML_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)),
        help="Directory where evaluation metrics will be saved.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--image-size", type=int, nargs=2, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--cache-dataset", action="store_true")
    parser.add_argument("--report-filename", default=DEFAULT_REPORT_FILENAME)
    parser.add_argument(
        "--confusion-matrix-filename",
        default=DEFAULT_CONFUSION_MATRIX_FILENAME,
    )
    parser.add_argument("--evaluation-filename", default=DEFAULT_EVALUATION_FILENAME)

    return parser.parse_args()


def main() -> None:
    """Evaluate the model from command-line arguments."""

    artifacts = evaluate_model(parse_args())
    print("Evaluation complete.")
    print(f"Metrics: {artifacts['evaluation_path']}")
    print(f"Classification report: {artifacts['report_path']}")
    print(f"Confusion matrix: {artifacts['confusion_matrix_path']}")


if __name__ == "__main__":
    main()
