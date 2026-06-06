"""Train a MobileNetV2 crop disease classifier on PlantVillage images."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import tensorflow as tf

try:
    from .data_loader import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_IMAGE_SIZE,
        DEFAULT_SEED,
        load_datasets,
    )
    from .preprocessing import prepare_datasets
except ImportError:  # Allows `python ml/src/train.py` in Colab or VS Code.
    from data_loader import (  # type: ignore[no-redef]
        DEFAULT_BATCH_SIZE,
        DEFAULT_IMAGE_SIZE,
        DEFAULT_SEED,
        load_datasets,
    )
    from preprocessing import prepare_datasets  # type: ignore[no-redef]


# Training defaults are configurable from CLI arguments.
DEFAULT_EPOCHS = 20
DEFAULT_FINE_TUNE_EPOCHS = 0
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_FINE_TUNE_LEARNING_RATE = 1e-5
DEFAULT_DROPOUT_RATE = 0.3
DEFAULT_OUTPUT_DIR = Path("ml") / "outputs"
DEFAULT_MODEL_FILENAME = "crop_disease_mobilenetv2.keras"
DEFAULT_WEIGHTS_FILENAME = "best_mobilenetv2.weights.h5"
DEFAULT_HISTORY_FILENAME = "training_history.json"
DEFAULT_CLASS_NAMES_FILENAME = "class_names.json"


def build_mobilenetv2_model(
    num_classes: int,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
) -> tf.keras.Model:
    """Build a transfer-learning classifier using MobileNetV2."""

    if num_classes <= 0:
        raise ValueError("num_classes must be greater than zero.")

    # Inputs are already normalized by preprocessing.normalize_images.
    input_shape = (*image_size, 3)
    inputs = tf.keras.Input(shape=input_shape, name="leaf_image")

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    # A small classification head keeps the model fast enough for web inference.
    features = base_model(inputs, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(
        features
    )
    features = tf.keras.layers.Dropout(dropout_rate, name="dropout")(features)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="disease_probabilities",
    )(features)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="crop_disease_mobilenetv2")


def compile_model(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    """Compile the model with optimizer, loss, and classification metrics."""

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy"),
        ],
    )
    return model


def build_callbacks(
    weights_path: Path,
    monitor: str = "val_accuracy",
) -> list[tf.keras.callbacks.Callback]:
    """Create callbacks for checkpointing and stable training."""

    weights_path.parent.mkdir(parents=True, exist_ok=True)

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(weights_path),
            monitor=monitor,
            save_best_only=True,
            save_weights_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=5,
            mode="max",
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def save_json(data: Any, output_path: Path) -> None:
    """Write JSON data with a stable, human-readable format."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def merge_histories(*histories: tf.keras.callbacks.History) -> dict[str, list[float]]:
    """Merge Keras History objects into one serializable dictionary."""

    merged: dict[str, list[float]] = {}

    for history in histories:
        for metric_name, values in history.history.items():
            merged.setdefault(metric_name, []).extend(float(value) for value in values)

    return merged


def fine_tune_model(
    model: tf.keras.Model,
    fine_tune_at: int,
    learning_rate: float,
) -> tf.keras.Model:
    """Unfreeze the MobileNetV2 backbone from a selected layer onward."""

    base_model = next(
        layer for layer in model.layers if isinstance(layer, tf.keras.Model)
    )
    base_model.trainable = True

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    return compile_model(model, learning_rate)


def train_model(args: argparse.Namespace) -> dict[str, Path | list[str]]:
    """Run the full training workflow and save model artifacts."""

    dataset_bundle = load_datasets(
        dataset_root=args.data_dir,
        image_size=tuple(args.image_size),
        batch_size=args.batch_size,
        seed=args.seed,
    )

    train_dataset, validation_dataset = prepare_datasets(
        train_dataset=dataset_bundle.train,
        validation_dataset=dataset_bundle.validation,
        seed=args.seed,
        use_augmentation=not args.disable_augmentation,
        cache=args.cache_dataset,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    model_path = output_dir / "models" / args.model_filename
    weights_path = output_dir / "models" / args.weights_filename
    history_path = output_dir / "metrics" / args.history_filename
    class_names_path = output_dir / "models" / args.class_names_filename

    model = build_mobilenetv2_model(
        num_classes=len(dataset_bundle.class_names),
        image_size=tuple(args.image_size),
        dropout_rate=args.dropout_rate,
    )
    compile_model(model, args.learning_rate)

    # Phase 1 trains only the classification head.
    initial_history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        callbacks=build_callbacks(weights_path),
    )

    histories = [initial_history]

    # Phase 2 optionally fine-tunes the upper backbone layers.
    if args.fine_tune_epochs > 0:
        fine_tune_model(model, args.fine_tune_at, args.fine_tune_learning_rate)
        fine_tune_history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=args.epochs + args.fine_tune_epochs,
            initial_epoch=args.epochs,
            callbacks=build_callbacks(weights_path),
        )
        histories.append(fine_tune_history)

    if weights_path.exists():
        model.load_weights(weights_path)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    save_json(dataset_bundle.class_names, class_names_path)
    save_json(merge_histories(*histories), history_path)

    return {
        "model_path": model_path,
        "weights_path": weights_path,
        "history_path": history_path,
        "class_names_path": class_names_path,
        "class_names": dataset_bundle.class_names,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for local or Google Colab training."""

    parser = argparse.ArgumentParser(
        description="Train MobileNetV2 for crop disease detection."
    )
    parser.add_argument(
        "--data-dir",
        default=os.getenv("PLANTVILLAGE_DATA_DIR"),
        required=os.getenv("PLANTVILLAGE_DATA_DIR") is None,
        help="Dataset root containing train/ and val/ folders.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("ML_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)),
        help="Directory where model, weights, and metrics will be saved.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--fine-tune-epochs", type=int, default=DEFAULT_FINE_TUNE_EPOCHS)
    parser.add_argument("--fine-tune-at", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--image-size", type=int, nargs=2, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--fine-tune-learning-rate",
        type=float,
        default=DEFAULT_FINE_TUNE_LEARNING_RATE,
    )
    parser.add_argument("--dropout-rate", type=float, default=DEFAULT_DROPOUT_RATE)
    parser.add_argument("--model-filename", default=DEFAULT_MODEL_FILENAME)
    parser.add_argument("--weights-filename", default=DEFAULT_WEIGHTS_FILENAME)
    parser.add_argument("--history-filename", default=DEFAULT_HISTORY_FILENAME)
    parser.add_argument("--class-names-filename", default=DEFAULT_CLASS_NAMES_FILENAME)
    parser.add_argument("--cache-dataset", action="store_true")
    parser.add_argument("--disable-augmentation", action="store_true")

    return parser.parse_args()


def main() -> None:
    """Train the model from command-line arguments."""

    artifacts = train_model(parse_args())
    print("Training complete.")
    print(f"Model: {artifacts['model_path']}")
    print(f"Best weights: {artifacts['weights_path']}")
    print(f"History: {artifacts['history_path']}")
    print(f"Class names: {artifacts['class_names_path']}")


if __name__ == "__main__":
    main()
