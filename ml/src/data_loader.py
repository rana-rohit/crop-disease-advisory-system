"""Dataset loading utilities for the crop disease model.

This module is intentionally small and importable so the same class ordering can
be reused by training, evaluation, export, and eventually the Flask backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import tensorflow as tf


# Shared image settings used by the PlantVillage MobileNetV2 pipeline.
DEFAULT_IMAGE_SIZE: tuple[int, int] = (224, 224)
DEFAULT_BATCH_SIZE = 32
DEFAULT_SEED = 42

# Dataset split folder names expected under the configured dataset root.
TRAIN_DIR_NAME = "train"
VALIDATION_DIR_NAME = "val"


@dataclass(frozen=True)
class DatasetBundle:
    """Container returned by the dataset loader."""

    train: tf.data.Dataset
    validation: tf.data.Dataset
    class_names: list[str]


def _as_path(path: str | Path) -> Path:
    """Convert path-like input into a resolved Path."""

    return Path(path).expanduser().resolve()


def resolve_dataset_dirs(dataset_root: str | Path) -> tuple[Path, Path]:
    """Resolve and validate the PlantVillage train and validation directories."""

    root_path = _as_path(dataset_root)
    train_dir = root_path / TRAIN_DIR_NAME
    validation_dir = root_path / VALIDATION_DIR_NAME

    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root_path}")
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not validation_dir.is_dir():
        raise FileNotFoundError(f"Validation directory not found: {validation_dir}")

    return train_dir, validation_dir


def validate_class_names(
    train_class_names: Sequence[str],
    validation_class_names: Sequence[str],
) -> None:
    """Ensure train and validation splits contain the same class folders."""

    train_set = set(train_class_names)
    validation_set = set(validation_class_names)

    if train_set != validation_set:
        missing_in_validation = sorted(train_set - validation_set)
        missing_in_train = sorted(validation_set - train_set)
        raise ValueError(
            "Train and validation class folders do not match. "
            f"Missing in validation: {missing_in_validation}; "
            f"missing in train: {missing_in_train}."
        )


def load_datasets(
    dataset_root: str | Path,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = DEFAULT_SEED,
) -> DatasetBundle:
    """Load train and validation datasets from a PlantVillage directory."""

    train_dir, validation_dir = resolve_dataset_dirs(dataset_root)

    # Categorical labels match the softmax output used by MobileNetV2 training.
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    # Validation data is not shuffled so metrics and reports are reproducible.
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        validation_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    train_class_names = list(train_dataset.class_names)
    validation_class_names = list(validation_dataset.class_names)
    validate_class_names(train_class_names, validation_class_names)

    return DatasetBundle(
        train=train_dataset,
        validation=validation_dataset,
        class_names=train_class_names,
    )
