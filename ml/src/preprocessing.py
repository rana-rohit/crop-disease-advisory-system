"""Preprocessing helpers for the crop disease image pipeline."""

from __future__ import annotations

import tensorflow as tf

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# TensorFlow chooses the best available prefetch parallelism at runtime.
AUTOTUNE = tf.data.AUTOTUNE


def build_data_augmentation_layer(seed: int = 42) -> tf.keras.Sequential:
    """Create lightweight augmentation layers for training images."""

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(0.08, seed=seed),
            tf.keras.layers.RandomZoom(0.12, seed=seed),
            tf.keras.layers.RandomContrast(0.12, seed=seed),
        ],
        name="leaf_data_augmentation",
    )


def normalize_images(
    images: tf.Tensor,
    labels: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Normalize image tensors for the ImageNet-pretrained MobileNetV2 model."""

    images = tf.cast(images, tf.float32)
    return preprocess_input(images), labels


def augment_images(
    images: tf.Tensor,
    labels: tf.Tensor,
    augmentation_layer: tf.keras.Model,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply image augmentation only during the training pipeline."""

    return augmentation_layer(images, training=True), labels


def prepare_training_dataset(
    dataset: tf.data.Dataset,
    augmentation_layer: tf.keras.Model | None = None,
    cache: bool = False,
) -> tf.data.Dataset:
    """Apply optional augmentation, normalization, and prefetching to training data."""

    prepared_dataset = dataset

    if cache:
        prepared_dataset = prepared_dataset.cache()

    if augmentation_layer is not None:
        prepared_dataset = prepared_dataset.map(
            lambda images, labels: augment_images(images, labels, augmentation_layer),
            num_parallel_calls=AUTOTUNE,
        )

    prepared_dataset = prepared_dataset.map(normalize_images, num_parallel_calls=AUTOTUNE)
    return prepared_dataset.prefetch(AUTOTUNE)


def prepare_validation_dataset(
    dataset: tf.data.Dataset,
    cache: bool = False,
) -> tf.data.Dataset:
    """Apply deterministic normalization and prefetching to validation data."""

    prepared_dataset = dataset

    if cache:
        prepared_dataset = prepared_dataset.cache()

    prepared_dataset = prepared_dataset.map(normalize_images, num_parallel_calls=AUTOTUNE)
    return prepared_dataset.prefetch(AUTOTUNE)


def prepare_datasets(
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
    seed: int = 42,
    use_augmentation: bool = True,
    cache: bool = False,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Prepare train and validation datasets for MobileNetV2 training."""

    augmentation_layer = build_data_augmentation_layer(seed) if use_augmentation else None
    prepared_train = prepare_training_dataset(train_dataset, augmentation_layer, cache)
    prepared_validation = prepare_validation_dataset(validation_dataset, cache)

    return prepared_train, prepared_validation
