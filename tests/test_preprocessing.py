"""
Tests for preprocessing parity between training and inference.
"""

import io

import numpy as np
import tensorflow as tf
from PIL import Image

from backend.services.image_service import preprocess_image, preprocess_image_array
from ml.src.preprocessing import normalize_images


def test_inference_preprocessing_matches_training_pipeline():
    """Backend inference must use the same MobileNetV2 preprocessing as training."""
    raw = np.arange(224 * 224 * 3, dtype=np.uint8).reshape((224, 224, 3))

    backend_processed = preprocess_image_array(raw)
    training_processed, _ = normalize_images(tf.convert_to_tensor(raw[None, ...]), tf.constant([[1.0]]))

    np.testing.assert_allclose(
        backend_processed,
        training_processed.numpy()[0],
        rtol=1e-6,
        atol=1e-6,
    )


def test_preprocess_image_returns_model_input_shape():
    image = Image.new("RGB", (256, 128), color="green")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    tensor = preprocess_image(buffer)

    assert tensor.shape == (1, 224, 224, 3)
    assert tensor.dtype == np.float32
    assert tensor.min() >= -1.0
    assert tensor.max() <= 1.0
