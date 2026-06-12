"""
Model loading utilities.
"""

from pathlib import Path
import json
import tensorflow as tf

ARTIFACT_DIR = Path(__file__).parent / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "crop_disease_mobilenetv2.keras"
CLASS_NAMES_PATH = ARTIFACT_DIR / "class_names.json"


def load_model():
    """
    Load trained MobileNetV2 model.
    """

    return tf.keras.models.load_model(MODEL_PATH)


def load_class_names():
    """
    Load disease class labels.
    """

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


MODEL = load_model()

CLASS_NAMES = load_class_names()