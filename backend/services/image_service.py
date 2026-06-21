"""
Image preprocessing service.
"""

import io
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from backend.config.settings import get_settings

def preprocess_image(image_bytes: io.BytesIO):
    """
    Convert in-memory image buffer into model input tensor.
    """
    settings = get_settings()
    
    # Open image from bytes
    image = Image.open(image_bytes)
    
    # Convert to RGB if necessary (e.g. RGBA or Grayscale)
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    # Resize to model input size
    image = image.resize(settings.IMAGE_SIZE)
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # Match the training pipeline in ml/src/preprocessing.py exactly.
    image_array = preprocess_input(image_array.astype("float32"))
    
    # Expand dims for batch size
    image_tensor = np.expand_dims(image_array, axis=0)
    
    return image_tensor

def preprocess_image_array(image_array: np.ndarray) -> np.ndarray:
    """
    Normalize a raw RGB uint8 image array using the production inference path.
    Used by tests to enforce train/inference preprocessing parity.
    """
    return preprocess_input(image_array.astype("float32"))
