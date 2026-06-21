"""
Model loading utilities.
"""

import json
import time
import tensorflow as tf
from backend.config.settings import get_settings
from backend.utils.logger import get_logger
from backend.utils.exceptions import ModelNotLoadedError

logger = get_logger(__name__)

MODEL = None
CLASS_NAMES = None
MODEL_METADATA = None

def init_model():
    """
    Load trained MobileNetV2 model and class names.
    Executed once during application startup.
    """
    global MODEL, CLASS_NAMES, MODEL_METADATA
    
    # If already loaded, skip
    if MODEL is not None and CLASS_NAMES is not None:
        return
        
    settings = get_settings()
    
    logger.info("Initializing ML Model...")
    start_time = time.time()
    
    try:
        MODEL = tf.keras.models.load_model(settings.MODEL_PATH)
        
        with open(settings.CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
            CLASS_NAMES = json.load(file)

        MODEL_METADATA = {
            "model_version": settings.MODEL_VERSION,
            "architecture": "MobileNetV2 transfer learning",
            "framework": f"TensorFlow/Keras {tf.__version__}",
            "model_path": settings.MODEL_PATH,
            "class_names_path": settings.CLASS_NAMES_PATH,
            "class_count": len(CLASS_NAMES),
            "input_shape": list(MODEL.input_shape[1:]),
            "output_shape": list(MODEL.output_shape[1:]),
            "preprocessing": settings.PREPROCESSING_MODE,
            "confidence_threshold": settings.CONFIDENCE_THRESHOLD,
            "top_k_predictions": settings.TOP_K_PREDICTIONS,
            "calibration_enabled": settings.CALIBRATION_ENABLED,
            "confidence_temperature": settings.CONFIDENCE_TEMPERATURE,
        }
            
        elapsed = time.time() - start_time
        logger.info(f"Model and class names loaded successfully in {elapsed:.2f}s")
        
    except Exception as e:
        logger.error(f"Failed to load model or class names: {e}")
        raise ModelNotLoadedError("Failed to initialize the ML model.") from e

def get_model():
    """Get the loaded model or raise an error if not loaded."""
    if MODEL is None:
        raise ModelNotLoadedError("Model is not initialized.")
    return MODEL

def get_class_names():
    """Get the loaded class names or raise an error if not loaded."""
    if CLASS_NAMES is None:
        raise ModelNotLoadedError("Class names are not initialized.")
    return CLASS_NAMES

def get_model_metadata():
    """Return metadata for the loaded model."""
    if MODEL_METADATA is None:
        raise ModelNotLoadedError("Model metadata is not initialized.")
    return MODEL_METADATA
