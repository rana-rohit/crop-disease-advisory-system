"""
Application settings and configuration.
"""

import os
from dataclasses import dataclass, field

@dataclass
class Settings:
    """Central configuration for the application."""
    
    # Model configuration
    MODEL_PATH: str = os.getenv("CROP_DISEASE_MODEL_PATH", "backend/model/artifacts/crop_disease_mobilenetv2.keras")
    CLASS_NAMES_PATH: str = os.getenv("CROP_DISEASE_CLASS_NAMES_PATH", "backend/model/artifacts/class_names.json")
    DISEASE_INFO_PATH: str = os.getenv("CROP_DISEASE_INFO_PATH", "backend/disease_data/disease_info.json")
    
    # Inference configuration
    IMAGE_SIZE: tuple = (224, 224)
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "80.0"))
    TOP_K_PREDICTIONS: int = int(os.getenv("TOP_K_PREDICTIONS", "3"))
    MODEL_VERSION: str = os.getenv("MODEL_VERSION", "mobilenetv2-plantvillage-v1")
    PREPROCESSING_MODE: str = "tf.keras.applications.mobilenet_v2.preprocess_input"
    CALIBRATION_ENABLED: bool = os.getenv("CALIBRATION_ENABLED", "False").lower() in ("true", "1", "t")
    CONFIDENCE_TEMPERATURE: float = float(os.getenv("CONFIDENCE_TEMPERATURE", "1.0"))
    
    # File upload validation
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "5"))
    MAX_CONTENT_LENGTH: int = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    MAX_IMAGE_PIXELS: int = int(os.getenv("MAX_IMAGE_PIXELS", "12000000"))
    ALLOWED_EXTENSIONS: set = field(default_factory=lambda: {"jpg", "jpeg", "png", "webp"})
    
    # Application configuration
    FLASK_ENV: str = os.getenv("FLASK_ENV", "production")
    FLASK_DEBUG: bool = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1", "t")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "5000"))
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "*")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# Singleton instance
_settings = None

def get_settings() -> Settings:
    """Get the application settings."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
