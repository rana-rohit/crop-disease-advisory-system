"""
Custom exceptions for the application.
"""

class AppError(Exception):
    """Base application exception."""
    status_code = 500

class ImageValidationError(AppError):
    """Raised when an uploaded image fails validation."""
    status_code = 400

class ModelNotLoadedError(AppError):
    """Raised when the model fails to load or is uninitialized."""
    status_code = 503

class PredictionError(AppError):
    """Raised when inference fails."""
    status_code = 500
