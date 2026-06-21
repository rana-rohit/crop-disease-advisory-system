"""
Prediction service.
"""

import numpy as np
from datetime import datetime, timezone

from backend.config.settings import get_settings
from backend.model.model_loader import get_model, get_class_names
from backend.services.image_service import preprocess_image
from backend.services.disease_info_service import get_disease_info
from backend.services.explainability_service import generate_gradcam_explanation
from backend.services.calibration_service import apply_temperature_scaling
from backend.services.inference_stats_service import (
    record_prediction,
    record_prediction_error,
    start_timer,
)

def predict_disease(image_buffer, include_xai: bool = False):
    """
    Predict disease from image buffer and return full advisory report.
    """
    settings = get_settings()
    start_time = start_timer()
    model = get_model()
    class_names = get_class_names()

    try:
        image_tensor = preprocess_image(image_buffer)

        predictions = model.predict(image_tensor, verbose=0)

        confidence_scores = predictions[0]
        if settings.CALIBRATION_ENABLED:
            confidence_scores = apply_temperature_scaling(
                confidence_scores,
                settings.CONFIDENCE_TEMPERATURE,
            )

        k = settings.TOP_K_PREDICTIONS
        top_indices = np.argsort(confidence_scores)[-k:][::-1]

        top_predictions = [
            {
                "disease": class_names[i],
                "confidence": round(float(confidence_scores[i] * 100), 4),
            }
            for i in top_indices
        ]

        primary_prediction = top_predictions[0]
        primary_class = primary_prediction["disease"]
        primary_confidence = primary_prediction["confidence"]
        top_index = int(top_indices[0])

        metadata = {
            "model_version": settings.MODEL_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "preprocessing": settings.PREPROCESSING_MODE,
            "confidence_threshold": settings.CONFIDENCE_THRESHOLD,
            "top_k": settings.TOP_K_PREDICTIONS,
            "calibration_enabled": settings.CALIBRATION_ENABLED,
            "confidence_temperature": settings.CONFIDENCE_TEMPERATURE,
        }

        if primary_confidence < settings.CONFIDENCE_THRESHOLD:
            result = {
                "prediction": {
                    "disease": "Unknown or Unrecognized Image",
                    "confidence": primary_confidence,
                },
                "top_predictions": top_predictions,
                "advisory": {
                    "crop": None,
                    "disease": "Unknown",
                    "symptoms": "N/A",
                    "treatment": "N/A",
                    "prevention": "Please upload a clear image of a supported crop leaf.",
                },
                "severity": None,
                "metadata": metadata,
            }
            record_prediction(start_time, include_xai=include_xai)
            return result

        advisory_info = get_disease_info(primary_class)

        result = {
            "prediction": primary_prediction,
            "top_predictions": top_predictions,
            "advisory": advisory_info,
            "severity": None,
            "metadata": metadata,
        }
        
        if include_xai:
            xai_data = generate_gradcam_explanation(model, image_tensor, top_index, primary_class)
            result["xai"] = xai_data

        record_prediction(start_time, include_xai=include_xai)
        return result
    except Exception:
        record_prediction_error()
        raise
