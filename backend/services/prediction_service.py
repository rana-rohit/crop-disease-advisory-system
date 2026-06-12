"""
Prediction service.
"""

import numpy as np

from backend.model.model_loader import (
    MODEL,
    CLASS_NAMES,
)

from backend.services.image_service import (
    preprocess_image,
)


def predict_disease(image_path):
    """
    Predict disease from leaf image.
    """

    image = preprocess_image(image_path)

    predictions = MODEL.predict(image)

    predicted_index = np.argmax(predictions)

    confidence = float(
        predictions[0][predicted_index]
    )

    disease_name = CLASS_NAMES[predicted_index]

    return {
        "disease": disease_name,
        "confidence": round(confidence * 100, 2),
    }