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

from backend.services.disease_info_service import (
    get_disease_info,
)

def predict_disease(image_path):
    """
    Predict disease from leaf image.
    """

    image = preprocess_image(image_path)

    predictions = MODEL.predict(
        image,
        verbose=0,
    )[0]

    top_indices = np.argsort(
        predictions
    )[::-1][:3]

    top_predictions = []

    for index in top_indices:

        top_predictions.append(
            {
                "disease": CLASS_NAMES[index],
                "confidence": round(
                    float(predictions[index]) * 100,
                    2,
                ),
            }
        )
    
    top_confidence = top_predictions[0]["confidence"]

    predicted_disease = top_predictions[0]["disease"]


    if top_confidence < 80:

        return {
            "prediction": {
                "disease": "Unknown Plant",
                "confidence": top_confidence,
            },

            "top_predictions": top_predictions,

            "advisory": {
                "crop": "Unknown",
                "disease": "Unknown",
                "symptoms": "Unable to identify a supported crop leaf.",
                "treatment": "Please upload a clearer image of a plant leaf.",
                "prevention": "Ensure the image contains a single crop leaf from the supported dataset."
            },
        }


    advisory = get_disease_info(
        predicted_disease
    )

    return {
        "prediction": top_predictions[0],
        "top_predictions": top_predictions,
        "advisory": advisory,
    }