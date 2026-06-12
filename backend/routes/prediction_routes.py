"""
Prediction routes.
"""

from flask import Blueprint
from flask import request
from flask import jsonify

from backend.services.prediction_service import (
    predict_disease,
)

prediction_bp = Blueprint(
    "prediction",
    __name__,
)


@prediction_bp.route(
    "/predict",
    methods=["POST"],
)
def predict():

    if "image" not in request.files:
        return jsonify(
            {
                "error": "No image uploaded"
            }
        ), 400

    image = request.files["image"]

    temp_path = "temp.jpg"

    image.save(temp_path)

    result = predict_disease(temp_path)

    return jsonify(result)