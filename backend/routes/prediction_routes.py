"""
API routes for prediction.
"""

import io
from flask import Blueprint, request, jsonify
from backend.services.prediction_service import predict_disease
from backend.utils.validators import validate_image_upload

prediction_bp = Blueprint("prediction", __name__)

@prediction_bp.route("/predict", methods=["POST"])
def predict():
    """
    Handle image upload and return disease prediction.
    """
    # Validate request
    file = validate_image_upload(request)

    # Read file into memory buffer
    image_bytes = file.read()
    image_buffer = io.BytesIO(image_bytes)

    # Check for XAI flag in query params or form data
    include_xai = request.args.get("include_xai", "false").lower() == "true"
    if "include_xai" in request.form:
        include_xai = request.form["include_xai"].lower() == "true"

    # Predict
    result = predict_disease(image_buffer, include_xai=include_xai)

    return jsonify(result)