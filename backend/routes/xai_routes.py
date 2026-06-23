"""
API routes for Explainable AI (XAI) validation.
"""

import io
import numpy as np
from flask import Blueprint, request, jsonify
from backend.utils.validators import validate_image_upload
from backend.model.model_loader import get_model, get_class_names
from backend.services.image_service import preprocess_image
from backend.services.explainability_service import generate_occlusion_validation, generate_gradcam_explanation

xai_bp = Blueprint("xai", __name__)

@xai_bp.route("/xai/validate", methods=["POST"])
def validate_prediction():
    """
    Perform Occlusion Sensitivity Analysis for a prediction.
    Requires an image and optionally a target class index.
    If no class index is provided, uses the model's top prediction.
    """
    file = validate_image_upload(request)
    
    image_bytes = file.read()
    image_buffer = io.BytesIO(image_bytes)
    
    model = get_model()
    class_names = get_class_names()
    image_tensor = preprocess_image(image_buffer)
    
    target_class_idx = request.form.get("class_index")
    
    if target_class_idx is not None:
        try:
            target_class_idx = int(target_class_idx)
            if target_class_idx < 0 or target_class_idx >= len(class_names):
                return jsonify({"error": "Invalid class_index."}), 400
        except ValueError:
            return jsonify({"error": "class_index must be an integer."}), 400
    else:
        # Default to top prediction
        preds = model.predict(image_tensor, verbose=0)
        target_class_idx = int(np.argmax(preds[0]))
        
    target_class_name = class_names[target_class_idx]
    
    result = generate_occlusion_validation(model, image_tensor, target_class_idx, target_class_name)
    
    return jsonify(result)

@xai_bp.route("/xai/gradcam", methods=["POST"])
def generate_gradcam():
    """
    Perform Grad-CAM analysis for a prediction.
    Requires an image and optionally a target class index.
    If no class index is provided, uses the model's top prediction.
    """
    file = validate_image_upload(request)
    
    image_bytes = file.read()
    image_buffer = io.BytesIO(image_bytes)
    
    model = get_model()
    class_names = get_class_names()
    image_tensor = preprocess_image(image_buffer)
    
    target_class_idx = request.form.get("class_index")
    
    if target_class_idx is not None:
        try:
            target_class_idx = int(target_class_idx)
            if target_class_idx < 0 or target_class_idx >= len(class_names):
                return jsonify({"error": "Invalid class_index."}), 400
        except ValueError:
            return jsonify({"error": "class_index must be an integer."}), 400
    else:
        # Default to top prediction
        preds = model.predict(image_tensor, verbose=0)
        target_class_idx = int(np.argmax(preds[0]))
        
    target_class_name = class_names[target_class_idx]
    
    result = generate_gradcam_explanation(model, image_tensor, target_class_idx, target_class_name)
    
    # We wrap it in {"xai": ...} to match the old payload structure the frontend expects
    return jsonify({"xai": result})
