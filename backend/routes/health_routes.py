"""
Health check routes.
"""

from flask import Blueprint, jsonify
from backend.model.model_loader import get_model, get_model_metadata
from backend.services.inference_stats_service import get_inference_stats

health_bp = Blueprint("health", __name__)

@health_bp.route("/health", methods=["GET"])
def health_check():
    """
    Simple health check endpoint to verify service and model status.
    """
    try:
        get_model()
        model_loaded = True
    except Exception:
        model_loaded = False
    
    return jsonify({
        "status": "ok",
        "model_loaded": model_loaded
    })

@health_bp.route("/model", methods=["GET"])
def model_info():
    """Return model metadata and lightweight inference statistics."""
    return jsonify({
        "model": get_model_metadata(),
        "inference": get_inference_stats()
    })
