"""
Orchestrator for Explainable AI generation.
Coordinates Grad-CAM and Occlusion Sensitivity.
"""

from datetime import datetime, timezone
import numpy as np
import tensorflow as tf

from backend.utils.exceptions import PredictionError
from backend.services.xai.gradcam import GradCAM
from backend.services.xai.occlusion import OcclusionSensitivity
from backend.services.xai.visualization import (
    apply_colormap, 
    overlay_heatmap, 
    encode_image_base64, 
    tensor_to_uint8_image,
    compute_heatmap_statistics
)

def generate_gradcam_explanation(model: tf.keras.Model, image_tensor: np.ndarray, class_index: int, class_name: str) -> dict:
    """
    Generate standard Grad-CAM heatmap and overlays.
    """
    gradcam = GradCAM()
    try:
        result = gradcam.generate_explanation(model, image_tensor, class_index)
    except Exception as e:
        raise PredictionError(f"Failed to generate Grad-CAM: {e}")
        
    heatmap = result["heatmap"]
    
    # Visualizations
    heatmap_rgb = apply_colormap(heatmap)
    original_uint8 = tensor_to_uint8_image(image_tensor)
    overlay_rgb = overlay_heatmap(original_uint8, heatmap_rgb, alpha=0.5)
    
    # Encodings
    heatmap_b64 = encode_image_base64(heatmap_rgb)
    overlay_b64 = encode_image_base64(overlay_rgb)
    
    # Statistics
    stats = compute_heatmap_statistics(heatmap)
    
    # Metadata
    metadata = {
        "method": "gradcam",
        "target_class_index": int(class_index),
        "target_class_name": class_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_layer": result["target_layer"],
        "target_parent_model": result.get("target_parent_model")
    }
    
    return {
        "metadata": metadata,
        "heatmap_statistics": stats,
        "visualizations": {
            "heatmap_base64": heatmap_b64,
            "overlay_base64": overlay_b64
        }
    }

def generate_occlusion_validation(model: tf.keras.Model, image_tensor: np.ndarray, class_index: int, class_name: str) -> dict:
    """
    Generate Occlusion Sensitivity analysis.
    """
    occlusion = OcclusionSensitivity()
    try:
        result = occlusion.generate_explanation(model, image_tensor, class_index)
    except Exception as e:
        raise PredictionError(f"Failed to generate Occlusion map: {e}")
        
    heatmap = result["heatmap"]
    
    # Visualizations
    heatmap_rgb = apply_colormap(heatmap, colormap_name="inferno") # Different colormap for occlusion
    original_uint8 = tensor_to_uint8_image(image_tensor)
    overlay_rgb = overlay_heatmap(original_uint8, heatmap_rgb, alpha=0.5)
    
    # Encodings
    heatmap_b64 = encode_image_base64(heatmap_rgb)
    overlay_b64 = encode_image_base64(overlay_rgb)
    
    # Statistics
    stats = compute_heatmap_statistics(heatmap)
    
    # Metadata
    metadata = {
        "method": "occlusion_sensitivity",
        "target_class_index": int(class_index),
        "target_class_name": class_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_confidence": float(result["base_confidence"]),
        "max_confidence_drop": float(result["max_confidence_drop"])
    }
    
    return {
        "metadata": metadata,
        "heatmap_statistics": stats,
        "visualizations": {
            "heatmap_base64": heatmap_b64,
            "overlay_base64": overlay_b64
        }
    }
