"""
Tests for the Explainable AI (XAI) subsystem.
"""

import io
from PIL import Image
import numpy as np

def test_xai_visualization_utilities():
    """Test the colormap and base64 encoding utilities."""
    from backend.services.xai.visualization import apply_colormap, overlay_heatmap, compute_heatmap_statistics, encode_image_base64
    
    # Dummy heatmap [0, 1]
    heatmap = np.array([[0.1, 0.5], [0.8, 1.0]], dtype=np.float32)
    
    stats = compute_heatmap_statistics(heatmap, threshold=0.5)
    assert stats["max_activation"] == 1.0
    assert stats["mean_activation"] == 0.6
    assert stats["activation_area_percentage"] == 50.0 # 2 out of 4 are > 0.5
    
    heatmap_rgb = apply_colormap(heatmap)
    assert heatmap_rgb.shape == (2, 2, 3)
    assert heatmap_rgb.dtype == np.uint8
    
    original = np.zeros((2, 2, 3), dtype=np.uint8)
    overlay = overlay_heatmap(original, heatmap_rgb, alpha=0.5)
    assert overlay.shape == (2, 2, 3)
    
    b64 = encode_image_base64(overlay)
    assert isinstance(b64, str)
    assert len(b64) > 0

def test_prediction_with_xai(client, sample_image):
    """Test the /predict endpoint with include_xai=true."""
    data = {
        "image": (sample_image, "test_leaf.jpg"),
        "include_xai": "true"
    }
    response = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 200

    json_data = response.get_json()
    assert "xai" in json_data
    xai = json_data["xai"]
    assert "metadata" in xai
    assert xai["metadata"]["method"] == "gradcam"
    assert xai["metadata"]["target_layer"] == "Conv_1"
    assert xai["metadata"]["target_parent_model"].startswith("mobilenetv2")
    assert "target_class_index" in xai["metadata"]
    assert "heatmap_statistics" in xai
    assert "visualizations" in xai
    assert len(xai["visualizations"]["heatmap_base64"]) > 100
    assert len(xai["visualizations"]["overlay_base64"]) > 100

def test_xai_validate_endpoint_valid(client, sample_image):
    """Test POST /xai/validate with a valid image."""
    data = {
        "image": (sample_image, "test_leaf.jpg")
    }
    response = client.post("/api/xai/validate", data=data, content_type="multipart/form-data")
    assert response.status_code == 200

    json_data = response.get_json()
    assert "metadata" in json_data
    assert json_data["metadata"]["method"] == "occlusion_sensitivity"
    assert "visualizations" in json_data
    assert len(json_data["visualizations"]["overlay_base64"]) > 100

def test_xai_validate_invalid_class_index(client, sample_image):
    """Test POST /xai/validate with an out-of-bounds class index."""
    data = {
        "image": (sample_image, "test_leaf.jpg"),
        "class_index": "999" # Way out of bounds for 38 classes
    }
    response = client.post("/api/xai/validate", data=data, content_type="multipart/form-data")
    
    assert response.status_code == 400
    assert "error" in response.get_json()
    
def test_xai_validate_missing_image(client):
    """Test POST /xai/validate without an image."""
    response = client.post("/api/xai/validate", data={})
    assert response.status_code == 400
