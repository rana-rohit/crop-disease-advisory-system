"""
Tests for the prediction routes.
"""

import io

def test_predict_no_image(client):
    """Test POST /predict without an image file."""
    response = client.post("/api/predict", data={})
    assert response.status_code == 400
    assert "error" in response.get_json()

def test_predict_empty_file(client):
    """Test POST /predict with an empty image file."""
    data = {
        "image": (io.BytesIO(b""), "empty.jpg")
    }
    response = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "error" in response.get_json()

def test_predict_valid_image(client, sample_image):
    """Test POST /predict with a valid image."""
    data = {
        "image": (sample_image, "test_leaf.jpg")
    }
    response = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 200

    json_data = response.get_json()
    assert "prediction" in json_data
    assert "top_predictions" in json_data
    assert "advisory" in json_data
    assert "metadata" in json_data
    assert len(json_data["top_predictions"]) == 3
    assert json_data["metadata"]["preprocessing"] == "tf.keras.applications.mobilenet_v2.preprocess_input"

def test_predict_corrupted_image(client):
    """Corrupted image bytes must be rejected before inference."""
    data = {
        "image": (io.BytesIO(b"not-a-real-image"), "corrupt.jpg")
    }
    response = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "error" in response.get_json()

def test_frontend_uses_backend_api_contract():
    """The browser client must call the registered backend route."""
    with open("frontend/js/api.js", "r", encoding="utf-8") as file:
        contents = file.read()

    assert "/api" in contents
    assert "/predict" in contents
