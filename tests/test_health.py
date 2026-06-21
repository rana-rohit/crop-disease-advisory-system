"""
Tests for the health check routes.
"""

def test_health_check(client):
    """Test the /api/health endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    
    data = response.get_json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True

def test_model_info_endpoint(client):
    """Test /api/model exposes model metadata and inference statistics."""
    response = client.get("/api/model")
    assert response.status_code == 200

    data = response.get_json()
    assert data["model"]["class_count"] == 38
    assert data["model"]["preprocessing"] == "tf.keras.applications.mobilenet_v2.preprocess_input"
    assert "prediction_requests" in data["inference"]
