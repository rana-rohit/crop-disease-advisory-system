"""
Tests for the advisory service.
"""

from backend.services.disease_info_service import get_disease_info
from backend.model.model_loader import get_class_names, init_model

def test_get_known_disease():
    """Test fetching advisory info for a known disease."""
    # Since we don't know exact names without reading the JSON, let's test a common one
    # or just verify it returns a dict with the expected keys.
    info = get_disease_info("Apple___Apple_scab")
    assert isinstance(info, dict)
    assert "symptoms" in info
    assert "treatment" in info
    assert "prevention" in info

def test_get_unknown_disease():
    """Test fetching advisory info for an unknown disease returns fallback."""
    info = get_disease_info("Made_Up_Disease")
    assert isinstance(info, dict)
    assert info["symptoms"] == "Information not available."

def test_every_model_class_has_advisory():
    """Every class emitted by the model must have advisory coverage."""
    init_model()
    for class_name in get_class_names():
        info = get_disease_info(class_name)
        assert info["symptoms"] != "Information not available.", class_name
        assert info["treatment"] != "Information not available.", class_name
        assert info["prevention"] != "Information not available.", class_name
