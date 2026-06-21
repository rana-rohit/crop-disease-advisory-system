"""
Pytest configuration and fixtures.
"""

import io
import pytest
from PIL import Image
from backend.app import create_app

@pytest.fixture
def app():
    """Create and configure a new app instance for each test."""
    app = create_app()
    app.config.update({
        "TESTING": True,
    })
    yield app

@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()

@pytest.fixture
def sample_image():
    """Create a sample 224x224 green image for testing."""
    img = Image.new("RGB", (224, 224), color="green")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer
