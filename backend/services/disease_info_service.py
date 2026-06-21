"""
Disease information service.
"""

import json
from backend.config.settings import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# In-memory cache for disease advisory data
_DISEASE_CACHE = None

def _load_disease_cache():
    """Load the disease JSON into memory if not already loaded."""
    global _DISEASE_CACHE
    if _DISEASE_CACHE is not None:
        return _DISEASE_CACHE
        
    settings = get_settings()
    try:
        with open(settings.DISEASE_INFO_PATH, "r", encoding="utf-8") as file:
            _DISEASE_CACHE = json.load(file)
            logger.info("Successfully loaded disease info cache.")
    except Exception as e:
        logger.error(f"Failed to load disease info from {settings.DISEASE_INFO_PATH}: {e}")
        _DISEASE_CACHE = {}
        
    return _DISEASE_CACHE

# Pre-load cache at module import
_load_disease_cache()

def get_disease_info(disease_name: str) -> dict:
    """
    Return disease advisory information from the in-memory cache.
    """
    cache = _load_disease_cache()

    return cache.get(
        disease_name,
        {
            "symptoms": "Information not available.",
            "treatment": "Information not available.",
            "prevention": "Information not available.",
        },
    )