"""
Centralized logging utility.
"""

import logging
import sys
from backend.config.settings import get_settings

def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger for the specified module.
    """
    settings = get_settings()
    
    logger = logging.getLogger(name)
    
    # Only configure if no handlers exist to avoid duplicate logs
    if not logger.handlers:
        logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
        
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Prevent propagation to the root logger
        logger.propagate = False
        
    return logger
