"""
Visualization utilities for Explainable AI.
Responsible for colormaps, overlays, statistics, and base64 encoding.
"""

import io
import base64
import numpy as np
import matplotlib.cm as cm
from PIL import Image

def compute_heatmap_statistics(heatmap: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Compute statistics for the generated heatmap.
    """
    max_act = float(np.max(heatmap))
    mean_act = float(np.mean(heatmap))
    
    # Calculate percentage of area where activation is above threshold
    active_pixels = np.sum(heatmap > threshold)
    total_pixels = heatmap.size
    area_pct = float(active_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0
    
    return {
        "max_activation": round(max_act, 4),
        "mean_activation": round(mean_act, 4),
        "activation_area_percentage": round(area_pct, 2)
    }

def apply_colormap(heatmap: np.ndarray, colormap_name: str = "jet") -> np.ndarray:
    """
    Apply a matplotlib colormap to a 2D heatmap [0, 1].
    Returns an RGB image array [0, 255].
    """
    import matplotlib as mpl
    cmap = mpl.colormaps[colormap_name]
    
    # Get RGBA colors and drop the alpha channel
    heatmap_colored = cmap(heatmap)[:, :, :3]
    
    # Convert to 0-255 uint8
    return np.uint8(255 * heatmap_colored)

def overlay_heatmap(original_image: np.ndarray, heatmap_rgb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Blend the original image and the RGB heatmap.
    Both arrays must be of type uint8 and shape (H, W, 3).
    """
    return np.uint8(alpha * heatmap_rgb + (1.0 - alpha) * original_image)

def encode_image_base64(image_array: np.ndarray, format: str = "PNG") -> str:
    """
    Convert a numpy image array (uint8) to a base64 encoded string.
    """
    img = Image.fromarray(image_array)
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def tensor_to_uint8_image(image_tensor: np.ndarray) -> np.ndarray:
    """
    Convert a normalized image tensor (1, H, W, 3) back to a uint8 image (H, W, 3).
    Assumes tensor is normalized to [0, 1] or roughly bounded. 
    MobileNetV2 preprocess_input scales to [-1, 1], so we need to reverse that.
    """
    img = image_tensor[0]
    # Reverse MobileNetV2 preprocessing (which was img = img / 127.5 - 1.0 or similar to [-1, 1])
    # The image_service uses tf.keras.applications.mobilenet_v2.preprocess_input
    # So we reverse it back to [0, 255]
    img = np.clip((img + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return img
