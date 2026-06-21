"""
Occlusion Sensitivity Analysis implementation.
"""

import numpy as np
import tensorflow as tf
from backend.services.xai.base import XAIMethod

class OcclusionSensitivity(XAIMethod):
    """
    Occlusion Sensitivity Analysis.
    Iteratively masks parts of the image and measures the drop in confidence.
    """

    def __init__(self, patch_size: int = 32, stride: int = 32, mask_value: float = 0.0, batch_size: int = 32):
        self.patch_size = patch_size
        self.stride = stride
        self.mask_value = mask_value
        self.batch_size = batch_size

    def generate_explanation(self, model: tf.keras.Model, image_tensor: np.ndarray, class_index: int) -> dict:
        img_shape = image_tensor.shape[1:3] # (H, W)
        H, W = img_shape
        
        # Base prediction confidence
        base_preds = model.predict(image_tensor, verbose=0)
        base_conf = float(base_preds[0, class_index])
        
        # Calculate grid dimensions
        h_steps = (H - self.patch_size) // self.stride + 1
        w_steps = (W - self.patch_size) // self.stride + 1
        
        heatmap = np.zeros((h_steps, w_steps))
        
        # Prepare batches of occluded images
        occluded_images = []
        indices = []
        
        for r in range(h_steps):
            for c in range(w_steps):
                h_start = r * self.stride
                w_start = c * self.stride
                
                # Copy original tensor
                img_copy = image_tensor.copy()
                # Apply occlusion mask
                img_copy[0, h_start:h_start+self.patch_size, w_start:w_start+self.patch_size, :] = self.mask_value
                
                occluded_images.append(img_copy[0])
                indices.append((r, c))
                
        occluded_images = np.array(occluded_images)
        
        # Batch inference
        num_patches = len(occluded_images)
        conf_drops = []
        
        for i in range(0, num_patches, self.batch_size):
            batch = occluded_images[i:i+self.batch_size]
            preds = model.predict(batch, verbose=0)
            
            # Extract confidence for target class
            batch_conf = preds[:, class_index]
            
            # Drop in confidence: base - occluded
            drops = base_conf - batch_conf
            conf_drops.extend(drops)
            
        # Map back to heatmap
        for (r, c), drop in zip(indices, conf_drops):
            heatmap[r, c] = max(0, drop) # Only care about drops
            
        # Normalize heatmap
        max_drop = np.max(heatmap)
        if max_drop > 0:
            heatmap = heatmap / max_drop
            
        # Resize to match original image dimensions
        heatmap_resized = tf.image.resize(
            heatmap[..., tf.newaxis], 
            img_shape, 
            method=tf.image.ResizeMethod.BILINEAR
        ).numpy().squeeze()
        
        return {
            "heatmap": heatmap_resized,
            "base_confidence": base_conf,
            "max_confidence_drop": max_drop
        }
