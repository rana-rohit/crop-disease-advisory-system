"""
Standard Grad-CAM implementation.
"""

import numpy as np
import tensorflow as tf
from backend.services.xai.base import XAIMethod

class GradCAM(XAIMethod):
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    """

    def __init__(self, layer_name: str = None):
        """
        Args:
            layer_name (str): The name of the target convolutional layer.
                              If None, attempts to auto-detect the last Conv2D layer.
        """
        self.layer_name = layer_name

    def _find_last_conv_layer(self, model: tf.keras.Model):
        if self.layer_name:
            for layer in model.layers:
                if layer.name == self.layer_name:
                    return model, layer
                if isinstance(layer, tf.keras.Model):
                    try:
                        return layer, layer.get_layer(self.layer_name)
                    except ValueError:
                        pass
            raise ValueError(f"Could not find layer named {self.layer_name}.")
            
        # Auto-detect last Conv2D layer, including nested Functional models.
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return model, layer
            if isinstance(layer, tf.keras.Model):
                for nested_layer in reversed(layer.layers):
                    if isinstance(nested_layer, tf.keras.layers.Conv2D):
                        return layer, nested_layer
                
        raise ValueError("Could not find a Conv2D layer in the model.")

    def generate_explanation(self, model: tf.keras.Model, image_tensor: np.ndarray, class_index: int) -> dict:
        parent_model, target_layer = self._find_last_conv_layer(model)
        target_layer_name = target_layer.name

        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            if parent_model is model:
                grad_model = tf.keras.models.Model(
                    model.inputs,
                    [target_layer.output, model.output]
                )
                conv_outputs, predictions = grad_model(image_tensor, training=False)
            else:
                nested_grad_model = tf.keras.models.Model(
                    parent_model.inputs,
                    [target_layer.output, parent_model.output]
                )
                nested_inputs = [image_tensor] if len(parent_model.inputs) == 1 else image_tensor
                conv_outputs, nested_outputs = nested_grad_model(nested_inputs, training=False)

                x = nested_outputs
                parent_seen = False
                for layer in model.layers:
                    if layer is parent_model:
                        parent_seen = True
                        continue
                    if not parent_seen or isinstance(layer, tf.keras.layers.InputLayer):
                        continue
                    try:
                        x = layer(x, training=False)
                    except TypeError:
                        x = layer(x)
                predictions = x

            # The score for the target class
            loss = predictions[:, class_index]

        # Compute gradients of the class score with respect to the feature map
        grads = tape.gradient(loss, conv_outputs)
        
        # Global average pooling of the gradients to compute weights
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiply each channel in the feature map by "how important this channel is"
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # ReLU on the heatmap (we only care about features that positively influence the class)
        heatmap = tf.maximum(heatmap, 0)
        max_value = tf.math.reduce_max(heatmap)
        heatmap = tf.where(max_value > 0, heatmap / max_value, tf.zeros_like(heatmap))
        
        # Resize heatmap to match original image dimensions using bilinear interpolation
        heatmap = heatmap.numpy()
        image_shape = image_tensor.shape[1:3] # (H, W)
        
        # tf.image.resize expects 4D or 3D tensor (H, W, C)
        heatmap_resized = tf.image.resize(
            heatmap[..., tf.newaxis], 
            image_shape, 
            method=tf.image.ResizeMethod.BILINEAR
        )
        heatmap_resized = tf.squeeze(heatmap_resized).numpy()
        
        return {
            "heatmap": heatmap_resized,
            "target_layer": target_layer_name,
            "target_parent_model": parent_model.name
        }
