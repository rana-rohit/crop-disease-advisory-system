"""
Base interfaces for Explainable AI methods.
"""

from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf

class XAIMethod(ABC):
    """
    Abstract base class for all Explainable AI methods (e.g., Grad-CAM, Occlusion).
    """

    @abstractmethod
    def generate_explanation(self, model: tf.keras.Model, image_tensor: np.ndarray, class_index: int) -> dict:
        """
        Generate the XAI explanation for the given image and target class.
        
        Args:
            model (tf.keras.Model): The trained classification model.
            image_tensor (np.ndarray): The preprocessed image tensor of shape (1, H, W, C).
            class_index (int): The target class index to explain.
            
        Returns:
            dict: A dictionary containing method-specific raw outputs (like heatmaps).
                  Visualizations and encodings are handled separately.
        """
        pass
