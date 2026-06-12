"""
Image preprocessing service.
"""

import numpy as np
import tensorflow as tf


IMAGE_SIZE = (224, 224)


def preprocess_image(image_path):
    """
    Convert image into model input tensor.
    """

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    image = tf.keras.utils.img_to_array(image)

    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    return image