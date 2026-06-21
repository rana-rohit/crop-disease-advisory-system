"""
Confidence calibration helpers.
"""

import numpy as np


def apply_temperature_scaling(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """
    Apply simple post-hoc temperature scaling to a probability vector.

    The model currently exports softmax probabilities instead of logits, so this
    function transforms probabilities back to log space before applying the
    temperature. A temperature of 1.0 is an identity transform.
    """
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero.")

    probabilities = np.asarray(probabilities, dtype=np.float32)
    if temperature == 1.0:
        return probabilities

    logits = np.log(np.clip(probabilities, 1e-8, 1.0))
    scaled = logits / temperature
    scaled = scaled - np.max(scaled)
    exp_scores = np.exp(scaled)
    return exp_scores / np.sum(exp_scores)
