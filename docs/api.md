# API Documentation

Base path: `/api`

## GET /health

Returns service readiness and model load status.

Response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

## GET /model

Returns model metadata and in-memory inference statistics.

Response:

```json
{
  "model": {
    "model_version": "mobilenetv2-plantvillage-v1",
    "architecture": "MobileNetV2 transfer learning",
    "class_count": 38,
    "preprocessing": "tf.keras.applications.mobilenet_v2.preprocess_input",
    "confidence_threshold": 80.0,
    "top_k_predictions": 3,
    "calibration_enabled": false,
    "confidence_temperature": 1.0
  },
  "inference": {
    "prediction_requests": 0,
    "xai_requests": 0,
    "prediction_errors": 0,
    "average_prediction_latency_ms": 0.0
  }
}
```

## POST /predict

Form data:

- `image`: JPG, JPEG, PNG, or WEBP.

Query parameters:

- `include_xai=true|false`

Response:

```json
{
  "prediction": {
    "disease": "Corn_(maize)___healthy",
    "confidence": 98.1234
  },
  "top_predictions": [
    {"disease": "Corn_(maize)___healthy", "confidence": 98.1234}
  ],
  "advisory": {
    "crop": "Corn",
    "disease": "Healthy",
    "symptoms": "No visible disease symptoms detected.",
    "treatment": "No treatment required.",
    "prevention": "Maintain balanced fertilization, irrigation, and field scouting."
  },
  "metadata": {
    "model_version": "mobilenetv2-plantvillage-v1",
    "preprocessing": "tf.keras.applications.mobilenet_v2.preprocess_input",
    "confidence_threshold": 80.0,
    "top_k": 3
  },
  "xai": {
    "metadata": {
      "method": "gradcam",
      "target_layer": "Conv_1",
      "target_parent_model": "mobilenetv2_1.00_224"
    },
    "heatmap_statistics": {
      "max_activation": 1.0,
      "mean_activation": 0.1234,
      "activation_area_percentage": 12.34
    },
    "visualizations": {
      "heatmap_base64": "...",
      "overlay_base64": "..."
    }
  }
}
```

## POST /xai/validate

Form data:

- `image`: JPG, JPEG, PNG, or WEBP.
- `class_index`: optional integer target class.

Returns occlusion sensitivity metadata and base64 visualizations.
