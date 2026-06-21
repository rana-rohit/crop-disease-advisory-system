# Explainability

## Methods

The application supports two explainability methods:

- Grad-CAM for prediction explanations.
- Occlusion sensitivity for validation-style analysis.

## Grad-CAM Flow

```text
image upload
  -> MobileNetV2 preprocessing
  -> prediction
  -> locate last Conv2D layer, including nested Functional models
  -> compute class gradients
  -> create heatmap
  -> resize to input image
  -> overlay heatmap on original image
  -> return base64 PNG payload
```

## Nested Model Support

The exported model wraps MobileNetV2 inside a top-level Functional model. The Grad-CAM implementation searches nested Keras models and reconstructs the prediction path from the nested convolutional feature map through the classifier head.

For the current artifact, the target layer is:

```text
parent model: mobilenetv2_1.00_224
target layer: Conv_1
```

## API Integration

Use:

```text
POST /api/predict?include_xai=true
```

The response includes:

- `xai.metadata`
- `xai.heatmap_statistics`
- `xai.visualizations.heatmap_base64`
- `xai.visualizations.overlay_base64`

## Frontend Integration

The result page shows:

- Original uploaded image.
- Grad-CAM overlay.
- Prediction.
- Confidence.
- Model metadata.

## Limitations

Grad-CAM is a localization aid, not proof of causal reasoning. It should be reviewed alongside confidence, top-3 predictions, and image quality.
