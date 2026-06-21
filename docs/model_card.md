# Model Card

## Model

- Name: Smart Crop Disease MobileNetV2
- Version: `mobilenetv2-plantvillage-v1`
- Framework: TensorFlow/Keras
- Architecture: MobileNetV2 transfer learning with global average pooling, dropout, and softmax classifier
- Classes: 38

## Intended Use

Classify supported crop leaf images into PlantVillage disease and healthy classes, then provide advisory information and explainability overlays for educational and portfolio demonstration use.

## Dataset

The model is trained on PlantVillage-style crop leaf images organized into train and validation folders.

Supported crops include apple, blueberry, cherry, corn, grape, orange, peach, bell pepper, potato, raspberry, soybean, squash, strawberry, and tomato.

## Metrics

Current validation artifacts report:

- Accuracy: approximately 95.98%
- Weighted F1: approximately 95.91%
- Macro F1: approximately 94.60%
- Top-3 validation accuracy: approximately 99.51%

Weaker classes include:

- `Potato___healthy`
- `Tomato___Early_blight`
- `Tomato___Target_Spot`
- `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot`

## Preprocessing

Training and inference both use:

```text
tf.keras.applications.mobilenet_v2.preprocess_input
```

This scales RGB image tensors into the input distribution expected by ImageNet-pretrained MobileNetV2.

## Confidence Calibration

The backend includes temperature-scaling support through:

- `CALIBRATION_ENABLED`
- `CONFIDENCE_TEMPERATURE`

Default behavior leaves model probabilities unchanged. Before enabling calibration, fit the temperature on a held-out calibration set.

## Limitations

- PlantVillage images are cleaner than real field images.
- Performance may drop with cluttered backgrounds, poor lighting, occlusion, multiple leaves, or non-leaf inputs.
- Unknown image detection uses confidence thresholding and is not a validated OOD classifier.
- Advisory guidance is educational and should not replace local agronomy expertise.

## Failure Cases

- Unsupported crops can still receive a high softmax confidence.
- Visually similar diseases may be confused.
- Severe image compression or blur can reduce reliability.
- Grad-CAM may highlight correlated visual regions rather than causal disease symptoms.
