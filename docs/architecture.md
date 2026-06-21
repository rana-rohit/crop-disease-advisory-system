# Architecture

## System View

```text
frontend/pages/index.html
  |
  | POST /api/predict?include_xai=true
  v
backend/routes
  |
  +-- validators.py
  +-- image_service.py
  +-- prediction_service.py
  +-- explainability_service.py
  +-- disease_info_service.py
  v
TensorFlow MobileNetV2 model artifact
```

## Backend Boundaries

- `backend/app.py` creates the Flask app, initializes the model, registers error handlers, and mounts blueprints.
- `backend/routes/` owns HTTP contracts only.
- `backend/services/` owns business logic: preprocessing, inference, advisory lookup, XAI, and observability.
- `backend/model/` owns model and metadata loading.
- `backend/config/` owns environment-driven settings.

## Dependency Flow

```text
routes -> services -> model/config/utils
```

Routes do not call TensorFlow directly except for XAI validation target-class selection. Prediction uses the prediction service as the orchestration boundary.

## Runtime Notes

TensorFlow model loading is process-local. Production deployments default to one Gunicorn worker to avoid multiplying memory usage unexpectedly. Increase `WEB_CONCURRENCY` only after measuring memory headroom.

## Critical Correctness Contracts

- Training and inference preprocessing both use `tf.keras.applications.mobilenet_v2.preprocess_input`.
- Advisory data must cover every class in `class_names.json`.
- Grad-CAM must support nested Keras Functional models such as exported MobileNetV2 backbones.
- `/api/health` must report the actual loaded model state.
