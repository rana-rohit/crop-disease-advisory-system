# Smart Crop Disease Detection and Explainable Advisory System

Production-oriented Flask + TensorFlow application for crop leaf disease classification, advisory retrieval, and Grad-CAM explainability.

## What This Project Demonstrates

- MobileNetV2 transfer-learning inference with top-3 predictions.
- Backend service architecture with validation, logging, model loading, and app factory.
- Explainable AI through Grad-CAM overlays and occlusion sensitivity validation.
- Advisory coverage for all 38 PlantVillage model classes.
- Model metadata, inference statistics, confidence calibration hooks, and health endpoints.
- Docker, Gunicorn, Render, Railway, and GitHub Actions readiness.
- Tests for prediction, preprocessing parity, XAI, route contracts, advisory coverage, and invalid images.

## Architecture

```text
Browser UI
  |
  | multipart image upload
  v
Flask API (/api)
  |
  +-- validation layer
  +-- preprocessing layer: MobileNetV2 preprocess_input
  +-- TensorFlow/Keras model
  +-- prediction service
  +-- advisory service
  +-- XAI service: Grad-CAM + occlusion
  +-- metadata/statistics endpoints
```

## Core Endpoints

- `GET /api/health` - service and model readiness.
- `GET /api/model` - model metadata and inference statistics.
- `POST /api/predict?include_xai=true` - prediction, advisory, metadata, and optional Grad-CAM.
- `POST /api/xai/validate` - occlusion sensitivity validation for a selected or predicted class.

See [docs/api.md](docs/api.md) for full contracts.

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m backend.app
```

Backend: `http://127.0.0.1:5000`

Frontend:

```bash
python -m http.server 5500
```

Open `http://localhost:5500/frontend/pages/index.html`.

## Tests

```bash
pytest -q
```

The suite validates model loading, health contracts, prediction flow, preprocessing parity, XAI generation, advisory coverage, invalid images, and frontend/backend route alignment.

## Deployment

Docker:

```bash
docker build -t smart-crop-disease .
docker run -p 5000:5000 smart-crop-disease
```

Render and Railway configuration files are included:

- `render.yaml`
- `railway.json`
- `Dockerfile`
- `gunicorn_config.py`

See [docs/deployment.md](docs/deployment.md).

## Documentation

- [Architecture](docs/architecture.md)
- [API](docs/api.md)
- [Deployment](docs/deployment.md)
- [Explainability](docs/xai.md)
- [Model Card](docs/model_card.md)

## Known Limitations

- PlantVillage images are controlled leaf images; real field performance may be lower.
- Unknown-image detection is confidence-threshold based, not a fully validated OOD detector.
- Confidence calibration support exists, but a calibrated temperature must be fitted on a held-out calibration set before enabling in production.
- Advisory content is educational and should not replace local agricultural expert guidance.
