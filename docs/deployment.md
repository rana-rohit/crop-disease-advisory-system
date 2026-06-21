# Deployment

## Docker

```bash
docker build -t smart-crop-disease .
docker run -p 5000:5000 smart-crop-disease
```

Health check:

```bash
curl http://localhost:5000/api/health
```

## Gunicorn

```bash
gunicorn -c gunicorn_config.py backend.app:app
```

Important environment variables:

- `PORT`: runtime port.
- `WEB_CONCURRENCY`: worker count. Default is `1` because TensorFlow is memory-heavy.
- `GUNICORN_THREADS`: default `2`.
- `GUNICORN_TIMEOUT`: default `180`.
- `CORS_ORIGINS`: comma-separated frontend origins, or `*` for development.

## Render

This repository includes `render.yaml`.

1. Create a new Render web service from the repository.
2. Use Docker environment.
3. Set `CORS_ORIGINS` to the deployed frontend origin.
4. Confirm health check path is `/api/health`.

## Railway

This repository includes `railway.json`.

1. Create a Railway project from the repository.
2. Railway builds from `Dockerfile`.
3. Set `CORS_ORIGINS` and any model path overrides if needed.

## CI

GitHub Actions workflow: `.github/workflows/ci.yml`.

The workflow installs dependencies and runs:

```bash
pytest -q
```

## Production Notes

- Keep large model artifacts under explicit artifact management for larger models.
- Set `FLASK_DEBUG=0`.
- Restrict `CORS_ORIGINS`.
- Monitor memory before increasing worker count.
