"""Gunicorn runtime configuration for production deployments."""

import multiprocessing
import os


bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"
workers = int(os.getenv("WEB_CONCURRENCY", "1"))
threads = int(os.getenv("GUNICORN_THREADS", "1"))
timeout = int(os.getenv("GUNICORN_TIMEOUT", "180"))
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", "30"))
preload_app = os.getenv("GUNICORN_PRELOAD_APP", "false").lower() in {"true", "1", "t"}

# TensorFlow is memory-heavy; default to one worker unless the host is sized for more.
if os.getenv("WEB_CONCURRENCY") is None:
    workers = min(1, multiprocessing.cpu_count())
