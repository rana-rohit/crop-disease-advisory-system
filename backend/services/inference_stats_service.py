"""
In-memory inference statistics for lightweight observability.
"""

from __future__ import annotations

from threading import Lock
from time import perf_counter


_LOCK = Lock()
_STATS = {
    "prediction_requests": 0,
    "xai_requests": 0,
    "prediction_errors": 0,
    "total_prediction_latency_ms": 0.0,
    "last_prediction_latency_ms": None,
}


def start_timer() -> float:
    return perf_counter()


def record_prediction(start_time: float, include_xai: bool = False) -> None:
    elapsed_ms = (perf_counter() - start_time) * 1000.0
    with _LOCK:
        _STATS["prediction_requests"] += 1
        if include_xai:
            _STATS["xai_requests"] += 1
        _STATS["total_prediction_latency_ms"] += elapsed_ms
        _STATS["last_prediction_latency_ms"] = round(elapsed_ms, 2)


def record_prediction_error() -> None:
    with _LOCK:
        _STATS["prediction_errors"] += 1


def get_inference_stats() -> dict:
    with _LOCK:
        stats = dict(_STATS)

    count = stats["prediction_requests"]
    avg = stats["total_prediction_latency_ms"] / count if count else 0.0
    stats["average_prediction_latency_ms"] = round(avg, 2)
    stats["total_prediction_latency_ms"] = round(stats["total_prediction_latency_ms"], 2)
    return stats
