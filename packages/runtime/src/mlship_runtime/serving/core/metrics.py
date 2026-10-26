"""Prometheus metrics. Latency percentiles (p50/p95/p99) come from the histogram.

With several uvicorn workers each process has its own counters; multiprocess mode
(``PROMETHEUS_MULTIPROC_DIR``, set by ``python -m mlship_runtime.serving``) aggregates
them on every scrape.
"""

import os
from collections.abc import Mapping

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    multiprocess,
)

REQUESTS = Counter(
    "mlship_http_requests_total", "HTTP requests by route and status.", ["route", "status"]
)
LATENCY = Histogram(
    "mlship_http_request_duration_seconds",
    "Request latency by route.",
    ["route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
PREDICTED_ROWS = Counter("mlship_predicted_rows_total", "Rows predicted.")
FEATURE_DRIFT = Gauge(
    "mlship_feature_drift_psi",
    "PSI of a feature over the recent requests, against the training baseline.",
    ["feature"],
    multiprocess_mode="max",
)
DRIFT_MONITORED_FEATURES = Gauge(
    "mlship_drift_monitored_features",
    "Features with a training baseline; 0 means drift monitoring is off.",
    multiprocess_mode="max",
)
MODEL_INFO = Gauge(
    "mlship_model_info",
    "The model version this instance serves (value is always 1).",
    ["name", "version"],
    multiprocess_mode="max",
)


def record_drift(scores: Mapping[str, float]) -> None:
    for feature, score in scores.items():
        FEATURE_DRIFT.labels(feature).set(score)


def render() -> tuple[bytes, str]:
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)  # type: ignore[no-untyped-call]
        return generate_latest(registry), CONTENT_TYPE_LATEST
    return generate_latest(), CONTENT_TYPE_LATEST
