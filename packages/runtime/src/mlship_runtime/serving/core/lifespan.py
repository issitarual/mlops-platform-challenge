"""Startup: the model is loaded once, before the instance reports ready."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from mlship_runtime import reference
from mlship_runtime.serving.core import metrics
from mlship_runtime.serving.services.drift import DriftMonitor
from mlship_runtime.serving.services.prediction_service import PredictionService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    engine = app.state.engine
    settings = app.state.settings
    await run_in_threadpool(engine.load)
    baseline = reference.load(settings.model_dir)
    drift = (
        DriftMonitor(
            baseline,
            [c.name for c in engine.columns],
            window_rows=settings.drift_window_rows,
            min_rows=settings.drift_min_rows,
        )
        if baseline is not None
        else None
    )
    app.state.prediction_service = PredictionService(
        engine,
        sample_rate=settings.prediction_log_sample_rate,
        exclude_columns=settings.prediction_log_exclude_columns,
        drift=drift,
    )
    metrics.MODEL_INFO.labels(engine.model_name, engine.model_version).set(1)
    metrics.DRIFT_MONITORED_FEATURES.set(len(drift.features) if drift is not None else 0)
    app.state.ready = True
    yield
    app.state.ready = False
