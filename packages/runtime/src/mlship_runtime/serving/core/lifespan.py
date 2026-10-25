"""Startup: the model is loaded once, before the instance reports ready."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from mlship_runtime.serving.core import metrics
from mlship_runtime.serving.services.prediction_service import PredictionService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    engine = app.state.engine
    settings = app.state.settings
    await run_in_threadpool(engine.load)
    app.state.prediction_service = PredictionService(
        engine,
        sample_rate=settings.prediction_log_sample_rate,
        exclude_columns=settings.prediction_log_exclude_columns,
    )
    metrics.MODEL_INFO.labels(engine.model_name, engine.model_version).set(1)
    app.state.ready = True
    yield
    app.state.ready = False
