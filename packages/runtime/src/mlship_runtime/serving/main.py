"""FastAPI application factory."""

from fastapi import FastAPI

from mlship_runtime.serving.api import health
from mlship_runtime.serving.api.v1 import predict
from mlship_runtime.serving.core.config import ServingSettings
from mlship_runtime.serving.core.lifespan import lifespan


def create_app(settings: ServingSettings | None = None) -> FastAPI:
    """Build the app. Health routes stay outside ``/v1``: they are infrastructure."""
    app = FastAPI(title="mlship model API", lifespan=lifespan)
    app.state.settings = settings or ServingSettings()
    app.state.ready = False
    app.include_router(health.router)
    app.include_router(predict.router)
    # E4: mount Prometheus /metrics.
    return app
