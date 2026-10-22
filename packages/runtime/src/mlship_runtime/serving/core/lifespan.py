"""Startup/shutdown: the model is loaded once, before the instance reports ready."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """E4: verify the model hash, load it via ``ModelEngine``, build the
    ``PredictionService`` into ``app.state`` and only then set ``ready = True``."""
    yield
