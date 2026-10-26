"""``POST /v1/predict``: 1 to N rows in one call, predicted in one vectorized pass.

The request model is generated from the model signature when the app is built, so
the OpenAPI page documents the real columns.
"""

import asyncio
from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from mlship_runtime.serving.api.deps import get_prediction_service
from mlship_runtime.serving.core import metrics, request_id
from mlship_runtime.serving.core.security import require_api_key
from mlship_runtime.serving.schemas.predict import ErrorResponse, ModelInfo, PredictionResponse
from mlship_runtime.serving.services.prediction_service import PredictionService

_ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse} for code in (401, 413, 422, 500, 503, 504)
}


def build_router(request_model: type[BaseModel], timeout_seconds: float) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["predict"], dependencies=[Depends(require_api_key)])

    @router.post("/predict", response_model=PredictionResponse, responses=_ERRORS)
    async def predict(
        body: request_model,  # type: ignore[valid-type]
        request: Request,
        service: Annotated[PredictionService, Depends(get_prediction_service)],
    ) -> PredictionResponse:
        rows = [row.model_dump() for row in cast(Any, body).rows]
        rid = request_id.of(request)
        try:
            predictions = await asyncio.wait_for(
                run_in_threadpool(service.predict, rows, rid), timeout_seconds
            )
        except TimeoutError as exc:
            raise HTTPException(
                status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Prediction took longer than {timeout_seconds}s.",
            ) from exc
        metrics.PREDICTED_ROWS.inc(len(rows))
        metrics.record_drift(service.drift_scores())
        return PredictionResponse(
            model=ModelInfo(name=service.model_name, version=service.model_version),
            predictions=predictions,
            request_id=rid,
        )

    return router
