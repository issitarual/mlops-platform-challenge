"""FastAPI dependencies: the only place routes get their services from."""

from fastapi import HTTPException, Request, status

from mlship_runtime.serving.services.prediction_service import PredictionService


def get_prediction_service(request: Request) -> PredictionService:
    service = getattr(request.app.state, "prediction_service", None)
    if not isinstance(service, PredictionService):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded yet.")
    return service
