"""Liveness (process is up) and readiness (model loaded) are separate on purpose."""

from fastapi import APIRouter, Request, Response, status

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
def ready(request: Request, response: Response) -> dict[str, str]:
    if not request.app.state.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "loading"}
    return {"status": "ready"}
