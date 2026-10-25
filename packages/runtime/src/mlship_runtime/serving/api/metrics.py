"""Prometheus scrape endpoint, outside ``/v1`` and without the API key (ADR-020)."""

from fastapi import APIRouter, Response

from mlship_runtime.serving.core import metrics

router = APIRouter(tags=["operations"])


@router.get("/metrics", include_in_schema=False)
def scrape() -> Response:
    body, content_type = metrics.render()
    return Response(body, media_type=content_type)
