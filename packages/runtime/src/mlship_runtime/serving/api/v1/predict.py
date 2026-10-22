"""Prediction endpoints (versioned API)."""

from fastapi import APIRouter, HTTPException, status

router = APIRouter(prefix="/v1", tags=["predict"])


@router.post("/predict")
def predict_one() -> None:
    """E4: one row in, one prediction out."""
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented yet.")


@router.post("/predict/batch")
def predict_batch() -> None:
    """E4: up to ``max_batch_rows`` rows, vectorized."""
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented yet.")
