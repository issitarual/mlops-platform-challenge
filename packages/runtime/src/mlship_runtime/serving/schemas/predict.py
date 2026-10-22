"""Response schemas. The request schema is generated from the model signature (E4)."""

from typing import Any

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    model_version: str
    predictions: list[Any]
