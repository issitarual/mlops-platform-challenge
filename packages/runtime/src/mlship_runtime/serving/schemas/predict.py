"""Request and response schemas. The row schema is generated from the model signature.

Pydantic validates every row and reports every error at once, with its location
(e.g. ``rows.3.MedInc``). Strict types: a JSON string is not silently turned into a
number, nor ``true`` into 1. ``NaN`` and ``Infinity``, which Python's JSON parser
accepts, are refused, and so is an unknown column. Without a column signature, rows
are free-form and the model itself validates them.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model

from mlship_runtime.serving.engines.model_engine import Column

_PYTHON_TYPES: dict[str, type] = {
    "double": float,
    "float": float,
    "long": int,
    "integer": int,
    "boolean": bool,
    "string": str,
    "datetime": datetime,
}


def build_request_model(columns: tuple[Column, ...], max_rows: int) -> type[BaseModel]:
    """``{"rows": [ {<column>: <value>, ...}, ... ]}`` with 1 to ``max_rows`` rows."""
    if columns:
        fields: dict[str, Any] = {}
        for column in columns:
            python_type = _PYTHON_TYPES.get(column.type, Any)
            annotation: Any = python_type if column.required else python_type | None
            default: Any = ... if column.required else None
            fields[column.name] = (annotation, Field(default, strict=True))
        row = create_model(
            "Row", __config__=ConfigDict(extra="forbid", allow_inf_nan=False), **fields
        )
    else:
        row = create_model("Row", __config__=ConfigDict(extra="allow"))
    return create_model(
        "PredictionRequest",
        rows=(list[row], Field(..., min_length=1, max_length=max_rows)),  # type: ignore[valid-type]
    )


class ModelInfo(BaseModel):
    name: str
    version: str


class PredictionResponse(BaseModel):
    model: ModelInfo
    predictions: list[Any]
    request_id: str


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str
    details: list[dict[str, Any]] = []
