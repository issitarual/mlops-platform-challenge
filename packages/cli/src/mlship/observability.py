"""Structured JSON logs with run/request/model context.

Named ``observability`` (not ``logging``) to avoid shadowing the standard library.
"""

import json
import logging
from contextvars import ContextVar
from datetime import UTC, datetime

_CONTEXT: dict[str, ContextVar[str | None]] = {
    "run_id": ContextVar("run_id", default=None),
    "request_id": ContextVar("request_id", default=None),
    "model_version": ContextVar("model_version", default=None),
}


def bind(**values: str) -> None:
    """Attach context fields (run_id, request_id, model_version) to later log lines."""
    for key, value in values.items():
        _CONTEXT[key].set(value)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, var in _CONTEXT.items():
            if (value := var.get()) is not None:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
