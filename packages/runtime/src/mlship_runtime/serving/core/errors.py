"""Every error is JSON with a code, a message and the request id; never a stack trace."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from mlship_runtime.serving.core import request_id

_CODES = {
    401: "unauthorized",
    404: "not_found",
    405: "method_not_allowed",
    413: "payload_too_large",
    503: "not_ready",
    504: "timeout",
}
INTERNAL_ERROR_MESSAGE = "The model failed to answer this request."


def payload(
    rid: str, error: str, message: str, details: list[dict[str, object]] | None = None
) -> dict[str, object]:
    return {
        "error": error,
        "message": message,
        "request_id": rid,
        "details": details or [],
    }


def install(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        details: list[dict[str, object]] = [
            {
                "loc": ".".join(str(p) for p in e["loc"] if p != "body"),
                "msg": e["msg"],
                "type": e["type"],
            }
            for e in exc.errors()
        ]
        return JSONResponse(
            payload(
                request_id.of(request),
                "invalid_request",
                f"{len(details)} error(s) in the request.",
                details,
            ),
            status_code=422,
        )

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        code = _CODES.get(exc.status_code, "http_error")
        return JSONResponse(
            payload(request_id.of(request), code, str(exc.detail)),
            status_code=exc.status_code,
            headers=getattr(exc, "headers", None),
        )
