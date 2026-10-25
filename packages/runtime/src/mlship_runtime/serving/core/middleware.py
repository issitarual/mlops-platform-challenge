"""Request plumbing: request id, access log, metrics, unexpected errors and payload limit."""

import json
import logging
import time
from collections.abc import Awaitable, Callable

from fastapi.responses import JSONResponse
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from mlship_runtime.serving.core import errors, metrics, request_id

access_log = logging.getLogger("mlship.access")
error_log = logging.getLogger("mlship.errors")


async def request_context(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Outermost layer, so every response is counted and logged: 413 and 500 included.

    An unhandled exception is turned into the JSON 500 here, not in an exception
    handler: Starlette runs that handler outside every middleware, where the response
    would miss the metrics, the access log and the request id header.
    """
    rid = request_id.assign(request)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        error_log.exception("unhandled error", extra={"request_id": rid})
        response = JSONResponse(
            errors.payload(rid, "internal_error", errors.INTERNAL_ERROR_MESSAGE),
            status_code=500,
        )
    elapsed = time.perf_counter() - started
    template = getattr(request.scope.get("route"), "path", "unmatched")
    metrics.REQUESTS.labels(template, str(response.status_code)).inc()
    metrics.LATENCY.labels(template).observe(elapsed)
    response.headers[request_id.HEADER] = rid
    model = getattr(request.app.state, "engine", None)
    access_log.info(
        "request",
        extra={
            "request_id": rid,
            "method": request.method,
            "path": template,
            "status": response.status_code,
            "latency_ms": round(elapsed * 1000, 2),
            "model_name": getattr(model, "model_name", None),
            "model_version": getattr(model, "model_version", None),
        },
    )
    return response


class BodySizeLimit:
    """Read the body up to ``max_bytes`` before the app sees it; 413 above that.

    Buffering (at most ``max_bytes``) is what makes the limit hold for chunked bodies
    too: an error raised while the app is reading would be turned into a 400 by the
    body parser.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = dict(scope.get("headers", [])).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            await self._reject(scope, send)
            return
        chunks: list[bytes] = []
        size = 0
        more = True
        while more:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > self.max_bytes:
                await self._reject(scope, send)
                return
            chunks.append(chunk)
            more = message.get("more_body", False)
        body = b"".join(chunks)
        delivered = False

        async def replay() -> Message:
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, replay, send)

    async def _reject(self, scope: Scope, send: Send) -> None:
        body = json.dumps(
            errors.payload(
                request_id.of_scope(scope),
                "payload_too_large",
                f"Request body is larger than {self.max_bytes} bytes.",
            )
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
