"""The request id: the client's own when it is safe to echo, a new one otherwise."""

import re
import uuid

from starlette.requests import Request
from starlette.types import Scope

HEADER = "X-Request-ID"
_SAFE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def assign(request: Request) -> str:
    incoming = request.headers.get(HEADER, "")
    request.state.request_id = incoming if _SAFE.match(incoming) else uuid.uuid4().hex
    return str(request.state.request_id)


def of(request: Request) -> str:
    return str(getattr(request.state, "request_id", ""))


def of_scope(scope: Scope) -> str:
    return str(scope.get("state", {}).get("request_id", ""))
