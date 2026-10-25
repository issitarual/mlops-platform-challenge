"""Optional API key (``MLSHIP_SERVING_API_KEY``). Production uses IAM or JWT at the edge."""

import secrets

from fastapi import HTTPException, Request, status

API_KEY_HEADER = "X-API-Key"


def require_api_key(request: Request) -> None:
    expected = request.app.state.settings.api_key
    if expected is None:
        return
    given = request.headers.get(API_KEY_HEADER, "")
    if not secrets.compare_digest(given.encode(), expected.get_secret_value().encode()):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, detail=f"Missing or invalid {API_KEY_HEADER}."
        )
