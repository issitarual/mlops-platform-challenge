"""``HttpProbe``: does a local HTTP endpoint answer 200? (never follows other schemes)."""

import urllib.error
import urllib.request
from urllib.parse import urlparse


class UrllibProbe:
    def __init__(self, timeout_seconds: float = 2.0) -> None:
        self._timeout = timeout_seconds

    def answers(self, url: str) -> bool:
        if urlparse(url).scheme not in ("http", "https"):
            return False
        try:
            # Scheme checked above: no file:// or other handlers.
            with urllib.request.urlopen(url, timeout=self._timeout) as response:  # noqa: S310
                return bool(response.status == 200)
        except (urllib.error.URLError, OSError, ValueError):
            return False
