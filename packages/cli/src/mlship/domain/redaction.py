"""Remove credentials embedded in URIs before anything is shown or logged."""

import re

_USERINFO = re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.\-]*://)[^/\s@]+@")


def redact(text: str) -> str:
    """``https://user:pass@host/x`` -> ``https://***@host/x``, anywhere in ``text``."""
    return _USERINFO.sub(r"\g<scheme>***@", text)
