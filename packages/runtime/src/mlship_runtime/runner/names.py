"""Names of params and metrics that MLflow accepts on every OS (ADR-019).

MLflow checks them only when logging, at the end of a run, and its rule depends on
the client OS (``:`` is refused on Windows). mlship applies the strictest rule up
front. The CLI has the same rule (``mlship.domain.names``); a contract
test keeps both identical.
"""

import posixpath
import re

MAX_NAME_LENGTH = 250
_PORTABLE = re.compile(r"^[/\w.\- ]+$")
NAME_RULE = (
    "letters, digits, '_', '-', '.', ' ' and '/', up to 250 characters, "
    "and not a relative path such as '../x'"
)


def is_valid_name(name: str) -> bool:
    if not name or len(name) > MAX_NAME_LENGTH or not _PORTABLE.match(name):
        return False
    normalized = posixpath.normpath(name)
    return normalized == name and normalized != "." and not normalized.startswith(("..", "/"))
