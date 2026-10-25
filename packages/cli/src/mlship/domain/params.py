"""Rules for ``--param`` values, checked before any work (ADR-019).

Params are hyperparameters: they are stored in ``run.json``, the manifest and MLflow,
and kept forever. Anything MLflow would refuse at the end of a run is refused here,
and so is anything that looks like a secret.
"""

import re
from collections.abc import Mapping

from mlship.domain.names import NAME_RULE, is_valid_name
from mlship.domain.validation import Severity, ValidationIssue

MAX_PARAMS = 50  # MLflow takes 100 params + tags per batch; mlship adds its own tags
MAX_VALUE_LENGTH = 6_000  # MLflow's limit for a param value
_SECRET_LIKE = re.compile(
    r"pass(word|wd)?|secret|token|api[_-]?key|access[_-]?key|private[_-]?key|credential",
    re.IGNORECASE,
)


def _issue(code: str, message: str, hint: str) -> ValidationIssue:
    return ValidationIssue(code, Severity.ERROR, message, "--param", hint)


def param_issues(params: Mapping[str, str]) -> list[ValidationIssue]:
    """Every problem with the params, all at once."""
    issues: list[ValidationIssue] = []
    if len(params) > MAX_PARAMS:
        issues.append(
            _issue(
                "PARAMS_TOO_MANY",
                f"{len(params)} params given; the limit is {MAX_PARAMS}.",
                "Pass hyperparameters only; read larger configuration from a file in the project.",
            )
        )
    for key, value in params.items():
        if not is_valid_name(key):
            issues.append(
                _issue(
                    "PARAM_NAME_INVALID", f"Param name {key!r} is not allowed.", f"Use {NAME_RULE}."
                )
            )
        if _SECRET_LIKE.search(key):
            issues.append(
                _issue(
                    "PARAM_LOOKS_SECRET",
                    f"{key!r} looks like a secret; params are stored in plain text, forever.",
                    "Never pass credentials to training: the container has no network, "
                    "so it cannot use them anyway.",
                )
            )
        if len(value) > MAX_VALUE_LENGTH:
            issues.append(
                _issue(
                    "PARAM_VALUE_TOO_LONG",
                    f"The value of {key!r} has {len(value)} characters; the limit is "
                    f"{MAX_VALUE_LENGTH}.",
                    "Put long values in a file in the project and pass its name.",
                )
            )
    return issues
