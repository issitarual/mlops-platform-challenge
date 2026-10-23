"""Problems found by the runner, reported all at once in ``result.json``."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Issue:
    code: str  # stable identifier, e.g. "METRIC_NOT_FINITE"
    message: str  # what is wrong
    hint: str  # how to fix it
    location: str = "train.py"


class RunnerError(Exception):
    """Stops the run early, carrying every issue found so far."""

    def __init__(self, issues: list[Issue]) -> None:
        super().__init__("; ".join(i.message for i in issues))
        self.issues = issues
