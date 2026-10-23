"""Reads ``result.json`` written inside the container: untrusted data (ADR-018).

Size-limited, schema-checked and truncated. Nothing in it is executed or trusted
beyond what is checked here; the exit code wins when the file is missing or invalid.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlship.domain.training import TrainingJobResult
from mlship.domain.validation import Severity, ValidationIssue

MAX_RESULT_BYTES = 1_000_000
MAX_ERRORS = 100
MAX_TEXT = 2_000
_SERIALIZATIONS = {"skops", "cloudpickle"}
_SIGNALS: dict[int, tuple[str, str]] = {
    6: ("SIGABRT", "A native library aborted; see the log above."),
    7: (
        "SIGBUS",
        "Often a damaged environment image, e.g. built while the disk was full. "
        "Delete the image (docker image rm <tag>), free disk space and train again.",
    ),
    9: ("SIGKILL", "The process was killed from outside (Docker, or a manual stop)."),
    11: (
        "SIGSEGV",
        "A native library crashed; check that package versions are compatible.",
    ),
    15: ("SIGTERM", "The process was asked to stop (Docker or a manual stop)."),
}


@dataclass(frozen=True, slots=True)
class ParsedResult:
    succeeded: bool
    issues: tuple[ValidationIssue, ...]
    metrics: dict[str, float]
    serialization: str | None


def _issue(code: str, message: str, hint: str, location: str = "result.json") -> ValidationIssue:
    return ValidationIssue(code, Severity.ERROR, message, location, hint)


def _text(value: Any) -> str:
    return str(value)[:MAX_TEXT] if isinstance(value, str) else ""


def _job_issue(job: TrainingJobResult, limits_hint: str) -> ValidationIssue | None:
    if job.timed_out:
        return _issue(
            "TRAIN_TIMEOUT",
            "Training exceeded the time limit and was stopped.",
            f"Speed up training or raise the limit ({limits_hint}).",
            "container",
        )
    if job.oom_killed:
        return _issue(
            "TRAIN_OUT_OF_MEMORY",
            "Training ran out of memory and was stopped.",
            f"Use less memory or raise the limit ({limits_hint}).",
            "container",
        )
    return None


def _missing_result_issue(job: TrainingJobResult) -> ValidationIssue:
    silent = "" if job.logs_tail.strip() else " It wrote no output before stopping."
    signal_number = job.exit_code - 128
    if signal_number in _SIGNALS:
        name, hint = _SIGNALS[signal_number]
        return _issue(
            "TRAIN_CRASHED",
            f"The training process was terminated by {name} (exit code {job.exit_code}).{silent}",
            hint,
            "container",
        )
    return _issue(
        "RESULT_MISSING",
        f"Training ended (exit code {job.exit_code}) without a result.{silent}",
        "See the log above for the error.",
    )


def parse_result(path: Path, job: TrainingJobResult, limits_hint: str) -> ParsedResult:
    stopped = _job_issue(job, limits_hint)
    if stopped is not None:
        return ParsedResult(False, (stopped,), {}, None)
    if path.is_symlink() or not path.is_file():  # a link could point at host files
        return ParsedResult(False, (_missing_result_issue(job),), {}, None)
    if path.stat().st_size > MAX_RESULT_BYTES:
        return ParsedResult(
            False,
            (
                _issue(
                    "RESULT_TOO_LARGE",
                    "result.json is larger than allowed.",
                    "This is an mlship bug.",
                ),
            ),
            {},
            None,
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return ParsedResult(
            False,
            (_issue("RESULT_INVALID", "result.json is not valid JSON.", "This is an mlship bug."),),
            {},
            None,
        )
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        return ParsedResult(
            False,
            (
                _issue(
                    "RESULT_INVALID", "result.json has an unknown format.", "This is an mlship bug."
                ),
            ),
            {},
            None,
        )

    errors = raw.get("errors")
    error_list: list[Any] = errors[:MAX_ERRORS] if isinstance(errors, list) else []
    issues = tuple(
        _issue(
            _text(e.get("code")) or "TRAIN_ERROR",
            _text(e.get("message")),
            _text(e.get("hint")),
            _text(e.get("location")) or "train.py",
        )
        for e in error_list
        if isinstance(e, dict)
    )
    metrics_value = raw.get("metrics")
    metrics_raw: dict[Any, Any] = metrics_value if isinstance(metrics_value, dict) else {}
    metrics = {
        k[:MAX_TEXT]: float(v)
        for k, v in metrics_raw.items()
        if isinstance(k, str)
        and isinstance(v, int | float)
        and not isinstance(v, bool)
        and math.isfinite(v)
    }
    model_value = raw.get("model")
    model: dict[Any, Any] | None = model_value if isinstance(model_value, dict) else None
    serialization = model.get("serialization") if model else None
    if serialization not in _SERIALIZATIONS:
        serialization = None

    mlmodel = path.parent / "model" / "MLmodel"
    model_saved = serialization is not None and mlmodel.is_file() and not mlmodel.is_symlink()
    succeeded = (
        raw.get("status") == "succeeded" and not issues and job.exit_code == 0 and model_saved
    )
    if not succeeded and not issues:
        issues = (
            _issue(
                "RESULT_INCONSISTENT",
                f"Training reported success but exit code was {job.exit_code} "
                "or the model is missing.",
                "See the log above.",
            ),
        )
    return ParsedResult(succeeded, issues, metrics, serialization)
