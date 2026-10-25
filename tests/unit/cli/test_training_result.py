import json
import sys
from pathlib import Path

import pytest

from mlship.domain.training import TrainingJobResult
from mlship.services.training_result import MAX_RESULT_BYTES, MAX_TEXT, parse_result

OK_JOB = TrainingJobResult(exit_code=0, timed_out=False, oom_killed=False, logs_tail="")
GOOD = {
    "schema_version": 1,
    "status": "succeeded",
    "errors": [],
    "metrics": {"rmse": 0.5},
    "model": {"path": "model", "serialization": "skops", "trusted_types": []},
}


@pytest.fixture
def output(tmp_path: Path) -> Path:
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "MLmodel").write_text("x")
    return tmp_path


def _parse(output: Path, payload: object, job: TrainingJobResult = OK_JOB):  # type: ignore[no-untyped-def]
    (output / "result.json").write_text(
        payload if isinstance(payload, str) else json.dumps(payload)
    )
    return parse_result(output / "result.json", job, "limits")


def test_good_result(output: Path) -> None:
    parsed = _parse(output, GOOD)
    assert parsed.succeeded and parsed.metrics == {"rmse": 0.5} and parsed.serialization == "skops"


def test_missing_result_uses_the_exit_code(tmp_path: Path) -> None:
    parsed = parse_result(tmp_path / "result.json", TrainingJobResult(2, False, False, ""), "l")
    assert not parsed.succeeded
    assert parsed.issues[0].code == "RESULT_MISSING"
    assert "exit code 2" in parsed.issues[0].message


@pytest.mark.parametrize(
    ("job", "code"),
    [
        (TrainingJobResult(137, True, False, ""), "TRAIN_TIMEOUT"),
        (TrainingJobResult(137, False, True, ""), "TRAIN_OUT_OF_MEMORY"),
    ],
)
def test_stopped_jobs_win_over_the_result_file(
    output: Path, job: TrainingJobResult, code: str
) -> None:
    parsed = _parse(output, GOOD, job)
    assert not parsed.succeeded
    assert [i.code for i in parsed.issues] == [code]


def test_invalid_json_is_not_trusted(output: Path) -> None:
    assert _parse(output, "{not json").issues[0].code == "RESULT_INVALID"


def test_unknown_schema_is_not_trusted(output: Path) -> None:
    assert _parse(output, {**GOOD, "schema_version": 9}).issues[0].code == "RESULT_INVALID"


def test_oversized_result_is_not_read(output: Path) -> None:
    parsed = _parse(output, "x" * (MAX_RESULT_BYTES + 1))
    assert parsed.issues[0].code == "RESULT_TOO_LARGE"


def test_untrusted_fields_are_sanitized(output: Path) -> None:
    payload = {
        **GOOD,
        "status": "failed",
        "errors": [{"code": "X", "message": "m" * (MAX_TEXT * 5), "hint": 3}],
        "metrics": {"rmse": "0.5", "ok": 1.0, "nan": float("nan"), "flag": True},
        "model": {"serialization": "exec-me"},
    }
    parsed = _parse(output, payload)
    assert len(parsed.issues[0].message) == MAX_TEXT
    assert parsed.issues[0].hint == ""
    assert parsed.metrics == {}, "a result that breaks the metric contract keeps no metrics"
    assert "RESULT_METRICS_INVALID" in {i.code for i in parsed.issues}
    assert parsed.serialization is None


@pytest.mark.parametrize(
    "metrics",
    [{"rmse (test)": 0.5}, {"../escape": 0.5}, {f"m{i}": 0.1 for i in range(101)}],
)
def test_metrics_that_mlflow_would_reject_fail_the_run(output: Path, metrics: object) -> None:
    parsed = _parse(output, {**GOOD, "metrics": metrics})
    assert not parsed.succeeded
    assert [i.code for i in parsed.issues] == ["RESULT_METRICS_INVALID"]


def test_success_claim_without_model_is_inconsistent(tmp_path: Path) -> None:
    (tmp_path / "result.json").write_text(json.dumps(GOOD))
    parsed = parse_result(tmp_path / "result.json", OK_JOB, "l")
    assert not parsed.succeeded
    assert parsed.issues[0].code == "RESULT_INCONSISTENT"


@pytest.mark.parametrize(
    ("exit_code", "name", "hint_word"),
    [(135, "SIGBUS", "disk"), (139, "SIGSEGV", "native"), (137, "SIGKILL", "killed")],
)
def test_signal_exit_codes_are_explained(
    tmp_path: Path, exit_code: int, name: str, hint_word: str
) -> None:
    job = TrainingJobResult(exit_code, False, False, "")
    issue = parse_result(tmp_path / "result.json", job, "l").issues[0]
    assert issue.code == "TRAIN_CRASHED"
    assert name in issue.message and hint_word in issue.hint
    assert "no output" in issue.message


def test_a_crash_with_logs_does_not_claim_silence(tmp_path: Path) -> None:
    job = TrainingJobResult(139, False, False, "Traceback ...\n")
    issue = parse_result(tmp_path / "result.json", job, "l").issues[0]
    assert "no output" not in issue.message


NO_SYMLINKS = pytest.mark.skipif(
    sys.platform == "win32", reason="creating symlinks needs extra rights on Windows"
)


@NO_SYMLINKS
def test_result_that_is_a_symlink_is_not_followed(output: Path) -> None:
    secret = output / "pretend-host-file.json"
    secret.write_text(json.dumps(GOOD))
    (output / "result.json").symlink_to(secret)
    parsed = parse_result(output / "result.json", OK_JOB, "l")
    assert not parsed.succeeded
    assert parsed.issues[0].code == "RESULT_MISSING"


@NO_SYMLINKS
def test_model_descriptor_that_is_a_symlink_is_not_a_saved_model(output: Path) -> None:
    (output / "elsewhere").write_text("x")
    (output / "model" / "MLmodel").unlink()
    (output / "model" / "MLmodel").symlink_to(output / "elsewhere")
    assert not _parse(output, GOOD).succeeded
