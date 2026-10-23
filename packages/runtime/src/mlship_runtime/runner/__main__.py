"""Entry point: ``python -m mlship_runtime.runner``. Exit 0 ok, 1 failed, 2 internal."""

import os
import platform
import sys
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mlship_runtime.runner.config import RunConfig, load_config
from mlship_runtime.runner.issues import Issue, RunnerError
from mlship_runtime.runner.loader import load_train_function
from mlship_runtime.runner.output_validation import validate_train_output
from mlship_runtime.runner.seeds import set_global_seeds
from mlship_runtime.runner.writer import SavedModel, save_model, write_json

RESULT_SCHEMA_VERSION = 1
DEFAULT_OUTPUT = Path("/output")


def _result(
    started: float,
    issues: list[Issue],
    metrics: Mapping[str, float] | None = None,
    saved: SavedModel | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "failed" if issues else "succeeded",
        "errors": [
            {"code": i.code, "message": i.message, "hint": i.hint, "location": i.location}
            for i in issues
        ],
        "metrics": dict(metrics or {}),
        "model": None
        if saved is None
        else {
            "path": "model",
            "serialization": saved.serialization,
            "trusted_types": list(saved.trusted_types),
        },
        "python_version": platform.python_version(),
        "duration_seconds": round(time.monotonic() - started, 3),
    }


def run(config: RunConfig, started: float) -> dict[str, Any]:
    set_global_seeds(config.seed)
    try:
        train = load_train_function(config.project_dir)
    except RunnerError as exc:
        return _result(started, exc.issues)
    try:
        output = train(config.data_dir, dict(config.params))
    except Exception as exc:
        traceback.print_exc()
        return _result(
            started,
            [
                Issue(
                    "TRAIN_RAISED",
                    f"train() raised {type(exc).__name__}: {exc}",
                    "See the traceback in the log above.",
                )
            ],
        )

    issues, predictions = validate_train_output(output)
    if issues:
        return _result(started, issues)
    try:
        saved, save_issues = save_model(
            output.model,  # type: ignore[attr-defined]
            output.input_example,  # type: ignore[attr-defined]
            predictions,
            config.output_dir / "model",
            config.serialization,
        )
    except Exception as exc:  # usually an input_example MLflow cannot describe
        traceback.print_exc()
        return _result(
            started,
            [
                Issue(
                    "MODEL_SAVE_FAILED",
                    f"Saving the model raised {type(exc).__name__}: {exc}",
                    "input_example should be a pandas DataFrame or a numpy array with "
                    "the columns the model expects. See the traceback in the log.",
                )
            ],
        )
    metrics = {k: float(v) for k, v in output.metrics.items()}  # type: ignore[attr-defined]
    write_json(config.output_dir / "metrics.json", metrics)
    return _result(started, save_issues, metrics, saved)


def main() -> int:
    started = time.monotonic()
    config_path = Path(os.environ.get("MLSHIP_RUN_CONFIG", "/mlship/run.json"))
    try:
        config = load_config(config_path)
    except (OSError, ValueError) as exc:
        issue = Issue("RUN_CONFIG_INVALID", str(exc), "This is an mlship bug.", "run.json")
        write_json(DEFAULT_OUTPUT / "result.json", _result(started, [issue]))
        return 2
    try:
        result = run(config, started)
    except Exception as exc:  # a bug in the runner, not in the user's code
        traceback.print_exc()
        issue = Issue("RUNNER_INTERNAL", f"{type(exc).__name__}: {exc}", "This is an mlship bug.")
        write_json(config.output_dir / "result.json", _result(started, [issue]))
        return 2
    write_json(config.output_dir / "result.json", result)
    print(f"mlship runner: {result['status']}", file=sys.stderr)
    return 0 if result["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
