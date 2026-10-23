"""Import the user's ``train.py`` and return its ``train`` function."""

import importlib.util
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mlship_runtime.runner.issues import Issue, RunnerError

TrainFunction = Callable[[Path, dict[str, Any]], object]


def load_train_function(project_dir: Path) -> TrainFunction:
    """Import ``project_dir/train.py`` (mounted read-only) and return ``train``."""
    path = project_dir / "train.py"
    if not path.is_file():
        raise RunnerError(
            [Issue("TRAIN_FILE_MISSING", "train.py was not found.", "Add train.py to the project.")]
        )
    # Helper modules next to train.py stay importable, as when running it locally.
    sys.path.insert(0, str(project_dir))
    spec = importlib.util.spec_from_file_location("train", path)
    if spec is None or spec.loader is None:
        raise RunnerError([Issue("TRAIN_IMPORT_FAILED", "train.py cannot be loaded.", "")])
    module = importlib.util.module_from_spec(spec)
    sys.modules["train"] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        traceback.print_exc()
        raise RunnerError(
            [
                Issue(
                    "TRAIN_IMPORT_FAILED",
                    f"Importing train.py raised {type(exc).__name__}: {exc}",
                    "See the traceback in the log. Code at module level runs at import.",
                )
            ]
        ) from exc
    train = getattr(module, "train", None)
    if not callable(train):
        raise RunnerError(
            [
                Issue(
                    "TRAIN_FUNCTION_MISSING",
                    "train.py has no function named train.",
                    "Define train(data_dir: Path, params: dict) -> TrainOutput.",
                )
            ]
        )
    return train
