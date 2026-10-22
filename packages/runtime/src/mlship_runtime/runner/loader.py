"""Import the user's ``train.py`` and return its ``train`` function."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

TrainFunction = Callable[[Path, dict[str, Any]], object]


def load_train_function(project_dir: Path) -> TrainFunction:
    """Import ``project_dir/train.py`` (mounted read-only) and return ``train``."""
    raise NotImplementedError("E2")
