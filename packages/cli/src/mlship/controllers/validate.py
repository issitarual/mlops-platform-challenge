"""``mlship validate``: report every contract error and warning at once."""

from pathlib import Path
from typing import Annotated

import typer

from mlship.controllers import presenters


def command(
    project: Annotated[Path, typer.Argument(help="Project directory.")] = Path(),
    data: Annotated[Path | None, typer.Option(help="Data directory to check too.")] = None,
    strict: Annotated[bool, typer.Option(help="Treat warnings as errors (CI).")] = False,
) -> None:
    """Check train.py, requirements.txt and data without running user code."""
    presenters.not_implemented("validate", "E5")
