"""``mlship train``: isolated, reproducible training run."""

from pathlib import Path
from typing import Annotated

import typer

from mlship.controllers import presenters


def command(
    data: Annotated[Path, typer.Option(help="Training data directory (mounted read-only).")],
    project: Annotated[Path, typer.Argument(help="Project directory.")] = Path(),
    param: Annotated[
        list[str] | None,
        typer.Option("--param", "-p", help="KEY=VALUE passed to train(); repeatable."),
    ] = None,
) -> None:
    """Validate, build the environment and train inside a locked-down container."""
    presenters.not_implemented("train", "E1-E3")
