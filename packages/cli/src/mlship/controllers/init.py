"""``mlship init``: scaffold a project that already satisfies the contract."""

from pathlib import Path
from typing import Annotated

import typer

from mlship.controllers import presenters


def command(
    path: Annotated[Path, typer.Argument(help="Directory for the new project.")] = Path(),
) -> None:
    """Create train.py and requirements.txt from a working template."""
    presenters.not_implemented("init", "E0/E5")
