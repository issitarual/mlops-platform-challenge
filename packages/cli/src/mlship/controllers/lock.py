"""``mlship lock``: resolve requirements.txt with the runtime into mlship.lock."""

from pathlib import Path
from typing import Annotated

import typer

from mlship.controllers import presenters
from mlship.controllers.context import load_services
from mlship.domain.errors import MlshipError, ProjectInvalidError


def command(
    project: Annotated[Path, typer.Argument(help="Project directory.")] = Path(),
) -> None:
    """Pin every dependency with hashes; commit mlship.lock next to requirements.txt."""
    services = load_services()
    try:
        resolved = services.validation.lock(project.resolve())
    except ProjectInvalidError as exc:
        presenters.validation_failed(exc.report)
        raise typer.Exit(code=1) from exc
    except MlshipError as exc:
        presenters.error(exc)
        raise typer.Exit(code=1) from exc
    presenters.lock_written(resolved)
