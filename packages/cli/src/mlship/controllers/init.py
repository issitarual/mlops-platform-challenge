"""``mlship init``: scaffold a project that already satisfies the contract."""

from pathlib import Path
from typing import Annotated

import typer

from mlship.controllers import presenters
from mlship.controllers.context import load_services
from mlship.domain.errors import MlshipError


def command(
    path: Annotated[Path, typer.Argument(help="Directory for the new project.")] = Path(),
    force: Annotated[bool, typer.Option(help="Overwrite files that already exist.")] = False,
) -> None:
    """Create train.py, requirements.txt, sample data and mlship.lock from a template."""
    services = load_services()
    try:
        created = services.scaffold.create(path, force=force)
    except MlshipError as exc:
        presenters.error(exc)
        raise typer.Exit(code=1) from exc
    presenters.project_created(path, created)
    try:
        presenters.lock_written(services.validation.lock(path.resolve()))
    except MlshipError as exc:
        presenters.lock_deferred(exc)
