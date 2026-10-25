"""``mlship runs``: list recent training runs and their metrics."""

from pathlib import Path
from typing import Annotated

import typer

from mlship.controllers import presenters
from mlship.controllers.context import load_services
from mlship.domain.errors import MlshipError
from mlship.domain.runs import resolve_model_name


def command(
    project: Annotated[Path, typer.Argument(help="Project directory.")] = Path(),
    model_name: Annotated[
        str | None,
        typer.Option(
            envvar="MLSHIP_MODEL_NAME",
            help="Registered model name (default: the project folder name).",
        ),
    ] = None,
    limit: Annotated[int, typer.Option(min=1, max=1000, help="How many runs to show.")] = 20,
) -> None:
    """Show recent runs of a model, newest first."""
    services = load_services()
    try:
        name = resolve_model_name(project, model_name)
        records = services.registry.list_runs(name, limit)
    except MlshipError as exc:
        presenters.error(exc)
        raise typer.Exit(code=1) from exc
    presenters.runs_table(name, records)
