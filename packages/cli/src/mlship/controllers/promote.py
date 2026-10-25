"""``mlship promote``: point an alias at a model version."""

from pathlib import Path
from typing import Annotated

import typer

from mlship.controllers import presenters
from mlship.controllers.context import load_services
from mlship.domain.errors import MlshipError
from mlship.domain.runs import resolve_model_name


def command(
    version: Annotated[int, typer.Argument(min=1, help="Registered model version.")],
    alias: Annotated[str, typer.Option(help="Alias to move.")] = "champion",
    project: Annotated[Path, typer.Option(help="Project directory.")] = Path(),
    model_name: Annotated[
        str | None,
        typer.Option(
            envvar="MLSHIP_MODEL_NAME",
            help="Registered model name (default: the project folder name).",
        ),
    ] = None,
) -> None:
    """Promote a model version. Serving always follows an explicit alias."""
    services = load_services()
    try:
        name = resolve_model_name(project, model_name)
        result = services.registry.promote(name, version, alias)
    except MlshipError as exc:
        presenters.error(exc)
        raise typer.Exit(code=1) from exc
    presenters.promotion(result)
