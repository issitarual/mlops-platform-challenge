"""``mlship serve``: run the REST API for the model behind an alias."""

import os
from pathlib import Path
from typing import Annotated

import typer

from mlship.controllers import presenters
from mlship.controllers.context import load_services
from mlship.domain.errors import MlshipError
from mlship.domain.runs import resolve_model_name

_FORWARDED = (
    "MLSHIP_SERVING_API_KEY",
    "MLSHIP_SERVING_MAX_BATCH_ROWS",
    "MLSHIP_SERVING_MAX_PAYLOAD_BYTES",
    "MLSHIP_SERVING_REQUEST_TIMEOUT_SECONDS",
    "MLSHIP_SERVING_PREDICTION_LOG_SAMPLE_RATE",
    "MLSHIP_SERVING_PREDICTION_LOG_EXCLUDE_COLUMNS",
)


def command(
    alias: Annotated[str, typer.Option(help="Model alias to serve.")] = "champion",
    port: Annotated[int, typer.Option(min=1, max=65535, help="Local port (127.0.0.1).")] = 8080,
    workers: Annotated[int, typer.Option(min=1, max=32, help="API worker processes.")] = 1,
    project: Annotated[Path, typer.Option(help="Project directory.")] = Path(),
    model_name: Annotated[
        str | None,
        typer.Option(
            envvar="MLSHIP_MODEL_NAME",
            help="Registered model name (default: the project folder name).",
        ),
    ] = None,
) -> None:
    """Resolve the alias, verify the model hash, build the image and start the API."""
    services = load_services()
    try:
        name = resolve_model_name(project, model_name)
        presenters.serving_started(name, alias)
        prepared = services.serving.prepare(project.resolve(), name, alias)
        presenters.serving_prepared(prepared)
        services.serving.run(
            prepared,
            port=port,
            workers=workers,
            environment={key: os.environ[key] for key in _FORWARDED if key in os.environ},
            on_log=presenters.serving_log,
            on_ready=presenters.serving_ready,
        )
    except MlshipError as exc:
        presenters.error(exc)
        raise typer.Exit(code=1) from exc
    except KeyboardInterrupt as exc:
        presenters.serving_stopped()
        raise typer.Exit(code=0) from exc
