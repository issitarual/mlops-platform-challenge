"""Shared setup for commands: settings and services, with configuration errors shown."""

import logging

import typer
from pydantic import ValidationError

from mlship.bootstrap import Services, build_services
from mlship.config import Settings
from mlship.controllers import presenters
from mlship.domain.errors import MlshipError


def load_services() -> Services:
    # MLflow's INFO lines ("waiting for model version", run links) would interleave
    # with the CLI output; mlship reports versions and runs itself.
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    try:
        return build_services(Settings())
    except ValidationError as exc:
        presenters.error(MlshipError("Invalid mlship configuration.", details=str(exc)))
        raise typer.Exit(code=2) from exc
