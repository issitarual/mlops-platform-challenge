"""``mlship serve``: run the REST API for the model behind an alias."""

import os
import signal
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Annotated

import typer

from mlship.controllers import presenters
from mlship.controllers.context import load_services
from mlship.domain.errors import MlshipError
from mlship.domain.runs import resolve_model_name
from mlship.domain.serving import SERVING_DEFAULT_PORT

_FORWARDED = (
    "MLSHIP_SERVING_API_KEY",
    "MLSHIP_SERVING_MAX_BATCH_ROWS",
    "MLSHIP_SERVING_MAX_PAYLOAD_BYTES",
    "MLSHIP_SERVING_REQUEST_TIMEOUT_SECONDS",
    "MLSHIP_SERVING_PREDICTION_LOG_SAMPLE_RATE",
    "MLSHIP_SERVING_PREDICTION_LOG_EXCLUDE_COLUMNS",
    "MLSHIP_SERVING_DRIFT_WINDOW_ROWS",
    "MLSHIP_SERVING_DRIFT_MIN_ROWS",
)


@contextmanager
def _stop_like_ctrl_c() -> Iterator[None]:
    """SIGTERM (``docker stop``, CI runners, systemd) and Windows Ctrl+Break stop the API
    the way Ctrl+C does, so the container is always removed on the way out."""

    def interrupt(signum: int, frame: FrameType | None) -> None:
        raise KeyboardInterrupt

    names = ("SIGTERM", "SIGBREAK")
    signals = [getattr(signal, n) for n in names if hasattr(signal, n)]
    previous = {s: signal.signal(s, interrupt) for s in signals}
    try:
        yield
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)


def command(
    alias: Annotated[str, typer.Option(help="Model alias to serve.")] = "champion",
    port: Annotated[
        int, typer.Option(min=1, max=65535, help="Local port (127.0.0.1).")
    ] = SERVING_DEFAULT_PORT,
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
    with _stop_like_ctrl_c():
        try:
            name = resolve_model_name(project, model_name)
            presenters.serving_started(name, alias)
            prepared = services.serving.prepare(project.resolve(), name, alias)
            presenters.serving_prepared(prepared)
            presenters.serving_dashboard(services.serving.dashboard())
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
