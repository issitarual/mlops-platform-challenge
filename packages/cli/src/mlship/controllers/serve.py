"""``mlship serve``: run the REST API for the model behind an alias."""

from typing import Annotated

import typer

from mlship.controllers import presenters


def command(
    alias: Annotated[str, typer.Option(help="Model alias to serve.")] = "champion",
    port: Annotated[int, typer.Option(help="Local port.")] = 8080,
    workers: Annotated[int, typer.Option(help="API worker processes.")] = 1,
) -> None:
    """Resolve the alias, verify the model hash, build the image and start the API."""
    presenters.not_implemented("serve", "E4")
