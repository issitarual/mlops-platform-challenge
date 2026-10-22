"""``mlship promote``: point an alias at a model version."""

from typing import Annotated

import typer

from mlship.controllers import presenters


def command(
    version: Annotated[int, typer.Argument(help="Registered model version.")],
    alias: Annotated[str, typer.Option(help="Alias to move.")] = "champion",
) -> None:
    """Promote a model version. Serving always follows an explicit alias."""
    presenters.not_implemented("promote", "E3")
