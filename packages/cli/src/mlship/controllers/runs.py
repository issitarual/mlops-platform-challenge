"""``mlship runs``: list recent training runs and their metrics."""

from typing import Annotated

import typer

from mlship.controllers import presenters


def command(
    limit: Annotated[int, typer.Option(help="How many runs to show.")] = 20,
) -> None:
    """Show recent runs, newest first."""
    presenters.not_implemented("runs", "E3")
