"""Typer application: registers the commands in workflow order."""

import typer

from mlship.controllers import init, lock, promote, runs, serve, train, validate

app = typer.Typer(
    name="mlship",
    help="From train.py to a REST API: isolated training, tracking and serving.",
    no_args_is_help=True,
    add_completion=False,
)
app.command("init")(init.command)
app.command("validate")(validate.command)
app.command("lock")(lock.command)
app.command("train")(train.command)
app.command("runs")(runs.command)
app.command("promote")(promote.command)
app.command("serve")(serve.command)


def main() -> None:
    app()
