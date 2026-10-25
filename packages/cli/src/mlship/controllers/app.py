"""Typer application: registers the commands in workflow order."""

import codecs
import io
import sys

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
    use_utf8_output()
    app()


def use_utf8_output() -> None:
    """Write UTF-8 even when the output is redirected.

    On Windows, a pipe or a file gets the ANSI code page (cp1252), where the arrows
    mlship prints and the emoji MLflow prints do not exist: ``mlship train > log.txt``
    or a CI log would crash with UnicodeEncodeError. A terminal already gets UTF-8.
    """
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper) and codecs.lookup(stream.encoding).name != "utf-8":
            stream.reconfigure(encoding="utf-8", errors="replace")
