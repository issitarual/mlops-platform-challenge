"""All terminal output lives here, so commands stay free of formatting code."""

from typing import NoReturn

import typer
from rich.console import Console

err_console = Console(stderr=True)


def not_implemented(command: str, epic: str) -> NoReturn:
    err_console.print(f"[yellow]mlship {command}[/yellow]: not implemented yet ({epic}).")
    raise typer.Exit(code=1)
