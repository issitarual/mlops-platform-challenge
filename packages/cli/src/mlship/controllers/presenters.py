"""All terminal output lives here, so commands stay free of formatting code."""

from typing import NoReturn

import typer
from rich.console import Console
from rich.table import Table

from mlship.domain.errors import MlshipError
from mlship.domain.training import TrainingOutcome
from mlship.domain.validation import Severity, ValidationIssue, ValidationReport

console = Console()
err_console = Console(stderr=True)


def not_implemented(command: str, epic: str) -> NoReturn:
    err_console.print(f"[yellow]mlship {command}[/yellow]: not implemented yet ({epic}).")
    raise typer.Exit(code=1)


def issues(items: tuple[ValidationIssue, ...] | list[ValidationIssue]) -> None:
    """Every issue at once: where, what, and how to fix it."""
    for item in items:
        colour = "red" if item.severity is Severity.ERROR else "yellow"
        err_console.print(
            f"[{colour}]{item.severity.value}[/{colour}] [bold]{item.code}[/bold] "
            f"[dim]{item.location}[/dim]\n  {item.message}",
            highlight=False,
        )
        if item.hint:
            err_console.print(f"  [green]fix:[/green] {item.hint}", highlight=False)


def validation_failed(report: ValidationReport) -> None:
    err_console.print(
        f"[red]Project is not valid[/red]: {len(report.errors)} error(s), "
        f"{len(report.warnings)} warning(s).\n"
    )
    issues(report.issues)


def error(exc: MlshipError) -> None:
    err_console.print(f"[red]error:[/red] {exc.message}", highlight=False)
    if exc.hint:
        err_console.print(f"  [green]fix:[/green] {exc.hint}", highlight=False)
    if exc.details:
        err_console.print(f"[dim]{exc.details}[/dim]", highlight=False)


def training_log(text: str) -> None:
    err_console.out(text, end="", style="dim", highlight=False)


def training_started() -> None:
    err_console.print(
        "[bold]Validating the project[/bold], then preparing the environment (lock, image)…"
    )


def training_interrupted() -> None:
    err_console.print("\n[yellow]Training interrupted[/yellow]; the container was removed.")


def training_outcome(outcome: TrainingOutcome) -> None:
    env = outcome.environment
    source = "reused" if env.cache_hit else "built"
    err_console.print(f"\nEnvironment [cyan]{env.image.tag}[/cyan] ({source})", highlight=False)
    if outcome.warnings:
        issues(outcome.warnings)
    if not outcome.succeeded:
        err_console.print(f"\n[red]Training failed[/red] (run {outcome.run_id})\n")
        issues(outcome.issues)
        err_console.print(f"\nRun files: {outcome.run_dir}", highlight=False)
        return
    table = Table("metric", "value", title=f"Run {outcome.run_id}", title_justify="left")
    for name, value in sorted(outcome.metrics.items()):
        table.add_row(name, f"{value:.6g}")
    console.print(table)
    if outcome.serialization == "cloudpickle":
        err_console.print(
            "[yellow]warning:[/yellow] model saved with cloudpickle (--allow-pickle): "
            "loading it can run arbitrary code. Recorded with the run."
        )
    console.print(f"Model and metrics: {outcome.run_dir / 'output'}", highlight=False)
