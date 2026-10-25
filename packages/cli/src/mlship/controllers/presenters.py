"""All terminal output lives here, so commands stay free of formatting code."""

from collections.abc import Sequence
from typing import NoReturn

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from mlship.domain.errors import MlshipError
from mlship.domain.redaction import redact
from mlship.domain.runs import Promotion, RunRecord, RunStatus
from mlship.domain.training import TrainingOutcome
from mlship.domain.validation import Severity, ValidationIssue, ValidationReport

console = Console()
_STATUS_STYLE = {
    RunStatus.FINISHED: "[green]finished[/green]",
    RunStatus.FAILED: "[red]failed[/red]",
    RunStatus.KILLED: "[yellow]killed[/yellow]",
    RunStatus.RUNNING: "[cyan]running[/cyan]",
}
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
    """Credentials in URIs are masked: messages may quote the tracking URI or tool output."""
    err_console.print(f"[red]error:[/red] {escape(redact(exc.message))}", highlight=False)
    if exc.hint:
        err_console.print(f"  [green]fix:[/green] {escape(redact(exc.hint))}", highlight=False)
    if exc.details:
        err_console.print(f"[dim]{escape(redact(exc.details))}[/dim]", highlight=False)


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
    version = outcome.model_version
    if version is not None:
        console.print(
            f"Registered [bold]{version.name}[/bold] version [bold]{version.version}[/bold] "
            f"(sha256 {version.model_sha256[:12]}…)",
            highlight=False,
        )
        console.print(
            f"Next: [cyan]mlship promote {version.version} --model-name {version.name}[/cyan]",
            highlight=False,
        )
    console.print(f"Run files and manifest: {outcome.run_dir}", highlight=False)


def runs_table(model_name: str, records: Sequence[RunRecord]) -> None:
    if not records:
        err_console.print(f"No runs for [bold]{model_name}[/bold] yet. Train with `mlship train`.")
        return
    metric_names = sorted({name for record in records for name in record.metrics})
    table = Table(
        "run",
        "started (UTC)",
        "status",
        "version",
        *metric_names,
        title=model_name,
        title_justify="left",
    )
    for record in records:
        table.add_row(
            record.run_id[:12],
            record.created_at.strftime("%Y-%m-%d %H:%M"),
            _STATUS_STYLE.get(record.status, str(record.status)),
            "" if record.model_version is None else f"v{record.model_version}",
            *(
                f"{record.metrics[name]:.6g}" if name in record.metrics else ""
                for name in metric_names
            ),
        )
    console.print(table)


def promotion(result: Promotion) -> None:
    current = result.current
    if not result.changed:
        console.print(
            f"[bold]{result.alias}[/bold] already points at {current.name} "
            f"v{current.version}; nothing changed."
        )
        return
    before = f"v{result.previous.version}" if result.previous else "nothing"
    console.print(
        f"[bold]{result.alias}[/bold]: {before} → [bold]v{current.version}[/bold] "
        f"({current.name}, run {current.run_id[:12]})",
        highlight=False,
    )
    err_console.print(
        "[dim]No automatic comparison with the previous version yet; "
        "check the metrics with `mlship runs`. Serving picks this up on the next deploy.[/dim]"
    )
