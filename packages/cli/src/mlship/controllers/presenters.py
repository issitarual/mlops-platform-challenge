"""All terminal output lives here, so commands stay free of formatting code."""

from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from mlship.domain.environment import ResolvedLock
from mlship.domain.errors import MlshipError
from mlship.domain.redaction import redact
from mlship.domain.runs import Promotion, RunRecord, RunStatus
from mlship.domain.serving import PreparedServing
from mlship.domain.training import TrainingOutcome
from mlship.domain.validation import (
    ResolutionCheck,
    Severity,
    ValidationIssue,
    ValidationReport,
)

console = Console()
_STATUS_STYLE = {
    RunStatus.FINISHED: "[green]finished[/green]",
    RunStatus.FAILED: "[red]failed[/red]",
    RunStatus.KILLED: "[yellow]killed[/yellow]",
    RunStatus.RUNNING: "[cyan]running[/cyan]",
}
err_console = Console(stderr=True)


def lock_written(resolved: ResolvedLock) -> None:
    if resolved.reused:
        console.print("mlship.lock is current; nothing changed.")
        return
    note = " (the previous one had been edited by hand)" if resolved.discarded_modified else ""
    console.print(f"[green]Wrote[/green] mlship.lock{note}. Commit it with requirements.txt.")


def lock_deferred(exc: MlshipError) -> None:
    err_console.print(
        "[yellow]mlship.lock was not created[/yellow]: "
        f"{escape(redact(exc.message))}\n  Run `mlship lock` when the package index is "
        "reachable.",
        highlight=False,
    )


def project_created(target: Path, created: Sequence[Path]) -> None:
    names = ", ".join(p.relative_to(target).as_posix() for p in created)
    console.print(f"[green]Created[/green] {escape(str(target))}: {escape(names)}")
    enter = "" if target.resolve() == Path.cwd().resolve() else f"  cd {escape(str(target))}\n"
    console.print(
        f"Next:\n{enter}  mlship validate --data data\n  mlship train --data data",
        highlight=False,
    )


def issues(items: tuple[ValidationIssue, ...] | list[ValidationIssue]) -> None:
    """Every issue at once: where, what, and how to fix it.

    Messages quote the user's files (``pandas[performance]``), so they are escaped
    as Rich markup, and redacted like every other message.
    """
    for item in items:
        colour = "red" if item.severity is Severity.ERROR else "yellow"
        err_console.print(
            f"[{colour}]{item.severity.value}[/{colour}] [bold]{item.code}[/bold] "
            f"[dim]{escape(item.location)}[/dim]\n  {escape(redact(item.message))}",
            highlight=False,
        )
        if item.hint:
            err_console.print(f"  [green]fix:[/green] {escape(redact(item.hint))}", highlight=False)


_RESOLUTION = {
    ResolutionCheck.PASSED: "[green]resolved[/green] together with the mlship runtime",
    ResolutionCheck.FAILED: "[red]do not resolve[/red] together with the mlship runtime",
    ResolutionCheck.OFFLINE: "[dim]not resolved (--offline)[/dim]",
    ResolutionCheck.SKIPPED: "[dim]not resolved: fix the errors above first[/dim]",
}


def validation_result(
    report: ValidationReport,
    resolution: ResolutionCheck,
    *,
    strict: bool,
    project: Path,
    data: Path | None,
) -> None:
    issues(report.issues)
    if report.issues:
        err_console.print()
    err_console.print(f"Dependencies: {_RESOLUTION[resolution]}")
    if data is None:
        err_console.print("Data: [dim]not checked (pass --data)[/dim]")
    summary = f"{len(report.errors)} error(s), {len(report.warnings)} warning(s)"
    if report.blocks(strict=strict):
        reason = " (warnings count as errors with --strict)" if not report.errors else ""
        err_console.print(f"[red]Project is not valid[/red]: {summary}{reason}.")
    else:
        err_console.print(f"[green]Project is valid[/green]: {summary}.")
        if not report.issues:
            target = "" if project.resolve() == Path.cwd().resolve() else f" {project}"
            folder = data if data is not None else "<data folder>"
            err_console.print(
                f"Next: mlship train{escape(str(target))} --data {escape(str(folder))}",
                highlight=False,
            )


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
    table = Table("metric", "value", title=f"Run {outcome.run_id[:12]}", title_justify="left")
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
        "started",
        "status",
        "version",
        *metric_names,
        title=f"{model_name} (times in UTC)",
        title_justify="left",
    )
    for record in records:
        table.add_row(
            record.run_id[:12],
            record.created_at.strftime("%m-%d %H:%M"),
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


def serving_started(model_name: str, alias: str) -> None:
    err_console.print(
        f"Resolving [bold]{alias}[/bold] of [bold]{model_name}[/bold], "
        "then verifying and packaging the model…"
    )


def serving_prepared(prepared: PreparedServing) -> None:
    model = prepared.model
    source = "reused" if prepared.environment_reused else "rebuilt from the run's lock"
    err_console.print(
        f"{prepared.alias} → [bold]{model.name} v{model.version}[/bold] "
        f"(run {model.run_id[:12]}, sha256 {model.model_sha256[:12]}… verified)\n"
        f"Environment [cyan]{prepared.environment_image.tag}[/cyan] ({source})\n"
        f"Serving image [cyan]{prepared.serving_image.tag}[/cyan]",
        highlight=False,
    )


def serving_log(text: str) -> None:
    err_console.out(text, end="", style="dim", highlight=False)


def serving_ready(url: str) -> None:
    console.print(
        f"\n[green]Ready[/green]: {url}/v1/predict  ·  docs {url}/docs  ·  metrics {url}/metrics"
        "\nPress Ctrl+C to stop.\n",
        highlight=False,
    )


def serving_stopped() -> None:
    err_console.print("\n[yellow]Stopped[/yellow]; the API container was removed.")
