"""``mlship validate``: report every contract error and warning at once."""

import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from mlship.controllers import presenters
from mlship.controllers.context import load_services
from mlship.domain.errors import MlshipError
from mlship.domain.redaction import redact

JSON_SCHEMA_VERSION = 1


class OutputFormat(StrEnum):
    TEXT = "text"
    JSON = "json"


def command(
    project: Annotated[Path, typer.Argument(help="Project directory.")] = Path(),
    data: Annotated[Path | None, typer.Option(help="Data directory to check too.")] = None,
    strict: Annotated[bool, typer.Option(help="Treat warnings as errors (CI).")] = False,
    offline: Annotated[
        bool,
        typer.Option(help="Skip resolving requirements.txt with the runtime (no network)."),
    ] = False,
    output: Annotated[
        OutputFormat, typer.Option("--format", help="text for people, json for CI.")
    ] = OutputFormat.TEXT,
) -> None:
    """Check train.py, requirements.txt and data without running user code."""
    services = load_services()
    try:
        report, resolution = services.validation.validate(
            project.resolve(), data.resolve() if data else None, offline=offline
        )
    except MlshipError as exc:
        presenters.error(exc)
        raise typer.Exit(code=2) from exc
    blocked = report.blocks(strict=strict)
    if output is OutputFormat.JSON:
        document = {
            "schema_version": JSON_SCHEMA_VERSION,
            "valid": not blocked,
            "strict": strict,
            "resolution": resolution.value,
            "errors": len(report.errors),
            "warnings": len(report.warnings),
            "issues": [
                {
                    "code": i.code,
                    "severity": i.severity.value,
                    "message": redact(i.message),
                    "location": i.location,
                    "hint": redact(i.hint),
                }
                for i in report.issues
            ],
        }
        typer.echo(json.dumps(document, indent=2))
    else:
        presenters.validation_result(report, resolution, strict=strict, project=project, data=data)
    raise typer.Exit(code=1 if blocked else 0)
