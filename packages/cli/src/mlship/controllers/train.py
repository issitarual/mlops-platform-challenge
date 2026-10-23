"""``mlship train``: isolated, reproducible training run."""

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from mlship.bootstrap import build_services
from mlship.config import Settings
from mlship.controllers import presenters
from mlship.domain.errors import MlshipError, ProjectInvalidError
from mlship.domain.validation import Severity, ValidationIssue


def parse_params(raw: list[str]) -> tuple[dict[str, str], list[ValidationIssue]]:
    """KEY=VALUE pairs; every malformed one is reported, not just the first."""
    params: dict[str, str] = {}
    problems: list[ValidationIssue] = []
    for item in raw:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            problems.append(
                ValidationIssue(
                    "PARAM_FORMAT",
                    Severity.ERROR,
                    f"{item!r} is not KEY=VALUE.",
                    "--param",
                    "Example: -p learning_rate=0.05",
                )
            )
            continue
        if key.strip() in params:
            problems.append(
                ValidationIssue(
                    "PARAM_DUPLICATE",
                    Severity.ERROR,
                    f"{key.strip()!r} is given twice.",
                    "--param",
                    "Pass each parameter once.",
                )
            )
        params[key.strip()] = value
    return params, problems


def command(
    data: Annotated[Path, typer.Option(help="Training data directory (mounted read-only).")],
    project: Annotated[Path, typer.Argument(help="Project directory.")] = Path(),
    param: Annotated[
        list[str] | None,
        typer.Option("--param", "-p", help="KEY=VALUE passed to train(); repeatable."),
    ] = None,
    allow_pickle: Annotated[
        bool,
        typer.Option(
            help="Allow models that are not scikit-learn estimators, saved with cloudpickle. "
            "Loading such a model can run arbitrary code; the choice is recorded."
        ),
    ] = False,
) -> None:
    """Validate, build the environment and train inside a locked-down container."""
    params, problems = parse_params(param or [])
    if problems:
        presenters.issues(problems)
        raise typer.Exit(code=2)
    try:
        services = build_services(Settings())
    except ValidationError as exc:
        presenters.error(MlshipError("Invalid mlship configuration.", details=str(exc)))
        raise typer.Exit(code=2) from exc

    presenters.training_started()
    try:
        outcome = services.training.train(
            project.resolve(),
            data.resolve(),
            params,
            allow_pickle=allow_pickle,
            on_log=presenters.training_log,
        )
    except ProjectInvalidError as exc:
        presenters.validation_failed(exc.report)
        raise typer.Exit(code=1) from exc
    except MlshipError as exc:
        presenters.error(exc)
        raise typer.Exit(code=1) from exc
    except KeyboardInterrupt as exc:
        presenters.training_interrupted()
        raise typer.Exit(code=130) from exc
    presenters.training_outcome(outcome)
    raise typer.Exit(code=0 if outcome.succeeded else 1)
