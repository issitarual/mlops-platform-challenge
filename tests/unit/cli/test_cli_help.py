import io
import sys

import pytest
from typer.testing import CliRunner

from mlship.controllers.app import app, use_utf8_output

runner = CliRunner()
COMMANDS = ["init", "validate", "lock", "train", "runs", "promote", "serve"]


def test_help_lists_every_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in COMMANDS:
        assert command in result.output


def test_redirected_output_on_windows_does_not_crash_on_non_ascii(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Windows pipe uses cp1252, which has no emoji: MLflow's run link would crash."""
    raw = io.BytesIO()
    pipe = io.TextIOWrapper(raw, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", pipe)
    use_utf8_output()
    print("\U0001f3c3 View run · v3 \u2192 v4", end="")  # no newline: Windows writes \r\n
    pipe.flush()
    assert raw.getvalue().decode("utf-8") == "\U0001f3c3 View run · v3 \u2192 v4"
