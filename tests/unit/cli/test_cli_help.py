from typer.testing import CliRunner

from mlship.controllers.app import app

runner = CliRunner()
COMMANDS = ["init", "validate", "train", "runs", "promote", "serve"]


def test_help_lists_every_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in COMMANDS:
        assert command in result.output


def test_unimplemented_command_exits_with_error() -> None:
    result = runner.invoke(app, ["runs"])
    assert result.exit_code == 1
