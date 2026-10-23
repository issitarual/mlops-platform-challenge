import subprocess
from pathlib import Path
from typing import Any

import pytest

from mlship.adapters.uv_lock_resolver import UvLockResolver
from mlship.domain.errors import LockResolutionError


class RecordingRunner:
    def __init__(self, returncode: int = 0, stderr: str = "") -> None:
        self.returncode = returncode
        self.stderr = stderr
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, command: list[str], **kwargs: Any) -> "subprocess.CompletedProcess[str]":
        self.calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, self.returncode, "", self.stderr)


@pytest.fixture
def files(tmp_path: Path) -> list[Path]:
    paths = [tmp_path / "mlship-runtime.in", tmp_path / "requirements.txt"]
    for p in paths:
        p.write_text("")
    return paths


@pytest.fixture(autouse=True)
def _uv_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")


def test_command_targets_the_image_platform_with_hashes_and_wheels_only(
    files: list[Path],
) -> None:
    runner = RecordingRunner()
    UvLockResolver("3.12", "x86_64-manylinux_2_28", runner=runner).resolve(
        files, files[0].parent / "lock.txt"
    )
    command, kwargs = runner.calls[0]
    assert command[:5] == ["/usr/bin/uv", "pip", "compile", "mlship-runtime.in", "requirements.txt"]
    for flag in ["--generate-hashes", "--no-build", "--no-header"]:
        assert flag in command
    assert command[command.index("--python-version") + 1] == "3.12"
    assert command[command.index("--python-platform") + 1] == "x86_64-manylinux_2_28"
    assert kwargs["cwd"] == files[0].parent
    assert "shell" not in kwargs


def test_failure_raises_with_uv_output(files: list[Path]) -> None:
    runner = RecordingRunner(returncode=1, stderr="No solution found: foo==9 does not exist")
    with pytest.raises(LockResolutionError) as info:
        UvLockResolver("3.12", "linux", runner=runner).resolve(files, files[0].parent / "l.txt")
    assert "foo==9" in info.value.details


def test_inputs_must_share_the_output_directory(files: list[Path], tmp_path: Path) -> None:
    elsewhere = tmp_path / "sub"
    elsewhere.mkdir()
    with pytest.raises(ValueError, match="one directory"):
        UvLockResolver("3.12", "linux", runner=RecordingRunner()).resolve(
            files, elsewhere / "lock.txt"
        )
