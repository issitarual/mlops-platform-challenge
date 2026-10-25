import subprocess
from typing import Any

import pytest

from mlship.adapters.tool_info import GitToolInfo


def _git(head: str = "f00d\n", status: str = "", code: int = 0) -> Any:
    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        output = head if "rev-parse" in command else status
        return subprocess.CompletedProcess(command, code, output, "")

    return runner


def test_clean_checkout_reports_commit_and_not_dirty() -> None:
    info = GitToolInfo(_git()).info()
    assert info.git_sha == "f00d" and info.git_dirty is False


def test_uncommitted_tool_changes_are_reported_as_dirty() -> None:
    assert GitToolInfo(_git(status=" M packages/cli/x.py\n")).info().git_dirty is True


def test_not_a_git_checkout_is_unknown_not_an_error() -> None:
    info = GitToolInfo(_git(code=128)).info()
    assert info.git_sha == "unknown" and info.git_dirty is None


def test_missing_git_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mlship.adapters.process.shutil.which", lambda _: None)
    info = GitToolInfo(_git()).info()
    assert info.git_sha == "unknown" and info.git_dirty is None


def test_status_is_limited_to_the_tool_packages() -> None:
    seen: list[list[str]] = []

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        seen.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    GitToolInfo(runner).info()
    status = next(c for c in seen if "status" in c)
    paths = status[status.index("--") + 1 :]
    assert len(paths) == 3 and all("packages" in p for p in paths)
