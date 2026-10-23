"""Error mapping of the Docker adapter, with a fake client (no Docker needed)."""

from pathlib import Path
from typing import Any

import docker.errors
import pytest

from mlship.adapters.docker_runtime import DockerRuntime
from mlship.domain.environment import ImageRef
from mlship.domain.errors import ContainerRunError, EnvironmentBuildError, ToolNotFoundError


class FakeImage:
    id = "sha256:feed"


class FakeImages:
    def __init__(self, build_error: Exception | None = None, present: bool = False) -> None:
        self.build_error = build_error
        self.present = present
        self.build_kwargs: dict[str, Any] = {}

    def get(self, tag: str) -> FakeImage:
        if not self.present:
            raise docker.errors.ImageNotFound(f"No such image: {tag}")
        return FakeImage()

    def build(self, **kwargs: Any) -> tuple[FakeImage, list[dict[str, str]]]:
        self.build_kwargs = kwargs
        if self.build_error is not None:
            raise self.build_error
        return FakeImage(), []


class FakeClient:
    def __init__(self, images: FakeImages) -> None:
        self.images = images

    def ping(self) -> bool:
        return True


def _runtime(images: FakeImages) -> DockerRuntime:
    return DockerRuntime("linux/amd64", client_factory=lambda: FakeClient(images))


def test_missing_image_is_none_not_an_error() -> None:
    assert _runtime(FakeImages()).find_image("mlship-env:abc") is None


def test_present_image_is_returned_with_its_id() -> None:
    found = _runtime(FakeImages(present=True)).find_image("mlship-env:abc")
    assert found == ImageRef(tag="mlship-env:abc", digest="sha256:feed")


def test_build_passes_tag_labels_and_platform(tmp_path: Path) -> None:
    images = FakeImages()
    image = _runtime(images).build_image(tmp_path, "mlship-env:abc", {"org.mlship.k": "v"})
    assert image == ImageRef(tag="mlship-env:abc", digest="sha256:feed")
    assert images.build_kwargs["tag"] == "mlship-env:abc"
    assert images.build_kwargs["labels"] == {"org.mlship.k": "v"}
    assert images.build_kwargs["platform"] == "linux/amd64"


def test_failed_build_carries_the_log_and_dockers_reason(tmp_path: Path) -> None:
    error = docker.errors.BuildError(
        "COPY failed: file not found", [{"stream": "Step 2/9 : COPY requirements.lock\n"}]
    )
    with pytest.raises(EnvironmentBuildError) as info:
        _runtime(FakeImages(build_error=error)).build_image(tmp_path, "t:1", {})
    assert "Step 2/9" in info.value.details
    assert "COPY failed" in info.value.details


def test_unreachable_daemon_becomes_a_clear_error() -> None:
    def unreachable() -> FakeClient:
        raise docker.errors.DockerException("Error while fetching server API version")

    runtime = DockerRuntime("linux/amd64", client_factory=unreachable)
    with pytest.raises(ToolNotFoundError) as info:
        runtime.find_image("t:1")
    assert "Docker Desktop" in info.value.hint


# --- run_training: isolation flags and lifecycle, with a fake container -------------

from mlship.domain.training import ResourceLimits, TrainingJobSpec  # noqa: E402


class FakeContainer:
    def __init__(self, statuses: list[str], exit_code: int = 0) -> None:
        self._statuses = statuses
        self.status = "created"
        self.attrs: dict[str, Any] = {"State": {"ExitCode": exit_code, "OOMKilled": False}}
        self.killed = False
        self.removed = False

    def start(self) -> None:
        self.status = "running"

    def logs(self, stream: bool = False, follow: bool = False) -> Any:
        return [b"hello\n", b"done\n"] if stream else b"hello\ndone\n"

    def reload(self) -> None:
        if self._statuses:
            self.status = self._statuses.pop(0)

    def kill(self) -> None:
        self.killed = True
        self.status = "exited"
        self.attrs["State"]["ExitCode"] = 137

    def wait(self, timeout: int) -> dict[str, int]:
        return {"StatusCode": self.attrs["State"]["ExitCode"]}

    def remove(self, force: bool) -> None:
        self.removed = True


class FakeContainers:
    def __init__(self, container: FakeContainer, create_error: Exception | None = None) -> None:
        self.container = container
        self.create_error = create_error
        self.create_kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> FakeContainer:
        self.create_kwargs = kwargs
        if self.create_error is not None:
            raise self.create_error
        return self.container


def _spec(tmp_path: Path, timeout: int = 60) -> TrainingJobSpec:
    for name in ("project", "data", "output"):
        (tmp_path / name).mkdir(exist_ok=True)
    (tmp_path / "run.json").write_text("{}")
    return TrainingJobSpec(
        image_tag="mlship-env:abc",
        project_dir=tmp_path / "project",
        data_dir=tmp_path / "data",
        config_file=tmp_path / "run.json",
        output_dir=tmp_path / "output",
        environment={"PYTHONHASHSEED": "42"},
        limits=ResourceLimits(
            cpus=2.0, memory="4g", pids=256, timeout_seconds=timeout, tmpfs_size="1g"
        ),
        run_id="0123456789abcdef",
    )


def _training_runtime(
    container: FakeContainer, create_error: Exception | None = None
) -> tuple[DockerRuntime, FakeContainers]:
    containers = FakeContainers(container, create_error)
    client = FakeClient(FakeImages())
    client.containers = containers
    return DockerRuntime("linux/amd64", client_factory=lambda: client), containers


def test_training_container_is_locked_down(tmp_path: Path) -> None:
    runtime, containers = _training_runtime(FakeContainer(["running", "exited"]))
    runtime.run_training(_spec(tmp_path), lambda _: None)
    kw = containers.create_kwargs

    assert kw["network_mode"] == "none"
    assert kw["read_only"] is True
    assert kw["cap_drop"] == ["ALL"]
    assert kw["security_opt"] == ["no-new-privileges"]
    assert kw["user"] != "0" and not str(kw["user"]).startswith("0:")
    assert kw["mem_limit"] == kw["memswap_limit"] == "4g"
    assert kw["nano_cpus"] == 2_000_000_000
    assert kw["pids_limit"] == 256
    assert kw["environment"] == {"PYTHONHASHSEED": "42"}
    assert set(kw["tmpfs"]) == {"/tmp"}
    modes = {v["bind"]: v["mode"] for v in kw["volumes"].values()}
    assert modes == {
        "/mlship/project": "ro",
        "/mlship/data": "ro",
        "/mlship/run.json": "ro",
        "/output": "rw",
    }
    assert "privileged" not in kw


def test_logs_are_streamed_and_container_removed(tmp_path: Path) -> None:
    container = FakeContainer(["running", "exited"])
    runtime, _ = _training_runtime(container)
    logs: list[str] = []
    result = runtime.run_training(_spec(tmp_path), logs.append)
    assert "".join(logs) == "hello\ndone\n"
    assert result.exit_code == 0 and not result.timed_out
    assert container.removed


def test_timeout_kills_the_container(tmp_path: Path) -> None:
    container = FakeContainer(["running"] * 1000)
    runtime, _ = _training_runtime(container)
    result = runtime.run_training(_spec(tmp_path, timeout=0), lambda _: None)
    assert result.timed_out
    assert container.killed
    assert result.exit_code == 137
    assert container.removed


def test_docker_refusing_the_container_is_a_clear_error(tmp_path: Path) -> None:
    refused = docker.errors.APIError("invalid mount config")
    runtime, _ = _training_runtime(FakeContainer([]), create_error=refused)
    with pytest.raises(ContainerRunError) as info:
        runtime.run_training(_spec(tmp_path), lambda _: None)
    assert "invalid mount config" in info.value.details


def test_selfcheck_runs_locked_down_and_passes_on_exit_zero() -> None:
    container = FakeContainer([], exit_code=0)
    runtime, containers = _training_runtime(container)
    assert runtime.verify_image("mlship-env:abc") is None
    kw = containers.create_kwargs
    assert kw["network_mode"] == "none" and kw["read_only"] is True
    assert kw["cap_drop"] == ["ALL"] and kw["user"] == "10001:10001"
    assert "volumes" not in kw, "the self-check needs no host files"
    assert container.removed


def test_selfcheck_failure_returns_the_log() -> None:
    container = FakeContainer([], exit_code=1)
    runtime, _ = _training_runtime(container)
    failure = runtime.verify_image("mlship-env:abc")
    assert failure is not None and "exit code 1" in failure and "hello" in failure
    assert container.removed
