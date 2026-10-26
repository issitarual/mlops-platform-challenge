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


class FakeNetwork:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.connected: list[tuple[Any, list[str]]] = []

    def connect(self, container: Any, aliases: list[str]) -> None:
        if self.error is not None:
            raise self.error
        self.connected.append((container, aliases))


class FakeNetworks:
    def __init__(self, network: FakeNetwork | None = None) -> None:
        self.network = network
        self.names: list[str] = []

    def get(self, name: str) -> FakeNetwork:
        self.names.append(name)
        if self.network is None:
            raise docker.errors.NotFound("no such network")
        return self.network


class FakeClient:
    def __init__(
        self, images: FakeImages, containers: Any = None, networks: FakeNetworks | None = None
    ) -> None:
        self.images = images
        self.containers = containers
        self.networks = networks or FakeNetworks()

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
        self.started = False

    def start(self) -> None:
        self.started = True
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

    def stop(self, timeout: int) -> None:
        self.status = "exited"

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
    container: FakeContainer,
    create_error: Exception | None = None,
    networks: FakeNetworks | None = None,
) -> tuple[DockerRuntime, FakeContainers]:
    containers = FakeContainers(container, create_error)
    client = FakeClient(FakeImages(), containers, networks)
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


# --- serving: image per version and a locked-down API container ---------------------

from mlship.adapters import docker_runtime  # noqa: E402
from mlship.domain.runs import ModelVersion  # noqa: E402
from mlship.domain.serving import (  # noqa: E402
    ServingImageSpec,
    ServingJobSpec,
    ServingLimits,
)

MODEL = ModelVersion("demo", 2, "run1", "a" * 64)


def _serving_spec(port: int = 9000) -> ServingJobSpec:
    return ServingJobSpec(
        image=ImageRef("mlship-serve-demo:v2", "sha256:s"),
        model=MODEL,
        host_port=port,
        workers=2,
        environment={"MLSHIP_SERVING_API_KEY": "k"},
        limits=ServingLimits(cpus=1.0, memory="1g", pids=64),
    )


def test_serving_image_bakes_the_verified_model_and_is_built_once(tmp_path: Path) -> None:
    images = FakeImages()
    runtime = _runtime(images)
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "MLmodel").write_text("x")
    spec = ServingImageSpec(ImageRef("mlship-env:abc", "sha256:" + "b" * 64), MODEL, False, ())
    image = runtime.ensure_serving_image(spec, tmp_path / "model")
    assert image.tag.startswith("mlship-serve-demo:v2-aaaaaaaaaaaa-bbbbbbbbbbbb-")
    assert images.build_kwargs["labels"]["org.mlship.model-sha256"] == "a" * 64

    images.present = True  # now it exists: no second build
    images.build_kwargs = {}
    runtime.ensure_serving_image(spec, tmp_path / "model")
    assert images.build_kwargs == {}


def test_api_container_is_locked_down_and_only_published_on_localhost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(docker_runtime, "_answers_ready", lambda _: True)
    container = FakeContainer(["running", "running", "exited"])
    runtime, containers = _training_runtime(container)
    ready: list[str] = []
    with pytest.raises(ContainerRunError, match="while serving"):
        runtime.run_serving(_serving_spec(), lambda _: None, ready.append)
    kw = containers.create_kwargs
    assert kw["ports"] == {"8080/tcp": ("127.0.0.1", 9000)}
    assert "network_mode" not in kw, "the API needs a network to receive requests"
    assert kw["read_only"] is True and kw["cap_drop"] == ["ALL"]
    assert kw["security_opt"] == ["no-new-privileges"] and kw["user"] == "10001:10001"
    assert kw["command"] == ["--port", "8080", "--workers", "2"]
    assert kw["environment"] == {"HOME": "/tmp", "MLSHIP_SERVING_API_KEY": "k"}
    assert ready == ["http://127.0.0.1:9000"] and container.removed


def test_api_that_dies_before_ready_is_an_error_with_its_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(docker_runtime, "_answers_ready", lambda _: False)
    container = FakeContainer(["exited"], exit_code=3)
    runtime, _ = _training_runtime(container)
    with pytest.raises(ContainerRunError, match="before it was ready") as info:
        runtime.run_serving(_serving_spec(), lambda _: None, lambda _: None)
    assert "hello" in info.value.details
    assert container.removed


def test_api_killed_for_memory_says_which_limit_to_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(docker_runtime, "_answers_ready", lambda _: True)
    container = FakeContainer(["running", "exited"], exit_code=137)
    container.attrs["State"]["OOMKilled"] = True
    runtime, _ = _training_runtime(container)
    with pytest.raises(ContainerRunError, match="exit code 137") as info:
        runtime.run_serving(_serving_spec(), lambda _: None, lambda _: None)
    assert "MLSHIP_SERVE_MEMORY_LIMIT" in info.value.hint
    assert container.removed


def test_cleanup_errors_never_hide_the_original_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docker_runtime, "_answers_ready", lambda _: False)
    container = FakeContainer(["exited"], exit_code=3)

    def already_gone(force: bool) -> None:
        raise docker.errors.NotFound("no such container")

    container.remove = already_gone  # type: ignore[method-assign]
    runtime, _ = _training_runtime(container)
    with pytest.raises(ContainerRunError, match="before it was ready"):
        runtime.run_serving(_serving_spec(), lambda _: None, lambda _: None)


def test_the_api_joins_the_monitoring_network_only_when_it_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(docker_runtime, "_answers_ready", lambda _: True)
    for network in (FakeNetwork(), None):
        container = FakeContainer(["running", "exited"])
        networks = FakeNetworks(network)
        runtime, _ = _training_runtime(container, networks=networks)
        with pytest.raises(ContainerRunError, match="while serving"):
            runtime.run_serving(_serving_spec(), lambda _: None, lambda _: None)
        assert networks.names == ["mlship-serving"] and container.started
        if network is not None:
            assert network.connected == [(container, ["mlship-api"])]


def test_a_broken_monitoring_network_is_a_clear_error_and_leaves_nothing_running() -> None:
    container = FakeContainer(["running"])
    networks = FakeNetworks(FakeNetwork(docker.errors.APIError("endpoint exists")))
    runtime, _ = _training_runtime(container, networks=networks)
    with pytest.raises(ContainerRunError, match="monitoring network") as info:
        runtime.run_serving(_serving_spec(), lambda _: None, lambda _: None)
    assert "--profile monitoring" in info.value.hint
    assert not container.started and container.removed
