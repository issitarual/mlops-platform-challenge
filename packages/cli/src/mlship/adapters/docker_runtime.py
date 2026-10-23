"""``ContainerRuntime`` implemented with the Docker SDK for Python.

The client is created on first use, so building the object never needs a Docker daemon.
"""

import contextlib
import os
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import docker
import docker.errors

from mlship.domain.environment import ImageRef
from mlship.domain.errors import ContainerRunError, EnvironmentBuildError, ToolNotFoundError
from mlship.domain.runs import ModelVersion
from mlship.domain.training import (
    CONFIG_MOUNT,
    CONTAINER_TMP,
    DATA_MOUNT,
    OUTPUT_MOUNT,
    PROJECT_MOUNT,
    ResourceLimits,
    TrainingJobResult,
    TrainingJobSpec,
)

_LOG_TAIL_CHARS = 4_000
_IMAGE_USER = "10001:10001"  # the non-root user baked into the image
_POLL_SECONDS = 0.5
_RUNNER_COMMAND = ["python", "-m", "mlship_runtime.runner"]
_SELFCHECK_COMMAND = ["python", "-m", "mlship_runtime.selfcheck"]
_SELFCHECK_LIMITS = ResourceLimits(
    cpus=1.0, memory="1g", pids=64, timeout_seconds=300, tmpfs_size="64m"
)


def _locked_down(limits: ResourceLimits) -> dict[str, Any]:
    """Container options shared by every job that runs image code (ADR-018)."""
    return {
        "network_mode": "none",
        "read_only": True,
        "tmpfs": {CONTAINER_TMP: f"rw,nosuid,nodev,size={limits.tmpfs_size},mode=1777"},
        "cap_drop": ["ALL"],
        "security_opt": ["no-new-privileges"],
        "mem_limit": limits.memory,
        "memswap_limit": limits.memory,
        "nano_cpus": int(limits.cpus * 1e9),
        "pids_limit": limits.pids,
        "working_dir": CONTAINER_TMP,
    }


def _container_user(output_dir: Path) -> str:
    """Non-root always. On Linux/macOS, the caller's uid, so outputs belong to them."""
    if sys.platform != "win32":
        if os.getuid() != 0:
            return f"{os.getuid()}:{os.getgid()}"
        output_dir.chmod(0o777)
    return _IMAGE_USER


class DockerRuntime:
    def __init__(self, platform: str, client_factory: Callable[[], Any] = docker.from_env) -> None:
        self._platform = platform
        self._client_factory = client_factory  # injectable, so error mapping is unit-tested
        self._client: Any = None

    def _docker(self) -> Any:
        if self._client is None:
            try:
                self._client = self._client_factory()
                self._client.ping()
            except docker.errors.DockerException as exc:
                self._client = None
                raise ToolNotFoundError(
                    "Could not connect to Docker.",
                    hint="Start Docker Desktop (or the Docker daemon) and try again.",
                    details=str(exc),
                ) from exc
        return self._client

    def find_image(self, tag: str) -> ImageRef | None:
        try:
            image = self._docker().images.get(tag)
        except docker.errors.ImageNotFound:
            return None
        return ImageRef(tag=tag, digest=image.id)

    def build_image(self, context_dir: Path, tag: str, labels: Mapping[str, str]) -> ImageRef:
        try:
            image, _ = self._docker().images.build(
                path=str(context_dir),
                tag=tag,
                labels=dict(labels),
                platform=self._platform,
                rm=True,
                forcerm=True,
            )
        except docker.errors.BuildError as exc:
            log = "".join(str(chunk.get("stream", "")) for chunk in exc.build_log)
            raise EnvironmentBuildError(
                "Building the environment image failed.",
                hint="See the build log below; dependency installation is the usual cause.",
                details=f"{log}\n{exc.msg}"[-_LOG_TAIL_CHARS:],
            ) from exc
        except docker.errors.APIError as exc:
            raise EnvironmentBuildError("Docker rejected the build.", details=str(exc)) from exc
        return ImageRef(tag=tag, digest=image.id)

    def run_training(
        self, spec: TrainingJobSpec, on_log: Callable[[str], None]
    ) -> TrainingJobResult:
        """Run the job locked down (ADR-018) and stream its logs."""
        try:
            container = self._docker().containers.create(
                image=spec.image_tag,
                command=_RUNNER_COMMAND,
                name=f"mlship-train-{spec.run_id[:12]}",
                labels={"org.mlship.run-id": spec.run_id},
                user=_container_user(spec.output_dir),
                environment=dict(spec.environment),  # nothing from the host leaks in
                volumes={
                    str(spec.project_dir): {"bind": PROJECT_MOUNT, "mode": "ro"},
                    str(spec.data_dir): {"bind": DATA_MOUNT, "mode": "ro"},
                    str(spec.config_file): {"bind": CONFIG_MOUNT, "mode": "ro"},
                    str(spec.output_dir): {"bind": OUTPUT_MOUNT, "mode": "rw"},
                },
                **_locked_down(spec.limits),
            )
        except docker.errors.APIError as exc:
            raise ContainerRunError(
                "Docker could not create the training container.",
                hint="On Docker Desktop, check that the project and data folders can be "
                "shared with containers (Settings > Resources > File sharing).",
                details=str(exc),
            ) from exc
        tail: deque[str] = deque(maxlen=200)

        def pump() -> None:
            for chunk in container.logs(stream=True, follow=True):
                text = chunk.decode("utf-8", errors="replace")
                tail.append(text)
                on_log(text)

        timed_out = False
        try:
            try:
                container.start()
            except docker.errors.APIError as exc:
                raise ContainerRunError(
                    "Docker could not start the training container.", details=str(exc)
                ) from exc
            reader = threading.Thread(target=pump, daemon=True)
            reader.start()
            deadline = time.monotonic() + spec.limits.timeout_seconds
            while True:
                container.reload()
                if container.status in ("exited", "dead"):
                    break
                if time.monotonic() > deadline:
                    timed_out = True
                    container.kill()
                    container.wait(timeout=30)
                    container.reload()
                    break
                time.sleep(_POLL_SECONDS)
            reader.join(timeout=5)
            state = container.attrs.get("State", {})
            return TrainingJobResult(
                exit_code=int(state.get("ExitCode", 1)),
                timed_out=timed_out,
                oom_killed=bool(state.get("OOMKilled", False)),
                logs_tail="".join(tail)[-_LOG_TAIL_CHARS:],
            )
        finally:
            container.remove(force=True)

    def verify_image(self, tag: str) -> str | None:
        """Run the self-check with the training lockdown; return its log if it fails."""
        try:
            container = self._docker().containers.create(
                image=tag,
                command=_SELFCHECK_COMMAND,
                user=_IMAGE_USER,
                environment={"HOME": CONTAINER_TMP},
                **_locked_down(_SELFCHECK_LIMITS),
            )
        except docker.errors.APIError as exc:
            return f"could not create the self-check container: {exc}"
        try:
            container.start()
            status = container.wait(timeout=_SELFCHECK_LIMITS.timeout_seconds)
            logs = container.logs().decode("utf-8", errors="replace")
        except Exception as exc:  # timeout or daemon error: the image is not trusted either way
            return f"self-check did not finish: {type(exc).__name__}: {exc}"
        finally:
            container.remove(force=True)
        exit_code = int(status.get("StatusCode", 1))
        if exit_code == 0:
            return None
        return f"exit code {exit_code}\n{logs}"[-_LOG_TAIL_CHARS:]

    def remove_image(self, tag: str) -> None:
        with contextlib.suppress(docker.errors.ImageNotFound):
            self._docker().images.remove(tag, force=True)

    def ensure_serving_image(
        self, base: ImageRef, model: ModelVersion, model_dir: Path
    ) -> ImageRef:
        raise NotImplementedError("E4")

    def run_serving(self, image: ImageRef, port: int, workers: int) -> None:
        raise NotImplementedError("E4")
