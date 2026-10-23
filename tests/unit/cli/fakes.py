"""In-memory implementations of the ports, for service tests without Docker or uv."""

from collections.abc import Mapping, Sequence
from pathlib import Path

from mlship.domain.environment import ImageRef
from mlship.domain.runs import ModelVersion
from mlship.domain.training import TrainingJobResult, TrainingJobSpec


class FakeLockResolver:
    def __init__(self, content: str = "numpy==2.0.0 \\\n    --hash=sha256:abc\n") -> None:
        self.content = content
        self.calls: list[list[str]] = []

    def resolve(self, requirement_files: Sequence[Path], output: Path) -> None:
        self.calls.append([f.read_text() for f in requirement_files])
        output.write_text(self.content)


class FakeRuntimePackager:
    def __init__(self, source: str = "s" * 64) -> None:
        self.source = source
        self.built: list[Path] = []

    def requirements(self) -> list[str]:
        return ["pydantic>=2", "fastapi>=0.1"]

    def source_sha256(self) -> str:
        return self.source

    def build_wheels(self, output_dir: Path) -> list[Path]:
        output_dir.mkdir(parents=True)
        wheel = output_dir / "mlship_runtime-0.1.0-py3-none-any.whl"
        wheel.write_bytes(b"wheel")
        self.built.append(wheel)
        return [wheel]


class FakeDockerfile:
    def __init__(self, text: str = "FROM python:3.12-slim@sha256:abc\n") -> None:
        self.text = text

    def render(self) -> str:
        return self.text


class FakeContainerRuntime:
    def __init__(self) -> None:
        self.images: dict[str, ImageRef] = {}
        self.builds: list[tuple[str, dict[str, str], list[str]]] = []

    def find_image(self, tag: str) -> ImageRef | None:
        return self.images.get(tag)

    def build_image(self, context_dir: Path, tag: str, labels: Mapping[str, str]) -> ImageRef:
        files = sorted(p.relative_to(context_dir).as_posix() for p in context_dir.rglob("*"))
        self.builds.append((tag, dict(labels), files))
        image = ImageRef(tag=tag, digest=f"sha256:{len(self.builds)}")
        self.images[tag] = image
        return image

    def run_training(self, spec: TrainingJobSpec) -> TrainingJobResult:
        raise NotImplementedError

    def ensure_serving_image(
        self, base: ImageRef, model: ModelVersion, model_dir: Path
    ) -> ImageRef:
        raise NotImplementedError

    def run_serving(self, image: ImageRef, port: int, workers: int) -> None:
        raise NotImplementedError
