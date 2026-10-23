"""Real uv and Docker. Run with: uv run pytest -m integration tests/integration"""

import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import docker
import pytest

from mlship.bootstrap import build_services
from mlship.config import Settings

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "california-housing"

pytestmark = pytest.mark.integration


@pytest.fixture
def project(tmp_path: Path) -> Path:
    shutil.copyfile(EXAMPLE / "requirements.txt", tmp_path / "requirements.txt")
    shutil.copyfile(EXAMPLE / "train.py", tmp_path / "train.py")
    return tmp_path


@pytest.fixture
def isolated_repository() -> Iterator[str]:
    """A fresh image repository per run, so the first prepare must really build."""
    repository = f"mlship-env-test-{uuid.uuid4().hex[:8]}"
    yield repository
    client = docker.from_env()
    for image in client.images.list(name=repository):
        client.images.remove(image.id, force=True)


def test_lock_resolves_the_example_with_hashes(project: Path) -> None:
    lock = build_services(Settings()).environment.lock(project)
    content = lock.path.read_text()
    assert "scikit-learn==" in content
    assert "mlflow-skinny==" in content
    assert "--hash=sha256:" in content


@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker not installed")
def test_first_prepare_builds_and_second_reuses(project: Path, isolated_repository: str) -> None:
    environment = build_services(Settings(image_repository=isolated_repository)).environment
    first = environment.prepare(project)
    second = environment.prepare(project)

    assert not first.cache_hit, "the first prepare must build a new image"
    # Same inputs: if this fails, the message shows which input changed.
    assert second.inputs == first.inputs
    assert second.cache_hit, f"{first.image.tag} was built but not found on the second run"
    assert second.image == first.image
