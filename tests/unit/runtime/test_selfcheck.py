"""The image self-check finds files that differ from what pip recorded."""

import base64
import hashlib
from importlib import metadata
from pathlib import Path

import pytest

from mlship_runtime.selfcheck import corrupted_files

CONTENT = b"Sentinel = sentinel\n"


def _record_hash(data: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


@pytest.fixture
def site(tmp_path: Path) -> Path:
    """A site-packages with one installed package, as pip leaves it."""
    (tmp_path / "pkg.py").write_bytes(CONTENT)
    info = tmp_path / "pkg-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: pkg\nVersion: 1.0\n")
    (info / "RECORD").write_text(
        f"pkg.py,sha256={_record_hash(CONTENT)},{len(CONTENT)}\npkg-1.0.dist-info/RECORD,,\n"
    )
    return tmp_path


def _check(site: Path) -> list[str]:
    return list(corrupted_files(metadata.distributions(path=[str(site)])))


def test_intact_installation_passes(site: Path) -> None:
    assert _check(site) == []


def test_truncated_file_is_reported(site: Path) -> None:
    (site / "pkg.py").write_bytes(CONTENT[:5])
    assert _check(site) == ["pkg: pkg.py (content differs)"]


def test_missing_file_is_reported(site: Path) -> None:
    (site / "pkg.py").unlink()
    assert _check(site) == ["pkg: pkg.py (FileNotFoundError)"]
