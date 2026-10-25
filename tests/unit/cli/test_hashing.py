import os
import sys
from pathlib import Path

import pytest

from mlship.adapters.hashing import sha256_file, sha256_tree, sha256_untrusted_tree
from mlship.domain.errors import ArtifactTooLargeError, UntrustedArtifactError

ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_sha256_file_matches_known_digest(tmp_path: Path) -> None:
    target = tmp_path / "abc.txt"
    target.write_bytes(b"abc")
    assert sha256_file(target) == ABC_SHA256


def test_sha256_tree_uses_sorted_posix_relative_keys(tmp_path: Path) -> None:
    (tmp_path / "b.csv").write_bytes(b"abc")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "a.csv").write_bytes(b"abc")
    result = sha256_tree(tmp_path)
    assert list(result) == ["b.csv", "nested/a.csv"]
    assert set(result.values()) == {ABC_SHA256}


def test_sha256_tree_can_skip_cache_directories(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_bytes(b"abc")
    (tmp_path / "pkg" / "__pycache__").mkdir()
    (tmp_path / "pkg" / "__pycache__" / "mod.pyc").write_bytes(b"compiled")
    assert list(sha256_tree(tmp_path, exclude_dirs=frozenset({"__pycache__"}))) == ["pkg/mod.py"]


def test_tree_can_keep_only_some_files(tmp_path: Path) -> None:
    (tmp_path / "train.py").write_bytes(b"abc")
    (tmp_path / "data.csv").write_bytes(b"abc")
    assert list(sha256_tree(tmp_path, include=lambda p: p.endswith(".py"))) == ["train.py"]


def test_untrusted_tree_hashes_regular_files(tmp_path: Path) -> None:
    (tmp_path / "MLmodel").write_bytes(b"abc")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "model.skops").write_bytes(b"abc")
    assert sha256_untrusted_tree(tmp_path, 10**6, 100) == {
        "MLmodel": ABC_SHA256,
        "nested/model.skops": ABC_SHA256,
    }


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need extra rights on Windows")
@pytest.mark.parametrize("target_is_dir", [False, True])
def test_untrusted_tree_refuses_links_to_files_and_directories(
    tmp_path: Path, target_is_dir: bool
) -> None:
    outside = tmp_path / "outside"
    if target_is_dir:
        outside.mkdir()
        (outside / "id_rsa").write_text("secret")
    else:
        outside.write_text("secret")
    model = tmp_path / "model"
    model.mkdir()
    (model / "MLmodel").write_text("x")
    (model / "innocent").symlink_to(outside, target_is_directory=target_is_dir)
    with pytest.raises(UntrustedArtifactError, match="innocent"):
        sha256_untrusted_tree(model, 10**6, 100)


@pytest.mark.skipif(sys.platform == "win32", reason="FIFOs do not exist on Windows")
def test_untrusted_tree_refuses_special_files(tmp_path: Path) -> None:
    if sys.platform != "win32":  # narrows the type for mypy; the test is skipped there
        os.mkfifo(tmp_path / "pipe")  # reading it would block the host forever
    with pytest.raises(UntrustedArtifactError, match="pipe"):
        sha256_untrusted_tree(tmp_path, 10**6, 100)


@pytest.mark.parametrize(("max_bytes", "max_files"), [(5, 100), (10**6, 2)])
def test_untrusted_tree_refuses_outputs_over_the_limits(
    tmp_path: Path, max_bytes: int, max_files: int
) -> None:
    for name in ("a", "b", "c"):
        (tmp_path / name).write_bytes(b"abc")
    with pytest.raises(ArtifactTooLargeError, match="exceeds the limit"):
        sha256_untrusted_tree(tmp_path, max_bytes, max_files)
