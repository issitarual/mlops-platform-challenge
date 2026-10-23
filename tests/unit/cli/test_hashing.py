from pathlib import Path

from mlship.adapters.hashing import sha256_file, sha256_tree

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
