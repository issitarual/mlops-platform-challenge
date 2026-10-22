"""Content hashing for lineage (code, data, lock, model)."""

import hashlib
from pathlib import Path

_CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file, streamed so large files do not load into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path) -> dict[str, str]:
    """Hash of every file under ``root``, keyed by POSIX relative path, sorted.

    Relative POSIX keys keep the result identical across machines and OSes.
    """
    files = sorted(p for p in root.rglob("*") if p.is_file())
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in files}
