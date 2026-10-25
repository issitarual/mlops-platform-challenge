"""Content hashing for lineage (code, data, lock, model)."""

import hashlib
import os
import stat
from collections.abc import Callable
from pathlib import Path

from mlship.domain.errors import ArtifactTooLargeError, UntrustedArtifactError

_CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file, streamed so large files do not load into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(
    root: Path,
    exclude_dirs: frozenset[str] = frozenset(),
    include: Callable[[str], bool] | None = None,
) -> dict[str, str]:
    """Hash of every file under ``root``, keyed by POSIX relative path, sorted.

    Relative POSIX keys keep the result identical across machines and OSes.
    ``exclude_dirs`` skips directories by name anywhere in the tree (e.g. caches);
    ``include`` keeps only the relative paths it accepts.
    """
    files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and not exclude_dirs.intersection(p.relative_to(root).parts[:-1])
    )
    keyed = {p.relative_to(root).as_posix(): p for p in files}
    return {
        key: sha256_file(path) for key, path in keyed.items() if include is None or include(key)
    }


def sha256_untrusted_tree(root: Path, max_bytes: int, max_files: int) -> dict[str, str]:
    """Like ``sha256_tree`` for output written by untrusted code: links are never followed.

    Anything that is not a regular file or a directory (symbolic link, device, FIFO,
    socket) raises ``UntrustedArtifactError``: a link could make the host read, hash and
    upload a file of its own (ADR-018). Size and file count are checked from metadata
    first, so an oversized output is refused before a single byte is read.
    """
    if root.is_symlink() or not root.is_dir():
        raise UntrustedArtifactError(f"{root.name} is not a plain directory.")
    regular: list[tuple[str, Path]] = []
    refused: list[str] = []
    total = 0
    for current, dirs, files in os.walk(root, followlinks=False):
        base = Path(current)
        for name in [*dirs, *files]:
            path = base / name
            info = path.lstat()
            relative = path.relative_to(root).as_posix()
            if stat.S_ISREG(info.st_mode):
                regular.append((relative, path))
                total += info.st_size
            elif not stat.S_ISDIR(info.st_mode):
                refused.append(relative)
        if len(regular) > max_files or total > max_bytes:
            raise ArtifactTooLargeError(
                f"The model output exceeds the limit ({max_files} files, {max_bytes} bytes).",
                hint="Save only what predict() needs. The platform can raise the limits "
                "(MLSHIP_MODEL_MAX_BYTES, MLSHIP_MODEL_MAX_FILES).",
            )
    if refused:
        raise UntrustedArtifactError(
            "The model output contains links or special files: " + ", ".join(sorted(refused)),
            hint="Save only regular files in the model; links are refused for safety.",
        )
    return {relative: sha256_file(path) for relative, path in sorted(regular)}


class Sha256Hasher:
    """``ContentHasher`` with the functions above."""

    def file(self, path: Path) -> str:
        return sha256_file(path)

    def tree(
        self,
        root: Path,
        exclude_dirs: frozenset[str] = frozenset(),
        include: Callable[[str], bool] | None = None,
    ) -> dict[str, str]:
        return sha256_tree(root, exclude_dirs, include)

    def untrusted_tree(self, root: Path, max_bytes: int, max_files: int) -> dict[str, str]:
        return sha256_untrusted_tree(root, max_bytes, max_files)
