"""Integrity of the model files, checked before anything reads them.

Same algorithm as the host (``mlship`` CLI): SHA-256 per file, then one SHA-256 of the
sorted ``{path: sha256}`` map; links and special files are refused. The two packages
cannot import each other, so a contract test keeps both identical.
"""

import hashlib
import json
import os
import stat
from pathlib import Path

_CHUNK = 1024 * 1024


class ModelIntegrityError(Exception):
    """The model files are not the ones the platform registered."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def tree_digest(files: dict[str, str]) -> str:
    canonical = json.dumps(dict(sorted(files.items())), separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def model_files_digest(root: Path) -> str:
    if root.is_symlink() or not root.is_dir():
        raise ModelIntegrityError(f"{root} is not a plain directory.")
    files: dict[str, str] = {}
    for current, dirs, names in os.walk(root, followlinks=False):
        for name in [*dirs, *names]:
            path = Path(current) / name
            mode = path.lstat().st_mode
            if stat.S_ISREG(mode):
                files[path.relative_to(root).as_posix()] = _sha256_file(path)
            elif not stat.S_ISDIR(mode):
                raise ModelIntegrityError(f"{path.relative_to(root)} is not a regular file.")
    return tree_digest(files)


def verify(root: Path, expected_sha256: str) -> None:
    if not expected_sha256:
        raise ModelIntegrityError("No expected model hash was given; refusing to load.")
    actual = model_files_digest(root)
    if actual != expected_sha256:
        raise ModelIntegrityError(
            f"Model files do not match the registered hash "
            f"(expected {expected_sha256[:12]}…, found {actual[:12]}…)."
        )
