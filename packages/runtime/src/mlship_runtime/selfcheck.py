"""Image self-check: ``python -m mlship_runtime.selfcheck``. Exit 0 ok, 1 corrupted.

Run by the host right after building an environment image. Every installed file is
compared with the sha256 pip recorded in the package's ``RECORD`` file, so
a truncated or altered file is found before training (e.g. a build on a full disk
can finish "successfully" and leave broken files behind).
"""

import base64
import csv
import hashlib
import importlib
import sys
from collections.abc import Iterable, Iterator
from importlib import metadata
from pathlib import Path

# Imported after the file check: loads the native extensions the runner needs.
_MODULES = ("numpy", "pandas", "mlflow.pyfunc", "skops.io", "mlship_runtime.runner.__main__")
_MAX_REPORTED = 20


def _sha256_record(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode("ascii")


def _record_entries(dist: metadata.Distribution) -> Iterator[tuple[str, str]]:
    """(relative path, urlsafe-b64 sha256) from RECORD, read directly.

    ``Distribution.files`` is not used: it silently skips files that no longer exist,
    which is exactly one of the failures this check is for.
    """
    record = dist.read_text("RECORD") or ""
    for row in csv.reader(record.splitlines()):
        if len(row) >= 2 and row[1].startswith("sha256="):
            yield row[0], row[1].removeprefix("sha256=")


def corrupted_files(
    distributions: Iterable[metadata.Distribution] | None = None,
) -> Iterator[str]:
    """Yield ``package: file (reason)`` for every file that does not match its RECORD."""
    for dist in metadata.distributions() if distributions is None else distributions:
        for relative, expected in _record_entries(dist):
            path = Path(str(dist.locate_file(relative)))
            name = f"{dist.metadata['Name']}: {relative}"
            try:
                actual = _sha256_record(path)
            except OSError as exc:
                yield f"{name} ({type(exc).__name__})"
                continue
            if actual != expected:
                yield f"{name} (content differs)"


def main() -> int:
    problems = list(corrupted_files())
    for module in _MODULES:
        try:
            importlib.import_module(module)
        except Exception as exc:
            problems.append(f"import {module} ({type(exc).__name__}: {exc})")
    if problems:
        print(f"mlship selfcheck: {len(problems)} problem(s)", file=sys.stderr)
        for problem in problems[:_MAX_REPORTED]:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print("mlship selfcheck: ok", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
