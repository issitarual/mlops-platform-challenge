# /// script
# requires-python = ">=3.12"
# dependencies = ["scikit-learn", "pandas"]
# ///
"""Download California Housing on the HOST and save it as CSV.

The training container has no network access, so data must exist before training.
Run with: ``uv run scripts/download_california_housing.py``

The CSV is hashed in memory and written only if it matches ``EXPECTED_SHA256``,
so a changed upstream file never silently becomes "the same" dataset.
"""

import hashlib
import sys
from pathlib import Path

from sklearn.datasets import fetch_california_housing

OUTPUT = Path(__file__).resolve().parents[1] / "examples/california-housing/data"
TARGET = OUTPUT / "california_housing.csv"
EXPECTED_SHA256 = "94ef369b909f5c725340b512b7c9a05c1550a554b6592d29303990e3ba521a1f"


def main() -> None:
    frame = fetch_california_housing(as_frame=True).frame
    content = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    if digest != EXPECTED_SHA256:
        sys.exit(
            "Dataset hash mismatch; nothing was written.\n"
            f"  expected: {EXPECTED_SHA256}\n"
            f"  got:      {digest}\n"
            "The upstream data or the pinned library versions changed. Review the change, "
            "then update EXPECTED_SHA256 on purpose."
        )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    TARGET.write_bytes(content)
    print(f"{len(frame)} rows -> {TARGET}")
    print(f"sha256: {digest} (verified)")


if __name__ == "__main__":
    main()
