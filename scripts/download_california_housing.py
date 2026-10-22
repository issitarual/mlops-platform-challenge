# /// script
# requires-python = ">=3.12"
# dependencies = ["scikit-learn", "pandas"]
# ///
"""Download California Housing on the HOST and save it as CSV.

The training container has no network access, so data must exist before training.
Run with: ``uv run scripts/download_california_housing.py``
"""

import hashlib
from pathlib import Path

from sklearn.datasets import fetch_california_housing

OUTPUT = Path(__file__).resolve().parents[1] / "examples/california-housing/data"


def main() -> None:
    frame = fetch_california_housing(as_frame=True).frame
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / "california_housing.csv"
    frame.to_csv(target, index=False)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    print(f"{len(frame)} rows -> {target}")
    print(f"sha256: {digest}")


if __name__ == "__main__":
    main()
