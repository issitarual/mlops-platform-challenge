"""Load scenarios for the model API, selected with ``--tags`` (see run.py).

- ``single``: one row per request, the online case;
- ``batch``: 100 rows per request, the vectorized path;
- ``drift``: one row per request with the first feature shifted, to show the drift
  panel and alert reacting (ADR-016).

Rows come from a CSV (``LOADTEST_DATA``) without its target column (``LOADTEST_TARGET``).
"""

import csv
import os
import random
from pathlib import Path

from locust import HttpUser, constant, tag, task

DATA = Path(
    os.environ.get("LOADTEST_DATA", "examples/california-housing/data/california_housing.csv")
)
TARGET = os.environ.get("LOADTEST_TARGET", "MedHouseVal")
BATCH_ROWS = 100
MAX_ROWS = 5_000


def _number(text: str) -> float | int:
    """Integers stay integers: the API types are strict (an int column refuses 3.0)."""
    try:
        return int(text)
    except ValueError:
        return float(text)


def _rows() -> list[dict[str, float | int]]:
    """A seeded sample spread over the whole file: the first rows of a dataset are rarely
    representative (California housing is ordered by location), and the drift monitor
    would rightly flag them as drift."""
    with DATA.open(newline="", encoding="utf-8") as handle:
        rows = [
            {k: _number(v) for k, v in row.items() if k != TARGET} for row in csv.DictReader(handle)
        ]
    return random.Random(7).sample(rows, min(MAX_ROWS, len(rows)))


ROWS = _rows()
_PICK = random.Random(42)  # the same request sequence on every run
FIRST = next(iter(ROWS[0]))


class ModelUser(HttpUser):
    wait_time = constant(0)

    def _predict(self, rows: list[dict[str, float | int]], name: str) -> None:
        with self.client.post(
            "/v1/predict", json={"rows": rows}, name=name, catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:200]}")

    @tag("single")
    @task
    def single_row(self) -> None:
        self._predict([_PICK.choice(ROWS)], "predict 1 row")

    @tag("batch")
    @task
    def batch(self) -> None:
        self._predict(_PICK.sample(ROWS, BATCH_ROWS), f"predict {BATCH_ROWS} rows")

    @tag("drift")
    @task
    def drifted_row(self) -> None:
        row = dict(_PICK.choice(ROWS))
        row[FIRST] = row[FIRST] * 3 + 10
        self._predict([row], "predict 1 drifted row")
