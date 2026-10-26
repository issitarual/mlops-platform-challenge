"""Reference distribution of the training features, the baseline for drift (ADR-016).

Computed by the runner from ``TrainOutput.reference_data`` and saved inside the model
folder, so it is covered by the model hash and travels in the serving image. Per
numeric feature it keeps the decile edges and the share of training rows in each of
the bins they define; serving compares recent requests against those shares (PSI).
"""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REFERENCE_FILE = "mlship_reference.json"
SCHEMA_VERSION = 1
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MIN_ROWS = 100
MAX_FILE_BYTES = 1_000_000


@dataclass(frozen=True, slots=True)
class FeatureReference:
    edges: tuple[float, ...]
    shares: tuple[float, ...]  # len(edges) + 1 bins: (-inf, e0], (e0, e1], ..., (e_n, inf)
    mean: float
    std: float

    def bin_of(self, value: float) -> int:
        low, high = 0, len(self.edges)
        while low < high:
            middle = (low + high) // 2
            if value <= self.edges[middle]:
                high = middle
            else:
                low = middle + 1
        return low


@dataclass(frozen=True, slots=True)
class Reference:
    rows: int
    features: Mapping[str, FeatureReference]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "rows": self.rows,
            "features": {
                name: {
                    "edges": list(f.edges),
                    "shares": list(f.shares),
                    "mean": f.mean,
                    "std": f.std,
                }
                for name, f in self.features.items()
            },
        }


def compute(data: Any, columns: Sequence[str]) -> Reference:
    """Decile bins of every numeric column; raises ValueError with what to fix."""
    import numpy as np
    import pandas as pd

    if not isinstance(data, pd.DataFrame):
        raise ValueError(
            f"reference_data is {type(data).__name__}; pass the training features as a "
            "pandas DataFrame, e.g. reference_data=x_train."
        )
    missing = [c for c in columns if c not in data.columns]
    if missing:
        raise ValueError(
            f"reference_data lacks the model input columns {missing}; pass the same "
            "columns as input_example."
        )
    if len(data) < MIN_ROWS:
        raise ValueError(
            f"reference_data has {len(data)} rows; at least {MIN_ROWS} are needed for a "
            "useful baseline. Pass the training features, e.g. reference_data=x_train."
        )
    features: dict[str, FeatureReference] = {}
    for name in columns:
        series = data[name]
        if series.dtype == bool or not pd.api.types.is_numeric_dtype(series):
            continue
        values = series.to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < MIN_ROWS:
            continue
        edges = tuple(sorted({float(e) for e in np.quantile(values, QUANTILES)}))
        bins = np.searchsorted(np.asarray(edges), values, side="left")
        counts = np.bincount(bins, minlength=len(edges) + 1)
        shares = tuple(float(c) / len(values) for c in counts)
        features[str(name)] = FeatureReference(
            edges, shares, float(values.mean()), float(values.std())
        )
    return Reference(len(data), features)


def save(reference: Reference, model_dir: Path) -> None:
    (model_dir / REFERENCE_FILE).write_text(json.dumps(reference.to_dict()), encoding="utf-8")


def load(model_dir: Path) -> Reference | None:
    """None when the model has no reference (drift monitoring is then off)."""
    path = model_dir / REFERENCE_FILE
    if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_FILE_BYTES:
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
            return None
        features: dict[str, FeatureReference] = {}
        for name, item in dict(raw["features"]).items():
            edges = tuple(float(e) for e in item["edges"])
            shares = tuple(float(s) for s in item["shares"])
            values = (*edges, *shares)
            if len(shares) != len(edges) + 1 or not all(math.isfinite(v) for v in values):
                continue
            features[str(name)] = FeatureReference(
                edges, shares, float(item["mean"]), float(item["std"])
            )
        return Reference(int(raw["rows"]), features)
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
