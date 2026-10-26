"""Online drift: PSI of each feature over the most recent requests (ADR-016).

Every value is placed in the decile bins of the training baseline; a sliding window
keeps the counts per bin, so each row costs O(features) and the PSI O(features x
bins). PSI = sum((actual - expected) * ln(actual / expected)) over the bins: under
0.1 no shift, 0.1 to 0.2 moderate, above 0.2 significant (the usual reading).
"""

import math
import threading
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

from mlship_runtime.reference import FeatureReference, Reference

_EPSILON = 1e-4


class _FeatureWindow:
    def __init__(self, baseline: FeatureReference, size: int) -> None:
        self.baseline = baseline
        self.bins: deque[int] = deque(maxlen=size)
        self.counts = [0] * len(baseline.shares)

    def add(self, value: float) -> None:
        if len(self.bins) == self.bins.maxlen:
            self.counts[self.bins[0]] -= 1
        index = self.baseline.bin_of(value)
        self.bins.append(index)
        self.counts[index] += 1

    def psi(self) -> float:
        total = len(self.bins)
        score = 0.0
        for count, share in zip(self.counts, self.baseline.shares, strict=True):
            actual = max(count / total, _EPSILON)
            expected = max(share, _EPSILON)
            score += (actual - expected) * math.log(actual / expected)
        return score


class DriftMonitor:
    """Thread-safe: predictions run in a thread pool."""

    def __init__(
        self, baseline: Reference, columns: Sequence[str], *, window_rows: int, min_rows: int
    ) -> None:
        self._windows = {
            name: _FeatureWindow(baseline.features[name], window_rows)
            for name in columns
            if name in baseline.features
        }
        self._min_rows = min_rows
        self._lock = threading.Lock()

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(self._windows)

    def observe(self, rows: Sequence[Mapping[str, Any]]) -> None:
        with self._lock:
            for row in rows:
                for name, window in self._windows.items():
                    value = row.get(name)
                    if isinstance(value, int | float) and not isinstance(value, bool):
                        number = float(value)
                        if math.isfinite(number):
                            window.add(number)

    def scores(self) -> dict[str, float]:
        """PSI per feature, only once its window holds enough rows to mean something."""
        with self._lock:
            return {
                name: window.psi()
                for name, window in self._windows.items()
                if len(window.bins) >= self._min_rows
            }
