"""Online drift: PSI against the training baseline, over a sliding window."""

import numpy as np
import pandas as pd
import pytest

from mlship_runtime import reference
from mlship_runtime.serving.services.drift import DriftMonitor

RNG = np.random.default_rng(7)
TRAINING = pd.DataFrame({"rooms": RNG.normal(5.0, 1.0, 5_000), "age": RNG.normal(30, 5, 5_000)})
BASELINE = reference.compute(TRAINING, ["rooms", "age"])


def _monitor(window: int = 1_000, minimum: int = 200) -> DriftMonitor:
    return DriftMonitor(BASELINE, ["rooms", "age", "unknown"], window_rows=window, min_rows=minimum)


def _rows(rooms_mean: float, n: int) -> list[dict[str, float]]:
    return [
        {"rooms": float(r), "age": float(a)}
        for r, a in zip(RNG.normal(rooms_mean, 1.0, n), RNG.normal(30, 5, n), strict=True)
    ]


def test_traffic_like_training_has_a_low_psi() -> None:
    monitor = _monitor()
    monitor.observe(_rows(5.0, 1_000))
    scores = monitor.scores()
    assert set(scores) == {"rooms", "age"}, "features without a baseline are ignored"
    assert all(score < 0.1 for score in scores.values()), scores


def test_a_shifted_feature_crosses_the_alert_threshold_alone() -> None:
    monitor = _monitor()
    monitor.observe(_rows(7.0, 1_000))
    scores = monitor.scores()
    assert scores["rooms"] > 0.2 and scores["age"] < 0.1, scores


def test_no_score_until_the_window_has_enough_rows() -> None:
    monitor = _monitor(minimum=200)
    monitor.observe(_rows(5.0, 199))
    assert monitor.scores() == {}
    monitor.observe(_rows(5.0, 1))
    assert set(monitor.scores()) == {"rooms", "age"}


def test_the_window_forgets_old_traffic() -> None:
    monitor = _monitor(window=500, minimum=100)
    monitor.observe(_rows(9.0, 500))
    assert monitor.scores()["rooms"] > 0.2
    monitor.observe(_rows(5.0, 500))
    assert monitor.scores()["rooms"] < 0.1


@pytest.mark.parametrize("value", [None, True, "7", float("nan"), float("inf")])
def test_values_that_are_not_finite_numbers_are_skipped(value: object) -> None:
    monitor = _monitor(minimum=1)
    monitor.observe([{"rooms": value, "age": 30.0}])
    assert "rooms" not in monitor.scores()
