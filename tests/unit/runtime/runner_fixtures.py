"""Shared helpers for runner tests."""

import json
from pathlib import Path

GOOD_TRAIN = """from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from mlship_sdk import TrainOutput


def train(data_dir: Path, params: dict) -> TrainOutput:
    frame = pd.read_csv(data_dir / \"data.csv\")
    x, y = frame[[\"a\", \"b\"]], frame[\"y\"]
    model = Ridge(alpha=float(params.get(\"alpha\", \"1.0\"))).fit(x, y)
    rmse = float(np.sqrt(np.mean((model.predict(x) - y) ** 2)))
    return TrainOutput(model=model, metrics={\"rmse\": rmse}, input_example=x.head(3))
"""


def write_config(path: Path, dirs: dict[str, Path], **overrides: object) -> Path:
    config = {
        "schema_version": 1,
        "project_dir": str(dirs["project"]),
        "data_dir": str(dirs["data"]),
        "output_dir": str(dirs["output"]),
        "params": {"seed": "42"},
        "seed": 42,
        "serialization": {"allow_pickle": False, "skops_trusted_types": []},
    }
    config.update(overrides)
    path.write_text(json.dumps(config))
    return path
