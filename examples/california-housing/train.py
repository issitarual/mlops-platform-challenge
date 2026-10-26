"""California Housing regression: a minimal project that satisfies the mlship contract."""

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlship_sdk import TrainOutput

TARGET = "MedHouseVal"
SPLIT_SEED = 42


def train(data_dir: Path, params: dict[str, Any]) -> TrainOutput:
    # CLI parameters arrive as strings; convert here, where their meaning is known.
    seed = int(params.get("seed", 42))
    learning_rate = float(params.get("learning_rate", 0.1))
    max_iter = int(params.get("max_iter", 200))

    df = pd.read_csv(data_dir / "california_housing.csv")
    x = df.drop(columns=[TARGET])
    y = df[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=SPLIT_SEED
    )

    # Preprocessing lives inside the model, so serving applies exactly the same steps.
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    learning_rate=learning_rate, max_iter=max_iter, random_state=seed
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    baseline = DummyRegressor(strategy="mean").fit(x_train, y_train).predict(x_test)
    metrics = {
        "rmse": float(root_mean_squared_error(y_test, predictions)),
        "baseline_rmse": float(root_mean_squared_error(y_test, baseline)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }
    return TrainOutput(
        model=model,
        metrics=metrics,
        input_example=x_test.head(5),
        reference_data=x_train,  # baseline for drift monitoring in serving
    )
