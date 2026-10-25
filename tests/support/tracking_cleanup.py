"""Remove what integration and end-to-end tests registered in a shared MLflow server."""

import contextlib
from collections.abc import Iterable

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient


def remove_models(tracking_uri: str, names: Iterable[str]) -> None:
    """Best effort: a cleanup failure must never hide the test result.

    Deletes each registered model (versions and aliases) and moves its experiment to
    MLflow's trash; artifacts are reclaimed by ``mlflow gc``.
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    for name in names:
        with contextlib.suppress(MlflowException):
            client.delete_registered_model(name)
        with contextlib.suppress(MlflowException):
            experiment = client.get_experiment_by_name(name)
            if experiment is not None:
                client.delete_experiment(experiment.experiment_id)
