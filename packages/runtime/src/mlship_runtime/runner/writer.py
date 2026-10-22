"""Persist the training result to ``/output`` (the only writable path)."""

from pathlib import Path

from mlship_sdk import TrainOutput


def write_outputs(result: TrainOutput, output_dir: Path) -> None:
    """Save the model as an MLflow pyfunc (offline), ``metrics.json`` and the manifest."""
    raise NotImplementedError("E2")
