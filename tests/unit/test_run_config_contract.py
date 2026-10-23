"""The host writes run.json; the runner must parse exactly what the host writes."""

import json
from pathlib import Path

from training_fakes import RecordingTrainingRuntime, make_project, make_service

from mlship_runtime.runner.config import parse_config


def test_host_run_json_is_valid_for_the_runner(tmp_path: Path) -> None:
    project, data = make_project(tmp_path)
    runtime = RecordingTrainingRuntime()
    make_service(runtime).train(project, data, {"alpha": "0.5"})
    config = parse_config(json.loads(runtime.specs[0].config_file.read_text()))
    assert config.params == {"alpha": "0.5", "seed": "42"}
    assert config.output_dir == Path("/output")
    assert config.serialization.allow_pickle is False
