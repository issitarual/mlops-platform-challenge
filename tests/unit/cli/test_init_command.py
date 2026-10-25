"""``mlship init``: the template is valid, trains as is, and never overwrites silently."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from fakes import FakeLockResolver
from typer.testing import CliRunner
from validation_fakes import FailingResolver, validation_service

from mlship.adapters.project_template import PackagedProjectTemplate
from mlship.adapters.runtime_packager import WorkspaceRuntimePackager
from mlship.controllers.app import app
from mlship.domain.errors import MlshipError
from mlship.services.scaffold_service import ScaffoldService
from mlship.services.validation.validator import ProjectValidator
from mlship_sdk import TrainOutput

runner = CliRunner()
FILES = {".gitignore", "README.md", "data/example.csv", "requirements.txt", "train.py"}


@pytest.fixture
def scaffold() -> ScaffoldService:
    return ScaffoldService(PackagedProjectTemplate())


def test_the_template_has_every_file_under_its_real_name() -> None:
    assert set(PackagedProjectTemplate().files()) == FILES


def test_a_new_project_passes_strict_validation(tmp_path: Path, scaffold: ScaffoldService) -> None:
    scaffold.create(tmp_path)
    validator = ProjectValidator(WorkspaceRuntimePackager().requirements())
    report = validator.validate(tmp_path, tmp_path / "data")
    assert report.issues == ()


def test_the_generated_train_py_trains_a_usable_model(
    tmp_path: Path, scaffold: ScaffoldService
) -> None:
    scaffold.create(tmp_path)
    spec = importlib.util.spec_from_file_location("scaffolded_train", tmp_path / "train.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = module.train(tmp_path / "data", {"seed": "42", "n_estimators": "20"})
    assert isinstance(output, TrainOutput)
    assert output.metrics["r2"] > 0.8
    assert len(output.model.predict(output.input_example)) == 5


def test_existing_files_are_all_listed_and_nothing_is_written(
    tmp_path: Path, scaffold: ScaffoldService
) -> None:
    (tmp_path / "train.py").write_text("mine")
    (tmp_path / "requirements.txt").write_text("mine")
    with pytest.raises(MlshipError) as info:
        scaffold.create(tmp_path)
    assert "requirements.txt, train.py" in info.value.message
    assert not (tmp_path / "README.md").exists()
    assert (tmp_path / "train.py").read_text() == "mine"


def test_force_overwrites_but_never_through_a_link(
    tmp_path: Path, scaffold: ScaffoldService
) -> None:
    (tmp_path / "train.py").write_text("mine")
    scaffold.create(tmp_path, force=True)
    assert (tmp_path / "train.py").read_text() != "mine"

    outside = tmp_path / "outside.txt"
    outside.write_text("keep")
    project = tmp_path / "linked"
    project.mkdir()
    try:
        (project / "README.md").symlink_to(outside)
    except OSError:
        pytest.skip("symbolic links need extra privileges on this system")
    with pytest.raises(MlshipError, match="symbolic links"):
        scaffold.create(project, force=True)
    assert outside.read_text() == "keep"


def _init(monkeypatch: pytest.MonkeyPatch, resolver: FakeLockResolver, *args: str) -> str:
    services = SimpleNamespace(
        scaffold=ScaffoldService(PackagedProjectTemplate()), validation=validation_service(resolver)
    )
    monkeypatch.setattr("mlship.controllers.init.load_services", lambda: services)
    result = runner.invoke(app, ["init", *args])
    assert result.exit_code == 0, result.output
    return " ".join(result.output.split())


def test_init_writes_the_lock_and_prints_the_next_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = _init(monkeypatch, FakeLockResolver(), str(tmp_path / "my-model"))
    assert "Wrote mlship.lock" in output and (tmp_path / "my-model" / "mlship.lock").is_file()
    assert "mlship validate --data data" in output and "cd " in output
    again = runner.invoke(app, ["init", str(tmp_path / "my-model")])
    assert again.exit_code == 1 and "--force" in again.output


def test_init_without_the_package_index_still_creates_the_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = _init(monkeypatch, FailingResolver(), str(tmp_path / "offline"))
    assert (tmp_path / "offline" / "train.py").is_file()
    assert not (tmp_path / "offline" / "mlship.lock").exists()
    assert "mlship.lock was not created" in output and "mlship lock" in output
