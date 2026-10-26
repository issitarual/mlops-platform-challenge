"""``serve`` end to end on the host side: alias, download, verification, images, run."""

from pathlib import Path

import pytest
from fakes import FakeModelRegistry, FakeRunRepository, FakeRuntimePackager
from training_fakes import RecordingTrainingRuntime, make_environment, make_project, make_service

from mlship.adapters.hashing import Sha256Hasher
from mlship.domain.errors import (
    EnvironmentMismatchError,
    MlshipError,
    ModelIntegrityError,
    ModelNotFoundError,
)
from mlship.domain.serving import THREAD_LIMIT_VARIABLES, ServingLimits
from mlship.services.ports import HttpProbe
from mlship.services.serving_service import (
    ALLOW_PICKLE_TAG,
    ENVIRONMENT_KEY_TAG,
    LOCK_SHA256_TAG,
    SERIALIZATION_TAG,
    ServingPolicy,
    ServingService,
)

LIMITS = ServingLimits(cpus=1.0, memory="1g", pids=64)


class World:
    """One trained, registered and promoted model, sharing fakes with serving."""

    def __init__(self, tmp_path: Path, *, allow_pickle: bool = False) -> None:
        self.project, self.data = make_project(tmp_path)
        self.runtime = RecordingTrainingRuntime()
        self.runs = FakeRunRepository()
        self.registry = FakeModelRegistry(self.runs)
        trainer = make_service(self.runtime, runs=self.runs, registry=self.registry)
        outcome = trainer.train(
            self.project, self.data, {}, model_name="demo", allow_pickle=allow_pickle
        )
        assert outcome.model_version is not None
        self.version = outcome.model_version
        self.registry.set_alias("demo", "champion", self.version.version, {})

    def serving(
        self,
        packager: FakeRuntimePackager | None = None,
        *,
        probe: HttpProbe | None = None,
        dashboard_url: str | None = None,
        allow_pickle_enabled: bool = True,
    ) -> ServingService:
        return ServingService(
            self.runs,
            self.registry,
            make_environment(self.runtime, packager),
            self.runtime,
            Sha256Hasher(),
            ServingPolicy(
                ("a.B",),
                LIMITS,
                model_max_bytes=10**6,
                model_max_files=100,
                allow_pickle_enabled=allow_pickle_enabled,
                dashboard_url=dashboard_url,
            ),
            probe,
        )


def test_the_champion_is_verified_and_packaged_on_its_training_environment(
    tmp_path: Path,
) -> None:
    world = World(tmp_path)
    builds_before = len(world.runtime.builds)
    prepared = world.serving().prepare(world.project, "demo", "champion")

    assert prepared.model.version == world.version.version
    assert prepared.environment_reused, "the training image is reused, not rebuilt"
    assert len(world.runtime.builds) == builds_before
    ((spec, files),) = world.runtime.serving_specs
    assert spec.base == prepared.environment_image and spec.model == world.version
    assert spec.allow_pickle is False and spec.skops_trusted_types == ("a.B",)
    assert files == ["MLmodel"]


def test_run_publishes_the_port_and_forwards_runtime_settings(tmp_path: Path) -> None:
    world = World(tmp_path)
    service = world.serving()
    prepared = service.prepare(world.project, "demo", "champion")
    ready: list[str] = []
    service.run(
        prepared,
        port=9000,
        workers=2,
        environment={"MLSHIP_SERVING_API_KEY": "k"},
        on_log=lambda _: None,
        on_ready=ready.append,
    )
    (job,) = world.runtime.serving_jobs
    assert (job.host_port, job.workers, job.limits) == (9000, 2, LIMITS)
    assert job.environment == {
        "MLSHIP_SERVING_API_KEY": "k",
        **dict.fromkeys(THREAD_LIMIT_VARIABLES, "1"),
    }
    assert ready == ["http://127.0.0.1:9000"]


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_native_thread_pools_are_one_thread_whatever_the_workers(
    tmp_path: Path, workers: int
) -> None:
    world = World(tmp_path)
    service = world.serving()
    service.run(
        service.prepare(world.project, "demo", "champion"),
        port=9000,
        workers=workers,
        environment={},
        on_log=lambda _: None,
        on_ready=lambda _: None,
    )
    assert world.runtime.serving_jobs[-1].environment == dict.fromkeys(THREAD_LIMIT_VARIABLES, "1")


def test_missing_alias_says_how_to_promote(tmp_path: Path) -> None:
    world = World(tmp_path)
    with pytest.raises(ModelNotFoundError) as info:
        world.serving().prepare(world.project, "demo", "challenger")
    assert "mlship promote" in info.value.hint


def test_a_model_changed_in_the_artifact_store_is_never_packaged(tmp_path: Path) -> None:
    world = World(tmp_path)
    stored = world.runs.store / world.version.run_id / "model" / "MLmodel"
    stored.write_text("flavors: {python_function: {loader_module: evil}}\n")
    with pytest.raises(ModelIntegrityError, match="does not match its registered hash"):
        world.serving().prepare(world.project, "demo", "champion")
    assert world.runtime.serving_specs == []


def test_a_deleted_training_image_is_rebuilt_from_the_runs_lock(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.runtime.images.clear()  # e.g. docker image prune
    prepared = world.serving().prepare(world.project, "demo", "champion")
    assert not prepared.environment_reused
    assert world.runtime.builds[-1][0] == prepared.environment_image.tag


def test_a_changed_tool_cannot_pass_for_the_training_environment(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.runtime.images.clear()
    changed = FakeRuntimePackager(source="t" * 64)  # different runtime code
    with pytest.raises(EnvironmentMismatchError):
        world.serving(changed).prepare(world.project, "demo", "champion")


def test_a_lock_changed_in_the_artifact_store_never_builds_an_environment(
    tmp_path: Path,
) -> None:
    world = World(tmp_path)
    world.runtime.images.clear()
    stored = world.runs.store / world.version.run_id / "mlship.lock"
    stored.write_text(stored.read_text() + "evil-package==1.0 --hash=sha256:00\n")
    builds_before = len(world.runtime.builds)
    with pytest.raises(ModelIntegrityError, match="lock file"):
        world.serving().prepare(world.project, "demo", "champion")
    assert len(world.runtime.builds) == builds_before
    assert world.runtime.serving_specs == []


def test_the_manifest_in_the_artifact_store_is_not_trusted(tmp_path: Path) -> None:
    world = World(tmp_path)
    (world.runs.store / world.version.run_id / "manifest.json").write_text("not json")
    (world.runs.store / world.version.run_id / "run.json").write_text("not json")
    world.serving().prepare(world.project, "demo", "champion")
    assert len(world.runtime.serving_specs) == 1


@pytest.mark.parametrize("tag", [LOCK_SHA256_TAG, ENVIRONMENT_KEY_TAG])
def test_runs_without_recorded_environment_tags_are_refused(tmp_path: Path, tag: str) -> None:
    world = World(tmp_path)
    del world.runs.runs[world.version.run_id]["tags"][tag]
    with pytest.raises(MlshipError, match="does not record its environment"):
        world.serving().prepare(world.project, "demo", "champion")


def test_runs_without_a_recorded_lock_are_refused(tmp_path: Path) -> None:
    world = World(tmp_path)
    (world.runs.store / world.version.run_id / "mlship.lock").unlink()
    with pytest.raises(MlshipError, match="lock file"):
        world.serving().prepare(world.project, "demo", "champion")


def test_pickle_is_allowed_only_when_the_operator_allowed_it_at_train_time(
    tmp_path: Path,
) -> None:
    world = World(tmp_path, allow_pickle=True)
    tags = world.runs.runs[world.version.run_id]["tags"]
    assert tags[ALLOW_PICKLE_TAG] == "true"
    service = world.serving()
    tags[SERIALIZATION_TAG] = "skops"
    service.prepare(world.project, "demo", "champion")
    tags[SERIALIZATION_TAG] = "cloudpickle"
    service.prepare(world.project, "demo", "champion")
    assert [spec.allow_pickle for spec, _ in world.runtime.serving_specs] == [False, True]


def test_a_model_that_claims_cloudpickle_without_the_operators_flag_is_not_served_with_pickle(
    tmp_path: Path,
) -> None:
    world = World(tmp_path)  # trained without --allow-pickle
    tags = world.runs.runs[world.version.run_id]["tags"]
    assert tags[ALLOW_PICKLE_TAG] == "false"
    tags[SERIALIZATION_TAG] = "cloudpickle"  # what a forged result.json would have produced
    world.serving().prepare(world.project, "demo", "champion")
    ((spec, _),) = world.runtime.serving_specs
    assert spec.allow_pickle is False


def test_runs_without_the_allow_pickle_tag_are_never_served_with_pickle(tmp_path: Path) -> None:
    world = World(tmp_path, allow_pickle=True)
    tags = world.runs.runs[world.version.run_id]["tags"]
    tags[SERIALIZATION_TAG] = "cloudpickle"
    del tags[ALLOW_PICKLE_TAG]  # a run recorded before the tag existed
    world.serving().prepare(world.project, "demo", "champion")
    ((spec, _),) = world.runtime.serving_specs
    assert spec.allow_pickle is False


def test_the_platform_can_forbid_pickle_at_serve_time_even_for_an_allowed_run(
    tmp_path: Path,
) -> None:
    world = World(tmp_path, allow_pickle=True)
    world.runs.runs[world.version.run_id]["tags"][SERIALIZATION_TAG] = "cloudpickle"
    world.serving(allow_pickle_enabled=False).prepare(world.project, "demo", "champion")
    ((spec, _),) = world.runtime.serving_specs
    assert spec.allow_pickle is False


def test_every_serve_downloads_and_verifies_again(tmp_path: Path) -> None:
    world = World(tmp_path)
    service = world.serving()
    service.prepare(world.project, "demo", "champion")
    cached = world.project / ".mlship" / "serving" / "demo" / "v1" / "model" / "MLmodel"
    cached.write_text("tampered locally")
    service.prepare(world.project, "demo", "champion")  # a fresh copy, so it passes
    assert cached.read_text() != "tampered locally"
