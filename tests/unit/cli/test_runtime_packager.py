from mlship.adapters.runtime_packager import WorkspaceRuntimePackager


def test_requirements_come_from_runtime_metadata_without_the_sdk() -> None:
    requirements = WorkspaceRuntimePackager().requirements()
    names = {r.split(">")[0].split("=")[0].split("<")[0].strip().lower() for r in requirements}
    assert {"mlflow-skinny", "numpy", "pandas", "skops", "fastapi"} <= names
    assert "mlship-sdk" not in names


def test_source_hash_is_stable() -> None:
    packager = WorkspaceRuntimePackager()
    assert packager.source_sha256() == packager.source_sha256()
