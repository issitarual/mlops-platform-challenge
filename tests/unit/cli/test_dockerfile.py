from mlship.adapters.dockerfile import EnvironmentDockerfile


def test_rendered_dockerfile_enforces_the_install_policy() -> None:
    text = EnvironmentDockerfile("python:3.12-slim@sha256:" + "0" * 64).render()
    assert text.count("FROM python:3.12-slim@sha256:" + "0" * 64) == 2  # build + final stage
    assert "--require-hashes" in text
    assert "--only-binary=:all:" in text
    assert text.count("pip uninstall --yes pip") == 2  # build venv and final image
    assert "USER 10001:10001" in text
    assert "MLFLOW_DISABLE_AGENT_HINT=1" in text
    assert "$PATH" in text and "$$" not in text
    assert not any(
        line.endswith("\\\\") for line in text.splitlines()
    )  # continuation: one backslash
