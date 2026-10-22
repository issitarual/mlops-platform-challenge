from fastapi.testclient import TestClient

from mlship_runtime.serving.main import create_app


def test_liveness_is_ok_before_model_loads() -> None:
    with TestClient(create_app()) as client:
        assert client.get("/health/live").status_code == 200


def test_readiness_is_unavailable_until_model_loads() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "loading"}
