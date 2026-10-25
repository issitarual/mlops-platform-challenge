"""The model API with a real saved model: contract, limits, errors, auth, metrics."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from serving_fixtures import saved_model, settings_for

from mlship_runtime.serving.main import create_app

ROW = {"rooms": 3.0, "age": 2}


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    return saved_model(tmp_path)


@pytest.fixture
def client(model_dir: Path) -> Iterator[TestClient]:
    with TestClient(create_app(settings_for(model_dir))) as test_client:
        yield test_client


def test_health_is_split_into_liveness_and_readiness(client: TestClient) -> None:
    assert client.get("/health/live").json() == {"status": "ok"}
    assert client.get("/health/ready").json() == {"status": "ready"}


def test_prediction_returns_one_value_per_row_and_the_model_version(client: TestClient) -> None:
    response = client.post("/v1/predict", json={"rows": [ROW, {"rooms": 4.0, "age": 1}]})
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["predictions"]) == 2
    assert body["model"] == {"name": "houses", "version": "3"}
    assert body["request_id"] == response.headers["X-Request-ID"]


def test_openapi_documents_the_real_columns(client: TestClient) -> None:
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    assert set(schemas["Row"]["properties"]) == {"rooms", "age"}
    assert schemas["Row"]["properties"]["age"]["type"] == "integer"


def test_every_field_error_is_reported_at_once_without_echoing_values(
    client: TestClient,
) -> None:
    rows = [{"rooms": "three", "age": 2}, {"age": 1.5}, {"rooms": 1.0, "age": 1, "extra": 9}]
    response = client.post("/v1/predict", json={"rows": rows})
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "invalid_request"
    locations = {d["loc"] for d in body["details"]}
    assert {"rows.0.rooms", "rows.1.rooms", "rows.1.age", "rows.2.extra"} <= locations
    assert "three" not in response.text


def test_booleans_and_numeric_strings_are_not_coerced(client: TestClient) -> None:
    response = client.post("/v1/predict", json={"rows": [{"rooms": True, "age": "2"}]})
    assert response.status_code == 422
    assert len(response.json()["details"]) == 2


def test_empty_and_oversized_batches_are_refused(client: TestClient) -> None:
    assert client.post("/v1/predict", json={"rows": []}).status_code == 422
    too_many = client.post("/v1/predict", json={"rows": [ROW] * 6})
    assert too_many.status_code == 422


def test_non_finite_numbers_are_refused_as_a_client_error(client: TestClient) -> None:
    response = client.post(
        "/v1/predict",
        content=b'{"rows": [{"rooms": NaN, "age": 1}, {"rooms": Infinity, "age": 1}]}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422, response.text
    assert {d["loc"] for d in response.json()["details"]} == {"rows.0.rooms", "rows.1.rooms"}


def test_payload_above_the_limit_is_refused_before_parsing(client: TestClient) -> None:
    response = client.post(
        "/v1/predict",
        content=b'{"rows": [' + b" " * 5_000 + b"]}",
        headers={"content-type": "application/json", "X-Request-ID": "big-1"},
    )
    assert response.status_code == 413
    assert response.json()["error"] == "payload_too_large"
    assert response.json()["request_id"] == response.headers["X-Request-ID"] == "big-1"


def test_chunked_payload_above_the_limit_is_refused_too(client: TestClient) -> None:
    def chunks() -> Iterator[bytes]:
        yield b'{"rows": ['
        for _ in range(10):
            yield b" " * 1_000
        yield b"]}"

    response = client.post(
        "/v1/predict", content=chunks(), headers={"content-type": "application/json"}
    )
    assert "content-length" not in response.request.headers
    assert response.status_code == 413, response.text


def test_chunked_payload_within_the_limit_is_served(client: TestClient) -> None:
    def chunks() -> Iterator[bytes]:
        yield b'{"rows": ['
        yield b'{"rooms": 3.0, "age": 2}'
        yield b"]}"

    response = client.post(
        "/v1/predict", content=chunks(), headers={"content-type": "application/json"}
    )
    assert response.status_code == 200, response.text


def test_internal_errors_hide_the_stack_trace(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*_: object) -> None:
        raise RuntimeError("secret internals at /opt/mlship/model")

    # As a real server would: the error is logged, the client gets the JSON response.
    with TestClient(create_app(settings_for(model_dir)), raise_server_exceptions=False) as c:
        monkeypatch.setattr(c.app.state.prediction_service, "predict", explode)  # type: ignore[attr-defined]
        response = c.post("/v1/predict", json={"rows": [ROW]})
        metrics = c.get("/metrics").text
    assert response.status_code == 500
    assert response.json()["error"] == "internal_error"
    assert response.json()["request_id"] == response.headers["X-Request-ID"]
    assert "secret internals" not in response.text and "Traceback" not in response.text
    assert 'mlship_http_requests_total{route="/v1/predict",status="500"}' in metrics


def test_slow_predictions_time_out(model_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    app = create_app(settings_for(model_dir, request_timeout_seconds=0.05))
    with TestClient(app) as slow_client:
        service = slow_client.app.state.prediction_service  # type: ignore[attr-defined]
        monkeypatch.setattr(service, "predict", lambda *_: time.sleep(0.5))
        response = slow_client.post("/v1/predict", json={"rows": [ROW]})
    assert response.status_code == 504 and response.json()["error"] == "timeout"


def test_api_key_protects_predictions_but_not_health_or_metrics(model_dir: Path) -> None:
    app = create_app(settings_for(model_dir, api_key="s3cret"))
    with TestClient(app) as keyed:
        assert keyed.post("/v1/predict", json={"rows": [ROW]}).status_code == 401
        wrong = keyed.post("/v1/predict", json={"rows": [ROW]}, headers={"X-API-Key": "nope"})
        assert wrong.status_code == 401
        ok = keyed.post("/v1/predict", json={"rows": [ROW]}, headers={"X-API-Key": "s3cret"})
        assert ok.status_code == 200
        assert keyed.get("/health/ready").status_code == 200
        assert keyed.get("/metrics").status_code == 200


def test_metrics_expose_latency_rows_and_the_served_version(client: TestClient) -> None:
    client.post("/v1/predict", json={"rows": [ROW, ROW]})
    text = client.get("/metrics").text
    assert 'mlship_http_request_duration_seconds_bucket{le="0.005",route="/v1/predict"}' in text
    assert "mlship_predicted_rows_total" in text
    assert 'mlship_model_info{name="houses",version="3"} 1.0' in text


def test_a_safe_request_id_is_kept_and_an_unsafe_one_replaced(client: TestClient) -> None:
    kept = client.get("/health/live", headers={"X-Request-ID": "abc-123"})
    assert kept.headers["X-Request-ID"] == "abc-123"
    replaced = client.get("/health/live", headers={"X-Request-ID": "x\nInjected: 1"})
    assert replaced.headers["X-Request-ID"] != "x\nInjected: 1"
