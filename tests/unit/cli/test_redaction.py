from mlship.domain.redaction import redact


def test_credentials_in_uris_are_masked_anywhere_in_the_text() -> None:
    text = "MLflow is not reachable at https://laura:s3cret@mlflow.example/api (retry)"
    assert redact(text) == "MLflow is not reachable at https://***@mlflow.example/api (retry)"


def test_text_without_credentials_is_unchanged() -> None:
    assert (
        redact("http://127.0.0.1:5000/health and a@b.c") == "http://127.0.0.1:5000/health and a@b.c"
    )
