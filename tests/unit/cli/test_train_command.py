from mlship.controllers.train import parse_params


def test_params_are_parsed() -> None:
    params, problems = parse_params(["alpha=0.5", "name=a=b"])
    assert params == {"alpha": "0.5", "name": "a=b"}
    assert problems == []


def test_every_malformed_param_is_reported() -> None:
    _, problems = parse_params(["alpha", "=1", "x=1", "x=2"])
    assert [p.code for p in problems] == ["PARAM_FORMAT", "PARAM_FORMAT", "PARAM_DUPLICATE"]
