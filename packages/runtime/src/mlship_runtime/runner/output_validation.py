"""Validate what ``train()`` returned, collecting every problem at once."""


def validate_train_output(output: object) -> list[str]:
    """Return all problems found (empty list = valid).

    Checks (E2): is a ``TrainOutput``; ``model`` has ``.predict``; metrics are finite
    numbers; ``model.predict(input_example)`` runs (smoke test).
    """
    raise NotImplementedError("E2")
