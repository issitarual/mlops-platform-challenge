"""Entry point: ``python -m mlship_runtime.runner``."""


def main() -> int:
    """Run one training job and return the process exit code.

    Steps (implemented in E2): load ``train.py`` -> set global seeds -> call
    ``train(data_dir, params)`` -> validate the output -> write model, metrics and
    manifest to ``/output``.
    """
    raise NotImplementedError("E2: training runner")


if __name__ == "__main__":
    raise SystemExit(main())
