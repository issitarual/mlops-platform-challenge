"""Global seeding for reproducibility (best effort, not bit-for-bit)."""


def set_global_seeds(seed: int) -> None:
    """Seed ``random`` and, if installed, ``numpy``. ``PYTHONHASHSEED`` is set by the host."""
    raise NotImplementedError("E2")
