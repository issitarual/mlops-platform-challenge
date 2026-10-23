"""Global seeding for reproducibility (best effort, not bit-for-bit; ADR-011).

``PYTHONHASHSEED`` and the thread limits (``OMP_NUM_THREADS`` and friends) are set by
the host as container environment variables: they only work if set before Python starts.
"""

import random


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        return
    np.random.seed(seed)
