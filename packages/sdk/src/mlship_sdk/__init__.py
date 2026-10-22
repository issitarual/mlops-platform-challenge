"""Contract between the data scientist's ``train.py`` and mlship.

This package has zero dependencies on purpose: it is imported by user code inside
the training image, so anything it pulls in would compete with the user's requirements.
"""

from mlship_sdk.contract import TrainOutput

__all__ = ["TrainOutput"]
