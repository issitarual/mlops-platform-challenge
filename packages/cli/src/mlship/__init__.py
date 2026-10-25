"""mlship: from a data scientist's train.py to a REST API."""

import os
from importlib.metadata import version

# Set before anything imports MLflow, which reads them at import time. Users can
# still opt in by setting them first.
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")  # usage telemetry is on by default
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # a banner printed on import

__version__ = version("mlship")
