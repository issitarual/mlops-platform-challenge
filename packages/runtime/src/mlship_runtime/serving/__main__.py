"""``python -m mlship_runtime.serving --port 8080 --workers 2``: the container entrypoint."""

import argparse
import os
from pathlib import Path

import uvicorn

from mlship_runtime.serving.core import logs

MULTIPROC_DIR = "/tmp/prometheus"  # noqa: S108 - tmpfs inside the container


def main() -> None:
    parser = argparse.ArgumentParser(prog="mlship_runtime.serving")
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104 - inside the container
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers > 1:
        os.environ.setdefault("PROMETHEUS_MULTIPROC_DIR", MULTIPROC_DIR)
        Path(os.environ["PROMETHEUS_MULTIPROC_DIR"]).mkdir(parents=True, exist_ok=True)
    logs.configure()
    uvicorn.run(
        "mlship_runtime.serving.main:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_config=None,
        server_header=False,
    )


if __name__ == "__main__":
    main()
