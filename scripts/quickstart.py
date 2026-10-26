"""One-command demo: install, MLflow, data, train, promote and serve California Housing.

Run from the repository root: ``uv run scripts/quickstart.py``

It only chains the documented commands (README, "Passo a passo"); nothing here is
needed to use mlship. Safe to re-run: every train creates a new model version and the
script promotes the one it just trained.
"""

import argparse
import io
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "examples/california-housing"
DATA = f"{PROJECT}/data"
MODEL_NAME = "california-housing"
REGISTERED = re.compile(rf"Registered {MODEL_NAME} version (\d+)")
SAMPLE = (
    '{"rows": [{"MedInc": 8.3, "HouseAge": 41.0, "AveRooms": 6.98, "AveBedrms": 1.02, '
    '"Population": 322.0, "AveOccup": 2.56, "Latitude": 37.88, "Longitude": -122.23}]}'
)


def step(number: int, total: int, title: str) -> None:
    print(f"\n[{number}/{total}] {title}", flush=True)


def fail(message: str) -> NoReturn:
    print(f"\nError: {message}", file=sys.stderr)
    raise SystemExit(1)


def run(args: list[str], *, capture: bool = False) -> str:
    """Run a command in the repository root, streaming its output; no shell involved."""
    process = subprocess.Popen(
        args,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if process.stdout is None:
        fail("Could not read the command output.")
    lines: list[str] = []
    for line in process.stdout:
        print(line, end="", flush=True)
        if capture:
            lines.append(line)
    if process.wait() != 0:
        fail(f"`{' '.join(args)}` failed (exit {process.returncode}). See the output above.")
    return "".join(lines)


def check_prerequisites() -> None:
    if shutil.which("uv") is None:
        fail("uv not found. Install it from https://docs.astral.sh/uv/ and try again.")
    if shutil.which("docker") is None:
        fail("docker not found. Install Docker and try again.")
    probe = subprocess.run(["docker", "info"], capture_output=True, check=False)
    if probe.returncode != 0:
        fail("Docker is not running. Open Docker Desktop (or start the daemon) and try again.")


def main() -> None:
    # Windows consoles default to cp1252 and cannot print arrows or ellipses from the CLI.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--no-serve",
        action="store_true",
        help="stop after promoting (do not start the API)",
    )
    options = parser.parse_args()
    total = 5 if options.no_serve else 6

    check_prerequisites()

    step(1, total, "Installing mlship (uv sync)")
    run(["uv", "sync", "--all-packages"])

    step(2, total, "Starting MLflow (docker compose up)")
    run(["docker", "compose", "up", "-d", "--wait", "mlflow"])

    step(3, total, "Downloading the example data")
    run(["uv", "run", "scripts/download_california_housing.py"])

    step(4, total, "Training in an isolated container (first run builds the image)")
    output = run(["uv", "run", "mlship", "train", PROJECT, "--data", DATA], capture=True)
    match = REGISTERED.search(output)
    if match is None:
        fail("Could not find the registered model version in the train output.")
    version = match.group(1)

    step(5, total, f"Promoting version {version} to the champion alias")
    run(["uv", "run", "mlship", "promote", version, "--project", PROJECT])

    if options.no_serve:
        print("\nDone. Start the API with: uv run mlship serve --project " + PROJECT)
        return

    step(6, total, "Serving the model (Ctrl+C to stop)")
    print("Once it says Ready, call it from another terminal:")
    print(
        "  curl -s http://127.0.0.1:8080/v1/predict "
        f"-H 'content-type: application/json' -d '{SAMPLE}'"
    )
    print("Docs: http://127.0.0.1:8080/docs\n", flush=True)
    run(["uv", "run", "mlship", "serve", "--project", PROJECT])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. MLflow keeps running: `docker compose down` to stop it.")
