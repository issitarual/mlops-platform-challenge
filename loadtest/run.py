"""Run a load scenario against a served model and report latency and throughput.

    uv run python loadtest/run.py --scenario single --users 20 --duration 60s --label "2 workers"

Prints a Markdown row (p50/p95/p99, requests/s, failures) with the machine it ran on.
With ``--max-p95-ms`` it exits 1 when p95 is above the limit or any request failed:
the CI smoke test uses it that way.
"""

import argparse
import csv
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

LOCUSTFILE = Path(__file__).with_name("locustfile.py")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="http://127.0.0.1:8080")
    parser.add_argument("--scenario", choices=("single", "batch", "drift"), default="single")
    parser.add_argument("--users", type=int, default=20)
    parser.add_argument("--duration", default="60s")
    parser.add_argument("--max-p95-ms", type=float, default=None)
    parser.add_argument("--label", default="", help="server setup, e.g. '2 workers'")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="mlship-load-") as tmp:
        prefix = Path(tmp) / "stats"
        command = [
            sys.executable, "-m", "locust", "-f", str(LOCUSTFILE), "--headless",
            "--host", args.host, "--users", str(args.users), "--spawn-rate", str(args.users),
            "--run-time", args.duration, "--tags", args.scenario,
            "--csv", str(prefix), "--only-summary", "--exit-code-on-error", "0",
        ]  # fmt: skip
        subprocess.run(command, check=True, env=os.environ.copy())  # noqa: S603
        with prefix.with_name("stats_stats.csv").open(newline="", encoding="utf-8") as handle:
            total = next(r for r in csv.DictReader(handle) if r["Name"] == "Aggregated")

    requests, failures = int(total["Request Count"]), int(total["Failure Count"])
    p50, p95, p99 = (float(total[q]) for q in ("50%", "95%", "99%"))
    machine = f"{platform.system()} {platform.machine()}, {os.cpu_count()} CPUs"
    print("| Scenario | Users | Requests | Req/s | p50 ms | p95 ms | p99 ms | Failures | Machine |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    scenario = f"{args.scenario} ({args.label})" if args.label else args.scenario
    print(
        f"| {scenario} | {args.users} | {requests} | {float(total['Requests/s']):.1f} "
        f"| {p50:.0f} | {p95:.0f} | {p99:.0f} | {failures} | {machine} |"
    )
    if args.max_p95_ms is not None and (failures or requests == 0 or p95 > args.max_p95_ms):
        print(f"FAILED: p95 {p95:.0f} ms (limit {args.max_p95_ms:.0f}), {failures} failure(s)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
