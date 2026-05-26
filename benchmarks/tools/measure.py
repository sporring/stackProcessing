#!/usr/bin/env python3
"""Run a benchmark command and append wall-time/peak-RSS data to CSV.

This wrapper keeps measurement independent of each backend. It uses only the
Python standard library. On Linux, ru_maxrss is reported in KiB. On macOS it is
reported in bytes, so the script normalizes to KiB.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path


FIELDNAMES = [
    "backend",
    "operation",
    "pixelType",
    "shape",
    "parameter",
    "repeat",
    "exitCode",
    "wallSeconds",
    "internalSeconds",
    "peakRssKiB",
    "command",
]


def peak_rss_kib(before: int, after: int) -> int:
    delta = max(0, after - before)
    if platform.system() == "Darwin":
        return int(delta / 1024)
    return int(delta)


def upgrade_existing_output(output: Path) -> None:
    if not output.exists():
        return
    with output.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames == FIELDNAMES:
            return
        rows = list(reader)

    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Measure a benchmark command and append one row to CSV.",
        allow_abbrev=False,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--operation", required=True)
    parser.add_argument("--pixel-type", required=True)
    parser.add_argument("--shape", required=True)
    parser.add_argument("--parameter", default="")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("missing command after --")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    upgrade_existing_output(output)
    with tempfile.NamedTemporaryFile(prefix="benchmark-internal-", suffix=".txt", dir=output.parent, delete=False) as handle:
        internal_path = Path(handle.name)

    env = os.environ.copy()
    env["BENCHMARK_INTERNAL_SECONDS_PATH"] = str(internal_path)

    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    start = time.perf_counter()
    completed = subprocess.run(command, env=env)
    elapsed = time.perf_counter() - start
    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss

    internal_seconds = ""
    try:
        text = internal_path.read_text().strip()
        if text:
            internal_seconds = f"{float(text):.9f}"
    finally:
        internal_path.unlink(missing_ok=True)

    write_header = not output.exists()

    with output.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "backend": args.backend,
                "operation": args.operation,
                "pixelType": args.pixel_type,
                "shape": args.shape,
                "parameter": args.parameter,
                "repeat": args.repeat,
                "exitCode": completed.returncode,
                "wallSeconds": f"{elapsed:.9f}",
                "internalSeconds": internal_seconds,
                "peakRssKiB": peak_rss_kib(before, after),
                "command": " ".join(command),
            }
        )

    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
