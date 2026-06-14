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
import signal
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


def command_option(command: list[str], name: str) -> str | None:
    for index, value in enumerate(command[:-1]):
        if value == name:
            return command[index + 1]
    return None


def trim_path_suffix(value: str) -> str:
    return value.rstrip("/\\")


def benchmark_preclean_paths(command: list[str]) -> list[Path]:
    output = command_option(command, "--output")
    if not output:
        return []

    paths = [Path(output)]
    command_text = " ".join(command)
    if "run-chunk-fft3d-zarr-roundtrip-io" in command:
        temp_base = command_option(command, "--temp-zarr") or (trim_path_suffix(output) + ".spectral.tmp.zarr")
        paths.extend(Path(trim_path_suffix(temp_base) + suffix) for suffix in [".xy", ".z", ".invz"])
    elif "run-chunk-fft3d-zarr-subchunked-roundtrip-io" in command:
        temp_base = command_option(command, "--temp-zarr") or (trim_path_suffix(output) + ".spectral-subchunked.tmp.zarr")
        paths.extend(Path(trim_path_suffix(temp_base) + suffix) for suffix in [".xy", ".invz"])
    elif "fft3d-zarr" in command_text and (temp_base := command_option(command, "--temp-zarr")):
        paths.append(Path(temp_base))

    return paths


def preclean_paths(paths: list[Path]) -> list[str]:
    cleaned = []
    seen = set()
    for path in paths:
        normalized = path.resolve()
        if normalized in seen:
            continue
        seen.add(normalized)
        if normalized.exists():
            if normalized.is_dir():
                import shutil

                shutil.rmtree(normalized)
            else:
                normalized.unlink()
        normalized.mkdir(parents=True, exist_ok=True)
        cleaned.append(str(normalized))
    return cleaned


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
    parser.add_argument("--repeat-index", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    repeat_index = args.repeat_index if args.repeat_index is not None else (args.repeat if args.repeat is not None else 1)

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
    precleaned = preclean_paths(benchmark_preclean_paths(command))
    if precleaned:
        env["BENCHMARK_PRECLEANED_OUTPUTS"] = os.pathsep.join(precleaned)
        env["BENCHMARK_SKIP_OUTPUT_CLEANING"] = "1"

    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    start = time.perf_counter()
    process = subprocess.Popen(command, env=env, start_new_session=(os.name == "posix"))
    try:
        exit_code = process.wait()
    except KeyboardInterrupt:
        terminate_process_tree(process)
        exit_code = 130
    finally:
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
                "repeat": repeat_index,
                "exitCode": exit_code,
                "wallSeconds": f"{elapsed:.9f}",
                "internalSeconds": internal_seconds,
                "peakRssKiB": peak_rss_kib(before, after),
                "command": " ".join(command),
            }
        )

    return exit_code


def terminate_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return

    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:
        process.terminate()

    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass

    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
    else:
        process.kill()
    process.wait()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
