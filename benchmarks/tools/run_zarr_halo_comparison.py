#!/usr/bin/env python3
"""Run a focused Zarr halo-layout comparison.

The experiment holds the logical image and operation fixed while varying the
Zarr chunk side length. Python/Dask uses 3D `map_overlap`; StackProcessing uses
its z-stream halo over full-XY slabs.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", default="1024x1024x1024")
    parser.add_argument("--pixel-type", default="UInt8", choices=["UInt8", "UInt16", "Float32"])
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--chunk-sides", default="32,64,128")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--input-root", default=str(ROOT / "tmp/benchmarks/input-omezarr-halo"))
    parser.add_argument("--output-root", default=str(ROOT / "tmp/benchmarks/output-zarr-halo"))
    parser.add_argument("--results", default=str(ROOT / "tmp/zarr-halo-comparison.csv"))
    parser.add_argument(
        "--stackprocessing-dll",
        default=str(ROOT / "benchmarks/StackProcessing.Benchmarks/bin/Release/net10.0/StackProcessing.Benchmarks.dll"),
    )
    parser.add_argument("--force-inputs", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def successful_keys(path: Path) -> set[tuple[str, str, str, str]]:
    if not path.exists():
        return set()
    keys = set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("exitCode") == "0":
                keys.add((row["backend"], row["parameter"], row["repeat"], row["pixelType"]))
    return keys


def run(command: list[str]) -> int:
    print(" ".join(command), flush=True)
    return subprocess.run(command, cwd=ROOT).returncode


def generate_input(args, chunk_side: int) -> Path:
    input_dir = Path(args.input_root) / f"{args.pixel_type}_{args.shape}_chunk{chunk_side}"
    if input_dir.exists() and not args.force_inputs:
        return input_dir

    command = [
        sys.executable,
        str(ROOT / "benchmarks/tools/generate_omezarr_ramp.py"),
        "--output",
        str(input_dir),
        "--shape",
        args.shape,
        "--pixel-type",
        args.pixel_type,
        "--chunk-shape",
        f"{chunk_side}x{chunk_side}x{chunk_side}",
        "--force",
    ]
    code = run(command)
    if code != 0:
        raise SystemExit(code)
    return input_dir


def measured_command(args, backend: str, chunk_side: int, repeat: int, input_dir: Path) -> list[str]:
    parameter = f"chunk={chunk_side}"
    output = Path(args.output_root) / backend / f"median_{args.pixel_type}_{args.shape}_{parameter.replace('=', '-')}_r{repeat:02d}"
    if output.exists():
        shutil.rmtree(output)

    if backend == "stackprocessing-zarr-halo":
        payload = [
            "dotnet",
            args.stackprocessing_dll,
            "run-zarr",
            "--operation",
            "median",
            "--pixel-type",
            args.pixel_type,
            "--input",
            str(input_dir),
            "--output",
            str(output),
            "--shape",
            args.shape,
            "--radius",
            str(args.radius),
        ]
    elif backend == "python-dask-omezarr-halo":
        payload = [
            sys.executable,
            str(ROOT / "benchmarks/python-dask-omezarr/bench.py"),
            "--operation",
            "median",
            "--pixel-type",
            args.pixel_type,
            "--input",
            str(input_dir),
            "--output",
            str(output),
            "--radius",
            str(args.radius),
        ]
    else:
        raise ValueError(backend)

    return [
        sys.executable,
        str(ROOT / "benchmarks/tools/measure.py"),
        "--output",
        args.results,
        "--backend",
        backend,
        "--operation",
        "median",
        "--pixel-type",
        args.pixel_type,
        "--shape",
        args.shape,
        "--parameter",
        parameter,
        "--repeat-index",
        str(repeat),
        "--",
    ] + payload


def main() -> int:
    args = parse_args()
    chunk_sides = [int(part) for part in args.chunk_sides.split(",") if part.strip()]
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    existing = successful_keys(Path(args.results)) if args.skip_existing else set()

    for chunk_side in chunk_sides:
        input_dir = generate_input(args, chunk_side)
        for repeat in range(1, args.repeat + 1):
            for backend in ["stackprocessing-zarr-halo", "python-dask-omezarr-halo"]:
                key = (backend, f"chunk={chunk_side}", str(repeat), args.pixel_type)
                if key in existing:
                    print(f"skip existing {backend} chunk={chunk_side} r{repeat:02d}")
                    continue
                code = run(measured_command(args, backend, chunk_side, repeat, input_dir))
                if code != 0:
                    return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
