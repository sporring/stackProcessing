#!/usr/bin/env python3
"""Run the StackProcessing chunk dilation parallel window benchmark grid."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SIZES = [256, 512, 1024]
WORKERS = [1, 2, 3, 4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="tmp/benchmarks/chunk-dilate-parallel-raw.csv")
    parser.add_argument("--input-root", default="tmp/benchmarks/input")
    parser.add_argument("--output-root", default="tmp/benchmarks/output-chunk-dilate-parallel")
    parser.add_argument("--dll", default="benchmarks/StackProcessing.Benchmarks/bin/Release/net10.0/StackProcessing.Benchmarks.dll")
    parser.add_argument("--pixel-type", default="UInt8")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--sizes", default="256,512,1024", help="Comma-separated cube sizes.")
    parser.add_argument("--workers", default="1,2,3,4", help="Comma-separated worker/window sizes.")
    parser.add_argument("--dotnet", default="dotnet")
    return parser.parse_args()


def parse_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    output.unlink(missing_ok=True)

    sizes = parse_ints(args.sizes)
    workers = parse_ints(args.workers)

    for repeat in range(1, args.repeats + 1):
        for size in sizes:
            shape = f"{size}x{size}x{size}"
            input_dir = Path(args.input_root) / f"{args.pixel_type}_{shape}"
            if not input_dir.exists():
                raise FileNotFoundError(f"Missing benchmark input directory: {input_dir}")

            for worker_count in workers:
                output_dir = Path(args.output_root) / f"{args.pixel_type}_{shape}_w{worker_count}"
                command = [
                    sys.executable,
                    "benchmarks/tools/measure.py",
                    "--output",
                    str(output),
                    "--backend",
                    f"stackprocessing-chunk-dilate-w{worker_count}",
                    "--operation",
                    "dilate-zonohedral",
                    "--pixel-type",
                    args.pixel_type,
                    "--shape",
                    shape,
                    "--parameter",
                    f"radius{args.radius}-workers{worker_count}",
                    "--repeat-index",
                    str(repeat),
                    "--",
                    args.dotnet,
                    args.dll,
                    "run-chunk-dilate",
                    "--input",
                    str(input_dir),
                    "--output",
                    str(output_dir),
                    "--radius",
                    str(args.radius),
                    "--workers",
                    str(worker_count),
                ]
                print(f"repeat {repeat}: {shape} workers={worker_count}", flush=True)
                subprocess.run(command, check=True)

    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
