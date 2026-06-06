#!/usr/bin/env python3
"""Run the StackProcessing chunk histogram parallelisation benchmark grid."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SIZES = [256, 512, 1024]
VARIANTS = ["dense", "sparse"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="tmp/benchmarks/chunk-histogram-parallel-raw.csv")
    parser.add_argument("--input-root", default="tmp/benchmarks/input")
    parser.add_argument("--dll", default="benchmarks/StackProcessing.Benchmarks/bin/Release/net10.0/StackProcessing.Benchmarks.dll")
    parser.add_argument("--pixel-type", default="UInt8")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--window-sizes", default="1,2,3,4", help="Comma-separated parallel window sizes. Worker count is inferred from window size.")
    parser.add_argument("--dotnet", default="dotnet")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    output.unlink(missing_ok=True)

    for repeat in range(1, args.repeats + 1):
        for size in SIZES:
            shape = f"{size}x{size}x{size}"
            input_dir = Path(args.input_root) / f"{args.pixel_type}_{shape}"
            if not input_dir.exists():
                raise FileNotFoundError(f"Missing benchmark input directory: {input_dir}")

            window_sizes = [int(value) for value in args.window_sizes.split(",") if value.strip()]
            for variant in VARIANTS:
                for window_size in window_sizes:
                    parameter = f"{variant}-window{window_size}"
                    command = [
                        sys.executable,
                        "benchmarks/tools/measure.py",
                        "--output",
                        str(output),
                        "--backend",
                        f"stackprocessing-chunk-reducer-window{window_size}",
                        "--operation",
                        f"histogram-{variant}",
                        "--pixel-type",
                        args.pixel_type,
                        "--shape",
                        shape,
                        "--parameter",
                        parameter,
                        "--repeat-index",
                        str(repeat),
                        "--",
                        args.dotnet,
                        args.dll,
                        "run-chunk-histogram",
                        "--pixel-type",
                        args.pixel_type,
                        "--input",
                        str(input_dir),
                        "--variant",
                        variant,
                        "--window-size",
                        str(window_size),
                    ]
                    print(f"repeat {repeat}: {shape} {variant} window-size={window_size}", flush=True)
                    subprocess.run(command, check=True)

    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
