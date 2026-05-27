#!/usr/bin/env python3
import argparse
import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate deterministic TIFF-stack benchmark inputs.")
    parser.add_argument("--cases", default=str(ROOT / "benchmarks/config/cases.csv"))
    parser.add_argument("--input-root", default=str(ROOT / "tmp/benchmarks/input"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_cases(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    args = parse_args()
    seen = sorted({(case["pixelType"], case["shape"]) for case in read_cases(args.cases)})
    input_root = Path(args.input_root)
    input_root.mkdir(parents=True, exist_ok=True)

    for pixel_type, shape in seen:
        output = input_root / f"{pixel_type}_{shape}"
        if output.exists() and any(output.glob("*.tif*")) and not args.force:
            print(f"exists {output}")
            continue
        command = [
            "dotnet",
            "run",
            "--project",
            str(ROOT / "benchmarks/StackProcessing.Benchmarks/StackProcessing.Benchmarks.fsproj"),
            "--",
            "generate",
            "--output",
            str(output),
            "--shape",
            shape,
            "--pixel-type",
            pixel_type,
            "--pattern",
            "ramp",
        ]
        print(" ".join(command), flush=True)
        completed = subprocess.run(command, cwd=ROOT)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
