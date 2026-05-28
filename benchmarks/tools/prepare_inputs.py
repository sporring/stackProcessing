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
    parser.add_argument("--pixel-types", default="", help="Optional comma-separated pixel type filter, for example UInt8,Float32.")
    parser.add_argument("--shapes", default="", help="Optional comma-separated shape filter, for example 256x256x256,512x512x512.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_cases(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def filter_cases(cases, pixel_types, shapes):
    selected = {item.strip().lower() for item in pixel_types.split(",") if item.strip()}
    selected_shapes = {item.strip().lower() for item in shapes.split(",") if item.strip()}
    return [
        case
        for case in cases
        if (not selected or case["pixelType"].lower() in selected)
        and (not selected_shapes or case["shape"].lower() in selected_shapes)
    ]


def main():
    args = parse_args()
    cases = filter_cases(read_cases(args.cases), args.pixel_types, args.shapes)
    seen = sorted({(case["pixelType"], case["shape"]) for case in cases})
    if not seen:
        print(f"prepare_inputs.py: no cases matched --pixel-types {args.pixel_types!r} --shapes {args.shapes!r}", file=sys.stderr)
        return 2

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
