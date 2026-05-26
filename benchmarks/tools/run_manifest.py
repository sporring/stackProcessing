#!/usr/bin/env python3
import argparse
import csv
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description="Run or print comparative benchmark commands from cases.csv.")
    parser.add_argument("--cases", default=str(ROOT / "benchmarks/config/cases.csv"))
    parser.add_argument(
        "--backend",
        required=True,
        choices=["stackprocessing", "python-skimage-scipy", "cpp-itk", "matlab", "python-dask-omezarr"],
    )
    parser.add_argument("--results", default=str(ROOT / "benchmarks/results/raw.csv"))
    parser.add_argument("--input-root", default=str(ROOT / "tmp/benchmarks/input"))
    parser.add_argument("--output-root", default=str(ROOT / "tmp/benchmarks/output"))
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--itk-exe", default=str(ROOT / "benchmarks/cpp-itk/build/benchmark_itk"))
    parser.add_argument("--matlab-exe", default="matlab")
    return parser.parse_args()


def read_cases(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def input_dir(args, case):
    return Path(args.input_root) / f"{case['pixelType']}_{case['shape']}"


def output_dir(args, case, repeat):
    parameter = case_parameter(case).replace("=", "-")
    return Path(args.output_root) / args.backend / f"{case['operation']}_{case['pixelType']}_{case['shape']}_{parameter}_r{repeat:02d}"


def case_parameter(case):
    name = case.get("parameterName", "")
    value = case.get("parameterValue", "")
    if not name:
        return "none"
    return f"{name}={value}"


def parameter_args(case):
    key = case.get("parameterName", "")
    value = case.get("parameterValue", "")
    if key == "radius":
        return ["--radius", value]
    if key == "threshold":
        return ["--threshold", value]
    if key == "sigma":
        return ["--sigma", value]
    if key == "window":
        return ["--window", value]
    return []


def backend_command(args, case, repeat):
    inp = str(input_dir(args, case))
    out = str(output_dir(args, case, repeat))
    common = ["--operation", case["operation"], "--pixel-type", case["pixelType"], "--input", inp, "--output", out]
    params = parameter_args(case)

    if args.backend == "stackprocessing":
        return ["dotnet", "run", "--project", str(ROOT / "benchmarks/StackProcessing.Benchmarks/StackProcessing.Benchmarks.fsproj"), "--", "run"] + common + params
    if args.backend == "python-skimage-scipy":
        return ["python3", str(ROOT / "benchmarks/python-skimage-scipy/bench.py")] + common + params
    if args.backend == "python-dask-omezarr":
        return ["python3", str(ROOT / "benchmarks/python-dask-omezarr/bench.py")] + common + params
    if args.backend == "cpp-itk":
        return [args.itk_exe] + common + params
    if args.backend == "matlab":
        matlab_args = {
            "operation": case["operation"],
            "pixelType": case["pixelType"],
            "input": inp,
            "output": out,
        }
        if case.get("parameterName") == "radius":
            matlab_args["radius"] = case["parameterValue"]
        if case.get("parameterName") == "threshold":
            matlab_args["threshold"] = case["parameterValue"]
        if case.get("parameterName") == "sigma":
            matlab_args["sigma"] = case["parameterValue"]
        if case.get("parameterName") == "window":
            matlab_args["window"] = case["parameterValue"]
        call = "addpath('%s'); bench_stack(%s)" % (
            str(ROOT / "benchmarks/matlab").replace("'", "''"),
            ",".join("'%s','%s'" % (k, str(v).replace("'", "''")) for k, v in matlab_args.items()),
        )
        return [args.matlab_exe, "-nodisplay", "-nojvm", "-batch", call]
    raise AssertionError(args.backend)


def measured_command(args, case, repeat):
    command = backend_command(args, case, repeat)
    return [
        "python3",
        str(ROOT / "benchmarks/tools/measure.py"),
        "--output",
        args.results,
        "--backend",
        args.backend,
        "--operation",
        case["operation"],
        "--pixel-type",
        case["pixelType"],
        "--shape",
        case["shape"],
        "--parameter",
        case_parameter(case),
        "--repeat",
        str(repeat),
        "--",
    ] + command


def main():
    args = parse_args()
    cases = read_cases(args.cases)
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    for repeat in range(1, args.repeat + 1):
        for case in cases:
            command = measured_command(args, case, repeat)
            printable = " ".join(shlex.quote(part) for part in command)
            print(printable, flush=True)
            if not args.dry_run:
                completed = subprocess.run(command, cwd=ROOT, env=os.environ.copy())
                if completed.returncode != 0:
                    return completed.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
