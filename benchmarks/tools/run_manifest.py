#!/usr/bin/env python3
import argparse
import csv
import os
import signal
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
    parser.add_argument("--pixel-types", default="", help="Optional comma-separated pixel type filter, for example UInt8,Float32.")
    parser.add_argument("--shapes", default="", help="Optional comma-separated shape filter, for example 256x256x256,512x512x512.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--itk-exe", default=str(ROOT / "benchmarks/cpp-itk/build/benchmark_itk"))
    parser.add_argument("--matlab-exe", default="matlab")
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
    if key == "kernelSize":
        return ["--kernel-size", value]
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
        if case.get("parameterName") == "kernelSize":
            matlab_args["kernelSize"] = case["parameterValue"]
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
        "--repeat-index",
        str(repeat),
        "--",
    ] + command


def terminate_process_tree(process):
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


def run_measured_command(command):
    process = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), start_new_session=(os.name == "posix"))
    try:
        return process.wait()
    except KeyboardInterrupt:
        terminate_process_tree(process)
        return 130


def main():
    args = parse_args()
    cases = filter_cases(read_cases(args.cases), args.pixel_types, args.shapes)
    if not cases:
        print(f"run_manifest.py: no cases matched --pixel-types {args.pixel_types!r} --shapes {args.shapes!r}", file=sys.stderr)
        return 2

    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    for repeat in range(1, args.repeat + 1):
        for case in cases:
            command = measured_command(args, case, repeat)
            printable = " ".join(shlex.quote(part) for part in command)
            print(printable, flush=True)
            if not args.dry_run:
                return_code = run_measured_command(command)
                if return_code != 0:
                    return return_code
    return 0


if __name__ == "__main__":
    sys.exit(main())
