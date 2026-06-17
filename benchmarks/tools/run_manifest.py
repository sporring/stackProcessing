#!/usr/bin/env python3
import argparse
import csv
import os
import shutil
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
        choices=[
            "stackprocessing",
            "stackprocessing-arraypool",
            "stackprocessing-arraypool-slice",
            "stackprocessing-arraypool-slice-reuse",
            "stackprocessing-byte-slice-reuse",
            "stackprocessing-byte-float32-slice-reuse",
            "stackprocessing-libtiff-direct-copy",
            "stackprocessing-libtiff-direct-threshold",
            "stackprocessing-libtiff-direct-threshold-intype",
            "stackprocessing-libtiff-strip-copy",
            "stackprocessing-libtiff-raw-strip-copy",
            "stackprocessing-native-libtiff-raw-strip-copy",
            "stackprocessing-tifflibrary-raw-strip-copy",
            "stackprocessing-imagesharp-copy",
            "stackprocessing-zarr",
            "stackprocessing-zarr-direct",
            "python-skimage-scipy",
            "cpp-itk",
            "matlab",
            "python-dask-omezarr",
        ],
    )
    parser.add_argument("--results", default=str(ROOT / "benchmarks/results/raw.csv"))
    parser.add_argument("--input-root", default=str(ROOT / "tmp/benchmarks/input"))
    parser.add_argument("--output-root", default=str(ROOT / "tmp/benchmarks/output"))
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--repeat-start", type=int, default=None, help="First repeat index to run. Defaults to 1.")
    parser.add_argument("--repeat-end", type=int, default=None, help="Last repeat index to run. Defaults to --repeat.")
    parser.add_argument("--pixel-types", default="", help="Optional comma-separated pixel type filter, for example UInt8,Float32.")
    parser.add_argument("--shapes", default="", help="Optional comma-separated shape filter, for example 256x256x256,512x512x512.")
    parser.add_argument("--operations", default="", help="Optional comma-separated operation filter, for example median,dilate.")
    parser.add_argument("--parameters", default="", help="Optional comma-separated parameter filter, for example radius=1,radius=2.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-outputs", action="store_true", help="Keep backend output stacks after each measured case.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip cases that already have a successful row in --results for the same backend, operation, pixel type, shape, parameter, and repeat.",
    )
    parser.add_argument("--itk-exe", default=str(ROOT / "benchmarks/cpp-itk/build/benchmark_itk"))
    parser.add_argument("--matlab-exe", default="matlab")
    parser.add_argument(
        "--stackprocessing-dll",
        default=str(ROOT / "benchmarks/StackProcessing.Benchmarks/bin/Debug/net10.0/StackProcessing.Benchmarks.dll"),
        help="Built StackProcessing benchmark DLL used by the stackprocessing backend.",
    )
    return parser.parse_args()


def read_cases(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def filter_cases(cases, pixel_types, shapes, operations, parameters):
    selected = {item.strip().lower() for item in pixel_types.split(",") if item.strip()}
    selected_shapes = {item.strip().lower() for item in shapes.split(",") if item.strip()}
    selected_operations = {item.strip().lower() for item in operations.split(",") if item.strip()}
    selected_parameters = {item.strip().lower() for item in parameters.split(",") if item.strip()}
    return [
        case
        for case in cases
        if (not selected or case["pixelType"].lower() in selected)
        and (not selected_shapes or case["shape"].lower() in selected_shapes)
        and (not selected_operations or case["operation"].lower() in selected_operations)
        and (not selected_parameters or case_parameter(case).lower() in selected_parameters)
    ]


def backend_supports_case(backend, case):
    if case["operation"] == "fftRoundtrip":
        return backend in {"stackprocessing", "cpp-itk"} and case["pixelType"] == "Float32"
    if backend == "stackprocessing-zarr":
        return case["pixelType"] in {"UInt8", "UInt16", "Float32"} and case["operation"] == "median"
    if backend == "stackprocessing-zarr-direct":
        return case["pixelType"] in {"UInt8", "UInt16", "Float32"} and case["operation"] in {"copy", "threshold"}
    if backend == "stackprocessing-libtiff-direct-copy":
        return case["pixelType"] in {"UInt8", "UInt16", "Float32"} and case["operation"] == "copy"
    if backend == "stackprocessing-libtiff-direct-threshold":
        return case["pixelType"] in {"UInt8", "UInt16", "Float32"} and case["operation"] == "threshold"
    if backend == "stackprocessing-libtiff-direct-threshold-intype":
        return case["pixelType"] in {"UInt8", "UInt16", "Float32"} and case["operation"] == "threshold"
    if backend == "stackprocessing-libtiff-strip-copy":
        return case["pixelType"] in {"UInt8", "UInt16", "Float32"} and case["operation"] == "copy"
    if backend == "stackprocessing-libtiff-raw-strip-copy":
        return case["pixelType"] in {"UInt8", "UInt16", "Float32"} and case["operation"] == "copy"
    if backend == "stackprocessing-native-libtiff-raw-strip-copy":
        return case["pixelType"] in {"UInt8", "UInt16", "Float32"} and case["operation"] == "copy"
    if backend == "stackprocessing-tifflibrary-raw-strip-copy":
        return case["pixelType"] in {"UInt8", "UInt16", "Float32"} and case["operation"] == "copy"
    if backend == "stackprocessing-imagesharp-copy":
        return case["pixelType"] in {"UInt8", "UInt16"} and case["operation"] == "copy"
    if backend == "python-dask-omezarr":
        return case["operation"] != "connectedComponents"
    return True


def input_dir(args, case):
    return Path(args.input_root) / f"{case['pixelType']}_{case['shape']}"


def output_dir(args, case, repeat):
    parameter = effective_case_parameter(args.backend, case).replace("=", "-")
    return Path(args.output_root) / args.backend / f"{case['operation']}_{case['pixelType']}_{case['shape']}_{parameter}_r{repeat:02d}"


def case_parameter(case):
    name = case.get("parameterName", "")
    value = case.get("parameterValue", "")
    if not name:
        return "none"
    return f"{name}={value}"


def effective_case_parameter(backend, case):
    if case["operation"] == "connectedComponents" and backend != "stackprocessing":
        return "none"
    return case_parameter(case)


def parameter_args(case, backend):
    if case["operation"] == "connectedComponents" and backend != "stackprocessing":
        return []

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
    params = parameter_args(case, args.backend)

    if args.backend == "stackprocessing":
        if case["operation"] == "fftRoundtrip":
            return [
                "dotnet",
                args.stackprocessing_dll,
                "run-chunk-fft3d-zarr-roundtrip-io",
                "--shape",
                case["shape"],
                "--input",
                inp,
                "--output",
                out,
            ]
        if case["operation"] == "copy":
            return [
                "dotnet",
                args.stackprocessing_dll,
                "run-stack-read-write",
                "--pixel-type",
                case["pixelType"],
                "--input",
                inp,
                "--output",
                out,
            ]
        return ["dotnet", args.stackprocessing_dll, "run"] + common + params
    if args.backend == "stackprocessing-arraypool":
        return ["dotnet", args.stackprocessing_dll, "run-arraypool"] + common + params
    if args.backend == "stackprocessing-arraypool-slice":
        return ["dotnet", args.stackprocessing_dll, "run-arraypool-slice"] + common + params
    if args.backend == "stackprocessing-arraypool-slice-reuse":
        return ["dotnet", args.stackprocessing_dll, "run-arraypool-slice-reuse"] + common + params
    if args.backend == "stackprocessing-byte-slice-reuse":
        return ["dotnet", args.stackprocessing_dll, "run-byte-slice-reuse"] + common + params
    if args.backend == "stackprocessing-byte-float32-slice-reuse":
        return ["dotnet", args.stackprocessing_dll, "run-byte-float32-slice-reuse"] + common + params
    if args.backend == "stackprocessing-libtiff-direct-copy":
        return ["dotnet", args.stackprocessing_dll, "run-libtiff-direct-copy"] + common
    if args.backend == "stackprocessing-libtiff-direct-threshold":
        return ["dotnet", args.stackprocessing_dll, "run-libtiff-direct-threshold"] + common + params
    if args.backend == "stackprocessing-libtiff-direct-threshold-intype":
        return ["dotnet", args.stackprocessing_dll, "run-libtiff-direct-threshold-intype"] + common + params
    if args.backend == "stackprocessing-libtiff-strip-copy":
        return ["dotnet", args.stackprocessing_dll, "run-libtiff-strip-copy"] + common
    if args.backend == "stackprocessing-libtiff-raw-strip-copy":
        return ["dotnet", args.stackprocessing_dll, "run-libtiff-raw-strip-copy"] + common
    if args.backend == "stackprocessing-native-libtiff-raw-strip-copy":
        return ["dotnet", args.stackprocessing_dll, "run-native-libtiff-raw-strip-copy"] + common
    if args.backend == "stackprocessing-tifflibrary-raw-strip-copy":
        return ["dotnet", args.stackprocessing_dll, "run-tifflibrary-raw-strip-copy"] + common
    if args.backend == "stackprocessing-imagesharp-copy":
        return ["dotnet", args.stackprocessing_dll, "run-imagesharp-copy"] + common
    if args.backend == "stackprocessing-zarr":
        return ["dotnet", args.stackprocessing_dll, "run-zarr"] + common + ["--shape", case["shape"]] + params
    if args.backend == "stackprocessing-zarr-direct":
        command = "run-zarr-direct-copy" if case["operation"] == "copy" else "run-zarr-direct-threshold"
        return ["dotnet", args.stackprocessing_dll, command] + common + ["--shape", case["shape"]] + params
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
        if case.get("parameterName") == "window" and args.backend == "stackprocessing":
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
        effective_case_parameter(args.backend, case),
        "--repeat-index",
        str(repeat),
        "--",
    ] + command


def successful_result_keys(path):
    result_path = Path(path)
    if not result_path.exists():
        return set()

    keys = set()
    with result_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("exitCode") != "0":
                continue
            keys.add(
                (
                    row["backend"],
                    row["operation"],
                    row["pixelType"],
                    row["shape"],
                    row["parameter"],
                    row["repeat"],
                )
            )
    return keys


def result_key(args, case, repeat):
    return (
        args.backend,
        case["operation"],
        case["pixelType"],
        case["shape"],
        effective_case_parameter(args.backend, case),
        str(repeat),
    )


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


def cleanup_case_output(args, case, repeat):
    if args.keep_outputs:
        return

    out = output_dir(args, case, repeat)
    for path in [out, Path(str(out) + "-labels")]:
        if path.exists():
            shutil.rmtree(path)


def main():
    args = parse_args()
    repeat_start = args.repeat_start if args.repeat_start is not None else 1
    repeat_end = args.repeat_end if args.repeat_end is not None else args.repeat
    if repeat_start < 1 or repeat_end < repeat_start:
        print("run_manifest.py: repeat range must satisfy 1 <= --repeat-start <= --repeat-end", file=sys.stderr)
        return 2

    cases = [
        case
        for case in filter_cases(read_cases(args.cases), args.pixel_types, args.shapes, args.operations, args.parameters)
        if backend_supports_case(args.backend, case)
    ]
    if not cases:
        print(
            "run_manifest.py: no cases matched "
            f"--pixel-types {args.pixel_types!r} --shapes {args.shapes!r} "
            f"--operations {args.operations!r} --parameters {args.parameters!r}",
            file=sys.stderr,
        )
        return 2

    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    existing_successes = successful_result_keys(args.results) if args.skip_existing else set()
    for repeat in range(repeat_start, repeat_end + 1):
        for case in cases:
            if result_key(args, case, repeat) in existing_successes:
                print(
                    "skip existing "
                    f"{args.backend} {case['operation']} {case['pixelType']} {case['shape']} "
                    f"{case_parameter(case)} r{repeat:02d}",
                    flush=True,
                )
                continue

            command = measured_command(args, case, repeat)
            printable = " ".join(shlex.quote(part) for part in command)
            print(printable, flush=True)
            if not args.dry_run:
                return_code = run_measured_command(command)
                cleanup_case_output(args, case, repeat)
                if return_code != 0:
                    return return_code
    return 0


if __name__ == "__main__":
    sys.exit(main())
