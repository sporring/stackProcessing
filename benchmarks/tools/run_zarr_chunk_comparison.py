#!/usr/bin/env python3
"""Run the focused StackProcessing-vs-Dask Zarr chunk-size comparison."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--input-zarr-root", required=True)
    parser.add_argument("--input-tiff-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--stackprocessing-dll", default=str(ROOT / "benchmarks/StackProcessing.Benchmarks/bin/Release/net10.0/StackProcessing.Benchmarks.dll"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--shape", default="1024x1024x1024")
    parser.add_argument("--pixel-type", default="UInt8", choices=["UInt8"])
    parser.add_argument("--chunk-sizes", default="64,128,256")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-inputs", action="store_true")
    return parser.parse_args()


def run(command: list[str], dry_run: bool) -> None:
    print("+ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def chunk_sizes(value: str) -> list[int]:
    sizes = [int(item) for item in value.split(",") if item.strip()]
    if not sizes:
        raise ValueError("at least one chunk size is required")
    return sizes


def tiff_input_dir(root: Path, pixel_type: str, shape: str) -> Path:
    return root / f"{pixel_type}_{shape}"


def zarr_input_dir(root: Path, pixel_type: str, shape: str, chunk_size: int) -> Path:
    return root / f"{pixel_type}_{shape}_chunk{chunk_size}"


def ensure_tiff_input(args, dry_run: bool) -> Path:
    path = tiff_input_dir(Path(args.input_tiff_root), args.pixel_type, args.shape)
    if path.exists() and any(path.glob("*.tif*")) and not args.force_inputs:
        return path

    if path.exists() and args.force_inputs and not dry_run:
        shutil.rmtree(path)

    run(
        [
            "dotnet",
            args.stackprocessing_dll,
            "generate",
            "--output",
            str(path),
            "--shape",
            args.shape,
            "--pixel-type",
            args.pixel_type,
        ],
        dry_run,
    )
    return path


def ensure_zarr_inputs(args, sizes: list[int], dry_run: bool) -> None:
    for size in sizes:
        path = zarr_input_dir(Path(args.input_zarr_root), args.pixel_type, args.shape, size)
        if path.exists() and not args.force_inputs:
            continue
        run(
            [
                args.python,
                "benchmarks/tools/generate_omezarr_ramp.py",
                "--output",
                str(path),
                "--shape",
                args.shape,
                "--pixel-type",
                args.pixel_type,
                "--chunk-shape",
                f"{size}x{size}x{size}",
                "--codec",
                "none",
                "--force",
            ],
            dry_run,
        )


def measured(args, backend: str, operation: str, chunk_size: int, repeat: int, input_path: Path, output_path: Path, kernel_size: int | None) -> list[str]:
    parameter_parts = [f"chunkSize={chunk_size}"]
    if kernel_size is not None:
        parameter_parts.append(f"kernelSize={kernel_size}")

    command = [
        args.python,
        "benchmarks/tools/measure.py",
        "--output",
        args.results,
        "--backend",
        backend,
        "--operation",
        operation,
        "--pixel-type",
        args.pixel_type,
        "--shape",
        args.shape,
        "--parameter",
        ";".join(parameter_parts),
        "--repeat-index",
        str(repeat),
        "--",
    ]

    if backend == "stackprocessing-zarr":
        command += [
            "dotnet",
            args.stackprocessing_dll,
            "run-zarr",
            "--operation",
            operation,
            "--pixel-type",
            args.pixel_type,
            "--shape",
            args.shape,
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--chunk-size",
            str(chunk_size),
            "--workers",
            str(args.workers),
        ]
        if kernel_size is not None:
            command += ["--kernel-size", str(kernel_size)]
    elif backend == "python-dask-skimage-zarr":
        command += [
            args.python,
            "benchmarks/python-dask-omezarr/bench.py",
            "--operation",
            operation,
            "--pixel-type",
            args.pixel_type,
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--chunk-size",
            str(chunk_size),
        ]
        if kernel_size is not None:
            command += ["--kernel-size", str(kernel_size)]
    else:
        raise ValueError(backend)

    return command


def main() -> int:
    args = parse_args()
    sizes = chunk_sizes(args.chunk_sizes)
    results = Path(args.results)
    output_root = Path(args.output_root)

    if not args.dry_run:
        results.parent.mkdir(parents=True, exist_ok=True)
        results.unlink(missing_ok=True)
        output_root.mkdir(parents=True, exist_ok=True)

    tiff_input = ensure_tiff_input(args, args.dry_run)
    ensure_zarr_inputs(args, sizes, args.dry_run)

    backends = ["stackprocessing-zarr", "python-dask-skimage-zarr"]
    simple_ops = ["copy", "zarrToTiff", "tiffToZarr", "threshold"]
    convolve_kernels = [3, 5, 7]

    for repeat in range(1, args.repeat + 1):
        for chunk_size in sizes:
            zarr_input = zarr_input_dir(Path(args.input_zarr_root), args.pixel_type, args.shape, chunk_size)
            for backend in backends:
                for operation in simple_ops:
                    input_path = tiff_input if operation == "tiffToZarr" else zarr_input
                    output_path = output_root / backend / f"{operation}_{args.pixel_type}_{args.shape}_chunk{chunk_size}_r{repeat:02d}"
                    run(measured(args, backend, operation, chunk_size, repeat, input_path, output_path, None), args.dry_run)
                    if not args.dry_run and output_path.exists():
                        shutil.rmtree(output_path)

                for kernel_size in convolve_kernels:
                    output_path = output_root / backend / f"convolve_{args.pixel_type}_{args.shape}_chunk{chunk_size}_k{kernel_size}_r{repeat:02d}"
                    run(measured(args, backend, "convolve", chunk_size, repeat, zarr_input, output_path, kernel_size), args.dry_run)
                    if not args.dry_run and output_path.exists():
                        shutil.rmtree(output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
