#!/usr/bin/env python3
import argparse
import os
import time
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import tifffile
from skimage import io, measure, morphology


def parse_args():
    parser = argparse.ArgumentParser(description="scikit-image/SciPy TIFF-stack benchmark backend.")
    parser.add_argument("--operation", required=True, choices=["copy", "threshold", "thresholdKernel", "thresholdKernelInType", "convolve", "median", "dilate", "connectedComponents"])
    parser.add_argument("--pixel-type", required=True, choices=["UInt8", "UInt16", "Float32"])
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--shape", default="256x256x256")
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=128.0)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--mode", choices=["3d", "slice"], default="3d")
    return parser.parse_args()


def parse_shape(value):
    parts = value.lower().split("x")
    if len(parts) != 3:
        raise ValueError(f"shape must be WxHxD, got {value!r}")
    return tuple(int(part) for part in parts)


def dtype_for_pixel_type(pixel_type):
    if pixel_type == "UInt8":
        return np.uint8
    if pixel_type == "UInt16":
        return np.uint16
    if pixel_type == "Float32":
        return np.float32
    raise ValueError(pixel_type)


def make_kernel_stack(args):
    width, height, depth = parse_shape(args.shape)
    dtype = dtype_for_pixel_type(args.pixel_type)
    values = np.arange(width * height * depth, dtype=np.uint64)
    if dtype == np.float32:
        values = np.mod(values, 256).astype(np.float32)
    else:
        values = values.astype(dtype, copy=False)
    return values.reshape((depth, height, width))


def run_threshold_kernel(args):
    stack = make_kernel_stack(args)
    start = time.perf_counter()
    if args.operation == "thresholdKernel":
        out = (stack >= args.threshold).astype(np.uint8)
    else:
        out = (stack >= args.threshold).astype(stack.dtype)
    elapsed = time.perf_counter() - start
    if out.size:
        checksum = int(out.reshape(-1)[0]) + int(out.reshape(-1)[out.size // 2]) + int(out.reshape(-1)[-1])
        if checksum == -1:
            print(checksum)
    write_internal_seconds(elapsed)


def read_stack(input_dir):
    paths = sorted(Path(input_dir).glob("*.tif*"))
    if not paths:
        raise FileNotFoundError(f"no TIFF files found in {input_dir}")
    images = [io.imread(path) for path in paths]
    return paths, np.stack(images, axis=0)


def write_stack(paths, output_dir, stack):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for z, source_path in enumerate(paths):
        output.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(output / source_path.name, stack[z], compression=None)


def prepare_output(output_dir):
    output = Path(output_dir)
    precleaned = {
        Path(path).resolve()
        for path in os.environ.get("BENCHMARK_PRECLEANED_OUTPUTS", "").split(os.pathsep)
        if path
    }
    if output.resolve() in precleaned:
        output.mkdir(parents=True, exist_ok=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    for old in output.glob("*.tif*"):
        old.unlink()


def per_slice(stack, func):
    return np.stack([func(stack[z]) for z in range(stack.shape[0])], axis=0)


def decomposed_ball(radius):
    try:
        return morphology.ball(radius, decomposition="sequence")
    except TypeError:
        return morphology.ball(radius)


def process(stack, args):
    if args.operation == "copy":
        return stack.copy()
    if args.operation == "threshold":
        return (stack >= args.threshold).astype(np.uint8)
    if args.operation == "convolve":
        kernel_size = max(1, args.kernel_size)
        kernel_shape = (1, kernel_size, kernel_size) if args.mode == "slice" else (kernel_size,) * 3
        kernel = np.full(kernel_shape, 1.0 / np.prod(kernel_shape), dtype=np.float32)
        convolved = ndi.convolve(stack, weights=kernel, output=np.float32, mode="constant", cval=0.0)
        return convolved.astype(stack.dtype, copy=False)
    if args.operation == "median":
        radius = max(1, args.radius)
        size = (1, 2 * radius + 1, 2 * radius + 1) if args.mode == "slice" else (2 * radius + 1,) * 3
        return ndi.median_filter(stack, size=size, mode="reflect")
    if args.operation == "dilate":
        radius = max(1, args.radius)
        mask = stack >= 128
        if args.mode == "slice":
            footprint = morphology.square(2 * radius + 1)
            return per_slice(mask, lambda image: morphology.binary_dilation(image, footprint=footprint)).astype(np.uint8)
        footprint = decomposed_ball(radius)
        return morphology.dilation(mask, footprint=footprint).astype(np.uint8)
    if args.operation == "connectedComponents":
        if args.mode == "slice":
            labels = per_slice(stack, lambda image: measure.label(image >= 128, connectivity=1))
        else:
            labels = measure.label(stack >= 128, connectivity=1)
        return np.mod(labels, 256).astype(np.uint8)
    raise ValueError(args.operation)


def main():
    args = parse_args()
    if args.operation in {"thresholdKernel", "thresholdKernelInType"}:
        run_threshold_kernel(args)
        return

    if not args.input or not args.output:
        raise ValueError("--input and --output are required unless using a kernel-only operation")

    prepare_output(args.output)
    start = time.perf_counter()
    paths, stack = read_stack(args.input)
    out = process(stack, args)
    write_stack(paths, args.output, out)
    write_internal_seconds(time.perf_counter() - start)


def write_internal_seconds(seconds):
    path = os.environ.get("BENCHMARK_INTERNAL_SECONDS_PATH")
    if path:
        Path(path).write_text(f"{seconds:.9f}")


if __name__ == "__main__":
    main()
