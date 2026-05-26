#!/usr/bin/env python3
import argparse
import os
import time
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
from skimage import io, measure, morphology


def parse_args():
    parser = argparse.ArgumentParser(description="scikit-image/SciPy TIFF-stack benchmark backend.")
    parser.add_argument("--operation", required=True, choices=["copy", "threshold", "smoothWGauss", "median", "dilate", "connectedComponents"])
    parser.add_argument("--pixel-type", required=True, choices=["UInt8", "UInt16", "Float32"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--threshold", type=float, default=128.0)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--mode", choices=["3d", "slice"], default="3d")
    return parser.parse_args()


def read_stack(input_dir):
    paths = sorted(Path(input_dir).glob("*.tif*"))
    if not paths:
        raise FileNotFoundError(f"no TIFF files found in {input_dir}")
    images = [io.imread(path) for path in paths]
    return paths, np.stack(images, axis=0)


def write_stack(paths, output_dir, stack):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for old in output.glob("*.tif*"):
        old.unlink()
    for z, source_path in enumerate(paths):
        io.imsave(output / source_path.name, stack[z], check_contrast=False)


def per_slice(stack, func):
    return np.stack([func(stack[z]) for z in range(stack.shape[0])], axis=0)


def process(stack, args):
    if args.operation == "copy":
        return stack.copy()
    if args.operation == "threshold":
        return (stack > args.threshold).astype(np.uint8) * np.uint8(255)
    if args.operation == "smoothWGauss":
        sigma = (0.0, args.sigma, args.sigma) if args.mode == "slice" else args.sigma
        return ndi.gaussian_filter(stack, sigma=sigma).astype(stack.dtype, copy=False)
    if args.operation == "median":
        radius = max(1, args.radius)
        size = (1, 2 * radius + 1, 2 * radius + 1) if args.mode == "slice" else (2 * radius + 1,) * 3
        return ndi.median_filter(stack, size=size, mode="reflect")
    if args.operation == "dilate":
        radius = max(1, args.radius)
        if args.mode == "slice":
            footprint = morphology.square(2 * radius + 1)
            return per_slice(stack, lambda image: morphology.dilation(image, footprint=footprint))
        footprint = morphology.cube(2 * radius + 1)
        return morphology.dilation(stack, footprint=footprint)
    if args.operation == "connectedComponents":
        if args.mode == "slice":
            labels = per_slice(stack, lambda image: measure.label(image > 0, connectivity=1))
        else:
            labels = measure.label(stack > 0, connectivity=1)
        return np.mod(labels, 256).astype(np.uint8)
    raise ValueError(args.operation)


def main():
    args = parse_args()
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
