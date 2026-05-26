#!/usr/bin/env python3
import argparse
import os
import shutil
import time
from pathlib import Path

import dask.array as da
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Dask/OME-Zarr chunk-native benchmark backend.")
    parser.add_argument("--operation", required=True, choices=["copy", "threshold", "smoothWGauss", "median", "dilate"])
    parser.add_argument("--pixel-type", required=True, choices=["UInt8", "UInt16", "Float32"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--threshold", type=float, default=128.0)
    return parser.parse_args()


def array_path(root):
    return Path(root) / "0"


def process(arr, args):
    if args.operation == "copy":
        return arr
    if args.operation == "threshold":
        return (arr > args.threshold).astype(np.uint8) * np.uint8(255)

    try:
        import scipy.ndimage as ndi
    except ImportError as exc:
        raise RuntimeError("smoothWGauss/median/dilate require scipy for this Dask backend") from exc

    radius = max(1, args.radius)
    if args.operation == "smoothWGauss":
        halo = max(1, int(4.0 * args.sigma + 0.5))
        return arr.map_overlap(
            ndi.gaussian_filter,
            depth={0: halo, 1: halo, 2: halo},
            boundary="reflect",
            dtype=arr.dtype,
            sigma=(args.sigma, args.sigma, args.sigma),
        )

    depth = {0: radius, 1: radius, 2: radius}
    if args.operation == "median":
        size = (2 * radius + 1, 2 * radius + 1, 2 * radius + 1)
        return arr.map_overlap(ndi.median_filter, depth=depth, boundary="reflect", dtype=arr.dtype, size=size)
    if args.operation == "dilate":
        size = (2 * radius + 1, 2 * radius + 1, 2 * radius + 1)
        return arr.map_overlap(ndi.grey_dilation, depth=depth, boundary="reflect", dtype=arr.dtype, size=size)
    raise ValueError(args.operation)


def write_ome_metadata(root, arr):
    try:
        import zarr
    except ImportError as exc:
        raise RuntimeError("OME-Zarr metadata writing requires zarr") from exc

    group = zarr.open_group(str(root), mode="a")
    group.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": [
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": [{"path": "0"}],
        }
    ]
    group.attrs["omero"] = {"channels": [{"label": "0"}]}


def main():
    args = parse_args()
    start = time.perf_counter()
    input_array = array_path(args.input)
    output_root = Path(args.output)
    output_array = array_path(output_root)

    if output_root.exists():
        shutil.rmtree(output_root)
    output_array.parent.mkdir(parents=True, exist_ok=True)

    arr = da.from_zarr(str(input_array))
    out = process(arr, args)
    out.to_zarr(str(output_array), overwrite=True)
    write_ome_metadata(output_root, out)
    write_internal_seconds(time.perf_counter() - start)


def write_internal_seconds(seconds):
    path = os.environ.get("BENCHMARK_INTERNAL_SECONDS_PATH")
    if path:
        Path(path).write_text(f"{seconds:.9f}")


if __name__ == "__main__":
    main()
