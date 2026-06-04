#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

import numpy as np
import tifffile
import zarr


def parse_shape(value):
    parts = value.lower().split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be WxHxD")
    width, height, depth = (int(part) for part in parts)
    return width, height, depth


def dtype_for(pixel_type):
    if pixel_type == "UInt8":
        return np.uint8
    if pixel_type == "UInt16":
        return np.uint16
    if pixel_type == "Float32":
        return np.float32
    raise ValueError(pixel_type)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a TIFF slice stack to minimal OME-Zarr.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shape", required=True, type=parse_shape)
    parser.add_argument("--pixel-type", required=True, choices=["UInt8", "UInt16", "Float32"])
    parser.add_argument("--chunk-shape", default="16x256x256", help="ZxYxX chunks; time and channel chunks are singleton")
    return parser.parse_args()


def tiff_files(input_dir):
    paths = sorted(Path(input_dir).glob("*.tif*"))
    if not paths:
        raise FileNotFoundError(f"no TIFF files found in {input_dir}")
    return paths


def parse_chunk_shape(value):
    parts = value.lower().split("x")
    if len(parts) != 3:
        raise ValueError("chunk shape must be ZxYxX")
    return tuple(int(part) for part in parts)


def write_ome_metadata(root):
    group = zarr.open_group(str(root), mode="a")
    group.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": [
                {"name": "t", "type": "time"},
                {"name": "c", "type": "channel"},
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
    width, height, depth = args.shape
    paths = tiff_files(args.input)
    if len(paths) != depth:
        raise RuntimeError(f"expected {depth} slices, found {len(paths)}")

    output_root = Path(args.output)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    array_path = output_root / "0"
    z_chunk, y_chunk, x_chunk = parse_chunk_shape(args.chunk_shape)
    arr = zarr.open(
        str(array_path),
        mode="w",
        shape=(1, 1, depth, height, width),
        chunks=(1, 1, z_chunk, y_chunk, x_chunk),
        dtype=dtype_for(args.pixel_type),
    )

    for z, path in enumerate(paths):
        img = tifffile.imread(path)
        arr[0, 0, z, :, :] = img.astype(dtype_for(args.pixel_type), copy=False)

    write_ome_metadata(output_root)


if __name__ == "__main__":
    main()
