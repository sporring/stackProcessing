#!/usr/bin/env python3
"""Generate a deterministic scalar OME-Zarr ramp volume.

This is intended for chunk-layout experiments where the same logical image is
written with different Zarr chunk shapes.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import zarr


def parse_shape(value: str) -> tuple[int, int, int]:
    parts = value.lower().split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be WxHxD")
    width, height, depth = (int(part) for part in parts)
    return width, height, depth


def parse_chunk_shape(value: str) -> tuple[int, int, int]:
    parts = value.lower().split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("chunk shape must be ZxYxX")
    z, y, x = (int(part) for part in parts)
    return z, y, x


def dtype_for(pixel_type: str):
    if pixel_type == "UInt8":
        return np.uint8
    if pixel_type == "UInt16":
        return np.uint16
    if pixel_type == "Float32":
        return np.float32
    raise ValueError(pixel_type)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shape", required=True, type=parse_shape)
    parser.add_argument("--pixel-type", required=True, choices=["UInt8", "UInt16", "Float32"])
    parser.add_argument("--chunk-shape", required=True, type=parse_chunk_shape, help="ZxYxX chunks")
    parser.add_argument(
        "--codec",
        choices=["default", "none"],
        default="default",
        help="Zarr codec pipeline. 'none' writes only the required bytes codec.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_ome_metadata(root: Path):
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


def main() -> int:
    args = parse_args()
    width, height, depth = args.shape
    z_chunk, y_chunk, x_chunk = args.chunk_shape
    dtype = dtype_for(args.pixel_type)
    codecs = None
    if args.codec == "none":
        if dtype == np.uint8:
            codecs = [{"name": "bytes"}]
        else:
            codecs = [{"name": "bytes", "configuration": {"endian": "little"}}]

    output_root = Path(args.output)
    if output_root.exists():
        if not args.force:
            print(f"exists {output_root}")
            return 0
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    array = zarr.open(
        str(output_root / "0"),
        mode="w",
        shape=(1, 1, depth, height, width),
        chunks=(1, 1, z_chunk, y_chunk, x_chunk),
        dtype=dtype,
        codecs=codecs,
    )

    for z0 in range(0, depth, z_chunk):
        z1 = min(depth, z0 + z_chunk)
        zz = np.arange(z0, z1, dtype=np.uint32)[:, None, None]
        yy = np.arange(height, dtype=np.uint32)[None, :, None]
        xx = np.arange(width, dtype=np.uint32)[None, None, :]
        block = (xx + yy + zz) % 256
        array[0, 0, z0:z1, :, :] = block.astype(dtype, copy=False)

    write_ome_metadata(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
