#!/usr/bin/env python3
"""Generate deterministic TIFF stack inputs for all benchmark backends."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile


def parse_shape(text: str) -> tuple[int, int, int]:
    parts = text.lower().split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be WxHxD")
    return tuple(int(part) for part in parts)  # type: ignore[return-value]


def dtype_for(pixel_type: str):
    key = pixel_type.lower()
    if key == "uint8":
        return np.uint8
    if key == "uint16":
        return np.uint16
    if key == "float32":
        return np.float32
    raise argparse.ArgumentTypeError(f"unsupported pixel type {pixel_type}")


def make_slice(width: int, height: int, z: int, depth: int, dtype, pattern: str):
    yy, xx = np.mgrid[0:height, 0:width]
    if pattern == "binary":
        values = ((xx + yy + z) % 17 < 8).astype(np.uint8) * 255
    else:
        values = (xx * 13 + yy * 7 + z * 31) % 65536

    if dtype == np.uint8:
        return (values % 256).astype(np.uint8)
    if dtype == np.uint16:
        return values.astype(np.uint16)
    if dtype == np.float32:
        return (values.astype(np.float32) / np.float32(65535.0))
    raise ValueError(dtype)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--shape", required=True, type=parse_shape)
    parser.add_argument("--pixel-type", required=True)
    parser.add_argument("--pattern", choices=["ramp", "binary"], default="ramp")
    args = parser.parse_args()

    width, height, depth = args.shape
    dtype = dtype_for(args.pixel_type)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    for z in range(depth):
        image = make_slice(width, height, z, depth, dtype, args.pattern)
        path = output / f"slice_{z:05d}.tiff"
        tifffile.imwrite(path, image, compression=None)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
