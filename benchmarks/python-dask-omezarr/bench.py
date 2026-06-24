#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import time
from pathlib import Path

import dask.array as da
import dask.delayed
import numpy as np
import tifffile


def parse_args():
    parser = argparse.ArgumentParser(description="Dask/OME-Zarr chunk-native benchmark backend.")
    parser.add_argument("--operation", required=True, choices=["copy", "zarrToTiff", "tiffToZarr", "tiffRechunkDrain", "threshold", "convolve", "median", "dilate"])
    parser.add_argument("--pixel-type", required=True, choices=["UInt8", "UInt16", "Float32"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=128.0)
    parser.add_argument("--chunk-size", type=int, default=128)
    return parser.parse_args()


def array_path(root):
    return Path(root) / "0"


def input_codecs(root):
    metadata = array_path(root) / "zarr.json"
    if not metadata.exists():
        return None
    with metadata.open() as handle:
        return json.load(handle).get("codecs")


def input_is_uncompressed(root):
    codecs = input_codecs(root)
    return bool(codecs) and all(codec.get("name") == "bytes" for codec in codecs)


def image_volume(arr):
    if arr.ndim == 5:
        return arr[0, 0, :, :, :]
    if arr.ndim == 3:
        return arr
    raise RuntimeError(f"expected a 3D or singleton 5D OME-Zarr array, got shape {arr.shape}")


def process(arr, args):
    if args.operation == "copy":
        return arr
    if args.operation == "threshold":
        return (arr >= args.threshold).astype(np.uint8)

    try:
        import scipy.ndimage as ndi
    except ImportError as exc:
        raise RuntimeError("convolve/median/dilate require scipy for this Dask backend") from exc

    radius = max(1, args.radius)
    if args.operation == "convolve":
        kernel_size = max(1, args.kernel_size)
        halo = kernel_size // 2
        kernel = np.full((kernel_size, kernel_size, kernel_size), 1.0 / float(kernel_size ** 3), dtype=np.float32)
        return arr.map_overlap(
            ndi.convolve,
            depth={0: halo, 1: halo, 2: halo},
            boundary=0,
            dtype=np.float32,
            weights=kernel,
            mode="constant",
            cval=0.0,
        ).astype(arr.dtype)

    depth = {0: radius, 1: radius, 2: radius}
    if args.operation == "median":
        size = (2 * radius + 1, 2 * radius + 1, 2 * radius + 1)
        return arr.map_overlap(ndi.median_filter, depth=depth, boundary="reflect", dtype=arr.dtype, size=size)
    if args.operation == "dilate":
        mask = arr >= 128
        footprint = ndi.generate_binary_structure(3, 1)
        try:
            from skimage import morphology

            footprint = morphology.ball(radius)
        except ImportError:
            grid = np.ogrid[-radius : radius + 1, -radius : radius + 1, -radius : radius + 1]
            footprint = sum(axis * axis for axis in grid) <= radius * radius
        return (
            mask.map_overlap(ndi.binary_dilation, depth=depth, boundary="none", dtype=bool, structure=footprint)
            .astype(np.uint8)
        )
    raise ValueError(args.operation)


def write_ome_metadata(root):
    try:
        import zarr
    except ImportError as exc:
        raise RuntimeError("OME-Zarr metadata writing requires zarr") from exc

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


def dtype_for_pixel_type(pixel_type):
    if pixel_type == "UInt8":
        return np.uint8
    if pixel_type == "UInt16":
        return np.uint16
    if pixel_type == "Float32":
        return np.float32
    raise ValueError(pixel_type)


def write_zarr_to_tiff_slices(arr, output_root, thick_depth):
    thick_depth = max(1, int(thick_depth))
    depth = int(arr.shape[0])
    for z0 in range(0, depth, thick_depth):
        thick = arr[z0 : min(depth, z0 + thick_depth), :, :].compute()
        for local_z in range(thick.shape[0]):
            z = z0 + local_z
            tifffile.imwrite(output_root / f"image_{z:03d}.tiff", thick[local_z, :, :], compression=None)


def read_tiff_slices_as_dask(input_root, pixel_type, thick_depth):
    paths = sorted(Path(input_root).glob("*.tif*"))
    if not paths:
        raise FileNotFoundError(f"no TIFF files found in {input_root}")

    first = tifffile.imread(paths[0])
    if first.ndim == 3:
        first_shape = first.shape
    elif first.ndim == 2:
        first_shape = (1, first.shape[0], first.shape[1])
    else:
        raise RuntimeError(f"expected 2D or 3D TIFF payloads, got shape {first.shape}")

    dtype = dtype_for_pixel_type(pixel_type)

    def read_thick(group):
        arrays = [tifffile.imread(path) for path in group]
        normalized = [array if array.ndim == 3 else array[None, :, :] for array in arrays]
        return np.concatenate(normalized, axis=0)

    thick_depth = max(1, int(thick_depth))
    groups = [paths[i : i + thick_depth] for i in range(0, len(paths), thick_depth)]
    thick_chunks = []
    for group in groups:
        delayed = dask.delayed(read_thick)(group)
        shape = (len(group) * first_shape[0], first_shape[1], first_shape[2])
        thick_chunks.append(da.from_delayed(delayed, shape=shape, dtype=dtype))
    return da.concatenate(thick_chunks, axis=0)


@dask.delayed
def consume_rechunked_block(block):
    arr = np.asarray(block)
    return int(arr.size)


def main():
    args = parse_args()
    output_root = Path(args.output)

    precleaned = {
        Path(path).resolve()
        for path in os.environ.get("BENCHMARK_PRECLEANED_OUTPUTS", "").split(os.pathsep)
        if path
    }
    if output_root.exists() and output_root.resolve() not in precleaned:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    if args.operation in {"tiffToZarr", "tiffRechunkDrain"}:
        out = read_tiff_slices_as_dask(args.input, args.pixel_type, args.chunk_size)
    else:
        arr = da.from_zarr(str(array_path(args.input)))
        volume = image_volume(arr)
        if args.operation == "zarrToTiff":
            write_zarr_to_tiff_slices(volume, output_root, args.chunk_size)
            write_internal_seconds(time.perf_counter() - start)
            return
        out = process(volume, args)

    output_array = array_path(output_root)
    out5 = out[None, None, :, :, :]
    chunk_size = max(1, args.chunk_size)
    out5 = out5.rechunk((1, 1, chunk_size, chunk_size, chunk_size))
    if args.operation == "tiffRechunkDrain":
        dask.compute(*[consume_rechunked_block(block) for block in out5.to_delayed().ravel()])
        write_internal_seconds(time.perf_counter() - start)
        return

    zarr_kwargs = {"compressors": []} if args.operation == "tiffToZarr" or input_is_uncompressed(args.input) else {}
    out5.to_zarr(str(output_array), overwrite=True, **zarr_kwargs)
    write_ome_metadata(output_root)
    write_internal_seconds(time.perf_counter() - start)


def write_internal_seconds(seconds):
    path = os.environ.get("BENCHMARK_INTERNAL_SECONDS_PATH")
    if path:
        Path(path).write_text(f"{seconds:.9f}")


if __name__ == "__main__":
    main()
