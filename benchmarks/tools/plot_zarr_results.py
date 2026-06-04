#!/usr/bin/env python3
"""Create OME-Zarr benchmark comparison figures."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path


BACKENDS = ["stackprocessing-zarr", "python-dask-omezarr"]
BACKEND_LABELS = {
    "stackprocessing-zarr": "StackProcessing-Zarr",
    "python-dask-omezarr": "Python/Dask-Zarr",
}
PIXEL_TYPES = ["UInt8", "UInt16", "Float32"]
PIXEL_LABELS = {"UInt8": "uint8", "UInt16": "uint16", "Float32": "float32"}
COLORS = {
    ("stackprocessing-zarr", "UInt8"): "#1f77b4",
    ("stackprocessing-zarr", "UInt16"): "#0b4f8a",
    ("python-dask-omezarr", "UInt8"): "#e15759",
    ("python-dask-omezarr", "UInt16"): "#b22222",
    ("python-dask-omezarr", "Float32"): "#7f1d1d",
}
MARKERS = {
    "UInt8": "o",
    "UInt16": "s",
    "Float32": "^",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="benchmarks/results/summary.csv")
    parser.add_argument("--output-dir", default="benchmarks/results/figures")
    parser.add_argument("--latex-dir", default="")
    return parser.parse_args()


def parse_shape(shape: str) -> tuple[int, int, int]:
    parts = shape.split("x")
    if len(parts) != 3:
        raise ValueError(f"expected WxHxD shape, got {shape!r}")
    return tuple(int(part) for part in parts)


def shape_voxels(shape: str) -> int:
    width, height, depth = parse_shape(shape)
    return width * height * depth


def shape_label(shape: str) -> str:
    width, height, depth = parse_shape(shape)
    if width == height == depth:
        return f"{width}^3"
    return shape


def finite_float(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["backend"] not in BACKENDS:
                continue
            if row["operation"] not in {"threshold", "median"}:
                continue

            internal = finite_float(row.get("medianInternalSeconds", ""))
            peak = finite_float(row.get("medianPeakRssMiB", ""))
            successes = int(row.get("successes") or 0)
            if successes <= 0 or internal is None or peak is None:
                continue

            enriched: dict[str, object] = dict(row)
            enriched["internal"] = internal
            enriched["peak_gib"] = peak / 1024.0
            enriched["voxels"] = shape_voxels(row["shape"])
            rows.append(enriched)
    return rows


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def panel_specs() -> list[tuple[str, str, str]]:
    return [
        ("threshold", "threshold=128", "threshold"),
        ("median", "radius=1", "median r=1"),
        ("median", "radius=2", "median r=2"),
        ("median", "radius=3", "median r=3"),
    ]


def metric_label(metric: str) -> str:
    if metric == "internal":
        return "internal seconds"
    if metric == "peak_gib":
        return "peak RSS (GiB)"
    raise ValueError(metric)


def figure_name(metric: str) -> str:
    if metric == "internal":
        return "zarr-runtime-by-size-and-parameter.pdf"
    if metric == "peak_gib":
        return "zarr-memory-by-size-and-parameter.pdf"
    raise ValueError(metric)


def plot_metric(rows: list[dict[str, object]], output_dir: Path, metric: str) -> Path:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.8), sharex=True)
    axes_flat = list(axes.flat)
    shapes = sorted({str(row["shape"]) for row in rows}, key=shape_voxels)
    ticks = [shape_voxels(shape) for shape in shapes]
    tick_labels = [shape_label(shape) for shape in shapes]

    handles_by_label = {}
    for ax, (operation, parameter, title) in zip(axes_flat, panel_specs()):
        panel_rows = [
            row
            for row in rows
            if row["operation"] == operation and row["parameter"] == parameter
        ]

        for backend in BACKENDS:
            for pixel_type in PIXEL_TYPES:
                series = [
                    row
                    for row in panel_rows
                    if row["backend"] == backend and row["pixelType"] == pixel_type
                ]
                if not series:
                    continue
                series = sorted(series, key=lambda row: int(row["voxels"]))
                label = f"{BACKEND_LABELS[backend]} {PIXEL_LABELS[pixel_type]}"
                (line,) = ax.plot(
                    [int(row["voxels"]) for row in series],
                    [float(row[metric]) for row in series],
                    marker=MARKERS[pixel_type],
                    linewidth=1.4,
                    markersize=4.0,
                    color=COLORS[(backend, pixel_type)],
                    label=label,
                )
                handles_by_label.setdefault(label, line)

        ax.set_title(title)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, rotation=20, ha="right")
        ax.grid(True, which="major", alpha=0.25)
        ax.grid(True, which="minor", axis="y", alpha=0.12)
        ax.set_ylabel(metric_label(metric))

    for ax in axes[-1, :]:
        ax.set_xlabel("input volume")

    labels = list(handles_by_label.keys())
    fig.legend(
        [handles_by_label[label] for label in labels],
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / figure_name(metric)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> int:
    args = parse_args()
    rows = load_rows(Path(args.summary))
    if not rows:
        raise SystemExit("No Zarr summary rows found.")

    output_dir = Path(args.output_dir)
    outputs = [
        plot_metric(rows, output_dir, "internal"),
        plot_metric(rows, output_dir, "peak_gib"),
    ]

    if args.latex_dir:
        latex_dir = Path(args.latex_dir)
        latex_dir.mkdir(parents=True, exist_ok=True)
        for output in outputs:
            shutil.copy2(output, latex_dir / output.name)

    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
