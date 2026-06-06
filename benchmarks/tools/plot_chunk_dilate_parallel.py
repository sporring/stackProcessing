#!/usr/bin/env python3
"""Plot median chunk dilation parallelisation benchmark results."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="tmp/benchmarks/chunk-dilate-radius3-parallel-raw.csv")
    parser.add_argument("--summary", default="tmp/benchmarks/chunk-dilate-radius3-parallel-summary.csv")
    parser.add_argument("--output-dir", default="notes/LMIP_Optimiser_and_Studio/figures")
    return parser.parse_args()


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def shape_size(shape: str) -> int:
    return int(shape.split("x", 1)[0])


def shape_label(shape: str) -> str:
    return f"{shape_size(shape)}^3"


def parse_parameter(parameter: str) -> tuple[int, int]:
    match = re.fullmatch(r"radius(\d+)-workers(\d+)", parameter)
    if match is None:
        raise ValueError(f"Unexpected dilation parameter: {parameter}")
    return int(match.group(1)), int(match.group(2))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["exitCode"] != "0":
            continue
        radius, workers = parse_parameter(row["parameter"])
        key = row["pixelType"], row["shape"], radius, workers
        grouped[key].append(row)

    summary = []
    for (pixel_type, shape, radius, workers), group in sorted(
        grouped.items(),
        key=lambda item: (item[0][2], shape_size(item[0][1]), item[0][3]),
    ):
        internal = [float(row["internalSeconds"]) for row in group]
        peak_mib = [float(row["peakRssKiB"]) / 1024.0 for row in group]
        wall = [float(row["wallSeconds"]) for row in group]
        summary.append(
            {
                "pixelType": pixel_type,
                "shape": shape,
                "radius": radius,
                "workers": workers,
                "n": len(group),
                "medianInternalSeconds": statistics.median(internal),
                "medianWallSeconds": statistics.median(wall),
                "medianPeakRssMiB": statistics.median(peak_mib),
            }
        )
    return summary


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pixelType",
        "shape",
        "radius",
        "workers",
        "n",
        "medianInternalSeconds",
        "medianWallSeconds",
        "medianPeakRssMiB",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot(summary: list[dict[str, object]], output_dir: Path) -> None:
    plt = setup_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    shapes = sorted({str(row["shape"]) for row in summary}, key=shape_size)
    colors = {shape: color for shape, color in zip(shapes, ["#1f77b4", "#ff7f0e", "#2ca02c"])}

    for metric, ylabel, output_name in [
        ("medianInternalSeconds", "median internal seconds", "chunk-dilate-radius3-parallel-runtime.pdf"),
        ("medianPeakRssMiB", "median peak RSS (MiB)", "chunk-dilate-radius3-parallel-memory.pdf"),
    ]:
        fig, ax = plt.subplots(figsize=(4.9, 3.1))
        for shape in shapes:
            points = [row for row in summary if row["shape"] == shape]
            points.sort(key=lambda row: int(row["workers"]))
            ax.plot(
                [int(row["workers"]) for row in points],
                [float(row[metric]) for row in points],
                marker="o",
                color=colors[shape],
                linewidth=1.4,
                label=shape_label(shape),
            )
        ax.set_xlabel("worker windows")
        ax.set_ylabel(ylabel)
        ax.set_xticks([1, 2, 3, 4])
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / output_name)
        plt.close(fig)


def main() -> int:
    args = parse_args()
    rows = read_rows(Path(args.input))
    summary = summarize(rows)
    write_summary(Path(args.summary), summary)
    plot(summary, Path(args.output_dir))
    print(f"wrote {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
