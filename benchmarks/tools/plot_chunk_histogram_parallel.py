#!/usr/bin/env python3
"""Plot median chunk histogram parallelisation benchmark results."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="tmp/benchmarks/chunk-histogram-parallel-raw.csv")
    parser.add_argument("--summary", default="tmp/benchmarks/chunk-histogram-parallel-summary.csv")
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
    size = shape_size(shape)
    return f"{size}^3"


def window_size(parameter: str) -> int:
    if "-window" in parameter:
        return int(parameter.rsplit("window", 1)[1])
    return int(parameter.rsplit("w", 1)[1])


def variant_name(operation: str) -> str:
    return operation.removeprefix("histogram-")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["exitCode"] != "0":
            continue
        key = (
            row["pixelType"],
            row["shape"],
            variant_name(row["operation"]),
            window_size(row["parameter"]),
        )
        grouped[key].append(row)

    summary = []
    for (pixel_type, shape, variant, window_size_value), group in sorted(grouped.items(), key=lambda item: (shape_size(item[0][1]), item[0][2], item[0][3])):
        internal = [float(row["internalSeconds"]) for row in group]
        peak_mib = [float(row["peakRssKiB"]) / 1024.0 for row in group]
        wall = [float(row["wallSeconds"]) for row in group]
        summary.append(
            {
                "pixelType": pixel_type,
                "shape": shape,
                "variant": variant,
                "windowSize": window_size_value,
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
        "variant",
        "windowSize",
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
    variants = ["dense", "sparse"]
    shapes = sorted({str(row["shape"]) for row in summary}, key=shape_size)
    colors = {shape: color for shape, color in zip(shapes, ["#1f77b4", "#ff7f0e", "#2ca02c"])}
    markers = {"dense": "o", "sparse": "s"}

    for metric, ylabel, output_name in [
        ("medianInternalSeconds", "median internal seconds", "chunk-histogram-parallel-runtime.pdf"),
        ("medianPeakRssMiB", "median peak RSS (MiB)", "chunk-histogram-parallel-memory.pdf"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.1), sharey=True)
        for ax, variant in zip(axes, variants):
            for shape in shapes:
                points = [
                    row
                    for row in summary
                    if row["variant"] == variant and row["shape"] == shape
                ]
                points.sort(key=lambda row: int(row["windowSize"]))
                ax.plot(
                    [int(row["windowSize"]) for row in points],
                    [float(row[metric]) for row in points],
                    marker=markers[variant],
                    color=colors[shape],
                    linewidth=1.4,
                    label=shape_label(shape),
                )
            ax.set_title(f"{variant} histogram")
            ax.set_xlabel("window size / workers")
            ax.set_xticks([1, 2, 3, 4])
            ax.grid(True, alpha=0.25)
        axes[0].set_ylabel(ylabel)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(shapes), frameon=False)
        fig.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.78, wspace=0.20)
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
