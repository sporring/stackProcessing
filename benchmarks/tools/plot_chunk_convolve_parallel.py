#!/usr/bin/env python3
"""Plot median chunk convolution parallelisation benchmark results."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="tmp/benchmarks/chunk-convolve-float32-1024-k7-parallel-raw.csv")
    parser.add_argument("--summary", default="tmp/benchmarks/chunk-convolve-float32-1024-k7-parallel-summary.csv")
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


def parse_parameter(parameter: str) -> tuple[int, int]:
    for pattern in [r"kernelSize=(\d+);workers=(\d+)", r"kernelSize(\d+)-workers(\d+)"]:
        match = re.fullmatch(pattern, parameter)
        if match is not None:
            return int(match.group(1)), int(match.group(2))
    raise ValueError(f"Unexpected convolution parameter: {parameter}")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["exitCode"] != "0":
            continue
        kernel_size, workers = parse_parameter(row["parameter"])
        key = row["pixelType"], row["shape"], row["operation"], kernel_size, workers
        grouped[key].append(row)

    summary = []
    for (pixel_type, shape, operation, kernel_size, workers), group in sorted(grouped.items(), key=lambda item: item[0]):
        internal = [float(row["internalSeconds"]) for row in group]
        peak_mib = [float(row["peakRssKiB"]) / 1024.0 for row in group]
        wall = [float(row["wallSeconds"]) for row in group]
        summary.append(
            {
                "pixelType": pixel_type,
                "shape": shape,
                "operation": operation,
                "kernelSize": kernel_size,
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
        "operation",
        "kernelSize",
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
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in summary:
        grouped[str(row["shape"])].append(row)

    for metric, ylabel, output_name in [
        ("medianInternalSeconds", "median internal seconds", "chunk-convolve-float32-1024-k7-parallel-runtime.pdf"),
        ("medianPeakRssMiB", "median peak RSS (MiB)", "chunk-convolve-float32-1024-k7-parallel-memory.pdf"),
    ]:
        fig, ax = plt.subplots(figsize=(4.9, 3.1))
        for shape, points in sorted(grouped.items(), key=lambda item: int(item[0].split("x")[0])):
            ordered = sorted(points, key=lambda row: int(row["workers"]))
            size = shape.split("x")[0]
            ax.plot(
                [int(row["workers"]) for row in ordered],
                [float(row[metric]) for row in ordered],
                marker="o",
                linewidth=1.4,
                label=f"{size}^3",
            )
        ax.set_xlabel("worker windows")
        ax.set_ylabel(ylabel)
        ax.set_xticks([1, 2, 3, 4])
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
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
