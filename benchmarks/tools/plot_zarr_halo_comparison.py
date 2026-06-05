#!/usr/bin/env python3
"""Plot the focused Zarr halo chunk-size comparison."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import statistics
from collections import defaultdict
from pathlib import Path


BACKEND_LABELS = {
    "stackprocessing-zarr-halo": "StackProcessing z-stream halo",
    "python-dask-omezarr-halo": "Python/Dask 3D overlap",
}
COLORS = {
    "stackprocessing-zarr-halo": "#1f77b4",
    "python-dask-omezarr-halo": "#e15759",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="tmp/zarr-halo-comparison.csv")
    parser.add_argument("--output-dir", default="benchmarks/results/figures")
    parser.add_argument("--latex-dir", default="")
    return parser.parse_args()


def chunk_side(parameter: str) -> int:
    key, value = parameter.split("=", 1)
    if key != "chunk":
        raise ValueError(parameter)
    return int(value)


def load_summary(path: Path):
    groups = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("exitCode") != "0":
                continue
            if row["backend"] not in BACKEND_LABELS:
                continue
            groups[(row["backend"], chunk_side(row["parameter"]))].append(row)

    summary = []
    for (backend, chunk), rows in sorted(groups.items()):
        internal = [float(row["internalSeconds"]) for row in rows if row.get("internalSeconds")]
        rss = [float(row["peakRssKiB"]) / 1024.0 for row in rows]
        if not internal:
            continue
        summary.append(
            {
                "backend": backend,
                "chunk": chunk,
                "seconds": statistics.median(internal),
                "rss_mib": statistics.median(rss),
                "n": len(rows),
            }
        )
    return summary


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


def plot_metric(rows, output_dir: Path, metric: str, ylabel: str, filename: str):
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    chunks = sorted({row["chunk"] for row in rows})

    for backend in BACKEND_LABELS:
        series = [row for row in rows if row["backend"] == backend]
        if not series:
            continue
        series = sorted(series, key=lambda row: row["chunk"])
        ax.plot(
            [row["chunk"] for row in series],
            [row[metric] for row in series],
            marker="o",
            linewidth=1.7,
            markersize=4.5,
            color=COLORS[backend],
            label=BACKEND_LABELS[backend],
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(chunks)
    ax.set_xticklabels([str(chunk) for chunk in chunks])
    ax.set_xlabel("Zarr chunk side length")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", axis="y", alpha=0.12)
    ax.legend(frameon=False)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / filename
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> int:
    args = parse_args()
    rows = load_summary(Path(args.input))
    if not rows:
        raise SystemExit("No successful halo-comparison rows found.")

    output_dir = Path(args.output_dir)
    outputs = [
        plot_metric(rows, output_dir, "seconds", "internal seconds", "zarr-halo-runtime-by-chunk-size.pdf"),
        plot_metric(rows, output_dir, "rss_mib", "peak RSS (MiB)", "zarr-halo-memory-by-chunk-size.pdf"),
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
