#!/usr/bin/env python3
"""Plot focused Zarr chunk-size comparison figures."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path


BACKENDS = ["stackprocessing-zarr", "python-dask-skimage-zarr"]
BACKEND_LABELS = {
    "stackprocessing-zarr": "StackProcessing",
    "python-dask-skimage-zarr": "Python/Dask/scikit-image",
}
COLORS = {
    "stackprocessing-zarr": "#1f77b4",
    "python-dask-skimage-zarr": "#e15759",
}
MARKERS = {
    "stackprocessing-zarr": "o",
    "python-dask-skimage-zarr": "s",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-dir", default="benchmarks/results/figures")
    parser.add_argument("--latex-dir", default="")
    parser.add_argument("--prefix", default="zarr-chunk-fast")
    parser.add_argument("--metrics", default="internal,peak", help="Comma-separated metrics: internal,peak")
    return parser.parse_args()


def finite_float(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_parameter(value: str) -> dict[str, str]:
    result = {}
    for item in value.split(";"):
        if "=" in item:
            key, val = item.split("=", 1)
            result[key] = val
    return result


def load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["backend"] not in BACKENDS:
                continue
            if row["pixelType"] != "UInt8" or row["shape"] != "1024x1024x1024":
                continue
            internal = finite_float(row.get("medianInternalSeconds", ""))
            peak = finite_float(row.get("medianPeakRssMiB", ""))
            successes = int(row.get("successes") or 0)
            if successes <= 0 or internal is None or peak is None:
                continue

            params = parse_parameter(row["parameter"])
            if "chunkSize" not in params:
                continue

            enriched: dict[str, object] = dict(row)
            enriched["internal"] = internal
            enriched["peak"] = peak
            enriched["chunk_size"] = int(params["chunkSize"])
            enriched["kernel_size"] = int(params["kernelSize"]) if "kernelSize" in params else None
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


def panel_specs():
    return [
        ("copy", None, "copy"),
        ("zarrToTiff", None, "Zarr -> TIFF slices"),
        ("tiffToZarr", None, "TIFF slices -> Zarr"),
        ("threshold", None, "threshold"),
        ("convolve", 3, "convolve k=3"),
        ("convolve", 5, "convolve k=5"),
        ("convolve", 7, "convolve k=7"),
        None,
    ]


def metric_label(metric: str) -> str:
    if metric == "internal":
        return "internal seconds"
    if metric == "peak":
        return "peak RSS (MiB)"
    raise ValueError(metric)


def plot_metric(rows: list[dict[str, object]], output_dir: Path, prefix: str, metric: str) -> Path:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(2, 4, figsize=(11.5, 5.8), sharex=True)
    ticks = [64, 128, 256]
    handles_by_label = {}

    for ax, spec in zip(axes.flat, panel_specs()):
        if spec is None:
            ax.axis("off")
            continue
        operation, kernel_size, title = spec
        panel_rows = [
            row
            for row in rows
            if row["operation"] == operation and (operation != "convolve" or row["kernel_size"] == kernel_size)
        ]

        for backend in BACKENDS:
            series = sorted(
                [row for row in panel_rows if row["backend"] == backend],
                key=lambda row: int(row["chunk_size"]),
            )
            if not series:
                continue
            label = BACKEND_LABELS[backend]
            (line,) = ax.plot(
                [int(row["chunk_size"]) for row in series],
                [float(row[metric]) for row in series],
                marker=MARKERS[backend],
                linewidth=1.4,
                markersize=4.0,
                color=COLORS[backend],
                label=label,
            )
            handles_by_label.setdefault(label, line)

        ax.set_title(title)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(tick) for tick in ticks])
        ax.set_yscale("log")
        ax.set_ylabel(metric_label(metric))
        ax.grid(True, which="major", alpha=0.25)
        ax.grid(True, which="minor", axis="y", alpha=0.12)

    for ax in axes[-1, :]:
        ax.set_xlabel("Zarr sub-volume edge")

    labels = list(handles_by_label.keys())
    fig.legend(
        [handles_by_label[label] for label in labels],
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=max(1, len(labels)),
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{prefix}-{metric}.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> int:
    args = parse_args()
    rows = load_rows(Path(args.summary))
    if not rows:
        raise SystemExit(f"no plottable rows in {args.summary}")

    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    output_dir = Path(args.output_dir)
    outputs = [plot_metric(rows, output_dir, args.prefix, metric) for metric in metrics]

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
