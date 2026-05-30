#!/usr/bin/env python3
"""Create publication-oriented PDF figures from benchmark summary rows."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


BACKEND_ORDER = ["stackprocessing", "cpp-itk", "python-skimage-scipy", "matlab"]
BACKEND_LABELS = {
    "stackprocessing": "StackProcessing",
    "cpp-itk": "C++ ITK",
    "python-skimage-scipy": "Python scipy/skimage",
    "matlab": "MATLAB",
}
BACKEND_COLORS = {
    "stackprocessing": "#1f77b4",
    "cpp-itk": "#2ca02c",
    "python-skimage-scipy": "#ff7f0e",
    "matlab": "#9467bd",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="benchmarks/results/summary.csv")
    parser.add_argument("--output-dir", default="benchmarks/results/figures")
    return parser.parse_args()


def finite_float(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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


def parameter_value(parameter: str) -> int | None:
    if "=" not in parameter:
        return None
    _, value = parameter.split("=", 1)
    try:
        return int(value)
    except ValueError:
        return None


def parameter_label(parameter: str) -> str:
    return "none" if parameter == "none" else parameter.replace("=", " ")


def load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            internal = finite_float(row.get("medianInternalSeconds", ""))
            wall = finite_float(row.get("medianWallSeconds", ""))
            startup = finite_float(row.get("medianStartupOverheadSeconds", ""))
            peak = finite_float(row.get("medianPeakRssMiB", ""))
            successes = int(row.get("successes") or 0)
            if successes <= 0 or internal is None or wall is None or peak is None:
                continue

            enriched: dict[str, object] = dict(row)
            enriched["internal"] = internal
            enriched["wall"] = wall
            enriched["startup"] = startup
            enriched["peak"] = peak
            enriched["voxels"] = shape_voxels(row["shape"])
            enriched["parameterValue"] = parameter_value(row["parameter"])
            rows.append(enriched)
    return rows


def backend_sequence(rows: list[dict[str, object]]) -> list[str]:
    present = {str(row["backend"]) for row in rows}
    ordered = [backend for backend in BACKEND_ORDER if backend in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def setup_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "plot_results.py requires matplotlib. Install it in the benchmark environment with "
            "`python3 -m pip install matplotlib`."
        ) from exc

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


def save(fig, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")


def plot_runtime_by_size(rows: list[dict[str, object]], output_dir: Path):
    plt = setup_matplotlib()
    representatives = [
        ("copy", "none"),
        ("threshold", "threshold=128"),
        ("uniformConvolve", "kernelSize=3"),
        ("uniformConvolve", "kernelSize=7"),
        ("median", "radius=1"),
        ("median", "radius=3"),
        ("dilate", "radius=1"),
        ("dilate", "radius=3"),
        ("connectedComponents", "window=256"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.2), sharex=True)
    backends = backend_sequence(rows)

    for ax, (operation, parameter) in zip(axes.flat, representatives):
        subset = [
            row
            for row in rows
            if row["operation"] == operation
            and row["parameter"] == parameter
            and row["pixelType"] == "UInt8"
        ]
        for backend in backends:
            points = sorted(
                [row for row in subset if row["backend"] == backend],
                key=lambda row: int(row["voxels"]),
            )
            if not points:
                continue
            ax.plot(
                [int(row["voxels"]) for row in points],
                [float(row["internal"]) for row in points],
                marker="o",
                linewidth=1.4,
                markersize=3.5,
                label=BACKEND_LABELS.get(backend, backend),
                color=BACKEND_COLORS.get(backend),
            )
        ax.set_title(f"{operation}\n{parameter_label(parameter)}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, which="major", alpha=0.25)

    for ax in axes[-1, :]:
        ax.set_xlabel("voxels")
    for ax in axes[:, 0]:
        ax.set_ylabel("internal seconds")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False)
    fig.suptitle("Runtime scaling by volume size, UInt8", y=1.03)
    save(fig, output_dir / "runtime-by-size.pdf")
    plt.close(fig)


def plot_complexity_scaling(rows: list[dict[str, object]], output_dir: Path):
    plt = setup_matplotlib()
    operations = [
        ("uniformConvolve", "kernelSize", "kernel size"),
        ("median", "radius", "radius"),
        ("dilate", "radius", "radius"),
    ]
    shapes = ["256x256x256", "512x512x512", "1024x1024x1024"]
    backends = backend_sequence(rows)

    fig, axes = plt.subplots(len(operations), len(shapes), figsize=(7.4, 5.8), sharey="row")
    for r, (operation, key, xlabel) in enumerate(operations):
        for c, shape in enumerate(shapes):
            ax = axes[r, c]
            subset = [
                row
                for row in rows
                if row["operation"] == operation
                and row["shape"] == shape
                and row["pixelType"] == "UInt8"
                and str(row["parameter"]).startswith(f"{key}=")
            ]
            for backend in backends:
                points = sorted(
                    [row for row in subset if row["backend"] == backend],
                    key=lambda row: int(row["parameterValue"] or 0),
                )
                if not points:
                    continue
                ax.plot(
                    [int(row["parameterValue"] or 0) for row in points],
                    [float(row["internal"]) for row in points],
                    marker="o",
                    linewidth=1.4,
                    markersize=3.5,
                    label=BACKEND_LABELS.get(backend, backend),
                    color=BACKEND_COLORS.get(backend),
                )
            ax.set_title(f"{operation}, {shape_label(shape)}")
            ax.set_yscale("log")
            ax.set_xlabel(xlabel)
            ax.grid(True, which="major", alpha=0.25)
        axes[r, 0].set_ylabel("internal seconds")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False)
    fig.suptitle("Runtime scaling by neighbourhood complexity, UInt8", y=1.03)
    save(fig, output_dir / "runtime-by-complexity.pdf")
    plt.close(fig)


def plot_memory_time_scatter(rows: list[dict[str, object]], output_dir: Path):
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8), sharey=True)
    shapes = ["256x256x256", "512x512x512", "1024x1024x1024"]
    backends = backend_sequence(rows)

    for ax, shape in zip(axes, shapes):
        subset = [row for row in rows if row["shape"] == shape]
        for backend in backends:
            points = [row for row in subset if row["backend"] == backend]
            if not points:
                continue
            ax.scatter(
                [float(row["peak"]) / 1024.0 for row in points],
                [float(row["internal"]) for row in points],
                s=14,
                alpha=0.75,
                label=BACKEND_LABELS.get(backend, backend),
                color=BACKEND_COLORS.get(backend),
                linewidths=0,
            )
        ax.set_title(shape_label(shape))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="major", alpha=0.25)
        ax.set_xlabel("peak RSS (GiB)")
    axes[0].set_ylabel("internal seconds")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False)
    fig.suptitle("Runtime and peak memory across benchmark cases", y=1.08)
    save(fig, output_dir / "runtime-vs-memory.pdf")
    plt.close(fig)


def plot_startup_overhead(rows: list[dict[str, object]], output_dir: Path):
    plt = setup_matplotlib()
    rows = [row for row in rows if row["startup"] is not None]
    backends = backend_sequence(rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))

    for index, backend in enumerate(backends, start=1):
        values = [float(row["startup"]) for row in rows if row["backend"] == backend]
        axes[0].scatter(
            [index] * len(values),
            values,
            s=10,
            alpha=0.45,
            color=BACKEND_COLORS.get(backend),
            linewidths=0,
        )
        if values:
            axes[0].plot(
                [index - 0.25, index + 0.25],
                [sorted(values)[len(values) // 2]] * 2,
                color="black",
                linewidth=1.0,
            )
    axes[0].set_xticks(range(1, len(backends) + 1), [BACKEND_LABELS.get(b, b) for b in backends], rotation=25, ha="right")
    axes[0].set_ylabel("wall - internal seconds")
    axes[0].set_title("Startup and wrapper overhead")
    axes[0].grid(True, axis="y", alpha=0.25)

    for backend in backends:
        points = [row for row in rows if row["backend"] == backend]
        axes[1].scatter(
            [float(row["internal"]) for row in points],
            [float(row["wall"]) for row in points],
            s=12,
            alpha=0.65,
            label=BACKEND_LABELS.get(backend, backend),
            color=BACKEND_COLORS.get(backend),
            linewidths=0,
        )
    max_value = max(max(float(row["internal"]), float(row["wall"])) for row in rows)
    min_value = min(min(float(row["internal"]), float(row["wall"])) for row in rows if float(row["internal"]) > 0)
    axes[1].plot([min_value, max_value], [min_value, max_value], color="black", linewidth=0.8, alpha=0.6)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("internal seconds")
    axes[1].set_ylabel("wall seconds")
    axes[1].set_title("Internal vs wall time")
    axes[1].grid(True, which="major", alpha=0.25)
    axes[1].legend(frameon=False)

    save(fig, output_dir / "startup-overhead.pdf")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    rows = load_rows(Path(args.input))
    if not rows:
        raise SystemExit(f"no plottable successful rows found in {args.input}")

    output_dir = Path(args.output_dir)
    plot_runtime_by_size(rows, output_dir)
    plot_complexity_scaling(rows, output_dir)
    plot_memory_time_scatter(rows, output_dir)
    plot_startup_overhead(rows, output_dir)

    print(f"wrote figures to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
