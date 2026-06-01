#!/usr/bin/env python3
"""Create publication-oriented PDF figures from benchmark summary rows."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


BACKEND_ORDER = [
    "stackprocessing",
    "stackprocessing-arraypool",
    "stackprocessing-arraypool-slice",
    "stackprocessing-arraypool-slice-reuse",
    "stackprocessing-byte-slice-reuse",
    "stackprocessing-byte-float32-slice-reuse",
    "cpp-itk",
    "python-skimage-scipy",
    "matlab",
]
BACKEND_LABELS = {
    "stackprocessing": "StackProcessing",
    "stackprocessing-arraypool": "StackProcessing ArrayPool",
    "stackprocessing-arraypool-slice": "StackProcessing ArrayPool slice",
    "stackprocessing-arraypool-slice-reuse": "StackProcessing ArrayPool slice reuse",
    "stackprocessing-byte-slice-reuse": "StackProcessing byte slice reuse",
    "stackprocessing-byte-float32-slice-reuse": "StackProcessing byte Float32 view",
    "cpp-itk": "C++ ITK",
    "python-skimage-scipy": "Python scipy/skimage",
    "matlab": "MATLAB",
}
BACKEND_COLORS = {
    "stackprocessing": "#1f77b4",
    "stackprocessing-arraypool": "#17becf",
    "stackprocessing-arraypool-slice": "#8c564b",
    "stackprocessing-arraypool-slice-reuse": "#7f7f7f",
    "stackprocessing-byte-slice-reuse": "#bcbd22",
    "stackprocessing-byte-float32-slice-reuse": "#e377c2",
    "cpp-itk": "#2ca02c",
    "python-skimage-scipy": "#ff7f0e",
    "matlab": "#9467bd",
}
BACKEND_MARKERS = {
    "stackprocessing": "o",
    "stackprocessing-arraypool": "P",
    "stackprocessing-arraypool-slice": "X",
    "stackprocessing-arraypool-slice-reuse": "*",
    "stackprocessing-byte-slice-reuse": "v",
    "stackprocessing-byte-float32-slice-reuse": "<",
    "cpp-itk": "s",
    "python-skimage-scipy": "^",
    "matlab": "D",
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


def volume_ticks(rows: list[dict[str, object]]) -> tuple[list[int], list[str]]:
    shapes = sorted({str(row["shape"]) for row in rows}, key=shape_voxels)
    return [shape_voxels(shape) for shape in shapes], [shape_label(shape) for shape in shapes]


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


def pixel_label(pixel_type: str) -> str:
    return {
        "UInt8": "uint8",
        "UInt16": "uint16",
        "Float32": "float32",
    }.get(pixel_type, pixel_type)


def pixel_slug(pixel_type: str) -> str:
    return pixel_label(pixel_type).lower()


def operation_label(operation: str) -> str:
    return {
        "connectedComponents": "connected components",
    }.get(operation, operation)


def load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            internal = finite_float(row.get("medianInternalSeconds", ""))
            wall = finite_float(row.get("medianWallSeconds", ""))
            wrapper = finite_float(row.get("medianWrapperOverheadSeconds", ""))
            peak = finite_float(row.get("medianPeakRssMiB", ""))
            successes = int(row.get("successes") or 0)
            if successes <= 0 or internal is None or wall is None or peak is None:
                continue

            enriched: dict[str, object] = dict(row)
            enriched["internal"] = internal
            enriched["wall"] = wall
            enriched["wrapper"] = wrapper
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


def draw_sequence(backends: list[str], draw_last: str | None = None) -> list[str]:
    if draw_last is None or draw_last not in backends:
        return backends
    return [backend for backend in backends if backend != draw_last] + [draw_last]


def setup_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
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


def top_legend(fig, handles, labels, y: float = 0.90):
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, y),
            ncol=min(4, len(labels)),
            frameon=False,
        )


def metric_value(row: dict[str, object], metric: str) -> float:
    value = float(row[metric])
    if metric == "peak":
        return value / 1024.0
    return value


def metric_ylabel(metric: str) -> str:
    if metric == "peak":
        return "peak RSS (GiB)"
    return "internal seconds"


def metric_title(metric: str) -> str:
    if metric == "peak":
        return "Peak memory pressure"
    return "Runtime"


def plot_by_size(rows: list[dict[str, object]], output_dir: Path, pixel_type: str, metric: str):
    plt = setup_matplotlib()
    representatives = [
        ("copy", "none"),
        ("threshold", "threshold=128"),
        ("convolve", "kernelSize=3"),
        ("convolve", "kernelSize=7"),
        ("median", "radius=1"),
        ("median", "radius=3"),
    ]
    if pixel_type == "UInt8":
        representatives.extend([
            ("dilate", "radius=1"),
            ("dilate", "radius=3"),
        ])
        representatives.append(("connectedComponents", "window=256"))

    if len(representatives) <= 6:
        rows_count, cols_count = 2, 3
    else:
        cols_count = 3
        rows_count = math.ceil(len(representatives) / cols_count)

    figure_height = 2.2 * rows_count + 0.5
    fig, axes = plt.subplots(rows_count, cols_count, figsize=(7.8, figure_height), sharex=True)
    backends = backend_sequence(rows)
    xticks, xticklabels = volume_ticks(rows)

    for ax, (operation, parameter) in zip(axes.flat, representatives):
        if operation == "connectedComponents" and pixel_type == "UInt8" and metric == "peak":
            lmip_windows = {
                "256x256x256": "window=256",
                "512x512x512": "window=64",
                "1024x1024x1024": "window=16",
            }
            subset = [
                row
                for row in rows
                if row["operation"] == operation
                and row["pixelType"] == pixel_type
                and (
                    (row["backend"] == "stackprocessing" and row["parameter"] == lmip_windows.get(str(row["shape"])))
                    or (row["backend"] != "stackprocessing" and row["parameter"] == parameter)
                )
            ]
            title_parameter = "window 256/64/16"
        else:
            subset = [
                row
                for row in rows
                if row["operation"] == operation
                and row["parameter"] == parameter
                and row["pixelType"] == pixel_type
            ]
            title_parameter = parameter_label(parameter)
        for backend in backends:
            points = sorted(
                [row for row in subset if row["backend"] == backend],
                key=lambda row: int(row["voxels"]),
            )
            if not points:
                continue
            ax.plot(
                [int(row["voxels"]) for row in points],
                [metric_value(row, metric) for row in points],
                marker=BACKEND_MARKERS.get(backend, "o"),
                linewidth=1.4,
                markersize=3.5,
                label=BACKEND_LABELS.get(backend, backend),
                color=BACKEND_COLORS.get(backend),
            )
        ax.set_title(f"{operation_label(operation)}\n{title_parameter}", pad=8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(xticks, xticklabels)
        ax.set_yscale("log")
        ax.grid(True, which="major", alpha=0.25)

    for ax in list(axes.flat)[len(representatives):]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("volume")
    for ax in axes[:, 0]:
        ax.set_ylabel(metric_ylabel(metric))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.78, hspace=0.62 if rows_count == 3 else 0.54, wspace=0.28)
    fig.suptitle(f"{metric_title(metric)} scaling by volume size, {pixel_label(pixel_type)}", y=0.98)
    top_legend(fig, handles, labels, y=0.91)
    prefix = "memory-pressure" if metric == "peak" else "runtime"
    output = output_dir / f"{prefix}-by-size-{pixel_slug(pixel_type)}.pdf"
    save(fig, output)
    plt.close(fig)


def plot_complexity_scaling(rows: list[dict[str, object]], output_dir: Path, pixel_type: str, metric: str):
    plt = setup_matplotlib()
    operations = [
        ("convolve", "kernelSize", "kernel size"),
        ("median", "radius", "radius"),
    ]
    if pixel_type == "UInt8":
        operations.append(("dilate", "radius", "radius"))
    shapes = ["256x256x256", "512x512x512", "1024x1024x1024"]
    backends = backend_sequence(rows)

    figure_height = 6.7 if len(operations) == 3 else 4.9
    fig, axes = plt.subplots(len(operations), len(shapes), figsize=(7.8, figure_height), sharey="row")
    for r, (operation, key, xlabel) in enumerate(operations):
        for c, shape in enumerate(shapes):
            ax = axes[r, c]
            subset = [
                row
                for row in rows
                if row["operation"] == operation
                and row["shape"] == shape
                and row["pixelType"] == pixel_type
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
                    [metric_value(row, metric) for row in points],
                    marker=BACKEND_MARKERS.get(backend, "o"),
                    linewidth=1.4,
                    markersize=3.5,
                    label=BACKEND_LABELS.get(backend, backend),
                    color=BACKEND_COLORS.get(backend),
                )
            if r == 0:
                ax.set_title(shape_label(shape), pad=8)
            ax.set_yscale("log")
            if r == len(operations) - 1:
                ax.set_xlabel("kernel size / radius")
            ax.grid(True, which="major", alpha=0.25)
        axes[r, 0].set_ylabel(f"{operation_label(operation)}\n{metric_ylabel(metric)}")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.08, top=0.78, hspace=0.42, wspace=0.24)
    fig.suptitle(f"{metric_title(metric)} scaling by neighbourhood complexity, {pixel_label(pixel_type)}", y=0.98)
    top_legend(fig, handles, labels, y=0.91)
    prefix = "memory-pressure" if metric == "peak" else "runtime"
    output = output_dir / f"{prefix}-by-complexity-{pixel_slug(pixel_type)}.pdf"
    save(fig, output)
    plt.close(fig)


def plot_memory_time_scatter(rows: list[dict[str, object]], output_dir: Path):
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 3.2), sharey=True)
    shapes = ["256x256x256", "512x512x512", "1024x1024x1024"]
    backends = backend_sequence(rows)

    for ax, shape in zip(axes, shapes):
        subset = [row for row in rows if row["shape"] == shape]
        raw_uint8_gib = shape_voxels(shape) / 1024.0**3
        for backend in backends:
            points = [row for row in subset if row["backend"] == backend]
            if not points:
                continue
            ax.scatter(
                [float(row["peak"]) / 1024.0 for row in points],
                [float(row["internal"]) for row in points],
                marker=BACKEND_MARKERS.get(backend, "o"),
                s=20,
                alpha=0.78,
                label=BACKEND_LABELS.get(backend, backend),
                color=BACKEND_COLORS.get(backend),
                linewidths=0.3,
                edgecolors="white",
            )
        ax.axvline(raw_uint8_gib, color="black", linestyle="--", linewidth=0.8, alpha=0.35)
        ax.text(
            raw_uint8_gib,
            0.98,
            f"{shape_label(shape)} UInt8",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            va="top",
            ha="right",
            fontsize=6,
            color="black",
            alpha=0.55,
        )
        ax.set_title(shape_label(shape), pad=8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="major", alpha=0.25)
        ax.set_xlabel("peak RSS (GiB)")
        left, right = ax.get_xlim()
        ax.set_xlim(min(left, raw_uint8_gib * 0.75), right)
    axes[0].set_ylabel("internal seconds")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.70, wspace=0.25)
    fig.suptitle("Runtime and peak memory across benchmark cases", y=0.98)
    top_legend(fig, handles, labels, y=0.89)
    save(fig, output_dir / "runtime-vs-memory.pdf")
    plt.close(fig)


def plot_wrapper_overhead(rows: list[dict[str, object]], output_dir: Path):
    plt = setup_matplotlib()
    rows = [row for row in rows if row["wrapper"] is not None]
    backends = backend_sequence(rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4))

    for index, backend in enumerate(backends, start=1):
        values = [float(row["wrapper"]) for row in rows if row["backend"] == backend]
        axes[0].scatter(
            [index] * len(values),
            values,
            marker=BACKEND_MARKERS.get(backend, "o"),
            s=20,
            alpha=0.70,
            color=BACKEND_COLORS.get(backend),
            linewidths=0.3,
            edgecolors="white",
            label=BACKEND_LABELS.get(backend, backend),
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
    axes[0].set_title("Wrapper overhead")
    axes[0].grid(True, axis="y", alpha=0.25)

    for backend in draw_sequence(backends, draw_last="stackprocessing"):
        points = [row for row in rows if row["backend"] == backend]
        axes[1].scatter(
            [float(row["internal"]) for row in points],
            [float(row["wall"]) for row in points],
            marker=BACKEND_MARKERS.get(backend, "o"),
            s=20,
            alpha=0.46,
            label=BACKEND_LABELS.get(backend, backend),
            color=BACKEND_COLORS.get(backend),
            linewidths=0.3,
            edgecolors="white",
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
    handles, labels = axes[1].get_legend_handles_labels()
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.22, top=0.72, wspace=0.35)
    fig.suptitle("Wrapper overhead", y=0.98)
    top_legend(fig, handles, labels, y=0.89)

    save(fig, output_dir / "wrapper-overhead.pdf")
    plt.close(fig)


def plot_connected_components_window_policy(rows: list[dict[str, object]], output_dir: Path):
    plt = setup_matplotlib()
    subset = [
        row
        for row in rows
        if row["backend"] == "stackprocessing"
        and row["operation"] == "connectedComponents"
        and row["pixelType"] == "UInt8"
        and str(row["parameter"]).startswith("window=")
    ]
    if not subset:
        return

    policies = [
        ("window=256", "fixed window 256", {"256x256x256": "window=256", "512x512x512": "window=256", "1024x1024x1024": "window=256"}),
        (
            "constant",
            "constant slab budget",
            {"256x256x256": "window=256", "512x512x512": "window=64", "1024x1024x1024": "window=16"},
        ),
    ]
    shapes = ["256x256x256", "512x512x512", "1024x1024x1024"]

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.2), sharex=True)
    for ax, metric in zip(axes, ["internal", "peak"]):
        for _, label, shape_parameters in policies:
            points = []
            for shape in shapes:
                parameter = shape_parameters[shape]
                matches = [row for row in subset if row["shape"] == shape and row["parameter"] == parameter]
                if matches:
                    points.append(matches[0])
            if not points:
                continue
            ax.plot(
                [int(row["voxels"]) for row in points],
                [metric_value(row, metric) for row in points],
                marker="o" if label.startswith("fixed") else "s",
                linewidth=1.4,
                markersize=3.5,
                label=label,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks([shape_voxels(shape) for shape in shapes], [shape_label(shape) for shape in shapes])
        ax.set_yscale("log")
        ax.set_xlabel("volume")
        ax.set_ylabel(metric_ylabel(metric))
        ax.set_title(metric_title(metric), pad=8)
        ax.grid(True, which="major", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.16, top=0.70, wspace=0.30)
    fig.suptitle("Connected-components window policy, uint8", y=0.98)
    top_legend(fig, handles, labels, y=0.88)
    save(fig, output_dir / "connected-components-window-policy.pdf")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    rows = load_rows(Path(args.input))
    if not rows:
        raise SystemExit(f"no plottable successful rows found in {args.input}")

    output_dir = Path(args.output_dir)
    for pixel_type in ["UInt8", "UInt16", "Float32"]:
        for metric in ["internal", "peak"]:
            plot_by_size(rows, output_dir, pixel_type, metric)
            plot_complexity_scaling(rows, output_dir, pixel_type, metric)
    plot_memory_time_scatter(rows, output_dir)
    plot_wrapper_overhead(rows, output_dir)
    plot_connected_components_window_policy(rows, output_dir)

    print(f"wrote figures to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
