#!/usr/bin/env python3
"""Summarize raw benchmark measurements for tables and plots."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    groups: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    with open(args.input, newline="") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["backend"],
                row["operation"],
                row["pixelType"],
                row["shape"],
                row["parameter"],
            )
            groups[key].append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "backend",
                "operation",
                "pixelType",
                "shape",
                "parameter",
                "n",
                "successes",
                "medianWallSeconds",
                "meanWallSeconds",
                "medianPeakRssMiB",
                "meanPeakRssMiB",
            ]
        )

        for key, rows in sorted(groups.items()):
            successful = [row for row in rows if row["exitCode"] == "0"]
            wall = [float(row["wallSeconds"]) for row in successful]
            rss_mib = [float(row["peakRssKiB"]) / 1024.0 for row in successful]
            writer.writerow(
                [
                    *key,
                    len(rows),
                    len(successful),
                    f"{median(wall):.9f}",
                    f"{statistics.mean(wall):.9f}" if wall else "nan",
                    f"{median(rss_mib):.3f}",
                    f"{statistics.mean(rss_mib):.3f}" if rss_mib else "nan",
                ]
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

