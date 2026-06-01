#!/usr/bin/env python3
"""Replace benchmark raw.csv rows with rows from another raw CSV."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


KEY_COLUMNS = ["backend", "operation", "pixelType", "shape", "parameter", "repeat"]


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[column] for column in KEY_COLUMNS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Existing raw CSV.")
    parser.add_argument("--replacement", required=True, help="Raw CSV rows to insert.")
    parser.add_argument("--output", required=True, help="Output raw CSV. May equal --input.")
    parser.add_argument("--backup", default="", help="Optional backup path when output equals input.")
    parser.add_argument(
        "--drop-backend",
        default="",
        help="Drop all existing rows for this backend before appending replacements.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    replacement_path = Path(args.replacement)
    output_path = Path(args.output)

    fields, rows = read_rows(input_path)
    replacement_fields, replacement_rows = read_rows(replacement_path)
    if fields != replacement_fields:
        raise SystemExit(f"CSV headers differ: {input_path} vs {replacement_path}")

    replacement_keys = {key(row) for row in replacement_rows}
    kept = []
    for row in rows:
        if args.drop_backend and row["backend"] == args.drop_backend:
            continue
        if key(row) in replacement_keys:
            continue
        kept.append(row)

    if input_path == output_path:
        backup = Path(args.backup) if args.backup else input_path.with_suffix(input_path.suffix + ".bak")
        shutil.copy2(input_path, backup)
        print(f"backup: {backup}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)
        writer.writerows(replacement_rows)

    print(f"kept: {len(kept)}")
    print(f"inserted: {len(replacement_rows)}")
    print(f"output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
