#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash benchmarks/run_zarr_filesystem_benchmark.sh [options]

Run only the focused Dask/OME-Zarr filesystem-sensitivity benchmark. This is
kept separate from run_lmip_pdf_benchmarks.sh because it is intended for
surveilled runs on an explicitly mounted slower filesystem.

The script measures copy, threshold, and median on the slow filesystem, combines
those rows with the corresponding fast-filesystem rows from
benchmarks/results/raw.zarr-none.csv, regenerates the filesystem comparison
summary, and copies the filesystem figures into the LMIP TeX figure directory.

Options:
  --repeat N              Repeats for the slow-filesystem Zarr rows. Default: 3.
  --dry-run               Print commands instead of running them.
  -h, --help              Show this help.

Environment overrides:
  PYTHON                  Python executable. Defaults to .venv-benchmarks/bin/python
                          when present, otherwise python3.
  DATE_STAMP              Output suffix. Default: current date as YYYYMMDD.
  SLOW_ROOT               Required mounted slow filesystem root.
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

repeat=3
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repeat)
      repeat="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "run_zarr_filesystem_benchmark.sh: unknown option $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ".venv-benchmarks/bin/python" ]]; then
    PYTHON=".venv-benchmarks/bin/python"
  else
    PYTHON="python3"
  fi
fi

date_stamp="${DATE_STAMP:-$(date +%Y%m%d)}"
slow_root="${SLOW_ROOT:-}"

if [[ -z "$slow_root" ]]; then
  echo "SLOW_ROOT is required for this surveilled benchmark." >&2
  echo "Example: SLOW_ROOT=/Volumes/slow bash benchmarks/run_zarr_filesystem_benchmark.sh" >&2
  exit 1
fi

input_root="tmp/benchmarks/input"
zarr_fast_raw="benchmarks/results/raw.zarr-none.csv"
zarr_slow_raw="benchmarks/results/raw.zarr-slow.csv"
zarr_filesystem_raw="benchmarks/results/raw.zarr-filesystems.csv"
zarr_filesystem_summary="benchmarks/results/summary.zarr-filesystems.csv"
figure_dir="benchmarks/results/figures"
tex_figure_dir="notes/LMIP_Optimiser_and_Studio/figures"
slow_omezarr_root="${slow_root}/benchmarks/input-omezarr-lmip-${date_stamp}"
slow_zarr_output_root="${slow_root}/benchmarks/output-lmip-zarr-${date_stamp}"

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$dry_run" -eq 0 ]]; then
    "$@"
  fi
}

echo "Using Python: $PYTHON"
echo "Date stamp:   $date_stamp"
echo "Slow root:    ${slow_root:-<unset>}"
echo "Dry run:      $dry_run"

if [[ "$dry_run" -eq 0 ]]; then
  if [[ ! -f "$zarr_fast_raw" ]]; then
    echo "Missing fast Zarr raw results: $zarr_fast_raw" >&2
    echo "Run benchmarks/run_lmip_pdf_benchmarks.sh first, or provide the fast raw file." >&2
    exit 1
  fi
  if [[ ! -d "$slow_root" ]]; then
    echo "Slow root does not exist or is not mounted: $slow_root" >&2
    echo "Mount the slow filesystem, or set SLOW_ROOT=/path/to/mount." >&2
    exit 1
  fi
fi

run mkdir -p "benchmarks/results" "$figure_dir" "$tex_figure_dir"

echo
echo "== Prepare slow-filesystem OME-Zarr inputs =="
if [[ "$dry_run" -eq 0 ]]; then
  "$PYTHON" - "$input_root" "$slow_omezarr_root" <<'PY'
import csv
import subprocess
import sys
from pathlib import Path

input_root = Path(sys.argv[1])
omezarr_root = Path(sys.argv[2])
cases = Path("benchmarks/config/special-cases.csv")
seen = sorted({(row["pixelType"], row["shape"]) for row in csv.DictReader(cases.open(newline=""))})
for pixel_type, shape in seen:
    command = [
        sys.executable,
        "benchmarks/tools/tiff_stack_to_omezarr.py",
        "--input",
        str(input_root / f"{pixel_type}_{shape}"),
        "--output",
        str(omezarr_root / f"{pixel_type}_{shape}"),
        "--shape",
        shape,
        "--pixel-type",
        pixel_type,
    ]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)
PY
else
  echo "+ convert special-case TIFF inputs to OME-Zarr stores under $slow_omezarr_root"
fi

echo
echo "== Run slow-filesystem Dask/OME-Zarr rows =="
run rm -f "$zarr_slow_raw" "$zarr_filesystem_raw" "$zarr_filesystem_summary"

run "$PYTHON" benchmarks/tools/run_manifest.py \
  --cases benchmarks/config/special-cases.csv \
  --backend python-dask-omezarr \
  --results "$zarr_slow_raw" \
  --input-root "$slow_omezarr_root" \
  --output-root "$slow_zarr_output_root" \
  --repeat "$repeat" \
  --operations copy,threshold,median

echo
echo "== Combine fast and slow Zarr rows =="
if [[ "$dry_run" -eq 0 ]]; then
  "$PYTHON" - "$zarr_fast_raw" "$zarr_slow_raw" "$zarr_filesystem_raw" <<'PY'
import csv
import sys
from pathlib import Path

fast = Path(sys.argv[1])
slow = Path(sys.argv[2])
output = Path(sys.argv[3])

fast_rows = list(csv.DictReader(fast.open(newline="")))
slow_rows = list(csv.DictReader(slow.open(newline="")))
if not fast_rows:
    raise SystemExit(f"no rows in {fast}")
if not slow_rows:
    raise SystemExit(f"no rows in {slow}")

fields = fast_rows[0].keys()
rows = []
for row in fast_rows:
    if row["backend"] == "python-dask-omezarr" and row["operation"] in {"copy", "threshold", "median"}:
        rows.append(row)
for row in slow_rows:
    copied = dict(row)
    if copied["backend"] == "python-dask-omezarr":
        copied["backend"] = "python-dask-omezarr-slow"
    rows.append(copied)

with output.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f"+ wrote {len(rows)} fast/slow Zarr rows -> {output}")
PY
else
  echo "+ combine $zarr_fast_raw and $zarr_slow_raw into $zarr_filesystem_raw with slow rows relabelled"
fi

run "$PYTHON" benchmarks/tools/summarize_results.py \
  --input "$zarr_filesystem_raw" \
  --output "$zarr_filesystem_summary"

run "$PYTHON" benchmarks/tools/plot_zarr_results.py \
  --summary "$zarr_filesystem_summary" \
  --output-dir "$figure_dir" \
  --latex-dir "$tex_figure_dir" \
  --filename-prefix "filesystem-"

echo
echo "Done."
echo "Slow raw:          $zarr_slow_raw"
echo "Filesystem raw:    $zarr_filesystem_raw"
echo "Filesystem summary:$zarr_filesystem_summary"
echo "Figures:           $figure_dir"
echo "TeX figures:       $tex_figure_dir"
