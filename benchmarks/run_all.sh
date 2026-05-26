#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash benchmarks/run_all.sh [options]

Run the comparative benchmark workflow end-to-end:
  1. Generate deterministic TIFF-stack inputs.
  2. Optionally build the C++/ITK backend.
  3. Run selected baseline TIFF-stack backends.
  4. Optionally convert inputs and run Dask/OME-Zarr special cases.
  5. Summarize raw measurements.

Options:
  --repeat N              Repeats per manifest case. Defaults to 3.
  --backends LIST         Comma-separated baseline backends.
                          Defaults to stackprocessing,python-skimage-scipy.
                          Valid baseline backends: stackprocessing,python-skimage-scipy,cpp-itk,matlab.
  --include-special       Also run python-dask-omezarr special cases.
  --cases PATH            Baseline cases CSV. Defaults to benchmarks/config/cases.csv.
  --special-cases PATH    Special cases CSV. Defaults to benchmarks/config/special-cases.csv.
  --input-root PATH       TIFF input root. Defaults to tmp/benchmarks/input.
  --output-root PATH      Output root. Defaults to tmp/benchmarks/output.
  --omezarr-root PATH     OME-Zarr input root. Defaults to tmp/benchmarks/input-omezarr.
  --results PATH          Raw output CSV. Defaults to benchmarks/results/raw.csv.
  --summary PATH          Summary output CSV. Defaults to benchmarks/results/summary.csv.
  --force-inputs          Regenerate TIFF inputs even if present.
  --skip-inputs           Do not generate TIFF inputs.
  --build-itk             Configure and build benchmarks/cpp-itk before running.
  --itk-exe PATH          C++/ITK executable path.
  --matlab-exe PATH       MATLAB executable. Defaults to matlab.
  --dry-run               Print commands without executing benchmark cases.
  -h, --help              Show this help.

Examples:
  bash benchmarks/run_all.sh --repeat 3

  bash benchmarks/run_all.sh \
    --repeat 3 \
    --backends stackprocessing,python-skimage-scipy,cpp-itk,matlab \
    --build-itk \
    --include-special
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

repeat=3
backends="stackprocessing,python-skimage-scipy"
include_special=0
cases="benchmarks/config/cases.csv"
special_cases="benchmarks/config/special-cases.csv"
input_root="tmp/benchmarks/input"
output_root="tmp/benchmarks/output"
omezarr_root="tmp/benchmarks/input-omezarr"
results="benchmarks/results/raw.csv"
summary="benchmarks/results/summary.csv"
force_inputs=0
skip_inputs=0
build_itk=0
dry_run=0
itk_exe="benchmarks/cpp-itk/build/benchmark_itk"
matlab_exe="matlab"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repeat)
      repeat="$2"
      shift 2
      ;;
    --backends)
      backends="$2"
      shift 2
      ;;
    --include-special)
      include_special=1
      shift
      ;;
    --cases)
      cases="$2"
      shift 2
      ;;
    --special-cases)
      special_cases="$2"
      shift 2
      ;;
    --input-root)
      input_root="$2"
      shift 2
      ;;
    --output-root)
      output_root="$2"
      shift 2
      ;;
    --omezarr-root)
      omezarr_root="$2"
      shift 2
      ;;
    --results)
      results="$2"
      shift 2
      ;;
    --summary)
      summary="$2"
      shift 2
      ;;
    --force-inputs)
      force_inputs=1
      shift
      ;;
    --skip-inputs)
      skip_inputs=1
      shift
      ;;
    --build-itk)
      build_itk=1
      shift
      ;;
    --itk-exe)
      itk_exe="$2"
      shift 2
      ;;
    --matlab-exe)
      matlab_exe="$2"
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
      echo "run_all.sh: unknown option $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "$repeat" =~ ^[0-9]+$ ]] || [[ "$repeat" -lt 1 ]]; then
  echo "run_all.sh: --repeat expects a positive integer" >&2
  exit 2
fi

mkdir -p "$(dirname "$results")" "$(dirname "$summary")" "$output_root"

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$dry_run" -eq 0 ]]; then
    "$@"
  fi
}

if [[ "$skip_inputs" -eq 0 ]]; then
  input_args=(python3 benchmarks/tools/prepare_inputs.py --cases "$cases" --input-root "$input_root")
  if [[ "$force_inputs" -eq 1 ]]; then
    input_args+=(--force)
  fi
  run_cmd "${input_args[@]}"
fi

if [[ "$build_itk" -eq 1 ]]; then
  run_cmd cmake -S benchmarks/cpp-itk -B benchmarks/cpp-itk/build
  run_cmd cmake --build benchmarks/cpp-itk/build --config Release
fi

IFS=',' read -r -a backend_array <<< "$backends"
for backend in "${backend_array[@]}"; do
  case "$backend" in
    stackprocessing|python-skimage-scipy|cpp-itk|matlab)
      ;;
    "")
      continue
      ;;
    *)
      echo "run_all.sh: unknown baseline backend '$backend'" >&2
      exit 2
      ;;
  esac

  manifest_args=(
    python3 benchmarks/tools/run_manifest.py
    --cases "$cases"
    --backend "$backend"
    --results "$results"
    --input-root "$input_root"
    --output-root "$output_root"
    --repeat "$repeat"
    --itk-exe "$itk_exe"
    --matlab-exe "$matlab_exe"
  )
  if [[ "$dry_run" -eq 1 ]]; then
    manifest_args+=(--dry-run)
    printf '+'
    printf ' %q' "${manifest_args[@]}"
    printf '\n'
    "${manifest_args[@]}"
  else
    run_cmd "${manifest_args[@]}"
  fi
done

if [[ "$include_special" -eq 1 ]]; then
  python3 - "$special_cases" "$input_root" "$omezarr_root" "$dry_run" <<'PY'
import csv
import subprocess
import sys
from pathlib import Path

cases_path = Path(sys.argv[1])
input_root = Path(sys.argv[2])
omezarr_root = Path(sys.argv[3])
dry_run = sys.argv[4] == "1"

seen = sorted({(row["pixelType"], row["shape"]) for row in csv.DictReader(cases_path.open(newline=""))})
for pixel_type, shape in seen:
    command = [
        "python3",
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
    if not dry_run:
        completed = subprocess.run(command)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
PY

  special_args=(
    python3 benchmarks/tools/run_manifest.py
    --cases "$special_cases"
    --backend python-dask-omezarr
    --results "$results"
    --input-root "$omezarr_root"
    --output-root "$output_root"
    --repeat "$repeat"
  )
  if [[ "$dry_run" -eq 1 ]]; then
    special_args+=(--dry-run)
    printf '+'
    printf ' %q' "${special_args[@]}"
    printf '\n'
    "${special_args[@]}"
  else
    run_cmd "${special_args[@]}"
  fi
fi

run_cmd python3 benchmarks/tools/summarize_results.py --input "$results" --output "$summary"

echo "raw results: $results"
echo "summary: $summary"
