#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash benchmarks/run_all.sh [options]

Run the comparative benchmark workflow end-to-end:
  1. Generate deterministic TIFF-stack inputs.
  2. Build selected compiled benchmark backends before measurement.
  3. Run selected baseline TIFF-stack backends.
  4. Optionally convert inputs and run Dask/OME-Zarr special cases.
  5. Summarize raw measurements.

Options:
  --repeat N              Repeats per manifest case. Defaults to 3.
  --repeat-start N        First repeat index to run. Defaults to 1.
  --repeat-end N          Last repeat index to run. Defaults to --repeat.
  --backends LIST         Comma-separated baseline backends.
                          Defaults to stackprocessing,python-skimage-scipy,cpp-itk,matlab.
                          Valid baseline backends: stackprocessing,stackprocessing-image-arraypool,stackprocessing-arraypool,stackprocessing-arraypool-slice,stackprocessing-arraypool-slice-reuse,stackprocessing-byte-slice-reuse,stackprocessing-byte-float32-slice-reuse,python-skimage-scipy,cpp-itk,matlab.
  --include-special       Also run python-dask-omezarr special cases.
  --cases PATH            Baseline cases CSV. Defaults to benchmarks/config/cases.csv.
  --special-cases PATH    Special cases CSV. Defaults to benchmarks/config/special-cases.csv.
  --pixel-types LIST      Optional comma-separated pixel type filter, for example UInt8 or UInt8,Float32.
  --shapes LIST           Optional comma-separated shape filter, for example 256x256x256.
  --operations LIST       Optional comma-separated operation filter, for example median,dilate.
  --parameters LIST       Optional comma-separated parameter filter, for example radius=1,radius=2.
  --input-root PATH       TIFF input root. Defaults to tmp/benchmarks/input.
  --output-root PATH      Output root. Defaults to tmp/benchmarks/output.
  --omezarr-root PATH     OME-Zarr input root. Defaults to tmp/benchmarks/input-omezarr.
  --results PATH          Raw output CSV. Defaults to benchmarks/results/raw.csv.
  --summary PATH          Summary output CSV. Defaults to benchmarks/results/summary.csv.
  --keep-outputs          Keep per-case output stacks. Defaults to deleting them after timing.
  --skip-existing         Skip successful rows already present in --results.
  --force-inputs          Regenerate TIFF inputs even if present.
  --skip-inputs           Do not generate TIFF inputs.
  --skip-builds           Do not prebuild compiled benchmark backends.
  --build-itk             Accepted for compatibility; cpp-itk is now built automatically when selected.
  --itk-exe PATH          C++/ITK executable path.
  --stackprocessing-dll PATH
                          Built StackProcessing benchmark DLL.
  --matlab-exe PATH       MATLAB executable. Defaults to matlab.
  --dry-run               Print commands without executing benchmark cases.
  -h, --help              Show this help.

Examples:
  bash benchmarks/run_all.sh --repeat 3

  bash benchmarks/run_all.sh \
    --repeat 3 \
    --backends stackprocessing,python-skimage-scipy,cpp-itk,matlab \
    --include-special
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

repeat=3
repeat_start=""
repeat_end=""
backends="stackprocessing,python-skimage-scipy,cpp-itk,matlab"
include_special=0
cases="benchmarks/config/cases.csv"
special_cases="benchmarks/config/special-cases.csv"
pixel_types=""
shapes=""
operations=""
parameters=""
input_root="tmp/benchmarks/input"
output_root="tmp/benchmarks/output"
omezarr_root="tmp/benchmarks/input-omezarr"
results="benchmarks/results/raw.csv"
summary="benchmarks/results/summary.csv"
force_inputs=0
skip_inputs=0
keep_outputs=0
skip_existing=0
build_itk=0
skip_builds=0
dry_run=0
itk_exe="benchmarks/cpp-itk/build/benchmark_itk"
stackprocessing_dll="benchmarks/StackProcessing.Benchmarks/bin/Debug/net10.0/StackProcessing.Benchmarks.dll"
matlab_exe="matlab"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repeat)
      repeat="$2"
      shift 2
      ;;
    --repeat-start)
      repeat_start="$2"
      shift 2
      ;;
    --repeat-end)
      repeat_end="$2"
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
    --pixel-types)
      pixel_types="$2"
      shift 2
      ;;
    --shapes)
      shapes="$2"
      shift 2
      ;;
    --operations)
      operations="$2"
      shift 2
      ;;
    --parameters)
      parameters="$2"
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
    --keep-outputs)
      keep_outputs=1
      shift
      ;;
    --skip-existing)
      skip_existing=1
      shift
      ;;
    --force-inputs)
      force_inputs=1
      shift
      ;;
    --skip-inputs)
      skip_inputs=1
      shift
      ;;
    --skip-builds)
      skip_builds=1
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
    --stackprocessing-dll)
      stackprocessing_dll="$2"
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
if [[ -n "$repeat_start" ]] && (! [[ "$repeat_start" =~ ^[0-9]+$ ]] || [[ "$repeat_start" -lt 1 ]]); then
  echo "run_all.sh: --repeat-start expects a positive integer" >&2
  exit 2
fi
if [[ -n "$repeat_end" ]] && (! [[ "$repeat_end" =~ ^[0-9]+$ ]] || [[ "$repeat_end" -lt 1 ]]); then
  echo "run_all.sh: --repeat-end expects a positive integer" >&2
  exit 2
fi

mkdir -p "$(dirname "$results")" "$(dirname "$summary")" "$output_root"

cleanup_internal_files() {
  if [[ "$dry_run" -eq 0 ]]; then
    find "$(dirname "$results")" -maxdepth 1 -type f -name 'benchmark-internal-*.txt' -delete
  fi
}

trap cleanup_internal_files EXIT
cleanup_internal_files

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$dry_run" -eq 0 ]]; then
    "$@"
  fi
}

IFS=',' read -r -a backend_array <<< "$backends"

has_backend() {
  local candidate="$1"
  local backend
  for backend in "${backend_array[@]}"; do
    if [[ "$backend" == "$candidate" ]]; then
      return 0
    fi
  done
  return 1
}

for backend in "${backend_array[@]}"; do
  case "$backend" in
    stackprocessing|stackprocessing-image-arraypool|stackprocessing-arraypool|stackprocessing-arraypool-slice|stackprocessing-arraypool-slice-reuse|stackprocessing-byte-slice-reuse|stackprocessing-byte-float32-slice-reuse|python-skimage-scipy|cpp-itk|matlab)
      ;;
    "")
      continue
      ;;
    *)
      echo "run_all.sh: unknown baseline backend '$backend'" >&2
      exit 2
      ;;
  esac
done

if [[ "$skip_builds" -eq 0 ]]; then
  if [[ "$skip_inputs" -eq 0 ]] || has_backend stackprocessing || has_backend stackprocessing-image-arraypool || has_backend stackprocessing-arraypool || has_backend stackprocessing-arraypool-slice || has_backend stackprocessing-arraypool-slice-reuse || has_backend stackprocessing-byte-slice-reuse || has_backend stackprocessing-byte-float32-slice-reuse; then
    run_cmd dotnet build benchmarks/StackProcessing.Benchmarks/StackProcessing.Benchmarks.fsproj --nologo
  fi
  if has_backend cpp-itk || [[ "$build_itk" -eq 1 ]]; then
    run_cmd cmake -S benchmarks/cpp-itk -B benchmarks/cpp-itk/build
    run_cmd cmake --build benchmarks/cpp-itk/build --config Release
  fi
fi

if [[ "$skip_inputs" -eq 0 ]]; then
  input_args=(python3 benchmarks/tools/prepare_inputs.py --cases "$cases" --input-root "$input_root")
  if [[ -n "$pixel_types" ]]; then
    input_args+=(--pixel-types "$pixel_types")
  fi
  if [[ -n "$shapes" ]]; then
    input_args+=(--shapes "$shapes")
  fi
  if [[ "$force_inputs" -eq 1 ]]; then
    input_args+=(--force)
  fi
  run_cmd "${input_args[@]}"
fi

for backend in "${backend_array[@]}"; do
  [[ -z "$backend" ]] && continue

  manifest_args=(
    python3 benchmarks/tools/run_manifest.py
    --cases "$cases"
    --backend "$backend"
    --results "$results"
    --input-root "$input_root"
    --output-root "$output_root"
    --repeat "$repeat"
    --itk-exe "$itk_exe"
    --stackprocessing-dll "$stackprocessing_dll"
    --matlab-exe "$matlab_exe"
  )
  if [[ -n "$pixel_types" ]]; then
    manifest_args+=(--pixel-types "$pixel_types")
  fi
  if [[ -n "$shapes" ]]; then
    manifest_args+=(--shapes "$shapes")
  fi
  if [[ -n "$operations" ]]; then
    manifest_args+=(--operations "$operations")
  fi
  if [[ -n "$parameters" ]]; then
    manifest_args+=(--parameters "$parameters")
  fi
  if [[ -n "$repeat_start" ]]; then
    manifest_args+=(--repeat-start "$repeat_start")
  fi
  if [[ -n "$repeat_end" ]]; then
    manifest_args+=(--repeat-end "$repeat_end")
  fi
  if [[ "$dry_run" -eq 1 ]]; then
    manifest_args+=(--dry-run)
  fi
  if [[ "$keep_outputs" -eq 1 ]]; then
    manifest_args+=(--keep-outputs)
  fi
  if [[ "$skip_existing" -eq 1 ]]; then
    manifest_args+=(--skip-existing)
  fi
  if [[ "$dry_run" -eq 1 ]]; then
    printf '+'
    printf ' %q' "${manifest_args[@]}"
    printf '\n'
    "${manifest_args[@]}"
  else
    run_cmd "${manifest_args[@]}"
  fi
done

if [[ "$include_special" -eq 1 ]]; then
  python3 - "$special_cases" "$input_root" "$omezarr_root" "$dry_run" "$pixel_types" "$shapes" <<'PY'
import csv
import subprocess
import sys
from pathlib import Path

cases_path = Path(sys.argv[1])
input_root = Path(sys.argv[2])
omezarr_root = Path(sys.argv[3])
dry_run = sys.argv[4] == "1"
pixel_types = {item.strip().lower() for item in sys.argv[5].split(",") if item.strip()}
shapes = {item.strip().lower() for item in sys.argv[6].split(",") if item.strip()}

seen = sorted({
    (row["pixelType"], row["shape"])
    for row in csv.DictReader(cases_path.open(newline=""))
    if not pixel_types or row["pixelType"].lower() in pixel_types
    if not shapes or row["shape"].lower() in shapes
})
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
  if [[ -n "$pixel_types" ]]; then
    special_args+=(--pixel-types "$pixel_types")
  fi
  if [[ -n "$shapes" ]]; then
    special_args+=(--shapes "$shapes")
  fi
  if [[ -n "$operations" ]]; then
    special_args+=(--operations "$operations")
  fi
  if [[ -n "$parameters" ]]; then
    special_args+=(--parameters "$parameters")
  fi
  if [[ -n "$repeat_start" ]]; then
    special_args+=(--repeat-start "$repeat_start")
  fi
  if [[ -n "$repeat_end" ]]; then
    special_args+=(--repeat-end "$repeat_end")
  fi
  if [[ "$dry_run" -eq 1 ]]; then
    special_args+=(--dry-run)
  fi
  if [[ "$keep_outputs" -eq 1 ]]; then
    special_args+=(--keep-outputs)
  fi
  if [[ "$dry_run" -eq 1 ]]; then
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
