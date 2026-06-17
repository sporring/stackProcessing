#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv-benchmarks/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${PYTHON:-python3}"
fi

REPEAT="${REPEAT:-3}"
CHUNK_SIZES="${CHUNK_SIZES:-64,128,256}"
SHAPE="${SHAPE:-1024x1024x1024}"
PIXEL_TYPE="${PIXEL_TYPE:-UInt8}"
WORKERS="${WORKERS:-3}"
DATE_STAMP="${DATE_STAMP:-$(date +%Y%m%d)}"

RAW="benchmarks/results/raw.zarr-chunk-fast.csv"
SUMMARY="benchmarks/results/summary.zarr-chunk-fast.csv"
INPUT_ZARR_ROOT="tmp/benchmarks/input-zarr-chunk-fast-${DATE_STAMP}"
INPUT_TIFF_ROOT="tmp/benchmarks/input-tiff-zarr-chunk-fast-${DATE_STAMP}"
OUTPUT_ROOT="tmp/benchmarks/output-zarr-chunk-fast-${DATE_STAMP}"
FIGURE_DIR="benchmarks/results/figures"
TEX_FIGURE_DIR="notes/LMIP_Optimiser_and_Studio/figures"
DLL="benchmarks/StackProcessing.Benchmarks/bin/Release/net10.0/StackProcessing.Benchmarks.dll"

dotnet build benchmarks/StackProcessing.Benchmarks/StackProcessing.Benchmarks.fsproj \
  -c Release \
  --nologo \
  --disable-build-servers \
  -p:UseSharedCompilation=false

"$PYTHON" benchmarks/tools/run_zarr_chunk_comparison.py \
  --results "$RAW" \
  --input-zarr-root "$INPUT_ZARR_ROOT" \
  --input-tiff-root "$INPUT_TIFF_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --stackprocessing-dll "$DLL" \
  --python "$PYTHON" \
  --shape "$SHAPE" \
  --pixel-type "$PIXEL_TYPE" \
  --chunk-sizes "$CHUNK_SIZES" \
  --repeat "$REPEAT" \
  --workers "$WORKERS" \
  "$@"

"$PYTHON" benchmarks/tools/summarize_results.py --input "$RAW" --output "$SUMMARY"
"$PYTHON" benchmarks/tools/plot_zarr_chunk_comparison.py \
  --summary "$SUMMARY" \
  --output-dir "$FIGURE_DIR" \
  --latex-dir "$TEX_FIGURE_DIR" \
  --prefix zarr-chunk-fast \
  --metrics internal,peak
