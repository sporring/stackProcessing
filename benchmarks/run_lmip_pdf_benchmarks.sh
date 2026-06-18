#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash benchmarks/run_lmip_pdf_benchmarks.sh [options]

Rerun the report-facing benchmarks from scratch, rebuild summaries, generate
the LMIP paper figures, and copy the generated PDFs into:

  notes/LMIP_Optimiser_and_Studio/figures

Options:
  --repeat N              Repeats for the main TIFF matrix. Default: 3.
  --parallel-repeat N     Repeats for Chunk worker-sweep figures. Default: 6.
  --convolve-repeat N     Repeats for the Chunk convolve worker sweep. Default: 3.
  --dry-run               Print commands instead of running them.
  -h, --help              Show this help.

Environment overrides:
  PYTHON                  Python executable. Defaults to .venv-benchmarks/bin/python
                          when present, otherwise python3.
  MATLAB_EXE              MATLAB executable. Default: matlab.
  MAIN_BACKENDS           Default: stackprocessing,python-skimage-scipy,cpp-itk,matlab.
  DATE_STAMP              Output suffix. Default: current date as YYYYMMDD.
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

repeat=3
parallel_repeat=6
convolve_repeat=3
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repeat)
      repeat="$2"
      shift 2
      ;;
    --parallel-repeat)
      parallel_repeat="$2"
      shift 2
      ;;
    --convolve-repeat)
      convolve_repeat="$2"
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
      echo "run_lmip_pdf_benchmarks.sh: unknown option $1" >&2
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
matlab_exe="${MATLAB_EXE:-matlab}"
main_backends="${MAIN_BACKENDS:-stackprocessing,python-skimage-scipy,cpp-itk,matlab}"

stackprocessing_dll="benchmarks/StackProcessing.Benchmarks/bin/Release/net10.0/StackProcessing.Benchmarks.dll"
itk_exe="benchmarks/cpp-itk/build/benchmark_itk"
input_root="tmp/benchmarks/input"
output_root="tmp/benchmarks/output-lmip-${date_stamp}"
main_raw="benchmarks/results/raw.csv"
main_summary="benchmarks/results/summary.csv"
figure_dir="benchmarks/results/figures"
tex_figure_dir="notes/LMIP_Optimiser_and_Studio/figures"

histogram_raw="tmp/benchmarks/chunk-histogram-parallel-raw-${date_stamp}.csv"
histogram_summary="tmp/benchmarks/chunk-histogram-parallel-summary-${date_stamp}.csv"
dilate_raw="tmp/benchmarks/chunk-dilate-radius3-parallel-raw-${date_stamp}.csv"
dilate_summary="tmp/benchmarks/chunk-dilate-radius3-parallel-summary-${date_stamp}.csv"
convolve_raw="tmp/benchmarks/chunk-convolve-float32-k7-parallel-all-sizes-raw-${date_stamp}.csv"
convolve_summary="tmp/benchmarks/chunk-convolve-float32-k7-parallel-summary-${date_stamp}.csv"

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$dry_run" -eq 0 ]]; then
    "$@"
  fi
}

run_shell() {
  printf '+ %s\n' "$*"
  if [[ "$dry_run" -eq 0 ]]; then
    bash -c "$*"
  fi
}

copy_main_figures_to_tex() {
  run cp \
    "$figure_dir/runtime-by-size-uint8.pdf" \
    "$figure_dir/runtime-by-size-uint16.pdf" \
    "$figure_dir/runtime-by-size-float32.pdf" \
    "$figure_dir/runtime-by-complexity-uint8.pdf" \
    "$figure_dir/runtime-by-complexity-uint16.pdf" \
    "$figure_dir/runtime-by-complexity-float32.pdf" \
    "$figure_dir/memory-pressure-by-size-uint8.pdf" \
    "$figure_dir/memory-pressure-by-size-uint16.pdf" \
    "$figure_dir/memory-pressure-by-size-float32.pdf" \
    "$figure_dir/memory-pressure-by-complexity-uint8.pdf" \
    "$figure_dir/memory-pressure-by-complexity-uint16.pdf" \
    "$figure_dir/memory-pressure-by-complexity-float32.pdf" \
    "$figure_dir/runtime-vs-memory.pdf" \
    "$figure_dir/wrapper-overhead.pdf" \
    "$figure_dir/connected-components-window-policy.pdf" \
    "$tex_figure_dir/"
}

echo "Using Python: $PYTHON"
echo "Date stamp:   $date_stamp"
echo "Dry run:      $dry_run"

run mkdir -p "benchmarks/results" "$figure_dir" "$tex_figure_dir" "$output_root"

echo
echo "== Build native and managed benchmark binaries =="
run bash lowlevel/build.sh
run dotnet build benchmarks/StackProcessing.Benchmarks/StackProcessing.Benchmarks.fsproj -c Release --nologo --disable-build-servers -p:UseSharedCompilation=false
run bash benchmarks/native-libtiff-shim/build-unix.sh "$(dirname "$stackprocessing_dll")"
run cmake -S benchmarks/cpp-itk -B benchmarks/cpp-itk/build -DCMAKE_BUILD_TYPE=Release
run cmake --build benchmarks/cpp-itk/build --config Release

echo
echo "== Main TIFF-stack benchmark matrix =="
run rm -f "$main_raw" "$main_summary"
run bash benchmarks/run_all.sh \
  --repeat "$repeat" \
  --backends "$main_backends" \
  --force-inputs \
  --results "$main_raw" \
  --summary "$main_summary" \
  --output-root "$output_root" \
  --stackprocessing-dll "$stackprocessing_dll" \
  --itk-exe "$itk_exe" \
  --matlab-exe "$matlab_exe" \
  --skip-builds

echo
echo "== Main TIFF-stack figures =="
run "$PYTHON" benchmarks/tools/plot_results.py \
  --input "$main_summary" \
  --output-dir "$figure_dir"
copy_main_figures_to_tex

echo
echo "== Chunk histogram worker sweep =="
run "$PYTHON" benchmarks/tools/run_chunk_histogram_parallel.py \
  --output "$histogram_raw" \
  --input-root "$input_root" \
  --dll "$stackprocessing_dll" \
  --repeats "$parallel_repeat"

run "$PYTHON" benchmarks/tools/plot_chunk_histogram_parallel.py \
  --input "$histogram_raw" \
  --summary "$histogram_summary" \
  --output-dir "$tex_figure_dir"
run cp "$tex_figure_dir/chunk-histogram-parallel-runtime.pdf" "$tex_figure_dir/chunk-histogram-parallel-memory.pdf" "$figure_dir/"

echo
echo "== Chunk dilation worker sweep =="
run "$PYTHON" benchmarks/tools/run_chunk_dilate_parallel.py \
  --output "$dilate_raw" \
  --input-root "$input_root" \
  --output-root "tmp/benchmarks/output-chunk-dilate-parallel-${date_stamp}" \
  --dll "$stackprocessing_dll" \
  --repeats "$parallel_repeat" \
  --radius 3

run "$PYTHON" benchmarks/tools/plot_chunk_dilate_parallel.py \
  --input "$dilate_raw" \
  --summary "$dilate_summary" \
  --output-dir "$tex_figure_dir"
run cp "$tex_figure_dir/chunk-dilate-radius3-parallel-runtime.pdf" "$tex_figure_dir/chunk-dilate-radius3-parallel-memory.pdf" "$figure_dir/"

echo
echo "== Chunk Float32 k=7 convolve worker sweep =="
run rm -f "$convolve_raw" "$convolve_summary"
for repeat_index in $(seq 1 "$convolve_repeat"); do
  for size in 256 512 1024; do
    shape="${size}x${size}x${size}"
    input_dir="$input_root/Float32_${shape}"
    for workers in 1 2 3 4; do
      output_dir="tmp/benchmarks/output-chunk-convolve-parallel-${date_stamp}/Float32_${shape}_w${workers}_r$(printf '%02d' "$repeat_index")"
      run "$PYTHON" benchmarks/tools/measure.py \
        --output "$convolve_raw" \
        --backend "stackprocessing-chunk-convolve-w${workers}" \
        --operation convolve \
        --pixel-type Float32 \
        --shape "$shape" \
        --parameter "kernelSize7-workers${workers}" \
        --repeat-index "$repeat_index" \
        -- \
        dotnet "$stackprocessing_dll" \
          run-chunk-convolve \
          --pixel-type Float32 \
          --input "$input_dir" \
          --output "$output_dir" \
          --kernel-size 7 \
          --workers "$workers"
      run rm -rf "$output_dir"
    done
  done
done

run "$PYTHON" benchmarks/tools/plot_chunk_convolve_parallel.py \
  --input "$convolve_raw" \
  --summary "$convolve_summary" \
  --output-dir "$tex_figure_dir"
run cp "$tex_figure_dir/chunk-convolve-float32-1024-k7-parallel-runtime.pdf" "$tex_figure_dir/chunk-convolve-float32-1024-k7-parallel-memory.pdf" "$figure_dir/"

echo
echo "Done."
echo "Main raw:       $main_raw"
echo "Main summary:   $main_summary"
echo "Figures:        $figure_dir"
echo "TeX figures:    $tex_figure_dir"
echo "Zarr comparison: run benchmarks/run_zarr_chunk_comparison_fast.sh and, when mounted, benchmarks/run_zarr_chunk_comparison_slow.sh"
