# Pending wrapper-overhead reruns

The previous cleanup removed wrapper-overhead outliers from `raw.csv`.
StackProcessing has now been rerun through the benchmark DLL in
`benchmarks/results/raw.stackprocessing-dll-rerun.csv` with 252 successful rows
and no missing StackProcessing cases. These Python/scikit-image/SciPy rows still
need to be rerun before regenerating final figures.

A first attempt on 2026-06-01 was stopped during repeat 2 before any row was
written, so restart both repeats from scratch in a separate file:

```sh
PATH="$PWD/.venv-benchmarks/bin:$PATH" bash benchmarks/run_all.sh \
  --skip-builds \
  --skip-inputs \
  --repeat 3 \
  --repeat-start 2 \
  --repeat-end 3 \
  --backends python-skimage-scipy \
  --operations median \
  --pixel-types uint16 \
  --shapes 1024x1024x1024 \
  --parameters radius=3 \
  --results benchmarks/results/raw.python-skimage-outlier-rerun.csv \
  --summary benchmarks/results/summary.python-skimage-outlier-rerun.csv
```

After both complete:

```sh
python3 benchmarks/tools/summarize_results.py \
  --input benchmarks/results/raw.csv \
  --output benchmarks/results/summary.csv

MPLCONFIGDIR=/private/tmp/matplotlib-cache \
  .venv-benchmarks/bin/python benchmarks/tools/plot_results.py \
  --input benchmarks/results/summary.csv \
  --output-dir benchmarks/results/figures

cp benchmarks/results/figures/*.pdf notes/LMIP_Optimiser_and_Studio/figures/
```

## In-memory threshold experiment

When the Python/skimage rerun has finished, pause the benchmark sequence before
starting the next backend reruns and run the standalone in-memory threshold
comparison. This isolates SimpleITK thresholding on an already materialized ITK
image from native F# loops over rented one-dimensional arrays, with both direct
array indexing and `Span`-based loops. The 1024^3 cases are included as stress
tests and may fail with an out-of-memory row rather than stopping the whole run.

Build once:

```sh
dotnet build benchmarks/InMemoryThreshold.Benchmarks/InMemoryThreshold.Benchmarks.fsproj --nologo
```

Run from the compiled DLL:

```sh
dotnet benchmarks/InMemoryThreshold.Benchmarks/bin/Debug/net10.0/InMemoryThreshold.Benchmarks.dll \
  --output benchmarks/results/in-memory-threshold.csv \
  --shapes 128x128x128,256x256x256,1024x1024x1024 \
  --pixel-types UInt8,UInt16,Float32 \
  --repeat 3 \
  --threshold 128
```
