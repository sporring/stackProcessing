# StackProcessing DLL rerun restart

StackProcessing benchmark commands now execute the already-built benchmark DLL:

```sh
dotnet benchmarks/StackProcessing.Benchmarks/bin/Debug/net10.0/StackProcessing.Benchmarks.dll run ...
```

This avoids the SDK/project-runner overhead from `dotnet run --no-build --project`.

The current full StackProcessing rerun writes to a separate file so the existing
comparative `raw.csv` remains intact until the rerun is complete:

```text
benchmarks/results/raw.stackprocessing-dll-rerun.csv
benchmarks/results/summary.stackprocessing-dll-rerun.csv
```

If the run is interrupted, resume it with:

```sh
caffeinate -dimsu bash benchmarks/run_all.sh \
  --skip-builds \
  --skip-existing \
  --repeat 3 \
  --backends stackprocessing \
  --results benchmarks/results/raw.stackprocessing-dll-rerun.csv \
  --summary benchmarks/results/summary.stackprocessing-dll-rerun.csv
```

The `--skip-existing` flag skips successful rows already present in the rerun
CSV for the same backend, operation, pixel type, shape, parameter, and repeat.

After the StackProcessing DLL rerun is complete, replace old StackProcessing
rows in `benchmarks/results/raw.csv` with rows from
`benchmarks/results/raw.stackprocessing-dll-rerun.csv`:

```sh
python3 benchmarks/tools/replace_raw_rows.py \
  --input benchmarks/results/raw.csv \
  --replacement benchmarks/results/raw.stackprocessing-dll-rerun.csv \
  --output benchmarks/results/raw.csv \
  --backup benchmarks/results/raw.before-stackprocessing-dll-replacement.csv \
  --drop-backend stackprocessing
```

Then regenerate `summary.csv` and figures.

## Remaining Python/skimage reruns

Two Python/scikit-image/SciPy rows still have wrapper overhead above 3 seconds.
They are long median cases, so rerun them after the StackProcessing DLL rerun:

```sh
PATH="$PWD/.venv-benchmarks/bin:$PATH" bash benchmarks/run_all.sh \
  --skip-builds \
  --repeat 3 \
  --repeat-start 2 \
  --repeat-end 2 \
  --backends python-skimage-scipy \
  --operations median \
  --pixel-types uint16 \
  --shapes 1024x1024x1024 \
  --parameters radius=3

PATH="$PWD/.venv-benchmarks/bin:$PATH" bash benchmarks/run_all.sh \
  --skip-builds \
  --repeat 3 \
  --repeat-start 3 \
  --repeat-end 3 \
  --backends python-skimage-scipy \
  --operations median \
  --pixel-types uint16 \
  --shapes 1024x1024x1024 \
  --parameters radius=3
```

Each Python/skimage median rerun should replace the corresponding existing row
in `benchmarks/results/raw.csv` first if we want to avoid duplicate repeats.
