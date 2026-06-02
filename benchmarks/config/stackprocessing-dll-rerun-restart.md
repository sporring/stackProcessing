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

As of 2026-06-01, this rerun is complete. The separate CSV has 252 successful
StackProcessing rows, i.e. all expected rows for three repeats. The command
below is kept only as a restart recipe if the file is later cleaned or extended:

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

Next, replace old StackProcessing
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
They are long median cases. A first attempt on 2026-06-01 was stopped during
repeat 2 before any row was written to
`benchmarks/results/raw.python-skimage-outlier-rerun.csv`, so these should be
rerun from scratch. Prefer writing to a separate file first:

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

After both rows complete, replace the corresponding existing repeat-2 and
repeat-3 rows in `benchmarks/results/raw.csv` to avoid duplicate repeats.

## MATLAB R2026a `-nojvm` rerun

MATLAB has been updated locally to R2026a. To make the wrapper-overhead
measurement fairer, the MATLAB benchmark command is again generated as:

```text
matlab -nodisplay -nojvm -batch ...
```

The R2026a MATLAB rerun should write to a separate file first, leaving the old
MATLAB rows in `benchmarks/results/raw.csv` untouched until the new run is known
to complete:

```sh
caffeinate -dimsu bash benchmarks/run_all.sh \
  --skip-builds \
  --skip-existing \
  --repeat 3 \
  --backends matlab \
  --matlab-exe /Applications/MATLAB_R2026a.app/bin/matlab \
  --results benchmarks/results/raw.matlab-r2026a-nojvm.csv \
  --summary benchmarks/results/summary.matlab-r2026a-nojvm.csv
```

If R2026a still crashes with `-nojvm`, stop the remaining MATLAB cases, discard
the separate R2026a result rows, and update `notes/UPSTREAM_BUGS.md` with the
new version and crash behavior.
