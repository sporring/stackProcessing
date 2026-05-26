# Cost Model Calibration

This note describes the developer workflow for collecting measurements, fitting the StackProcessing cost model, inspecting fit quality, and requesting targeted follow-up measurements.

For ordinary users, the root [README.md](../README.md) is enough. This note is for maintaining and improving the runtime cost model used by StackProcessing and the Optimiser.

## Goals

The cost model should estimate runtime and memory from user-visible pipeline structure:

- read and write format
- pixel type
- cast source and target type
- image size and shape
- local window/slab parameters
- operator family
- reducer behaviour
- IO and native-operation costs

The model is fitted from measured probe programs. The important design split is:

- `collect` gathers durable measurement evidence.
- `fit` selects evidence and writes a model.
- `inspect` checks coverage and fit quality, then optionally writes the next collection request.

This lets us recollect only weak or missing evidence instead of rerunning the whole probe ladder after every modelling change.

## Measurement Store

Raw measurements are appended to:

```text
measurements/stackprocessing-probe.jsonl
```

This file is intentionally ignored by git because it grows quickly.

Generated scratch data usually goes below:

```text
tmp/
```

Fitted models are written to:

```text
models/fitted/stackprocessing.operator-cost.json
```

Local overlays may be written to:

```text
models/local/stackprocessing.operator-cost.json
```

The repository fallback model lives in:

```text
models/default/stackprocessing.operator-cost.json
```

## General Measurement Rules

Use unoptimized execution for calibration. The point is to measure the cost of the implementation, not the cost after a changing optimizer decision.

Use `-j 1` for timing runs. Parallel probe graphs compete for CPU, memory bandwidth, disk IO, and SimpleITK worker threads, which makes the evidence noisier.

Prefer the current larger shapes:

```text
256x256x256,512x512x128,1024x1024x64
```

Avoid small `64x64x64` measurements for model fitting unless a specific debugging task needs them. They are often dominated by fixed overhead and can pollute operator fits.

On macOS, use `caffeinate` for long collection runs:

```bash
caffeinate -dimsu dotnet run --project src/StackProcessing.Probe/StackProcessing.Probe.fsproj -- \
  collect --family io \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --noisy-type Float32 --repeat 6 -j 1
```

## Clean Start

When starting a new fitting cycle, clean generated scratch files and archive existing fitted/local models.

```bash
rm -rf tmp/*
mkdir -p models/archive
mv models/fitted models/archive/fitted_$(date +%Y%m%d_%H%M%S) 2>/dev/null || true
mv models/local models/archive/local_$(date +%Y%m%d_%H%M%S) 2>/dev/null || true
```

If the measurement store itself is stale or intentionally being reset, move it aside rather than deleting it silently:

```bash
mkdir -p measurements/archive
mv measurements/stackprocessing-probe.jsonl measurements/archive/stackprocessing-probe_$(date +%Y%m%d_%H%M%S).jsonl 2>/dev/null || true
```

## Ladder Families

The current calibration ladder is:

```text
io -> io-cast -> sources -> singleton -> neighbourhood -> geometry -> fourier -> keypoints -> dependency -> reducers
```

There is also a `window-slab` family for measuring scaffolding such as window-to-slab, slab-to-window, and singleton-on-slab behaviour. It is useful for implementation experiments, but it is not part of the implicit `--up-to` fit ladder unless requested explicitly.

Typical families:

| Family | Purpose |
| ------ | ------- |
| `io` | Read/write behaviour by format, pixel type, and shape. |
| `io-cast` | Explicit and implicit read/cast combinations. |
| `sources` | Synthetic source stages such as zero, noise, and coordinate images. |
| `singleton` | Per-slice image operations such as cast, threshold, scalar ops, and intensity transforms. |
| `window-slab` | Window/slab scaffolding experiments. |
| `neighbourhood` | Local 3D operations requiring windows or slabs. |
| `reducers` | Stages that summarize streams into scalar/statistical outputs. |

## Standard Collect/Fit/Inspect Loop

For a ladder step, run:

```bash
caffeinate -dimsu dotnet run --project src/StackProcessing.Probe/StackProcessing.Probe.fsproj -- \
  collect --family io \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --noisy-type Float32 --repeat 6 -j 1

dotnet run --project src/StackProcessing.Probe/StackProcessing.Probe.fsproj -- \
  fit --up-to io \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --model-output models/fitted/stackprocessing.operator-cost.json

dotnet run --project src/StackProcessing.Probe/StackProcessing.Probe.fsproj -- \
  inspect --max-step io --min-repeats 6 \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --suggest tmp/inspect/io-request.json
```

Then move up one family at a time:

```bash
caffeinate -dimsu dotnet run --project src/StackProcessing.Probe/StackProcessing.Probe.fsproj -- \
  collect --family io-cast \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --noisy-type Float32 --repeat 6 -j 1

dotnet run --project src/StackProcessing.Probe/StackProcessing.Probe.fsproj -- \
  fit --up-to io-cast \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --model-output models/fitted/stackprocessing.operator-cost.json

dotnet run --project src/StackProcessing.Probe/StackProcessing.Probe.fsproj -- \
  inspect --max-step io-cast --min-repeats 6 \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --suggest tmp/inspect/io-cast-request.json
```

For `sources`:

```bash
caffeinate -dimsu dotnet run --project src/StackProcessing.Probe/StackProcessing.Probe.fsproj -- \
  collect --family sources \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --noisy-type Float32 --repeat 6 -j 1

dotnet run --project src/StackProcessing.Probe/StackProcessing.Probe.fsproj -- \
  fit --up-to sources \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --model-output models/fitted/stackprocessing.operator-cost.json

dotnet run --project src/StackProcessing.Probe/StackProcessing.Probe.fsproj -- \
  inspect --max-step sources --min-repeats 6 \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --suggest tmp/inspect/sources-request.json
```

For `singleton`:

```bash
caffeinate -dimsu dotnet run --project src/StackProcessing.Probe/StackProcessing.Probe.fsproj -- \
  collect --family singleton \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --noisy-type Float32 --repeat 6 -j 1

dotnet run --project src/StackProcessing.Probe/StackProcessing.Probe.fsproj -- \
  fit --up-to singleton \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --model-output models/fitted/stackprocessing.operator-cost.json

dotnet run --project src/StackProcessing.Probe/StackProcessing.Probe.fsproj -- \
  inspect --max-step singleton --min-repeats 6 \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --suggest tmp/inspect/singleton-request.json
```

The same pattern continues for later families.

## Request-Based Collection

If `inspect` reports that fit quality or coverage is weak, it writes a request JSON. Collect that request directly:

```bash
caffeinate -dimsu dotnet run --project src/StackProcessing.Probe/StackProcessing.Probe.fsproj -- \
  collect --request tmp/inspect/singleton-request.json -j 1
```

Then immediately refit and reinspect the same scope:

```bash
dotnet run --project src/StackProcessing.Probe/StackProcessing.Probe.fsproj -- \
  fit --up-to singleton \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --model-output models/fitted/stackprocessing.operator-cost.json

dotnet run --project src/StackProcessing.Probe/StackProcessing.Probe.fsproj -- \
  inspect --max-step singleton --min-repeats 6 \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --suggest tmp/inspect/singleton-request.json
```

Request files may carry the active shape scope. If a request appears to revisit old or noisy shapes, regenerate it with explicit `--shapes`.

## Fitting With A Fixed Lower Ladder

When diagnosing whether a higher ladder step is pushing cost down into lower steps, use `--fixed-through`:

```bash
dotnet run --project src/StackProcessing.Probe/StackProcessing.Probe.fsproj -- \
  fit --up-to sources --fixed-through io-cast \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --model-output models/fitted/stackprocessing.operator-cost.json

dotnet run --project src/StackProcessing.Probe/StackProcessing.Probe.fsproj -- \
  inspect --max-step sources --fixed-through io-cast --min-repeats 6 \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --suggest tmp/inspect/sources-request.json
```

This is diagnostic. If the lower levels are not yet genuinely stable, freezing them can make the upper-level fit worse or hide model problems.

## Shape Filtering

Both `fit` and `inspect` should be given the same active shape scope used for the current ladder climb:

```bash
--shapes 256x256x256,512x512x128,1024x1024x64
```

This prevents old measurements from small or experimental shapes from contaminating the fit. In particular, old `64x64x64` singleton measurements can create noisy flagged rows because fixed overhead dominates such small runs.

## IO And IO-Cast Notes

The `io` family measures user-visible read/write behaviour by pixel type and file format.

Important modelling details:

- TIFF is measured for the pixel types StackProcessing exposes as TIFF-compatible.
- `.mha` covers the wider scalar set and complex images.
- SimpleITK writes are requested without compression by default so measurements match ordinary DSL behaviour.
- Read/write cost should be format-aware, for example `read-tiff-uint8` and `read-mha-uint8` should not be treated as the same entity.

The `io-cast` family compares:

```text
read<T>
read<diskT> --> cast<diskT,T>
```

Implicit read-cast and explicit cast can have different mechanics and timings. The model should preserve that distinction rather than blindly rewriting one into the other.

## Interpreting Inspect

`inspect` reports:

- coverage by ladder family
- record counts
- graph counts
- measurement counts
- minimum and median repeats
- fit quality, including R2 where available
- flagged prediction counts
- next collection suggestions

Good signs:

- coverage sufficient through selected scope
- min repeats at or above requested threshold
- R2 plausible for the current family
- flagged prediction ratio below the active threshold
- no repeated request for the exact same members after multiple rounds

Warning signs:

- flagged rows repeatedly concentrate in a lower ladder step after adding a higher family
- repeated request runs do not change R2 or flagged count
- `inspect` suggests old shapes that should be outside the active scope
- a family has enough repeats but poor fit quality
- the same tiny or unusually fast rows dominate discrepancy requests

When repeated collection does not improve the fit, suspect a missing model term before collecting indefinitely.

## Sample Validation

After the ladder looks plausible, validate on user-facing samples with discrepancy reporting:

```bash
rm -f tmp/costDiscrepancies.csv

dotnet run --project src/StackProcessing.RunSamples/RunSamples.fsproj -- \
  --skip-build --repeat 1 -j 1 --debug-level 1 --cost-discrepancies \
  --cost-flags tmp/costDiscrepancies.csv \
  --cost-model models/fitted/stackprocessing.operator-cost.json --no-optimize
```

For an individual sample:

```bash
cd samples/someSample
dotnet run -- -d 1 --cost-discrepancies \
  --cost-flags tmp/costDiscrepancies.csv \
  --cost-model models/fitted/stackprocessing.operator-cost.json \
  --no-optimize
```

Relative `--cost-flags` and `--cost-model` paths are resolved from the repository root when possible.

## Local Updates

For targeted local updates after discrepancy flags:

```bash
dotnet run --project src/StackProcessing.Probe/StackProcessing.Probe.fsproj -- \
  local-update --shape 256x256x256 --repeat 3 -j 1
```

Or target specific operators:

```bash
dotnet run --project src/StackProcessing.Probe/StackProcessing.Probe.fsproj -- \
  local-update --operators SmoothWMedian --shape 256x256x256 --repeat 3 -j 1
```

Local updates write an overlay model to:

```text
models/local/stackprocessing.operator-cost.json
```

Use this for focused correction, not as a replacement for a properly climbed ladder.

## Legacy Bottom-Up Commands

Older notes and scripts may refer to `bottom-up` and `calibrate --estimate-only`.

Example:

```bash
dotnet run --project src/StackProcessing.Probe/StackProcessing.Probe.fsproj -- \
  bottom-up --size 128 --noisy-type Float32 --repeat 3 -j 1
```

The current preferred workflow is `collect`, `fit`, and `inspect`. Use legacy commands only when testing compatibility or reproducing older measurements.

## Main Outputs

Important files include:

```text
measurements/stackprocessing-probe.jsonl
tmp/analysis/costEvidence.csv
tmp/analysis/diagnostics.csv
tmp/analysis/subsetDiagnostics.csv
tmp/analysis/matrix.csv
tmp/analysis/vectors.csv
tmp/analysis/sampleEstimates.csv
tmp/costDiscrepancies.csv
models/fitted/stackprocessing.operator-cost.json
models/local/stackprocessing.operator-cost.json
models/default/stackprocessing.operator-cost.json
```

Do not clean `tmp/` during a single `collect` or `local-update` run. The command may still need its generated `runJson_*` directories before appending durable records.

