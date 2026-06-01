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

For singleton and higher ladder families, make sure the representative user-facing TIFF types are well covered:

- `UInt8`: masks, binary images, labels, and 8-bit grayscale.
- `UInt16`: common microscopy and synchrotron grayscale TIFF data.
- `Int32`: representative signed-integer data and label-like intermediate values.
- `Float32`: high-end microscopy/scientific TIFF data and compact floating-point processing.

Do not restrict StackProcessing itself to these types. The point is calibration emphasis: Probe should strongly cover the common large-image cases while retaining broader type support where the library exposes it. `Float64` remains useful as an explicit high-precision anchor, but it should not accidentally stand in for the normal floating TIFF workflow. Complex Float32 is represented separately as `ComplexFloat32` so FFT-like workflows can model the 8-byte/pixel complex path instead of always widening to `System.Numerics.Complex`/complex Float64.

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

Build Probe once before timing-oriented calibration runs, then execute the built DLL directly:

```bash
dotnet build src/StackProcessing.Probe/StackProcessing.Probe.fsproj --nologo
```

```bash
dotnet src/StackProcessing.Probe/bin/Debug/net10.0/StackProcessing.Probe.dll <command> [args]
```

Avoid `dotnet run` for cost-model evidence. Even with `--no-build`, it still goes through the SDK/project runner and can add unpredictable fixed overhead to short probe rows. That overhead is not part of StackProcessing's execution model and should not be fitted as operator cost. The same rule applies to Studio-generated pipelines once compiled: run the built artifact that represents the workflow, not a project-runner convenience command.

Use `-j 1` for timing runs. Parallel probe graphs compete for CPU, memory bandwidth, disk IO, and SimpleITK worker threads, which makes the evidence noisier.

Prefer the current larger shapes:

```text
256x256x256,512x512x128,1024x1024x64
```

Avoid small `64x64x64` measurements for model fitting unless a specific debugging task needs them. They are often dominated by fixed overhead and can pollute operator fits.

On macOS, use `caffeinate` for long collection runs:

```bash
caffeinate -dimsu dotnet src/StackProcessing.Probe/bin/Debug/net10.0/StackProcessing.Probe.dll \
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
caffeinate -dimsu dotnet src/StackProcessing.Probe/bin/Debug/net10.0/StackProcessing.Probe.dll \
  collect --family io \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --noisy-type Float32 --repeat 6 -j 1

dotnet src/StackProcessing.Probe/bin/Debug/net10.0/StackProcessing.Probe.dll \
  fit --up-to io \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --model-output models/fitted/stackprocessing.operator-cost.json

dotnet src/StackProcessing.Probe/bin/Debug/net10.0/StackProcessing.Probe.dll \
  inspect --max-step io --min-repeats 6 \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --suggest tmp/inspect/io-request.json
```

Then move up one family at a time:

```bash
caffeinate -dimsu dotnet src/StackProcessing.Probe/bin/Debug/net10.0/StackProcessing.Probe.dll \
  collect --family io-cast \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --noisy-type Float32 --repeat 6 -j 1

dotnet src/StackProcessing.Probe/bin/Debug/net10.0/StackProcessing.Probe.dll \
  fit --up-to io-cast \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --model-output models/fitted/stackprocessing.operator-cost.json

dotnet src/StackProcessing.Probe/bin/Debug/net10.0/StackProcessing.Probe.dll \
  inspect --max-step io-cast --min-repeats 6 \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --suggest tmp/inspect/io-cast-request.json
```

For `sources`:

```bash
caffeinate -dimsu dotnet src/StackProcessing.Probe/bin/Debug/net10.0/StackProcessing.Probe.dll \
  collect --family sources \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --noisy-type Float32 --repeat 6 -j 1

dotnet src/StackProcessing.Probe/bin/Debug/net10.0/StackProcessing.Probe.dll \
  fit --up-to sources \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --model-output models/fitted/stackprocessing.operator-cost.json

dotnet src/StackProcessing.Probe/bin/Debug/net10.0/StackProcessing.Probe.dll \
  inspect --max-step sources --min-repeats 6 \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --suggest tmp/inspect/sources-request.json
```

For `singleton`:

```bash
caffeinate -dimsu dotnet src/StackProcessing.Probe/bin/Debug/net10.0/StackProcessing.Probe.dll \
  collect --family singleton \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --noisy-type Float32 --repeat 6 -j 1

dotnet src/StackProcessing.Probe/bin/Debug/net10.0/StackProcessing.Probe.dll \
  fit --up-to singleton \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --model-output models/fitted/stackprocessing.operator-cost.json

dotnet src/StackProcessing.Probe/bin/Debug/net10.0/StackProcessing.Probe.dll \
  inspect --max-step singleton --min-repeats 6 \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --suggest tmp/inspect/singleton-request.json
```

The same pattern continues for later families.

## Standalone Ladder-Climb Protocol

When calibrating a new computing environment, use a self-contained ladder climb. Each ladder family is treated as a local convergence problem:

```text
for family in ladder:
    collect initial family evidence
    repeat:
        fit up to family
        inspect up to family and write request
        if inspect says sufficient:
            accept family and move on
        if request repeats without improvement:
            mark plateau and decide whether to accept or diagnose model terms
        collect request
    cleanup generated scratch
```

The key idea is that one family should be made locally stable before moving upward. Higher families can otherwise push cost back into lower terms and make interpretation ambiguous.

Probe implements this protocol as the `climb` command:

```bash
dotnet src/StackProcessing.Probe/bin/Debug/net10.0/StackProcessing.Probe.dll \
  --log tmp/climb/climb.log \
  climb --through singleton \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --repeat 6 --min-repeats 6 --max-request-rounds 3 -j 1
```

`--log PATH` is a global Probe option, so it can be used with any command. It tees both stdout and stderr to the file while still printing to the console. For long macOS climbs, combine it with `caffeinate`:

```bash
caffeinate -dimsu dotnet src/StackProcessing.Probe/bin/Debug/net10.0/StackProcessing.Probe.dll \
  --log tmp/climb/climb.log \
  climb --through singleton \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --repeat 6 --min-repeats 6 --max-request-rounds 3 -j 1
```

By default, `climb` starts at `io`, walks the implicit ladder, performs one initial `collect --family`, then repeats `fit`, `inspect`, and `collect --request` until the step converges or plateaus. It cleans generated probe scratch between accepted or plateaued families while keeping the durable measurement store and fitted model.

Recommended ladder:

```text
io
io-cast
sources
singleton
neighbourhood
geometry
fourier
keypoints
dependency
reducers
```

Optional/experimental:

```text
window-slab
```

Keep `window-slab` explicit rather than part of the default climb unless the goal is to study slab/window scaffolding.

### Per-Step Template

For a family named `FAMILY` and a request file `tmp/inspect/FAMILY-request.json`:

```bash
caffeinate -dimsu dotnet src/StackProcessing.Probe/bin/Debug/net10.0/StackProcessing.Probe.dll \
  collect --family FAMILY \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --noisy-type Float32 --repeat 6 -j 1

dotnet src/StackProcessing.Probe/bin/Debug/net10.0/StackProcessing.Probe.dll \
  fit --up-to FAMILY \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --model-output models/fitted/stackprocessing.operator-cost.json

dotnet src/StackProcessing.Probe/bin/Debug/net10.0/StackProcessing.Probe.dll \
  inspect --max-step FAMILY --min-repeats 6 \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --suggest tmp/inspect/FAMILY-request.json
```

If `inspect` requests more evidence:

```bash
caffeinate -dimsu dotnet src/StackProcessing.Probe/bin/Debug/net10.0/StackProcessing.Probe.dll \
  collect --request tmp/inspect/FAMILY-request.json -j 1
```

Then rerun `fit` and `inspect`.

### Convergence Criteria

Accept a ladder step when:

- coverage is sufficient through the selected scope,
- every family through the current step has `minRepeats >= 6`,
- `inspect` reports fit quality sufficient,
- flagged prediction ratio is below the active threshold,
- the request file is not asking for repeated collection of the same members,
- the improvement from the last request round is small but already within tolerance.

In practice, a good inspect line looks like:

```text
coverage looks sufficient through the selected ladder scope.
fit quality elapsedMilliseconds R2=...
fit quality flagged predictions=.../...
fit quality looks sufficient through the selected ladder scope.
```

### Plateau Criteria

Stop repeating a step aggressively when any of these happen:

- the same members are requested for two or three consecutive request rounds,
- R2 changes only in the third or fourth decimal place,
- flagged prediction count does not decrease meaningfully,
- the flagged rows concentrate in a lower family that was already accepted,
- the flagged rows are mostly old or tiny shapes outside the active shape scope,
- collected record count grows but evidence-row count and fit quality barely move.

Suggested hard limit:

```text
initial family collect + at most 3 request rounds
```

For a clean machine-to-machine comparison, prefer this aggressive cap. If a family has not converged after three targeted request rounds, treat it as a modelling question rather than a measurement-volume question.

### Plateau Actions

When a step plateaus:

1. Re-run `fit` and `inspect` with explicit `--shapes`.
2. Inspect whether the request is stuck on old shapes or old graph names.
3. Try `--fixed-through` one level below the current family to see whether the strain is moving down the ladder.
4. Check whether the flagged rows share a missing model factor, such as file format, pixel type, cast direction, write-vs-ignore, coordinate axis, or window radius.
5. If the model explanation is clear, update the model terms before collecting more.
6. If the model explanation is not clear but the fit is usable, mark the step accepted-with-warning and continue.

Example diagnostic freeze:

```bash
dotnet src/StackProcessing.Probe/bin/Debug/net10.0/StackProcessing.Probe.dll \
  fit --up-to sources --fixed-through io-cast \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --model-output models/fitted/stackprocessing.operator-cost.json

dotnet src/StackProcessing.Probe/bin/Debug/net10.0/StackProcessing.Probe.dll \
  inspect --max-step sources --fixed-through io-cast --min-repeats 6 \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --suggest tmp/inspect/sources-request.json
```

### Cleanup Between Steps

Do not clean `tmp/` during a running `collect` command. Once `collect` has appended durable records and the following `fit`/`inspect` has completed, it is fine to clean generated scratch before moving to the next family:

```bash
rm -rf tmp/runJson_* tmp/probingGraphs tmp/probeInputs tmp/analysis
mkdir -p tmp/inspect
```

Do not remove:

```text
measurements/stackprocessing-probe.jsonl
models/fitted/stackprocessing.operator-cost.json
```

unless intentionally starting a new calibration cycle.

### Environment Comparison

For comparing machines, keep these fixed:

- same git commit,
- same SimpleITK version and native library,
- same .NET SDK/runtime,
- same shapes,
- same repeat count,
- same ladder order,
- same max request rounds,
- same active model terms,
- same optimizer-off probe execution,
- same `-j 1` timing policy.

Record environment metadata with the final model:

```text
machine name
CPU
RAM
storage type
operating system
.NET version
SimpleITK version
git commit
date
shape scope
repeat count
ladder families completed
families accepted with warning
```

This makes it possible to compare fitted coefficients across machines without confusing hardware differences with procedure differences.

## Request-Based Collection

If `inspect` reports that fit quality or coverage is weak, it writes a request JSON. Collect that request directly:

```bash
caffeinate -dimsu dotnet src/StackProcessing.Probe/bin/Debug/net10.0/StackProcessing.Probe.dll \
  collect --request tmp/inspect/singleton-request.json -j 1
```

Then immediately refit and reinspect the same scope:

```bash
dotnet src/StackProcessing.Probe/bin/Debug/net10.0/StackProcessing.Probe.dll \
  fit --up-to singleton \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --model-output models/fitted/stackprocessing.operator-cost.json

dotnet src/StackProcessing.Probe/bin/Debug/net10.0/StackProcessing.Probe.dll \
  inspect --max-step singleton --min-repeats 6 \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --suggest tmp/inspect/singleton-request.json
```

Request files may carry the active shape scope. If a request appears to revisit old or noisy shapes, regenerate it with explicit `--shapes`.

## Fitting With A Fixed Lower Ladder

When diagnosing whether a higher ladder step is pushing cost down into lower steps, use `--fixed-through`:

```bash
dotnet src/StackProcessing.Probe/bin/Debug/net10.0/StackProcessing.Probe.dll \
  fit --up-to sources --fixed-through io-cast \
  --shapes 256x256x256,512x512x128,1024x1024x64 \
  --model-output models/fitted/stackprocessing.operator-cost.json

dotnet src/StackProcessing.Probe/bin/Debug/net10.0/StackProcessing.Probe.dll \
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

dotnet src/StackProcessing.RunSamples/bin/Debug/net10.0/RunSamples.dll \
  --skip-build --repeat 1 -j 1 --debug-level 1 --cost-discrepancies \
  --cost-flags tmp/costDiscrepancies.csv \
  --cost-model models/fitted/stackprocessing.operator-cost.json --no-optimize
```

For an individual sample:

```bash
cd samples/someSample
dotnet bin/Debug/net10.0/someSample.dll -d 1 --cost-discrepancies \
  --cost-flags tmp/costDiscrepancies.csv \
  --cost-model models/fitted/stackprocessing.operator-cost.json \
  --no-optimize
```

Relative `--cost-flags` and `--cost-model` paths are resolved from the repository root when possible.

## Local Updates

For targeted local updates after discrepancy flags:

```bash
dotnet src/StackProcessing.Probe/bin/Debug/net10.0/StackProcessing.Probe.dll \
  local-update --shape 256x256x256 --repeat 3 -j 1
```

Or target specific operators:

```bash
dotnet src/StackProcessing.Probe/bin/Debug/net10.0/StackProcessing.Probe.dll \
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
dotnet src/StackProcessing.Probe/bin/Debug/net10.0/StackProcessing.Probe.dll \
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
