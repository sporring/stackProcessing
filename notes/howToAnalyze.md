# How to analyze

Probe calibration should measure unoptimized execution. Keep `-j 1` for timing
runs so graphs do not compete for CPU, memory bandwidth, or SimpleITK worker
threads.

## 1. Build calibration data

Probe clears `tmp/` by default, generates calibration input stacks, emits
controlled bottom-up calibration graphs, runs them with the optimizer off,
repeats the measurements, and writes the current estimates. The generated
inputs are a binary moving-box shape stack and a noisy gray-valued stack.

```bash
dotnet run --project src/StackProcessing.Probe/StackProcessing.Probe.fsproj -- bottom-up --size 128 --noisy-type Float32 --repeat 3 -j 1
```

For the first scale-model fitting run, gather the same layers at several cubic
image sizes:

```bash
dotnet run --project src/StackProcessing.Probe/StackProcessing.Probe.fsproj -- bottom-up --sizes 64,128,256 --noisy-type Float32 --repeat 3 -j 1
```

The fit treats the empty graph as the common intercept measurement and fixes
`Ignore` to zero cost. The starter layer is therefore the anchor for separating
startup/shutdown overhead from read, write, and later stage costs.
Use `--keep-tmp` only when deliberately adding measurements to an existing
working directory. The normal calibration run should start fresh.

## 2. Validate on sample workloads

After calibration, run the user-facing sample JSON graphs, also with the
optimizer off. These sample runs are validation data: they test whether the
learned coefficients predict real pipelines.

```bash
dotnet run --project src/StackProcessing.RunSamples/RunSamples.fsproj -- --json --repeat 3 -j 1 --optimize false
dotnet run --project src/StackProcessing.Probe/StackProcessing.Probe.fsproj -- calibrate --estimate-only
```

The second command refreshes `tmp/analysis/sampleEstimates.csv` using the
available measurements without emitting or running another probe batch. By
default it uses the latest `tmp/probingGraphs/bottomup_*` calibration root, so
older greedy `calibration_*` folders do not pollute the validation fit. Pass
`--probe-json-root PATH` to validate against a specific probe root.

To flag pipelines where the runtime model and measured cost disagree strongly,
run the graph with debug level 1 and cost discrepancy reporting:

```bash
dotnet run --project samples/someSample/someSample.fsproj -- -d 1 --cost-discrepancies --cost-model models/fitted/stackprocessing.operator-cost.json --no-optimize
```

If `--cost-model` is omitted, StackProcessing tries `STACKPROCESSING_COST_MODEL`,
then `~/.stackprocessing/cost/stackprocessing.operator-cost.json`, then
`models/fitted/stackprocessing.operator-cost.json`, and finally the repository
fallback in `models/default/`.

## Main outputs

```text
tmp/analysis/frozenCoefficients.csv
tmp/analysis/sampleEstimates.csv
tmp/analysis/greedyCoverage.csv
tmp/analysis/probeTargets.csv
tmp/analysis/probePlan.csv
tmp/analysis/diagnostics.csv
tmp/analysis/subsetDiagnostics.csv
tmp/analysis/matrix.csv
tmp/analysis/vectors.csv
tmp/analysis/costEvidence.csv
models/fitted/stackprocessing.operator-cost.json
models/default/stackprocessing.operator-cost.json
```

Do not clean `tmp/` between calibration and validation. The timestamped
`runJson_*` folders and `tmp/probeInputs` are the measurement evidence and
input data used by Probe.
