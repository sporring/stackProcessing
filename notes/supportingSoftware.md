# Supporting Software

This note collects developer-facing support tooling for StackProcessing. The root `README.md` is intended for general users; build, test, calibration, and benchmark details live here and in the more focused notes linked below.

## Build

StackProcessing is a .NET 10 F# solution.

Build the full solution from the repository root:

```bash
dotnet build StackProcessing.sln
```

Build a specific project:

```bash
dotnet build src/StackProcessing/StackProcessing.fsproj
dotnet build src/Studio/Studio.fsproj
```

## Test

Run all tests:

```bash
dotnet test
```

Run specific test projects:

```bash
dotnet test tests/SlimPipeline.Tests/SlimPipeline.Tests.fsproj
dotnet test tests/StackProcessing.Tests/StackProcessing.Tests.fsproj
dotnet test tests/Studio.Tests/Studio.Tests.fsproj
```

The main test stack uses Expecto, YoloDev.Expecto.TestSdk, Microsoft.NET.Test.Sdk, and coverlet.

## Cost Measurement And Calibration

StackProcessing includes tools for measuring operation costs and fitting a runtime model. These tools are mainly for maintainers, optimizer work, and performance analysis.

The short version:

- `StackProcessing.RunSamples` runs sample workflows and can collect repeat timings.
- `StackProcessing.Probe` can collect controlled measurements, fit cost models, inspect fit quality, and request targeted additional measurements.
- raw measurement files can grow large and should stay out of git.

Example inspection command:

```bash
dotnet run --project src/StackProcessing.Probe/StackProcessing.Probe.fsproj -- \
  inspect --max-step io --min-repeats 3
```

For the current calibration workflow, ladder strategy, shape filtering, fixed-through fitting, and logging conventions, see:

- [CostModelCalibration.md](CostModelCalibration.md)
- [SlimPipeline.md](SlimPipeline.md)
- [StackProcessing.md](StackProcessing.md)
- [dsl-stage-graph-enrichment.md](dsl-stage-graph-enrichment.md)
- [ChunkPayload.md](ChunkPayload.md)
- [ChunkFftStreamingSummary.md](ChunkFftStreamingSummary.md)
- [NativeTiffLowLevelNotes.md](NativeTiffLowLevelNotes.md)

## Benchmarks

The benchmark side project compares read-process-write workflows across StackProcessing and external environments.

Start with:

```bash
source .venv-benchmarks/bin/activate
bash benchmarks/run_all.sh --repeat 3
```

See [../benchmarks/README.md](../benchmarks/README.md) for the current benchmark matrix, backend setup, timing columns, and fairness rules.

## Supporting Dependencies

StackProcessing builds on:

- FSharp.Control.AsyncSeq for asynchronous streams.
- Avalonia, NodeEditorAvalonia, PanAndZoom, and CommunityToolkit.Mvvm for Studio.
- Plotly.NET for charts and reports.
- PureHDF and ZarrNET for HDF5/NeXus and Zarr-style array storage.
- Expecto, YoloDev.Expecto.TestSdk, Microsoft.NET.Test.Sdk, and coverlet for tests.
- DIKU.Graph for graph algorithms used in the core.
- Low-level native helper libraries from `lowlevel/` for selected Chunk kernels.
