# Chunk Trimmed Publish Notes

Trimmed publish is shelved for now.

We tried two variants on macOS/arm64 with .NET 10:

- `StackProcessing.Benchmarks` with `PublishTrimmed=true` and `TrimMode=partial`.
- A small throwaway `Chunk.Core`-only probe under `tmp/benchmarks/ChunkCoreTrimmedProbe`.

Both trimmed publish attempts spent a long time after ordinary compilation in a
single `dotnet publish` process with low partial-core CPU use. The full
benchmark publish did not reach a useful artifact. The `Chunk.Core`-only probe
also stalled, and one attempt failed in ILLink's `ComputeManagedAssemblies`
task-host path.

The successful untrimmed self-contained `Chunk.Core` probe provides the useful
baseline:

- publish size: `86M`, 196 files
- `256^3`, 3 FFT iterations: about `0.707 s/iter`
- Chunk peak live storage: `192 MiB`
- process working set after the run: about `388 MiB`

The main expected memory pressure in the FFT path is data storage and FFTW work:
input chunks, interleaved complex work buffers, output chunks, and native FFT
plans. Trimming may reduce deployed file size, but it is unlikely to reduce the
dominant runtime memory in these experiments. Until the .NET trimming/linker
path is faster and more reliable here, it is not a practical optimization knob
for Chunk benchmarking.
