# Chunk FFT Streaming Notes

This note summarizes the Chunk FFT design and current optimization state.

## Goal

The FFT path should be a streaming Chunk pipeline that:

- reads scalar Float32 slices as `Chunk<float32>`;
- performs per-slice XY FFTs;
- stores complex64 data as interleaved `float32` real/imaginary pairs;
- uses temporary chunk storage when an axis reorganization is needed;
- performs the Z-direction FFT without full-volume materialization;
- writes complex64-interleaved output chunks.

Complex64 is the preferred representation. Complex128 is not the default for
larger-than-memory workflows because it doubles memory traffic and working-set
size.

## Representation

The XY stage emits `Chunk<float32>` where each logical complex64 pixel is stored
as two adjacent `float32` values. An input slice of logical size
`width x height x 1` becomes an interleaved output chunk of size
`(2 * width) x height x 1`.

Complex helpers use the same convention:

- real/imaginary composition
- polar composition
- real
- imaginary
- modulus
- argument
- conjugate

## Current Public Shape

The public FFT surface separates slice-local XY transforms from separable 3D
transforms:

- `fft` is the explicit XY-only stage.
- `fftRealXY` is the real-to-Hermitian-packed XY stage.
- `fft3DComplexXY` performs complex XY followed by complex Z over a batch.
- `fft3DRealXY` performs real XY followed by complex Z and emits spectral
  chunks with compact/Hermitian metadata.
- `invFft3DRealXY` performs inverse complex Z followed by complex-to-real XY
  and emits real `float32` chunks.

The separable 3D stages reuse native FFTW plan caches inside the batch. The
Zarr-backed Z passes and subchunked round trips are explicit workspace stages
for larger-than-memory experiments, not hidden behaviour inside a local image
filter.

## Full Pipeline Shape

```text
TIFF Float32 slices
  -> read
  -> real-to-complex FFT over XY slices
  -> temporary complex64 chunk storage
  -> z-axis FFT over chunk columns or reorganized tiles
  -> final complex64-interleaved chunks
```

The z pass must work by bounded chunk tiles or a transposed/chunk-reorganized
workspace. It must not read the full volume, transform it, and write it back.

## Native Wrapper

The active backend calls the `lowlevel` native helper through the F# low-level
wrapper:

- `fftwfComplexXYInplace`
- `fftwfComplexZInplace`

The wrapper gives direct pinned-buffer access to the interleaved `float32`
chunk memory. Plan creation is guarded because FFTW plan creation is not
thread-safe.

## Streaming Shape

XY FFT is slice-local and parallelizes naturally over Chunk slices.

Z FFT is global along z for each x-y complex column. The Chunk representation
does not make that dependency local; it makes the dependency explicit so the
implementation can choose a bounded temporary layout.

The expected shape is two-pass:

```text
XY pass
  -> temporary complex chunk workspace
  -> Z pass over bounded columns/tiles
```

## Performance Questions

The remaining optimization work is mostly constant-factor and layout work:

1. reduce temporary chunk IO overhead
2. reduce managed allocation around encoded chunk reads/writes
3. keep FFTW plan and buffer reuse on the hot paths
4. tune chunk/tile sizes for the Z pass
5. keep XY parallelism from overwhelming temporary storage bandwidth

## Measurement Rules

- Validate correctness on small examples first.
- Keep full 3D FFT benchmarks separate from local-window image filters.
- Compare chunk sizes explicitly; too-small chunks can make temporary storage
  overhead dominate.
- Measure raw temporary chunk copy costs separately from FFT execution.
