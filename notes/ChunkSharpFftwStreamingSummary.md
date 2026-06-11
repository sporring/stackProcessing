# Chunk FFT Streaming Notes

This note summarizes the current restart point for implementing a fast
streaming 3D FFT around `Chunk<>`, complex64 data, Zarr, and FFTW/SharpFFTW.

## Goal

Replace the older Image/SimpleITK-oriented FFT path with a chunk-native
pipeline that:

- reads scalar Float32 slices as `Chunk<float32>`;
- performs per-slice XY FFTs;
- writes an intermediate complex64 Zarr workspace;
- reads vertical x-y chunk columns from that workspace;
- performs the Z-direction FFT;
- writes final complex64 Zarr chunks;
- avoids full-volume materialization.

The preferred complex representation is complex64, stored as interleaved
`float32` real/imaginary pairs. Complex128 is supported by Zarr.net, but should
not be the default for memory-intensive StackProcessing workflows.

## Implemented Shape

The chunk-native FFT experiment currently has three benchmark entry points in
`benchmarks/StackProcessing.Benchmarks/Program.fs`:

- `run-chunk-fft-xy-float32-zarr`
- `run-chunk-fft-z-complex64-zarr`
- `run-chunk-fft-native-float32-zarr`

The intended full pipeline is:

```text
TIFF Float32 slices
  -> readChunkSlice
  -> ChunkFunctions.fftXYFloat32ToComplex64InterleavedParallelCollect workers
  -> writeZarrComplex64InterleavedFloat32, uncompressed, chunkZ = chunkSize
  -> fftZComplex64InterleavedZarrTiles
  -> final uncompressed complex64 Zarr
```

The XY stage emits `Chunk<float32>` where each logical complex64 pixel is stored
as two adjacent `float32` values. Thus an input slice of logical size
`width x height x 1` becomes an interleaved output chunk of size
`(2 * width) x height x 1`.

The Z stage reads x-y tiles across the whole depth, e.g. for a 512^3 image with
128^3 Zarr chunks it processes 128 x 128 x 512 complex columns at a time, then
writes final 128^3 complex64 chunks. We explicitly do not want to collect the
whole image in memory.

## SharpFFTW Versus Native Wrapper

The older Image path in `StackImageFunctions.fs` uses `SharpFFTW` and still
round-trips through `Image<>`/array-shaped representations.

The newer chunk-native experiment does not yet use SharpFFTW directly in the
hot path. It calls a thin native wrapper in `libspnth.dylib` via `NativeSp`:

- `fftwfComplexXYInplace`
- `fftwfComplexZInplace`

This was chosen because it gives direct pinned-buffer access to the interleaved
`float32` chunk memory and avoids `Image<ComplexFloat32>`/`Array2D` conversion.
For the restart, one open choice is whether to keep this thin C wrapper or
rework the same zero-copy shape through SharpFFTW if SharpFFTW exposes enough
plan/buffer control.

## Important Fixes Already Made

### Avoided Image and Array2D in the Chunk Path

The first chunk-native XY path avoided:

- `Image<>`;
- `Array2D`;
- `ComplexFloat32[,]`;
- image-to-array-to-image round-tripping.

The hot path is now essentially:

```text
Chunk<float32> input span
  -> rented/pinned interleaved float32 output buffer
  -> FFTW complex in-place transform
  -> Chunk<float32> output
```

### Wrote XY Scratch in Real 3D Zarr Chunks

An early version wrote one z-plane at a time into Zarr. That risks repeatedly
touching the same 3D Zarr chunk and is the wrong IO shape.

The writer was changed to collect `chunkZ` transformed XY slices and write one
full complex64 Zarr slab/chunk region at a time. For example:

```text
chunk shape: [1, 1, 128, 128, 128]
```

rather than a pile of plane-shaped writes.

This was a major structural improvement.

### FFTW Planner Thread Safety

Parallel XY FFT initially crashed for larger runs with an FFTW planner
assertion. FFTW plan creation is not thread-safe, so native plan creation was
guarded with a mutex.

This fixed the crash, but it also exposed that per-slice plan creation is still
architecturally untidy.

## Measurements and Lessons

### 256^3, chunk size 64

After the slab-writing fix:

- XY-only, 1 worker: about `0.69 s` internal.
- XY-only, 3 workers: about `0.61 s` internal.
- Full chunk FFT, 1 worker: about `1.48 s` internal.
- Full chunk FFT, 3 workers: about `1.15 s` internal.

The full 256^3 result looked promising.

### 512^3, chunk size 64

This chunk size was too small for the full workflow:

- C++/ITK full FFT/Zarr write: about `3.34 s` internal.
- SP chunk FFT, 1 worker: about `9.36 s`.
- SP chunk FFT, 3 workers: about `10.43 s`.
- XY-only improved with workers, but the full run got worse.

The likely reason is IO/buffer pressure and too many Zarr chunks. Parallelizing
XY helps compute, but the rest of the pipeline becomes the bottleneck.

### 512^3, chunk size 128

Using 128^3 chunks was much more reasonable:

- C++/ITK full FFT/Zarr write: about `3.33 s`.
- SP chunk FFT, 1 worker: about `8.92 s`.
- SP chunk FFT, 3 workers: about `6.79 s`.

This suggests that larger chunks reduce Zarr overhead enough that XY
parallelism matters again. Memory pressure rises, but still remains far below
the full-volume ITK approach.

### Z Parallelization

The Z phase is deliberately not parallelized for now. It reads large vertical
tiles, so memory pressure grows quickly with worker count. The current sensible
default is:

- parallelize XY slices;
- keep Z serial;
- tune tile/chunk size first.

### FFTW Planning Cost

A small native probe suggested `FFTW_ESTIMATE` planning cost is tiny compared
with execution:

- planning per 256 x 256 slice was on the order of tens of microseconds;
- the main cost is not plan creation in Estimate mode.

Plan reuse is still cleaner, and may matter for non-Estimate planning, but it
does not explain the current multi-second gap to C++/ITK.

## Suspicious Zarr Read/Write Baseline

A same-grid Zarr copy benchmark for complex64 512^3 with 128^3 chunks measured
around `8.2 s`. That cannot be treated as a disk lower bound, because C++/ITK
does TIFF read + full 3D FFT + complex64 Zarr write in about `3.3 s`.

The same-grid copy path currently uses Zarr.NET abstractions:

```text
ReadChunkEncodedAsync -> byte[] allocation
WriteChunkEncodedAsync -> store abstraction -> File.WriteAllBytesAsync
```

This is not the same as a raw local file copy. It may include avoidable managed
allocations, store overhead, and debug read-back code in `LocalFileSystemStore`.

The conclusion is that the Zarr.NET copy benchmark is measuring the current
managed Zarr path, not the physical IO limit.

## Current Hypotheses

The remaining deficit versus C++/ITK is probably a mix of:

1. Zarr.NET write/copy overhead for large uncompressed chunks.
2. Extra managed buffer allocation and copying around encoded chunks.
3. Intermediate scratch IO, which C++/ITK does not need because it holds the
   full volume in memory.
4. Less mature FFTW plan/buffer reuse in the chunk-native path.
5. Possible loss from writing/reading the complex64 scratch volume in the
   middle of the transform.

The two-pass design is still the correct streaming shape. The question is how
much constant factor we can remove.

## Next Experiments

### Establish a True IO Lower Bound

Add a raw local Zarr chunk-file copy baseline:

- copy metadata and `0/c/...` chunk files directly;
- no Zarr.NET chunk decode/encode;
- no `ReadChunkEncodedAsync`/`WriteChunkEncodedAsync`;
- no debug read-back;
- measure 512^3 complex64, 128^3 chunks.

If that lands near the C++/ITK write time, Zarr.NET needs a faster raw encoded
chunk path.

### Optimize Zarr.NET Encoded Chunk Writes

Likely useful upstream changes:

- remove or compile-gate debug read-back in `LocalFileSystemStore.WriteAsync`;
- expose `WriteAsync(string key, ReadOnlyMemory<byte>)` to avoid forcing exact
  arrays;
- add a same-store/same-grid encoded chunk copy fast path;
- avoid extra `ToArray` when the caller already owns a complete `byte[]`;
- consider direct `FileStream` writes for large chunk payloads.

### Decide SharpFFTW Integration Shape

For the next focused session, decide whether the Chunk FFT should:

- continue using the thin native `fftwf_*` wrapper, or
- move to SharpFFTW while preserving the same interleaved chunk-buffer layout.

The deciding criterion should be whether SharpFFTW can expose reusable plans
over pinned or native-aligned buffers without forcing array-shaped copies.

### Keep Chunk Size as a First-Class Parameter

For FFT, 64^3 chunks were too small for 512^3. 128^3 looked better. Future
benchmark grids should include chunk size explicitly because it strongly
affects both IO overhead and memory pressure.

### Keep Z Serial Until IO Is Understood

Do not parallelize the Z phase yet. It is memory-heavy, and current timings are
already dominated enough by IO/buffer movement that adding Z workers would
confuse the diagnosis.

## Restart Summary

The chunk-native streaming FFT is structurally in the right shape:

```text
slice-local XY FFT -> chunked complex64 scratch -> tile-column Z FFT -> final Zarr
```

The best current benchmark is roughly:

```text
512^3, chunk 128^3:
C++/ITK:     ~3.3 s, high memory
SP chunk w1: ~8.9 s, lower memory
SP chunk w3: ~6.8 s, lower memory
```

The main open problem is not whether the two-pass algorithm is conceptually
valid. It is reducing the constant factors in scratch Zarr IO and buffer
movement, while preserving the low-memory streaming design.
