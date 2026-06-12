# Chunk Speedup Notes

This note records active performance guidance for the Chunk runtime.

## Current Priority

The main speedup target is to keep hot loops close to Chunk memory:

- operate over typed spans from `Chunk.span<'T>`
- keep conversion boundaries explicit
- prefer `float32` for storage and arithmetic-heavy image data when precision
  allows
- use native C++ wrappers for kernels where F# generic loops add measurable
  overhead
- use `ArrayPool<byte>` ownership through `Chunk.create`/`Chunk.decRef`

The preferred storage types are `uint8`, `uint16`, `float32`, and
complex64-interleaved `float32`.

## Affine Resampling

Affine resampling is Chunk-backed and correct enough for current samples, but
it remains a performance candidate. The next optimization pass should focus on:

1. **Specialized float32 trilinear sampling**

   The common path should avoid generic conversion callbacks in the inner loop.
   Seven interpolations per voxel are enough that small function-call overheads
   become visible.

2. **Cache-friendly chunk lookups**

   Resolve source chunk coordinates and neighbouring chunk references once for
   the output neighbourhood where possible, rather than repeating dictionary or
   layout work for every neighbour sample.

3. **Flat span indexing**

   Keep source and destination access as flat span indexing:

   ```fsharp
   pixels[(z * height + y) * width + x]
   ```

   This is the shape that maps cleanly to both managed loops and native helper
   routines.

4. **Coarse parallelism**

   Parallelize by output rows, slices, or chunk regions only after scalar access
   is already cheap. Small parallel jobs inside a per-pixel loop generally lose
   to scheduling overhead and memory-bandwidth contention.

## Native Wrappers

The current native wrapper strategy is intentionally narrow: use C++ where it
buys a direct speed or memory win, keep the public F# surface Chunk-shaped, and
avoid duplicating pipeline logic in native code.

Good native-wrapper candidates:

- single-axis convolution over primitive chunks
- vector-component convolution for structure tensor smoothing
- median nth-element filters
- signed distance band
- 2D resampling/Euler transform helpers
- FFT/InvFFT and future z-axis FFT work

Less attractive candidates:

- simple scalar maps where F# span loops are already memory-bandwidth bound
- reducers where worker-local F# accumulation is clear and fast
- orchestration logic such as source/sink composition, parallel collection, and
  resource release

## Chunk Layout

`Chunk<'T>` should remain an owned payload, not a storage-format object. Layout
metadata belongs in `ChunkLayout`, source metadata, or stage-specific context.

This keeps algorithms simple:

- local chunk functions see size, bytes, and typed spans
- IO stages know external tiling and file paths
- streaming stages know z-order, windows, and emit ranges
- optimizers can reason about payload memory separately from storage layout

## FFT

FFT is the main open performance experiment. The active shape is:

```text
Float32 Chunk slices
  -> XY FFT to complex64-interleaved Float32 chunks
  -> temporary complex Chunk storage when needed
  -> z-axis FFT pass
  -> complex64-interleaved output
```

The current public stage is XY-only and is marked as such. A full 3D FFT needs
the z-axis pass wired without whole-volume materialization. The intended design
is a two-pass chunked transform with explicit temporary chunk storage, not a
read-full-volume shortcut.

## Measurement Rules

- Measure realistic volumes; tiny shapes are dominated by fixed overhead.
- Separate IO, allocation, native execution, and pipeline overhead when
  possible.
- Run timing probes with controlled parallelism.
- Prefer benchmark evidence over intuition when choosing between F# spans and
  native helpers.
