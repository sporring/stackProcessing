# ArrayPool, Zarr, SIMD, and Chunk restart notes

These notes summarize the lessons learned before returning to the branch where
`Image` may become ArrayPool-backed instead of SimpleITK-backed. They are meant
as a restart checklist rather than as a polished design document.

## Current evidence

- Direct OME-Zarr copy and threshold are now chunk-native. The fast path is:
  `Zarr.NET decoded chunk bytes -> Chunk<byte> -> optional typed span/SIMD view -> Zarr.NET decoded chunk write`.
- The report-facing Zarr benchmark now uses uncompressed Zarr v3 chunks
  (`bytes` codec only) to stress disk IO and chunk/scaffold overhead. A separate
  codec-aligned `zstd(level=0)` run remains useful, because compressible ramp
  data can make compressed writes faster than raw writes.
- `Chunk<>` is not the factor-2 slowdown. A tight raw no-`Chunk<>` threshold
  path and the logical-byte-length `Chunk<>` path are within noise of each
  other. The important fix was avoiding extra copies and respecting logical
  payload length when buffers come from `ArrayPool<byte>`.
- For 1024^3 uncompressed copy/threshold, StackProcessing-Zarr is faster and
  lower-memory than Python/Dask-Zarr, but the runtime gap narrows for large
  float32 thresholding because raw IO dominates.
- Median filtering remains the meaningful neighbourhood comparison. Dask uses
  3D `map_overlap`; StackProcessing uses z-stream windows/slabs. The halo
  layout and task/chunk granularity dominate more than the file format alone.

## Chunk ownership

The current `StackCore.Chunk<'T>` stores:

```fsharp
type ChunkStorage<'T> =
    { Bytes: byte[]
      ByteLength: int
      Release: unit -> unit }
```

The distinction between physical array length and logical byte length is
essential. `ArrayPool<byte>.Rent(n)` may return a larger buffer. Hot paths should
use `Chunk.memory` or `Chunk.span<'T>` so only the logical payload is read or
written. Returning the underlying `byte[]` directly is only safe when its length
equals `ByteLength`.

Ownership belongs next to the buffer. A chunk stage that rents memory should
wrap the release action in the chunk and release it immediately after the write
or after the downstream consumer no longer needs it.

## SIMD and flat loops

For singleton pixel operations, use flat 1D buffers first:

- `System.Numerics.Vector<byte>` is excellent for `uint8 -> uint8` threshold
  masks with the 0/1 convention.
- `Vector<uint16>` and `Vector<float32>` are useful when the output type is
  natural for the lane type. If the operation must produce `bool[]` or pack
  masks, the win may disappear.
- `Span<T>` is mainly a zero-copy typed view and slicing tool. It does not
  automatically improve cache locality; locality comes from contiguous access
  and loop structure.
- `MethodImplOptions.AggressiveOptimization` and simple span rewrites were
  mostly neutral in the threshold tests. The real wins came from avoiding
  conversions and using portable SIMD where it naturally fits.
- Unsafe `NativePtr` loops can be faster than scalar managed loops, but they
  are not the first cross-platform path to adopt.

For neighbourhood operations:

- SimpleITK/ITK is still very strong, especially when it can use optimized and
  threaded C++ filters.
- Native .NET can compete for fixed, type-natural kernels if the stencil is
  unrolled and vectorized along contiguous x-runs.
- Median filtering is a different beast. Sparse/rolling histogram ideas are
  promising for larger discrete neighbourhoods, but the general native median
  path is not yet a replacement for tuned library implementations.

## ArrayPool image branch guidance

The safest migration path is incremental:

1. Keep `Image` semantics stable at the public boundary.
2. Move ownership into a small buffer-owner type as close as possible to the
   rented array.
3. Use flat 1D storage as the internal representation.
4. Add typed span views for hot loops.
5. Keep explicit `toITK`/`ofITK` wrappers for filters that should remain in
   SimpleITK.
6. Convert simple singleton operations first: copy, threshold, add/multiply
   scalar, simple casts, map/fold/iter over flat buffers.
7. Preserve SimpleITK for complex neighbourhood filters until a native version
   is proven faster and memory-safe.

Avoid a branch-wide rewrite that merely replaces SimpleITK images with managed
arrays but still converts through SimpleITK for most operations. The earlier
experiments showed that such a path can improve memory but fail to deliver the
expected time win.

## Things to watch

- AsyncSeq lifetime: pooled buffers must not be returned before downstream
  consumers have finished. Use ownership wrappers and tests with poisoned
  returns.
- Power-of-two or oversized pool buffers: always track requested/logical length
  separately from physical array length.
- Hidden conversions: if an operation crosses ArrayPool -> SimpleITK -> ArrayPool
  in a hot loop, measure it explicitly. Conversion can dominate.
- Output type conventions: threshold outputs should use `uint8` 0/1 unless the
  surrounding API is explicitly changed to support true bool images.
- Dask comparison fairness: Dask should be allowed to use natural Dask/Zarr
  expressions. StackProcessing should likewise use natural DSL/chunk stages.
  Note concurrency differences in text rather than forcing both into an
  unnatural single-threaded shape.

## Suggested first tests after branch restart

- Rebase or port the current `Chunk<>` byte-backed/logical-length design.
- Re-run the focused threshold microbenchmark:
  typed array direct, byte-backed span, `Chunk<>`, and SimpleITK filter.
- Re-run Zarr direct copy/threshold on uncompressed 512^3 and 1024^3 only.
- Add a poison-on-return AsyncSeq test for pooled slice/window/slab lifetimes.
- Only then start replacing more `ImageFunctions` singleton operations.

