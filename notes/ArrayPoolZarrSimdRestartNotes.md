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
- A first chunk-only TIFF pipeline now exists. BitMiracle/libtiff.net reads
  scanlines directly into StackProcessing-owned `ArrayPool<byte>` buffers,
  exposes them as `Chunk<'T>`, and returns the buffers through `Chunk.decRef`.
  The same chunk path can write TIFF slices back out.
- Chunk-native histogram reducers are now the first complete non-image pipeline:
  `readChunkSlices -> ChunkFunctions.histogram...Reducer -> compact histogram`.
  Dense and sparse variants both work serially and through the new native
  parallel reducer.
- Median filtering remains the meaningful neighbourhood comparison. Dask uses
  3D `map_overlap`; StackProcessing uses z-stream windows/slabs. The halo
  layout and task/chunk granularity dominate more than the file format alone.

## Chunk ownership

The current `StackCore.Chunk<'T>` stores:

```fsharp
type Chunk<'T when 'T: equality> =
    { Size: uint64 * uint64 * uint64
      Bytes: byte[]
      ByteLength: int
      Release: unit -> unit
      RefCount: int ref }
```

The distinction between physical array length and logical byte length is
essential. `ArrayPool<byte>.Rent(n)` may return a larger buffer. Hot paths should
use `Chunk.span<'T>` or byte spans over `Bytes[0..ByteLength)` so only the
logical payload is read or written. Returning the underlying `byte[]` directly
is only safe when its length equals `ByteLength`.

Ownership belongs next to the buffer. A chunk stage that rents memory should
wrap the release action in the chunk and release it immediately after the write
or after the downstream consumer no longer needs it. `Chunk.incRef` and
`Chunk.decRef` are intentionally explicit: `decRef` means the caller releases
its claim, not necessarily that the buffer is returned immediately.

## Chunk-native parallelization

The chunk histogram experiments introduced a small native parallel layer:

- `Stage.parallelReduce` splits a stream into non-overlapping windows,
  creates worker-local accumulators, folds each worker's share of the window,
  and merges the worker states into the final reducer state.
- `Stage.parallelMap` covers independent window maps.
- `Stage.parallelCollect` is the corresponding shape for neighbourhood-like
  operators that emit zero, one, or many outputs per input window.

The important design choice is that parallelism is expressed in StackProcessing
terms: window size is also the worker count; windows have explicit size,
stride, and padding; chunks have explicit
ownership; reducers define how to create, update, merge, finish, and release
items. This keeps `ArrayPool` lifetimes local and avoids shared mutable
structures in hot loops. For histograms, dense reducers use worker-local count
arrays and vector-friendly adders, while sparse reducers use worker-local
dictionaries and merge them afterwards rather than updating one concurrent
dictionary from all workers.

This is related to `AsyncSeq` parallel map helpers, but it is not only a wrapper
around them. `AsyncSeq.mapAsyncParallel` variants are useful for ordinary async
maps, especially when IO or independent latency dominates. The StackProcessing
parallel reducers need additional domain structure: deterministic window
geometry, bounded in-flight chunks, `Chunk.decRef` at the correct ownership
boundary, worker-local reducer states, and a final merge. Ordered versus
unordered output is also an algorithmic choice: unordered parallel maps are only
transparent for consumers that do not care about stream order, or for reducers
whose merge operation is commutative as well as associative.

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
