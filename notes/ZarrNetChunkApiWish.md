# Zarr.NET chunk API wish

## Motivation

StackProcessing needs a chunk-native Zarr path for larger-than-memory algorithms. This is not a custom storage layout: Zarr is already fundamentally a chunked array format, where metadata defines a chunk grid and each chunk is stored under a chunk key.

The present Zarr.NET region API is convenient and general:

```csharp
ReadRegionAsync(start, end, ...)
WriteRegionAsync(start, end, data, ...)
```

For full chunk operations, however, this introduces overhead that is unnecessary when the caller already wants to process one complete chunk at a time. In StackProcessing benchmarks, direct chunk-region copy and threshold have excellent memory behaviour, but they still pay public region API overhead once per chunk:

- region validation,
- region shape calculation,
- chunk-coordinate enumeration,
- partial region copy/scatter logic,
- partial chunk read-modify-write checks,
- `PixelRegion` / `RegionResult` wrapping,
- task/semaphore overhead around operations that are already chunk-granular.

For non-neighbourhood operations such as copy, threshold, scalar arithmetic, and many pointwise transforms, the natural LMIP path is:

```text
Zarr file -> decoded chunk bytes -> 1D/SIMD operation -> decoded chunk bytes -> Zarr file
```

For pure copy where source and target metadata/codecs match, an even lower path is possible:

```text
encoded chunk bytes -> encoded chunk bytes
```

This mirrors the way Dask naturally schedules work at chunk granularity.

## Proposed public API shape

The exact names are flexible, but the important part is to expose native chunk references and full-chunk read/write operations.

```csharp
public readonly record struct ZarrChunkRef(
    long[] ChunkCoord,
    long[] Origin,
    long[] Shape,
    string Key);
```

Possible array-level API:

```csharp
IAsyncEnumerable<ZarrChunkRef> EnumerateChunksAsync(
    CancellationToken ct = default);

IAsyncEnumerable<ZarrChunkRef> EnumerateChunksAsync(
    long[] regionStart,
    long[] regionEnd,
    CancellationToken ct = default);

Task<byte[]> ReadChunkDecodedAsync(
    ZarrChunkRef chunk,
    CancellationToken ct = default);

Task WriteChunkDecodedAsync(
    ZarrChunkRef chunk,
    byte[] decodedData,
    CancellationToken ct = default);

Task<byte[]?> ReadChunkEncodedAsync(
    ZarrChunkRef chunk,
    CancellationToken ct = default);

Task WriteChunkEncodedAsync(
    ZarrChunkRef chunk,
    byte[] encodedData,
    CancellationToken ct = default);
```

The decoded API should return the full logical chunk buffer in array order. Edge chunks may either:

- return padded full chunk buffers, matching the current internal behaviour, or
- return a truncated valid shape with `chunk.Shape`.

For StackProcessing, either is usable as long as `Shape` and layout are explicit.

## Useful guarantees

For full-chunk algorithms, callers benefit from these guarantees:

- `ReadChunkDecodedAsync` does not allocate or copy through a region output buffer.
- `WriteChunkDecodedAsync` does not read-modify-write when the supplied data covers the whole chunk.
- Encoded chunk copy can bypass decode/encode when source and destination array metadata/codecs are compatible.
- Chunk coordinates and keys are stable enough to build diagnostics and benchmark reports.
- The API works for `uint8`, `uint16`, `float32`, and later complex types supported by Zarr metadata.

## StackProcessing use cases

### Copy

If source and destination chunks/codecs match:

```text
for chunk in source.EnumerateChunksAsync():
    bytes = source.ReadChunkEncodedAsync(chunk)
    destination.WriteChunkEncodedAsync(chunk, bytes)
```

If metadata differs but chunk shapes match:

```text
decoded = source.ReadChunkDecodedAsync(chunk)
destination.WriteChunkDecodedAsync(chunk, decoded)
```

### Threshold

For `uint8`, a fast path can operate directly on `byte[]` using `Vector<byte>`:

```text
decoded = source.ReadChunkDecodedAsync(chunk)
output = threshold_uint8_simd(decoded)
destination.WriteChunkDecodedAsync(chunk, output)
```

For `uint16` and `float32`, StackProcessing can use:

```csharp
MemoryMarshal.Cast<byte, ushort>(decoded)
MemoryMarshal.Cast<byte, float>(decoded)
```

and later add SIMD paths for those types.

### Neighbourhood operations

Neighbourhood operations still need halo handling. A chunk iterator remains useful, but the caller must request neighbouring chunks or use a region/overlap helper. This is a separate layer from the full-chunk pointwise API.

### FFT and affine resampling

Chunk-native APIs also support the planned chunked separable FFT and affine resampling paths, where chunks are saved, revisited along another direction, or transposed without requiring the whole image in memory.

## Why this is upstream-friendly

This is not a StackProcessing-specific modification of Zarr semantics. It exposes the native abstraction of the Zarr format: chunks. Region APIs remain the ergonomic high-level interface; chunk APIs provide the low-level building block needed by larger-than-memory and distributed array systems.

The benefit should also apply outside StackProcessing:

- fast copy between compatible arrays,
- chunk-local pointwise transforms,
- batch validation/checksum tools,
- custom codecs or diagnostics,
- lower-overhead pipelines for cloud/object stores,
- better parity with Dask-style chunk scheduling.

## Benchmark target

The current target comparison is:

```text
Python/Dask-Zarr:
  da.from_zarr(...).map_blocks(...).to_zarr(...)

StackProcessing/Zarr.NET desired:
  EnumerateChunksAsync -> ReadChunkDecodedAsync -> 1D/SIMD op -> WriteChunkDecodedAsync
```

For `1024^3` threshold, the chunk-region API already gives good memory behaviour, but speed is still limited by per-region overhead and missing SIMD for wider types. A direct chunk API should reduce that overhead while keeping the memory bound close to one or a few chunks.
