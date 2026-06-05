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

## Hot path issues found during StackProcessing benchmarks

The first direct chunk API experiments in StackProcessing showed good memory
behaviour, but a later byte-backed `Chunk<T>` implementation unexpectedly lost
roughly a factor of two in copy and threshold benchmarks. A focused in-memory
microbenchmark isolated the native processing kernel:

```text
typed array -> threshold -> byte output
byte[] -> MemoryMarshal.Cast<byte,T> -> threshold -> byte output
```

For `uint16`, the byte-backed span path was essentially identical to typed
arrays. For `float32`, it was only about 7--8% slower. For `uint8` with
`Vector<byte>`, it was indistinguishable. This suggests that byte-backed chunk
storage and typed span views are not the cause of the larger benchmark
regression.

Inspection of the local Zarr.NET implementation pointed to three more likely
hot-path issues.

### 1. Decoded chunk writes always do a round-trip decode

In `ZarrArray.WriteChunkAsync`, the decoded write path encodes the supplied
chunk, then immediately decodes the encoded bytes again before writing:

```csharp
var encoded = await _pipeline.EncodeAsync(decodedData, ct).ConfigureAwait(false);

try
{
    var roundTrip = await _pipeline.DecodeAsync(encoded, ct).ConfigureAwait(false);
    // debug logging only
}
catch (Exception ex)
{
    // debug logging only
}

await _store.WriteAsync(key, encoded, ct).ConfigureAwait(false);
```

For the benchmark OME-Zarr arrays, the codec pipeline is:

```json
[
  { "name": "bytes", "configuration": { "endian": "little" } },
  { "name": "zstd", "configuration": { "level": 0, "checksum": false } }
]
```

Thus every decoded write pays:

```text
decoded bytes -> zstd encode -> zstd decode again -> filesystem write
```

The round-trip decode is useful as a diagnostic, but it should not be paid in
the normal write hot path. It should either be removed, placed behind an
explicit validation/debug option, or compiled out of release builds.

### 2. Decoded write requires an exact-size `byte[]`

`WriteChunkDecodedAsync` currently accepts only `byte[]` and checks
`decodedData.Length` against the exact full chunk byte count:

```csharp
if (decodedData.Length != expectedBytes)
    throw new ArgumentException(...);
```

This is clear and safe for ordinary arrays, but awkward for `ArrayPool<byte>`,
because rented arrays may be larger than the requested logical length. To use
pooled output safely, the API needs an overload such as:

```csharp
Task WriteChunkDecodedAsync(
    ZarrChunkRef chunk,
    ReadOnlyMemory<byte> decodedData,
    CancellationToken ct = default);
```

or:

```csharp
Task WriteChunkDecodedAsync(
    ZarrChunkRef chunk,
    ArraySegment<byte> decodedData,
    CancellationToken ct = default);
```

The implementation should validate the logical memory length, not the physical
array length. Once the returned task completes, callers should be able to reuse
or return the buffer. This is compatible with the Zarr standard: the standard
defines the stored bytes and metadata, not the lifetime or ownership model of
the caller's temporary buffer.

### 3. Zstd level 0 appears to be clamped to level 1

The benchmark metadata uses `zstd` with `level: 0`, but `ZstdCodec` currently
constructs the compressor as:

```csharp
_level = Math.Clamp(level, 1, 22);
```

This silently turns level 0 into level 1. If `ZstdSharp` supports level 0, it
would be better to preserve it. If it does not, the behaviour should be
documented explicitly, because Python/Zarr may be writing or measuring a
different compression level.

## Suggested Zarr.NET changes before rerunning benchmarks

1. Remove or guard the round-trip decode in `WriteChunkAsync`.
2. Add a decoded chunk write overload accepting `ReadOnlyMemory<byte>` or
   `ArraySegment<byte>`.
3. Ensure the store write path consumes/copies the supplied memory before the
   returned task completes, so callers can return pooled buffers safely.
4. Revisit zstd level handling, especially level 0.
5. Keep the chunk API as the public low-level route, since StackProcessing needs
   to compare chunk-native execution fairly against Python/Dask-Zarr.
