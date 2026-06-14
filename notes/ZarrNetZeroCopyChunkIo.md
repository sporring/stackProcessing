# Zarr.NET Zero-Copy Chunk IO Wish

Status: implemented for the compact Chunk FFT/convolution pipeline's local
uncompressed full-chunk Zarr path.

## Problem

StackProcessing can now represent compact Hermitian FFT spectra as
`SpectralChunk` values backed by interleaved `float32` complex64 storage. The
natural persistent form is an OME-Zarr `complex64` array whose stored X size is
`realWidth / 2 + 1`, plus metadata that records the original real-domain size.

The original Zarr.NET API forced a payload copy at the read/write boundary:

- `OmeZarrWriter.WriteRegionAsync` and `ResolutionLevelNode.WriteRegionAsync`
  take `byte[]`.
- `ZarrArray.WriteChunkDecodedAsync(ReadOnlyMemory<byte>)` exists, but currently
  funnels into `ToExactArray(...)` because the store API is `byte[]`-based.
- `IZarrStore.WriteAsync` accepts only `byte[]`.
- reads return newly allocated `byte[]` buffers rather than an owned/rented
  buffer that StackProcessing can wrap directly.

For large compact FFT intermediates this means extra copies such as:

```text
SpectralChunk bytes -> temporary byte[] -> Zarr chunk buffer -> file
file -> decoded byte[] -> SpectralChunk bytes
```

## Zarr.NET API Used

The useful narrow addition is a whole-chunk memory-oriented API:

```csharp
Task WriteChunkDecodedAsync(
    long[] chunkCoord,
    ReadOnlyMemory<byte> decodedData,
    bool allowBorrowedBuffer,
    CancellationToken ct = default);
```

At the store layer this has matching memory-aware methods, otherwise the
array-level overload would still copy:

```csharp
Task WriteAsync(
    string key,
    ReadOnlyMemory<byte> data,
    CancellationToken ct = default);

Task<IMemoryOwner<byte>?> ReadOwnedAsync(
    string key,
    CancellationToken ct = default);
```

For local uncompressed Zarr this allows the fast path:

```text
SpectralChunk.Chunk.Bytes -> Zarr chunk file
Zarr chunk file -> Chunk-owned or ArrayPool-owned buffer
```

## Scope

The StackProcessing compact spectral Zarr writer now creates chunks that span
the full packed XY plane and a configurable number of Z slices:

```text
chunk_shape = [1, 1, chunkZ, height, realWidth / 2 + 1]
```

It writes those slabs with `WriteChunkDecodedAsync(..., allowBorrowedBuffer =
true)`.

The compact spectral reader uses `ReadChunkDecodedAsync` into an ArrayPool
buffer, then splits the full Zarr chunk into per-slice `SpectralChunk` values
for the current stream contract.

## Scope

Use whole decoded chunks, not arbitrary regions.

This is enough for the FFT/convolution theorem pipeline because StackProcessing
chooses Zarr chunk geometry that matches the handoff units. Region writes still
need read/modify/write chunk assembly and are not the primary target.

Compression can remain on the existing copying path at first. The important
first win is uncompressed `bytes` codec storage for compact complex
intermediates.

## StackProcessing Usage

The compact FFT pipeline should aim for:

```text
real TIFF slices
  -> fft3DRealXY
  -> write compact complex64 Zarr chunks
  -> read compact complex64 Zarr chunks
  -> spectral multiply / inverse path
  -> real TIFF slices
```

The current StackProcessing metadata convention is:

```text
complex_storage = "complex64_interleaved"
spectral_layout = "hermitian_packed"
packed_axis = 0
real_size = [realWidth, height, depth]
stored_complex_size = [realWidth / 2 + 1, height, depth]
```

`fftshift` is not part of this compact representation. A shift stage would be a
conversion from compact Hermitian storage to a full complex spectrum, optionally
followed by centering.
