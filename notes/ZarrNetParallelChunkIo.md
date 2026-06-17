# Zarr.NET Parallel Chunk IO Wish

The current StackProcessing Zarr copy benchmarks show a strong chunk-count effect.
For a 1024^3 uint8 volume, 64^3 chunks mean 4096 chunk files, while 256^3 chunks
mean only 64 files. StackProcessing is competitive for larger chunks, but much
slower for 64^3 chunks. This points to per-file transaction overhead rather than
raw byte throughput.

The important observation is that our copy path uses Zarr.NET's encoded chunk
APIs:

- `ReadChunkEncodedAsync`
- `WriteChunkEncodedAsync`

For uncompressed Zarr v3 arrays using only the `bytes` codec, these calls are
effectively raw payload reads and writes. They avoid compression and codec work,
but the caller currently drives them one chunk at a time. This serializes
thousands of small file opens, reads, creates, writes, and closes.

## Desired Zarr.NET Support

Add bounded parallel encoded chunk IO helpers to Zarr.NET, for example:

- `ReadChunksEncodedAsync(...)`
- `WriteChunksEncodedAsync(...)`
- `CopyChunksEncodedAsync(...)`

The key behavior should be:

- Accept an enumerable of `ZarrChunkRef` or chunk coordinates.
- Use bounded concurrency with a caller-provided `maxDegreeOfParallelism`.
- Preserve cancellation support.
- Avoid read-modify-write for full chunks.
- Use direct decoded-byte storage when the codec pipeline is a no-op bytes path.
- Avoid unnecessary payload copies when `ReadOnlyMemory<byte>` or caller-owned
  buffers are available.

For local filesystem stores, a good default might be modest parallelism such as
8 or 16. For remote stores, the best value may be higher and should remain
caller-configurable.

## Why This Matters

Python/Dask naturally schedules one task per chunk and can overlap many small
chunk reads and writes. StackProcessing currently preserves streaming semantics,
but its encoded Zarr copy path does not yet exploit chunk-level concurrency.
For many small chunk files, this leaves the filesystem underutilized and exposes
per-file latency directly.

For pure same-grid Zarr copy, an even faster specialized path may be possible:
copy raw chunk payload files and metadata directly when the source and target
are both uncompressed, non-sharded, same-grid Zarr arrays. That optimization is
separate from the general parallel chunk IO API, but the two ideas complement
each other.

## Related Bug To Fix While There

`LocalFileSystemStore.WriteAsync` currently contains benchmark-visible debug
readback. After writing a chunk, it logs a sample of the payload for the first
few calls. For chunk keys under `0/c/`, it then reads the same file back from
disk with `File.ReadAllBytesAsync` and logs another sample.

This should not happen in the default storage path. It adds hidden IO to normal
writes and pollutes benchmark measurements. A storage backend should only
perform verification reads when a caller explicitly requests diagnostics or
verification.

Suggested fix:

- Remove the default readback/logging from `LocalFileSystemStore.WriteAsync`, or
  guard it behind an explicit opt-in diagnostics setting.
- Keep the write path as one payload write and no follow-up read.
- Add a small test that writes a chunk through `LocalFileSystemStore`, verifies
  the payload through an explicit test read, and confirms that diagnostics are
  disabled by default.

## Benchmark Target

The immediate benchmark target is:

- 1024^3 uint8
- chunk sizes 64^3, 128^3, and 256^3
- uncompressed Zarr v3, `bytes` codec only
- operations: copy and threshold

Success means the 64^3 copy case approaches Python/Dask/Zarr timing without
changing the external StackProcessing DSL or Studio-facing stages.
