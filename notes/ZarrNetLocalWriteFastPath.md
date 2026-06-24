# Zarr.NET LocalFileSystemStore WriteManyAsync Focus

Status: handoff note from StackProcessing/Zarr.NET benchmark experiments, 2026-06-24.

## Goal

Optimize the local filesystem batch writer in Zarr.NET, especially:

```text
LocalFileSystemStore.WriteManyAsync(...)
```

The StackProcessing side should keep assembling valid full chunk buffers and hand them to Zarr.NET. Zarr.NET should own Zarr layout, key semantics, path safety, and local filesystem write strategy.

## Current Evidence

Focused case:

- Pixel type: `UInt8`
- Shape: `1024x1024x1024`
- Zarr chunks: `64x64x64`
- Output: uncompressed Zarr v3 local filesystem store
- Chunk files: `4096`
- Payload: about `1.0G`

Relevant medians:

| Path | Median |
| --- | ---: |
| Original Zarr.NET thick writer | `5.411 s` |
| Parallel StackProcessing slab split experiment | `5.356 s` |
| Zarr.NET decoded batch path, small SP batch | `4.123 s` |
| Zarr.NET decoded batch path, larger SP batch | `3.496 s` |
| Updated Zarr.NET decoded batch path | `2.859 s` |
| Benchmark-only direct local writer ceiling | `2.527 s` |

The direct-local benchmark writes the same chunk payloads and produces a readable Zarr output, but it bypasses the general store abstraction. That makes it a useful ceiling, not an implementation we want in StackProcessing.

Full TIFF-to-Zarr comparison after cleanup:

| Path | Internal Median |
| --- | ---: |
| StackProcessing/Zarr.NET `tiffToZarr` | `7.638 s` |
| Python/Dask/Zarr `tiffToZarr` | `6.087 s` |

No-store diagnostics on the same TIFF input:

| Path | Internal Median |
| --- | ---: |
| SP TIFF read + Zarr-shaped buffer split + drain | `1.069 s` |
| Python/Dask TIFF read + rechunk + drain | `2.973 s` |

This strongly suggests the remaining gap is not TIFF read speed and not the basic memory reshaping. The write path is where the remaining hot-loop reduction should happen.

## Why Focus On WriteManyAsync

The current Zarr.NET fast path already avoids compression, encoding, and read-modify-write for no-op bytes pipelines:

```text
ZarrArray.WriteChunksDecodedAsync(...)
  -> IZarrStore.WriteManyAsync(...)
  -> LocalFileSystemStore.WriteManyAsync(...)
```

That is the right API shape. The remaining overhead is likely inside the local store batch implementation and the immediate planning around it.

Current local batch writing still does per chunk:

- store-key to absolute-path resolution,
- slash replacement,
- `Path.Combine`,
- `Path.GetFullPath`,
- root traversal check,
- `Path.GetDirectoryName`,
- directory de-duplication through `HashSet<string>`,
- `FileStream` construction,
- one file write.

Most of these are appropriate for arbitrary store keys. They are also visible at 4096 chunks per GiB, especially when repeated in many batches.

## Proposed Zarr.NET Change

Keep the public API small:

```csharp
Task WriteManyAsync(
    IEnumerable<ZarrStoreWrite> writes,
    int maxDegreeOfParallelism,
    CancellationToken ct = default);
```

Optimize the `LocalFileSystemStore` override for repeated local chunk writes.

Suggested implementation steps:

1. Add a fast path for trusted local relative keys inside `LocalFileSystemStore.WriteManyAsync`.
   - The public method can still guard against unsafe keys.
   - Internally, avoid `Path.GetFullPath` for keys that are already normalized relative store keys.
   - Keep the existing safe `ResolveKey` path for unusual keys, absolute-looking keys, `..`, empty segments, or metadata paths if needed.

2. Cache created directories on the store instance.
   - A `ConcurrentDictionary<string, byte>` or lock-protected `HashSet<string>` is enough.
   - Call `Directory.CreateDirectory` only the first time a directory is observed by this store.
   - This should help repeated batches writing the same `.../z/y` directories.

3. Avoid per-batch directory `HashSet<string>` allocation when possible.
   - With a store-level directory cache, the batch can check/cache directories as it plans writes.
   - This removes a temporary `HashSet<string>` and avoids repeated de-duplication work.

4. Preserve synchronous bounded parallel writes for local files.
   - The current `Parallel.ForEachAsync` plus synchronous `stream.Write(...)` is reasonable.
   - The direct-local benchmark used bounded synchronous writes and was fast.
   - Avoid switching back to per-file async writes unless measurement shows a win.

5. Keep the borrowed-buffer contract explicit.
   - `WriteManyAsync` must consume all `ReadOnlyMemory<byte>` payloads before returning.
   - StackProcessing returns pooled buffers immediately after `WriteChunksDecodedAsync` returns, so this contract matters.

## Possible Fast Key Resolver

The main cost to reduce is not safety; it is applying the most expensive safety path to every ordinary chunk key.

Conceptually:

```csharp
private string ResolveKeyForWriteMany(string key)
{
    if (IsSafeRelativeStoreKey(key))
        return Path.Join(_rootPath, key.Replace('/', Path.DirectorySeparatorChar));

    return ResolveKey(key);
}
```

`IsSafeRelativeStoreKey` should reject:

- empty keys,
- rooted/absolute paths,
- `..` segments,
- backslash traversal surprises,
- leading `/`,
- any platform-specific invalid form Zarr.NET wants to disallow.

The fallback remains the existing `ResolveKey`.

## Directory Cache Sketch

Conceptually:

```csharp
private readonly object _directoryCacheLock = new();
private readonly HashSet<string> _createdDirectories = new(StringComparer.Ordinal);

private void EnsureDirectoryCreated(string directory)
{
    lock (_directoryCacheLock)
    {
        if (!_createdDirectories.Add(directory))
            return;
    }

    Directory.CreateDirectory(directory);
}
```

If races matter, it is fine for two threads to call `Directory.CreateDirectory` for the same directory occasionally. The important part is avoiding thousands of repeated calls across batches.

## ZarrArray Follow-Up

After `WriteManyAsync` is optimized, consider a smaller follow-up in `ZarrArray`:

- cache chunk counts for the array,
- cache expected full chunk byte length,
- avoid recomputing immutable write invariants per batch.

This is secondary. `LocalFileSystemStore.WriteManyAsync` is the better first target because it is the hot local filesystem boundary and benefits all Zarr.NET callers.

## Validation Plan

Use the focused benchmarks first:

```text
run-zarr-thick-writeonly
UInt8 1024x1024x1024
chunkSize=64
```

Compare against:

- current Zarr.NET decoded batch path: about `2.86 s`,
- direct-local benchmark ceiling: about `2.53 s`.

Then validate:

1. Normal StackProcessing Zarr readback succeeds.
2. Output has `4096` chunk files.
3. Output size is about `1.0G`.
4. Full `tiffToZarr` improves for `UInt8 1024x1024x1024 chunkSize=64`.
5. Existing Zarr.NET store tests still cover unsafe key rejection and path traversal behavior.

## Non-Goals

Do not move Zarr chunk path construction into StackProcessing.

Do not add StackProcessing-specific APIs to Zarr.NET.

Do not reintroduce a separate hardcoded direct-local writer in SP. The benchmark-only direct writer is useful as a ceiling, but the production version belongs in Zarr.NET.
