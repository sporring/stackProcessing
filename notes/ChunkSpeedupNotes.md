# Chunk Speedup Notes

The main speedup target is exactly analogous to the `Image.convolve` lesson: `resampleAffine` still does scalar pixel access in the hot loop.

Hot path:

- [StackAffineResampler.fs](/Users/jrh630/repositories/stackProcessing/src/StackProcessing.Core/StackAffineResampler.fs:227) allocates an output `Array2D`.
- For every output pixel it calls `trilinearSample`.
- `trilinearSample` fetches 8 voxels.
- Each voxel goes through `getVoxel`, then `ch[lx,ly,lz]`.
- That lands in [Image.fs](/Users/jrh630/repositories/stackProcessing/src/Image/Image.fs:1102), which calls `Image.Get`.
- `Image.Get` does SimpleITK scalar access plus type dispatch/vector construction at [Image.fs](/Users/jrh630/repositories/stackProcessing/src/Image/Image.fs:1026).

So for a 256x256x256 output, that is roughly 134 million scalar `Image.Get` calls. That will dominate everything.

Suggested speedups, in priority order:

1. **Cache chunks as bulk arrays, not `Image<'T>`**

   Change `ChunkCache<'T>` from `Dictionary<int64, Image<'T>>` to something like:

   ```fsharp
   type ChunkData<'T> =
       { Pixels: 'T[,,]
         W: int
         H: int
         D: int }
   ```

   Load with `_readChunk`, immediately call `toArray3D()`, store the array, then dispose/decrement the image if appropriate. `Image.toArray3D()` already has the fast bulk-copy path for scalar supported types at [Image.fs](/Users/jrh630/repositories/stackProcessing/src/Image/Image.fs:778).

2. **Prefer flattened chunk arrays**

   For even less overhead, store `Pixels: 'T[]` and index manually:

   ```fsharp
   pixels[(lz * h + ly) * w + lx]
   ```

   That avoids `Array3D` bounds/indexer overhead in the innermost loop.

3. **Reduce per-voxel dictionary/division work**

   `getVoxel` currently recomputes chunk coordinates, packs keys, dictionary-lookups, and calls chunk dimensions for each of the 8 neighbors. A faster `trilinearSample` should compute `x0/x1`, `y0/y1`, `z0/z1`, resolve the needed chunk data once per neighbor coordinate, then read raw arrays.

4. **Add a `float32` specialized path**

   The sample uses `float32`. The generic `lerp: 'T -> 'T -> float32 -> 'T` means 7 function calls per output pixel. A specialized float32 trilinear path can do direct arithmetic and will likely matter once scalar `Image.Get` is gone.

5. **Optional later: parallelize output rows/slices**

   After chunk access is array-backed, the per-slice loops are pure CPU work. `Array.Parallel` or row partitioning could help, but do this after fixing scalar access because the current SimpleITK scalar calls are the bigger bottleneck.

Strongest recommendation: first implement `ChunkCache` as bulk copied chunk arrays. That should be the same class of win as the `convolve` change, and it keeps the algorithm intact.

## Chunk Representation Cleanup

Chunks are not only an affine-resampling implementation detail. They are also the natural representation used by chunked volume formats such as Zarr, HDF5, and NeXus. That makes the current split between `ChunkInfo`, path-based chunk helpers, HDF5/Zarr layout metadata, and `ChunkData<'T>` somewhat awkward.

At the moment StackProcessing has clean central representations for:

```fsharp
Image<'T>
Window<'T>
Slab<'T>
```

but chunks are represented more indirectly:

- `ChunkInfo` describes an on-disk chunk layout.
- `ChunkData<'T>` stores cached array-backed chunk pixels for affine resampling.
- TIFF chunk folders, Zarr arrays, HDF5/NeXus datasets, and affine-resampling caches each carry some chunk logic locally.

A possible cleanup is to introduce a first-class chunk model in `StackProcessing.Core`.

```fsharp
type ChunkIndex = int * int * int

type ChunkLayout =
    { VolumeSize: uint64 * uint64 * uint64
      ChunkSize: uint64 * uint64 * uint64
      ChunkCounts: int * int * int
      PixelType: string
      Components: uint }

type ChunkStorage<'T> =
    | ImageChunk of Image<'T>
    | ArrayChunk of 'T[,,]

type Chunk<'T> =
    { Index: ChunkIndex
      Origin: uint64 * uint64 * uint64
      Size: uint64 * uint64 * uint64
      Data: ChunkStorage<'T> }
```

The storage choice should probably remain explicit:

- `ImageChunk` is useful when the next operation is SimpleITK-backed or file-IO oriented.
- `ArrayChunk` is useful for random access, interpolation, cache-heavy algorithms, and inner loops.

With this split, `StackIO` could own layout discovery and chunk read/write mechanics for TIFF chunk folders, Zarr, HDF5, and NeXus. Algorithm modules such as `StackAffineResampler` could then depend on a chunk cache API rather than path conventions and low-level `_readChunk` helpers.

This is a larger cleanup than the immediate speed fix. It touches IO, Zarr/HDF5/NeXus layout mapping, affine resampling, and cache semantics. The practical order should be:

1. Keep the immediate speedup local by making `ChunkCache` array-backed.
2. Extract a common `ChunkLayout` once the fast path is stable.
3. Introduce `Chunk<'T>` only when multiple chunked backends can share it without forcing unnecessary conversions between `Image<'T>` and arrays.
