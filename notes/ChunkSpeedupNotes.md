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

and chunks now have a first-class type-level hook:

```fsharp
Chunk<'T>
```

Some implementation paths still represent chunks more indirectly:

- `ChunkInfo` describes an on-disk chunk layout.
- `ChunkData<'T>` stores cached array-backed chunk pixels for affine resampling.
- TIFF chunk folders, Zarr arrays, HDF5/NeXus datasets, and affine-resampling caches each carry some chunk logic locally.

The first cosmetic cleanup has introduced a first-class chunk model in `StackProcessing.Core`.

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

Fully using this model is still a larger cleanup than the immediate speed fix. It touches IO, Zarr/HDF5/NeXus layout mapping, affine resampling, and cache semantics. The practical order should be:

1. Keep the immediate speedup local by making `ChunkCache` array-backed.
2. Extract a common `ChunkLayout` once the fast path is stable.
3. Connect the existing `Chunk<'T>` type to multiple chunked backends only when they can share it without forcing unnecessary conversions between `Image<'T>` and arrays.

## Updated Direction After SimpleITK Ownership Cleanup

The SimpleITK ownership cleanup changes the priority of this discussion. `Image<'T>` is now more explicitly the wrapper around SimpleITK-backed images, with separate constructors for deep-copy, aliasing, and consuming temporary SimpleITK values. That makes the chunk representation gap more visible: chunked algorithms currently pass through path conventions, temporary files, `ChunkInfo`, and ad-hoc caches instead of consistently flowing through the new chunk resource type.

The current FFT/inverse FFT path is the strongest example. `FFT` first applies a 2D FFT slice-wise, writes intermediate chunks to a temporary folder, reads those chunks back into a full volume, performs the Z-direction transform, writes chunks again, and finally reads chunks back as slices. This works, but it is structurally awkward:

- chunks are represented as files plus `ChunkInfo`, not as pipeline values;
- memory use is described only approximately by the surrounding stage, while the full-volume read inside the chunked operation is hidden;
- cleanup is tied to temporary directories rather than resource ownership;
- the optimizer cannot reason about chunk layout, chunk storage, or conversions between `Image<'T>` and array-backed chunks.

With the new insight, the right long-term direction is to wire the new first-class `Chunk<'T>` in gradually, while keeping the first version deliberately small. It should not try to solve all Zarr/HDF5/NeXus semantics at once. The current minimal model is:

```fsharp
type ChunkIndex = int * int * int

type ChunkLayout =
    { VolumeSize: uint64 * uint64 * uint64
      ChunkSize: uint64 * uint64 * uint64
      ChunkCounts: int * int * int
      PixelType: string
      Components: uint }

type ChunkStorage<'T when 'T : equality> =
    | ImageChunk of Image<'T>
    | ArrayChunk of 'T[,,]

type Chunk<'T when 'T : equality> =
    { Index: ChunkIndex
      Origin: uint64 * uint64 * uint64
      Size: uint64 * uint64 * uint64
      Data: ChunkStorage<'T> }
```

This mirrors the role of `Window<'T>` and `Slab<'T>`: it names the representation and gives the pipeline something real to account for. `ImageChunk` should be used when the next operation is SimpleITK/Image-level. `ArrayChunk` should be used when the next operation is random access, interpolation, or custom managed loops.

The immediate FFT concern is not that it must be rewritten at once. The safer interpretation is:

1. Do not add more path-based chunk algorithms.
2. Keep `FFT`/`invFFT` working as-is until benchmarks and calibration settle.
3. Use the newly introduced `ChunkLayout`, `ChunkStorage<'T>`, and `Chunk<'T>` in `StackProcessing.Core`.
4. Refactor affine resampling first, because it already has an array-backed chunk cache and will validate the representation with low risk.
5. Then revisit `FFT`/`invFFT` as chunk-native stages that expose the intermediate chunk stream instead of hiding full-volume reconstruction inside a directory operation.

The FFT path also raises a second question: some operations are genuinely global along one axis. A chunk representation does not magically make a Z-direction FFT local; it only makes the dependency and memory shape explicit. The optimizer should be able to see that an XY FFT is slice-local, while the Z FFT requires all chunks along each `(x,y)` column group or a transposed/chunk-reorganized representation. That is exactly the kind of distinction that is hard to express while chunks are only filenames.

So the recommended next step is not a sweeping rewrite. It is to promote the existing `ChunkData<'T>` idea into a small shared chunk model, use it first where the code already wants arrays, and let FFT become the second consumer once the type has proven itself.
