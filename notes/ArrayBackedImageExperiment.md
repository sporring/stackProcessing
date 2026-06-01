# Array-Backed Image Experiment

## Motivation

StackProcessing currently uses `Image<'T>` as a SimpleITK-backed image wrapper. This is a good bridge to ITK algorithms, but it makes SimpleITK the default storage representation even when the operation is naturally array-based. The recent TIFF slice reader and streaming morphology work suggests that some overhead comes from repeatedly crossing this boundary:

```text
TIFF slice -> managed array -> SimpleITK image -> managed array/slice -> TIFF slice
```

The experiment is to make the internal backbone array-like while keeping the public DSL unchanged. SimpleITK should remain available as an accelerator for operations where ITK is still the best implementation.

## Goal

Build drop-in replacements for selected `Image` and `StackProcessing` internals so that existing DSL code keeps the same surface syntax, while the execution path can stay in pooled managed arrays for simple streaming stages.

The first target pipeline should be deliberately narrow:

```text
TIFF slices
  -> array-backed image/window/slab
  -> threshold
  -> connected-component scaffolding
  -> SimpleITK only for per-slab connected components
  -> array-side collision/relabel/statistics
  -> optional TIFF slices
```

This tests the likely benefit without requiring a full rewrite of Image or all algorithms.

## External Interface

The external benchmark/interface should remain TIFF slices for now.

- Continue to use the direct libtiff path for TIFF read/write.
- Keep the benchmark comparable with existing `cpp-itk`, `python-skimage-scipy`, MATLAB, and StackProcessing rows.
- Do not introduce a new user-visible file format requirement for ordinary stack input/output.

The first question is not whether a new storage format is better than TIFF. It is whether StackProcessing can avoid unnecessary SimpleITK representation crossings after the TIFF slices have entered the pipeline.

## Internal Scratch Format

For multipass algorithms, experiment with a minimal raw-array scratch format:

```text
volume.raw
volume.json
```

The sidecar should contain enough information to reconstruct the array without guessing:

- width, height, depth
- pixel type
- component count
- byte order
- layout convention
- chunk/slab shape if the data is partitioned
- optional checksum or version field

This is not intended as an exchange format. It is a low-overhead internal format for measuring the lower bound of array-close read/write. If this proves useful, a later OME-Zarr or Zarr backend can be considered for a proper chunked scientific storage format.

## Representation Sketch

The possible direction is:

```fsharp
type ImageStorage<'T> =
    | SimpleITKImage of itk.simple.Image
    | PooledArray2D of PooledBuffer<'T>
    | PooledArray3D of PooledBuffer<'T>

type Image<'T> =
    { Name: string
      Index: uint64
      Storage: ImageStorage<'T> }
```

This exact type is only illustrative. The important points are:

- array storage must have explicit ownership and disposal;
- SimpleITK storage remains possible;
- conversions must be explicit about deep copy versus aliasing;
- the optimizer should eventually remove unnecessary `array -> SimpleITK -> array` roundtrips.

## First Experiment

Use `UInt8` only and compare:

```text
read -> threshold -> connectedComponents
```

against the current StackProcessing implementation.

Recommended initial shapes:

- `256x256x256`
- `512x512x512`

Recommended measurements:

- wall time
- internal time
- peak memory
- number of SimpleITK conversions
- size of raw scratch written/read, if used

The connected-component operation may still call SimpleITK for per-slab labeling. The array-backed experiment should focus on keeping thresholding, slab stitching, label translation, and statistics in managed arrays.

## Implemented Minimal Scaffold

The first scaffold is implemented as a benchmark-only backend rather than as a replacement of the production `Image<'T>` type. This keeps the experiment isolated while preserving the external benchmark interface:

```text
TIFF slices -> pooled flat array -> operation -> TIFF slices
```

It is exposed through:

```bash
bash benchmarks/run_all.sh \
  --backends stackprocessing-arraypool \
  --operations copy,threshold,connectedComponents \
  --pixel-types UInt8 \
  --shapes 256x256x256
```

The benchmark executable also accepts the direct command:

```bash
dotnet benchmarks/StackProcessing.Benchmarks/bin/Debug/net10.0/StackProcessing.Benchmarks.dll \
  run-arraypool --operation threshold --pixel-type UInt8 \
  --input tmp/benchmarks/input/UInt8_256x256x256 \
  --output tmp/benchmarks/output/stackprocessing-arraypool/threshold_UInt8_256x256x256_threshold-128_r01 \
  --threshold 128
```

The scaffold currently includes:

- `PooledVolume<'T>`, a flat `ArrayPool<'T>`-backed volume with explicit `incRefCount` and `decRefCount`.
- Direct TIFF slice read/write into and out of pooled arrays.
- Copy implemented as a `Span` copy.
- Threshold implemented as a flat array loop into a pooled `uint8` mask.
- Connected components implemented as `uint8` TIFF -> pooled mask -> SimpleITK connected components -> pooled `uint8` label output.
- A slice-streaming variant, `stackprocessing-arraypool-slice`, which rents and returns slice-sized buffers as it streams.
- A reusable slice-streaming variant, `stackprocessing-arraypool-slice-reuse`, which rents the input slice, optional output slice, read scanline, and write row buffers once per run and reuses them for every slice.
- A byte-oriented `UInt8` variant, `stackprocessing-byte-slice-reuse`, which keeps TIFF input and output in reusable `byte[]` page buffers and avoids the `byte[] -> uint8[] -> byte[]` staging path.

This is intentionally a lower-bound experiment for representation crossings. It does not yet implement the production slab-wise connected-component merge, local relabelling, or raw+sidecar scratch. Its connected-component path is therefore best interpreted first for the full-volume `256^3` case, where the current StackProcessing benchmark also uses the full-volume shortcut.

`Span` is used where it maps directly to the intended operation. Copy uses `Span.CopyTo`, while threshold uses typed flat loops over the rented backing array. The important point is not that `Span` is magic; it is that the loops operate on contiguous memory without `Image.GetPixel`/`SetPixel`, without Array2D indexing, and without unnecessary SimpleITK allocation for simple map stages.

The first `1024^3` `UInt8` timing suggests that reusable buffers help most when there is actual per-pixel work. Compared with the first slice-streaming backend, reusing buffers made copy essentially neutral, but reduced threshold from roughly 6.4 seconds to roughly 4.7 seconds internal time. The likely interpretation is that copy is dominated by TIFF read/write, whereas threshold benefits from avoiding repeated output-buffer and row-buffer churn in the map stage.

The byte-oriented variant indicates that typed-array staging is also significant for `UInt8`. On the `1024^3` copy benchmark, keeping the data as reusable `byte[]` pages reduced internal time from roughly 6.1 seconds for the typed reusable ArrayPool path to roughly 4.1 seconds. Threshold was also roughly 4.1 seconds, suggesting that once the data stays byte-oriented the threshold loop is nearly hidden by TIFF read/write. This does not generalize directly to `UInt16` or `Float32`, but it is strong evidence that a future array-backed IO layer should avoid unnecessary byte-to-typed-to-byte copies for `UInt8` slice-local pipelines.

For `Float32`, the corresponding experiment uses byte-owned storage with a typed `Span<float32>` view created by `MemoryMarshal.Cast<byte,float32>`. On the `1024^3` copy benchmark this reduced internal time compared with current StackProcessing, but threshold was not a win over the current streaming implementation. The comparison is subtle: copy writes `Float32` output, while threshold writes `UInt8` output. The byte-storage threshold path therefore reads `Float32` pages, views them as floats, writes a `UInt8` page, and avoids SimpleITK, but it is still governed by the cost of reading 4 GiB of input and writing 1 GiB of output. The useful lesson is that byte-owned storage plus typed span views is viable, but each operation needs to be compared with the actual input and output byte counts.

The simpler `ArrayPool<float32>` slice-reuse test is closer to a drop-in image replacement. It rents a typed `float32[]` slice, copies bytes from TIFF into it, operates on the typed array, and writes via byte staging. On the `1024^3` benchmark it improved copy time and roughly matched current StackProcessing threshold time, while reducing peak memory by about a factor of two. This supports a conservative conclusion: typed ArrayPool storage is a general memory win and can improve simple copy-like paths, but the major time win is clearest for `UInt8` byte-oriented pipelines. Extending the byte-oriented style to every pixel type increases low-level complexity and is not automatically a speed win.

## Expected Wins

Likely wins:

- cheap operations such as threshold avoid SimpleITK allocation/conversion;
- collision detection and relabeling can use flat array loops;
- multipass scratch can avoid TIFF metadata and per-slice overhead;
- peak memory may drop by avoiding simultaneous managed and SimpleITK copies;
- pipelines can fuse more naturally once array-backed stages are visible.

Likely non-wins:

- SimpleITK may still beat managed code for heavy kernels such as full connected-component labeling, FFT, affine resampling, and general convolution;
- raw scratch is not a replacement for TIFF as an external format;
- ArrayPool only helps if ownership is disciplined and buffers are returned promptly.

## Success Criterion

The experiment is successful if the DSL-visible pipeline remains unchanged and the array-backed path shows a measurable reduction in time or peak memory for:

```text
TIFF -> threshold -> connected components
```

without making the SimpleITK-backed path harder to use for algorithms where ITK remains the right engine.
