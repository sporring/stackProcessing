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
