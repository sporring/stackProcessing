# Image Project

The `src/Image` project is StackProcessing's typed wrapper around SimpleITK. It provides the low-level image value used by `StackProcessing.Core`, while keeping most direct SimpleITK interop, pixel type conversion, bulk array import/export, and image arithmetic in one place.

The public namespace is:

```fsharp
namespace FSharp
module Image
module ImageFunctions
```

The project targets `net10.0`, references the bundled SimpleITK C# bindings from `lib/`, and exposes a generated signature file at [src/Image/Image.fsi](/Users/jrh630/repositories/stackProcessing/src/Image/Image.fsi:1).

## Role In StackProcessing

`Image<'T>` is the image object that flows through StackProcessing stages. It wraps an `itk.simple.Image` and adds:

- a static F# pixel type, such as `uint8`, `uint16`, `float32`, `float`, `ComplexFloat32`, or `System.Numerics.Complex`
- a `Name`
- a mutable slice/volume `index`
- reference counting used by StackProcessing streaming stages
- memory facts and debug accounting
- typed conversion to and from arrays, files, vectors, and SimpleITK images

`Image` is intentionally lower level than the public StackProcessing DSL. It knows how to represent and manipulate images, but it does not know about plans, streaming windows, slabs, cost fitting, Studio graphs, or the optimiser. Those belong in `SlimPipeline`, `StackProcessing.Core`, and the probe/optimizer projects.

## Main Files

- [Image.fs](/Users/jrh630/repositories/stackProcessing/src/Image/Image.fs:1): the `Image<'T>` wrapper, SimpleITK interop helpers, array conversion, file IO, scalar pixel access, reference counting, and basic operators.
- [ImageIO.fs](/Users/jrh630/repositories/stackProcessing/src/Image/ImageIO.fs:1): low-level TIFF and SimpleITK image-file helpers used by StackProcessing IO without exposing SimpleITK in `StackProcessing.Core`.
- [ImageFunctions.fs](/Users/jrh630/repositories/stackProcessing/src/Image/ImageFunctions.fs:1): functional wrappers around SimpleITK filters and image algorithms.
- [Image.fsi](/Users/jrh630/repositories/stackProcessing/src/Image/Image.fsi:1): generated/public interface.
- [Image.fsproj](/Users/jrh630/repositories/stackProcessing/src/Image/Image.fsproj:1): project definition and SimpleITK native-library handling.
- [tests/Image.Tests](/Users/jrh630/repositories/stackProcessing/tests/Image.Tests/Tests.fs:1): unit tests for arithmetic, casting, indexing, vector helpers, array helpers, complex support, iteration, and wrapper coverage.

## Image<'T>

`Image<'T>` is a typed owner/wrapper for a SimpleITK image:

```fsharp
type Image<'T when 'T: equality>
```

Common construction paths:

- `Image<'T>(size, ?optionalNumberComponents, ?optionalName, ?optionalIndex)`
- `Image<'T>.ofSimpleITK`
- `Image<'T>.ofSimpleITKAlias`
- `Image<'T>.ofSimpleITKNDispose`
- `Image<'T>.ofFile`
- `Image<'T>.ofArray2D`
- `Image<'T>.ofArray3D`
- `Image<'T>.ofArray4D`
- `Image<'T>.constant2D`
- `Image<'T>.coordinateAxis2D`

Common export paths:

- `toSimpleITK`
- `toFile`
- `toArray2D`
- `toArray3D`
- `toArray4D`

The wrapper preserves the image index through many conversions and operators. This is important when 2D slices are streamed through StackProcessing and later reassembled, written, or inspected.

## Pixel Types

The project maps F# types to SimpleITK pixel IDs using `InternalHelpers.fromType<'T>`. Supported scalar types include:

- `uint8`, `int8`
- `uint16`, `int16`
- `uint`, `int`
- `uint64`, `int64`
- `float32`, `float`
- `ComplexFloat32`
- `System.Numerics.Complex`

Vector-valued images are represented as `Image<'T list>` where supported by the conversion helpers.

The type mapping is central to the read/cast/write model used by StackProcessing's cost calibration. Explicit casts use `Image<'T>.castTo<'S>()`, while many file reads also cast through `Image<'T>.ofSimpleITK`.

## SimpleITK Ownership

`Image<'T>` now makes SimpleITK ownership explicit:

- `ofSimpleITK` is the safe constructor. It deep-copies or casts into an independent `Image<'T>` and does not dispose the argument.
- `ofSimpleITKAlias` is an internal fast path for exact pixel-type matches where aliasing is acceptable. It does not cast, copy deeply, or dispose the argument.
- `ofSimpleITKNDispose` consumes a temporary SimpleITK image. If the type already matches, ownership of the wrapper is transferred into the returned `Image<'T>`; if a cast is needed, the temporary is disposed after the deep-copy/cast path.
- `ofCastITK` always returns an independent SimpleITK image and leaves the argument alone.

This split is important because SimpleITK wrapper copies may share pixel containers until copy-on-write, while cast filters allocate new images. The API names are deliberately explicit so hot paths can use alias/consume behavior without making ordinary `ofSimpleITK` unsafe.

## Reference Counting

`Image<'T>` currently owns mutable reference-count state:

- `incRefCount`
- `decRefCount`
- `getNReferences`

StackProcessing stages follow the rule that a stage releases its consumed input image after producing the output image, unless the image is retained for reuse first. This makes `Image` slightly object-oriented inside an otherwise functional pipeline style.

This is a deliberate implementation compromise: SimpleITK images are native resources, and StackProcessing needs predictable release points when streaming large images. Longer term, the wrapper may benefit from separating immutable image metadata from the managed native ownership handle, but the present model is explicit and works with the streaming DSL.

## Memory Facts And Debugging

`ImageFacts` describes the backend, pixel type, component size, image size, voxel count, and estimated memory. It is used to keep memory estimates explicit and close to the image representation.

Debug support includes:

- total image count
- peak image count
- estimated image memory
- process RSS sampling
- printable memory facts

This support is useful when validating StackProcessing's memory model against actual runs.

## Bulk Array Paths

Bulk array conversion is one of the most important performance boundaries in the project.

Preferred paths:

- `Image<'T>.ofArray2D`
- `Image<'T>.ofArray3D`
- `Image<'T>.toArray2D`
- `Image<'T>.toArray3D`
- `Image<'T>.ofFlatArray`
- `Image<'T>.toFlatArray`

For supported scalar types these use SimpleITK buffer access/import paths where possible, avoiding per-pixel `Get` and `Set` calls.

The flat-array helpers are the preferred path for new hot loops. They keep the same SimpleITK bulk-copy boundary, but avoid the extra `Array2D`/`Array3D` wrapper and its multidimensional indexer overhead. Use the inline helpers from `Image.InternalHelpers` for indexing:

```fsharp
let i2 = flatIndex2 width x y
let i3 = flatIndex3 width height x y z
let i4 = flatIndex4 width height depth x y z t
```

This convention is intentionally simple: x is the fastest-moving coordinate, then y, then z. If an operation immediately turns an `Array2D` or `Array3D` back into an `Image<'T>`, prefer building the flat output array directly and returning it with `Image<'T>.ofFlatArray`.

Scalar pixel access through:

```fsharp
image.Get [x; y; z]
image.Set [x; y; z] value
image[x, y]
image[x, y, z]
```

is convenient but should not be used in hot loops over large images. The chunk/resampling speedup notes are a good example: millions of SimpleITK scalar `Get` calls can dominate runtime, while bulk array extraction changes the cost class of the algorithm.

## ImageFunctions

`ImageFunctions` collects functional wrappers around SimpleITK filters and image operations. It is intentionally not the streaming DSL; it operates on already materialized `Image<'T>` values.

Main groups:

- scalar-image arithmetic, such as image/scalar add, subtract, multiply, divide, and power
- reductions, such as `sum`, `prod`, and `computeStats`
- intensity and unary functions, such as `sqrtImage`, `logImage`, `expImage`, `shiftScale`, `normalizeImage`, and intensity windowing
- comparisons and mask logic
- FFT and complex-image helpers
- 2D resampling and transforms
- convolution and Gaussian kernels
- finite difference filters and gradient helpers
- median, bilateral, Sobel, Laplacian, and gradient magnitude filters
- binary and grayscale morphology
- connected components and label statistics
- signed distance maps and watershed

StackProcessing.Core lifts many of these functions into streaming `Stage`s, adding reference-count handling, memory models, and cost terms.

Binary morphology now also contains native paths for spherical approximations. `binaryDilateZonohedralNative`, `binaryErodeZonohedralNative`, and the valid-slice helpers use a composition of one-dimensional line operations rather than a dense ball footprint. These functions are still Image-level operations over materialized images or explicit slice windows; StackProcessing.Core is responsible for turning the same idea into streaming stages with release-aware window handling.

## Complex And Vector Images

Complex images are supported with:

- `Image<ComplexFloat32>`, mapped to SimpleITK `sitkComplexFloat32`
- `Image<System.Numerics.Complex>`
- `ofComplexFloat32Array2D`
- `ofComplexFloat32Array3D`
- `ofComplexArray2D`
- `ofComplexArray3D`
- `toComplexFloat32Array2D`
- `toComplexFloat32Array3D`
- `toComplexArray2D`
- `toComplexArray3D`
- `Re`, `Im`, `modulus`, `arg`, `toComplex`, `polarToComplex`, `conjugate`

`System.Numerics.Complex` is double precision and maps to SimpleITK `sitkComplexFloat64`, so it uses 16 bytes per pixel. `ComplexFloat32` stores real and imaginary parts as `float32` and maps to `sitkComplexFloat32`, so it uses 8 bytes per pixel. The Float32 complex path is important for large TIFF and FFT-like workflows where widening to complex Float64 would double memory traffic and working-set size.

Vector images are supported through `Image<'T list>` helpers, including image-list zip/unzip conversions. This is useful for gradient-vector and multi-component operations, while keeping the main scalar image path simple.

## IO Scope

`Image<'T>.ofFile` and `toFile` are thin SimpleITK-backed image file operations. `ImageIO` contains lower-level single-image and slice helpers, including direct TIFF scalar read/write paths and SimpleITK file-information/slice-read wrappers. This keeps SimpleITK and libtiff details inside the Image project.

StackProcessing's higher-level streaming IO lives in `StackProcessing.Core.StackIO`, where file stacks, slab reads, OME-Zarr, NeXus/HDF5, chunk orchestration, and write stages are modeled.

The distinction matters:

- `Image` reads or writes a single SimpleITK image object.
- `ImageIO` provides low-level image/slice IO primitives and owns direct TIFF/SimpleITK interop.
- `StackIO` reads or writes streams, stacks, slabs, chunks, and plan-aware stages using Image-level helpers.

## Design Boundaries

`Image` should own:

- SimpleITK pixel type mapping
- single-image creation and conversion
- bulk array import/export
- direct SimpleITK filter wrappers
- direct TIFF scalar slice IO
- native image ownership and reference counting
- low-level memory facts

`Image` should not own:

- streaming window/slab semantics
- plan composition
- cost fitting
- DSL graph rewrites
- Studio-specific graph/code generation
- multi-file stack scheduling

Keeping this boundary clean prevents Image from becoming a second pipeline framework.

## Testing

The `Image.Tests` project covers the wrapper and filter surface from several angles:

- arithmetic operators and support operations
- casting and pixel type conversion
- indexing and scalar access
- array helper roundtrips
- vector helper conversion
- complex image conversion
- iteration and folds
- memory/debug helper behavior
- wrapper coverage against SimpleITK behavior

Run with:

```sh
dotnet test tests/Image.Tests/Image.Tests.fsproj
```

## Performance Guidance

Use SimpleITK filter wrappers for whole-image operations. Use bulk array conversion for custom hot loops. Prefer flat arrays over `Array2D`/`Array3D` when the operation is immediately rewrapped as an image or when the loop is in a measured hot path. Avoid per-pixel `Image.Get`/`Image.Set` in performance-sensitive code unless the operation is sparse or diagnostic.

The most important practical rule:

> Cross the SimpleITK boundary once per image or slab when possible, not once per pixel.
