# StackProcessing

`StackProcessing` is the image-domain binding layer between the `Image` project and `SlimPipeline`. It turns typed SimpleITK-backed `Image<'T>` values into memory-aware, deferred, streaming stages and plans.

The user-facing module is [src/StackProcessing/StackProcessing.fs](/Users/jrh630/repositories/stackProcessing/src/StackProcessing/StackProcessing.fs:1). Most implementation lives in `src/StackProcessing.Core`.

## Role

`Image` owns single-image representation and SimpleITK interop.

`SlimPipeline` owns generic asynchronous streaming, stages, plans, cost models, graphs, and deferred execution.

`StackProcessing` binds them:

```text
Image<'T>
    +
SlimPipeline.Stage / SlimPipeline.Plan
    =
StackProcessing image stream DSL
```

Its job is to make image processing feel like a high-level functional DSL while still preserving:

- bounded memory use
- explicit native-resource release
- stream cardinality
- cost and memory models
- source metadata
- graph/debug information
- IO-aware read/write stages

## Public Shape

The public module re-exports the main stream and image concepts:

```fsharp
type Stage<'S,'T> = SlimPipeline.Stage<'S,'T>
type Window<'T> = SlimPipeline.Window<'T>
type Slab<'T> = StackCore.Slab<'T>
type Chunk<'T> = StackCore.Chunk<'T>
type Image<'T> = Image.Image<'T>
```

Core also contains a first-class `Chunk<'T>` representation for block-backed workflows. It currently provides a type-level hook beside `Window<'T>` and `Slab<'T>`; older path-based chunk machinery is still used by some implementations while the chunk-native interface matures.

It also re-exports the main composition and execution functions:

```fsharp
source
debug
>=>       // Plan composition
-->       // Stage composition
>=>>      // synchronized fan-out
>>=>      // pair combine
>>=>>     // paired branch mapping
sink
drain
```

From the user's point of view, a pipeline looks like:

```fsharp
open StackProcessing

source availableMemory
|> read<float32> "input" ".tiff"
>=> smoothWGauss<float32> 1.5 None
>=> cast<float32,uint8>
>=> write "output" ".tiff"
|> sink
```

Nothing is processed until `sink` or `drain` runs.

## Binding Image To SlimPipeline

The central binding is in [StackCore.fs](/Users/jrh630/repositories/stackProcessing/src/StackProcessing.Core/StackCore.fs:1).

StackCore defines image-specific resource operations:

```fsharp
ResourceOps<Image<'T>>
```

These tell SlimPipeline how to:

- retain an image
- release an image
- estimate image memory

This is where the generic `ResourceOps<'T>` abstraction becomes concrete for SimpleITK-backed images.

The key lifecycle rule is:

> A stage releases an input image after consuming it, unless the image was explicitly retained or copied first.

That rule lets StackProcessing stream large images without relying on the garbage collector to discover native-memory pressure.

## Stages As Lifted Image Operations

Most image algorithms start as ordinary functions over materialized images:

```fsharp
Image<'S> -> Image<'T>
```

StackProcessing lifts them into stages:

```fsharp
Stage<Image<'S>, Image<'T>>
```

The lifting adds:

- input release after successful computation
- memory estimates
- cost model labels
- element-size transformation
- graph nodes
- compatibility with `>=>` and `-->`

Examples include:

- `cast`
- arithmetic and scalar image operations
- thresholding
- intensity transforms
- smoothing
- morphology
- FFT
- connected components
- signed distance maps
- object measurement

The raw image algorithms generally live in `ImageFunctions`, while the lifted streaming stages live in `StackImageFunctions`.

## Streaming Binary Morphology

Binary dilation has two implementations with deliberately different roles:

- `dilate` is the older SimpleITK-backed spherical dilation path. It builds local slabs from z-windows, applies the image-level morphology operation, and emits the valid central slices.
- `dilateZonohedral` is the streaming benchmark path for binary `UInt8` masks. It approximates a spherical structuring element by composing one-dimensional line dilations using the Jensen/Gorpho zonohedral coefficients and a van Herk/Gil-Werman style line pass.

The zonohedral path is closer to StackProcessing's preferred streaming shape. It keeps only the slices required by the current line halo, copies each input slice into a flat buffer once, emits only valid output slices, and releases consumed images through the usual window resource rules. This avoids materializing a whole 3D slab for each local neighbourhood. `erodeZonohedral` mirrors the same structure with an all-foreground line predicate, while `openingZonohedral` and `closingZonohedral` are stage compositions of erosion/dilation and dilation/erosion.

The benchmark comparison now records that the four tools use different, library-native choices for binary dilation:

- StackProcessing: streaming zonohedral/VHGW line approximation.
- Python/scikit-image/SciPy: scikit-image's 3D `ball(..., decomposition="sequence")` footprint.
- MATLAB: `strel("sphere", r)` with MATLAB's structuring-element decomposition machinery.
- C++/ITK: ITK's binary ball structuring element, currently kept as the exact-ball comparison point.

## IO As Plan Sources And Stage Sinks

IO is handled mostly by [StackIO.fs](/Users/jrh630/repositories/stackProcessing/src/StackProcessing.Core/StackIO.fs:1).

Read functions usually have the shape:

```fsharp
Plan<unit, unit> -> Plan<unit, Image<'T>>
```

They create source plans because they establish:

- source length
- elements per slice
- source shape metadata
- memory peak
- IO cost model
- read format and pixel type context

Examples:

- `read`
- `readVolume`
- `readRandom`
- `readRange`
- `readSlab`
- `readZarrSlab`
- `readNexusSlab`

Write functions are usually stages:

```fsharp
Stage<Image<'T>, Image<'T>>
```

or reducers:

```fsharp
Stage<Image<'T>, unit>
```

They consume image streams while preserving enough stream structure for continued composition when appropriate.

## Windows

`Window<'T>` is generic in SlimPipeline, but StackProcessing uses it primarily as:

```fsharp
Window<Image<'T>>
```

This represents a local z-neighbourhood of slices. It supports one-dimensional streaming over the z-axis while expressing 3D algorithms that require halos.

The important benefit is that many 3D local operations only need a small number of adjacent slices in memory, rather than the whole volume.

Example structure:

```text
slice stream
  -> window of adjacent slices
  -> local operation
  -> emitted slice stream
```

Padding, emit ranges, and release counts are carried by the window machinery so stages can release consumed images correctly.

## Slabs

`Slab<'T>` is defined in StackCore:

```fsharp
type Slab<'T> =
    { Image: Image<'T>
      EmitRange: uint * uint }
```

A slab is a small 3D image built from a window of adjacent 2D slices. It exists because some image operations are more naturally or efficiently expressed as operations on a 3D SimpleITK image rather than as a list of slices.

The usual pattern is:

```fsharp
window
--> windowToSlabWithRange
--> mapSlabWithStage imageStage
--> slabWithRangeToWindow
--> slabSkipTakeM
--> flattenList ()
```

This lets StackProcessing apply ordinary image stages to local 3D slabs while still returning to a stream of slices.

## Chunks

Chunks are the block-oriented counterpart to windows and slabs. A window is a z-neighbourhood in the active stream, and a slab is a small 3D image assembled from adjacent streamed slices. A chunk is a bounded 3D block from a larger volume or chunked backing store.

`StackCore` defines the high-level chunk shape:

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

The storage choice is explicit. `ImageChunk` is for workflows that should stay close to Image/SimpleITK operations or file IO. `ArrayChunk` is for random access, interpolation, and managed hot loops where per-pixel SimpleITK access would be too expensive.

Some older implementation pieces still use related local forms:

- `ChunkInfo` describes existing on-disk chunk layouts.
- `ChunkData<'T>` is the current in-memory cached representation used by chunk-backed affine resampling.
- path-based chunk folders are still used by the existing FFT/inverse FFT implementation.

Chunks are useful when an operation cannot be expressed as a simple forward stream of z-windows. For example, affine resampling may need to fetch source voxels from several spatial blocks while producing one output slice. In that case StackProcessing writes or reads bounded chunks, caches only the blocks needed for the current work, and avoids materializing the full input volume.

Conceptually:

```text
Window<Image<'T>>
    streaming local z-neighbourhood

Slab<'T>
    small adjacent z-neighbourhood packed as one Image<'T>

Chunk<'T>
    bounded 3D block with explicit storage, origin, size, and index
```

The chunk type is intentionally minimal at present. It is a structural placeholder for the next cleanup: making affine resampling and FFT-like multi-pass algorithms expose their memory shape through typed chunk streams rather than hidden temporary directories.

## Internal And Public Composition

StackProcessing uses two composition levels:

```fsharp
-->   // Stage composition
>=>   // Plan composition
```

`-->` is mostly used inside Core to build compound stages from smaller stages.

`>=>` is the user-facing plan composition operator. It is where SlimPipeline updates the deferred plan with memory, cost, graph, and cardinality information.

This split keeps the public DSL compact:

```fsharp
>=> smoothWGauss sigma None
```

rather than exposing implementation scaffolding:

```fsharp
>=> window ...
>=> windowToSlab ...
>=> mapSlabWithStage ...
>=> slabToWindow ...
>=> flatten ...
```

Future optimizer work may enrich the internal stage graph so the optimizer can still see this structure without making the DSL unpleasant.

## Cost And Memory Binding

StackProcessing supplies domain-specific cost models to SlimPipeline.

For example:

- reads are tagged by format and pixel type
- writes are tagged by format and pixel type
- casts are tagged by source and target type
- image operators are tagged by operator family
- windowed operations carry window/stride context

The probe tools collect measurements, fit a model, and load calibration coefficients back into the runtime. SlimPipeline then turns stage cost terms into estimated time and memory during plan composition.

This is why StackProcessing stages are not just functions. They are functions plus enough modelling information to support memory checks, inspection, and future optimization.

## Source Families

StackProcessing's probe ladder groups operations into families such as:

- `io`
- `io-cast`
- `sources`
- `singleton`
- `window-slab`
- `neighbourhood`
- reducers and higher-level operations

These families are not part of the core runtime abstraction, but they are important for fitting and inspecting the cost model. They reflect the way the runtime's image stages are measured and calibrated.

## Studio Binding

Studio builds user-facing graphs and generates StackProcessing DSL code.

The boundary is:

- Studio expresses user intent and UI graph structure.
- StackProcessing provides the typed DSL functions.
- SlimPipeline executes deferred plans.
- The Optimiser should work from structured stage/plan metadata, not from fragile generated-code string rewrites.

This keeps Studio useful as an authoring environment without making it responsible for execution semantics.

## Important Modules

The implementation is split into focused Core modules:

- `StackCore`: aliases, resource operations, composition helpers, windows/slabs, sink/drain bindings.
- `StackIO`: stack, volume, slab, TIFF, MHA, OME-Zarr, NeXus/HDF5, chunk, and write stages.
- `StackImageFunctions`: lifted image operations and image-processing stages.
- `StackObjects`: streamed connected objects and object measurements.
- `StackPoints`: point-set IO and point operations.
- `StackRegistration`: affine registration and distance metrics.
- `StackAffineResampler`: chunk-backed affine resampling.
- `StackManifest`, `StackStitching`, `StackSerialSections`: metadata-driven workflows.
- `StackOptimizer`: candidate selection helpers over costed stages.

The top-level `StackProcessing.fs` module re-exports these pieces as the public API.

## Design Boundaries

`Image` should own:

- SimpleITK image representation
- single-image algorithms
- file IO for one image object
- array conversion
- native image reference counting

`SlimPipeline` should own:

- generic stream execution
- stages and plans
- deferred composition
- graph and cost metadata
- memory/time modelling infrastructure

`StackProcessing` should own:

- image stream stages
- stack/volume IO
- window/slab/chunk image adaptation
- lifted image algorithms
- domain cost labels
- user-facing image DSL

This boundary is the reason StackProcessing can remain both image-specific and pipeline-aware without either `Image` or `SlimPipeline` absorbing all of the complexity.

## Mental Model

The shortest useful mental model is:

```text
Image<'T>
    one SimpleITK-backed image or slice

Window<Image<'T>>
    adjacent streamed slices

Slab<'T>
    a small 3D image made from adjacent slices

Chunk<'T>
    a bounded 3D block used for chunked IO, random-access caches, and future chunk-native multi-pass algorithms

Stage<Image<'S>, Image<'T>>
    a reusable image stream operation

Plan<unit, Image<'T>>
    a deferred image stream computation
```

StackProcessing is the layer that makes those pieces work together.
