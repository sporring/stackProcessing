# StackProcessing

`StackProcessing` is the Chunk-based image-stream DSL. It binds `Chunk<'T>` and
`ChunkFunctions` to `SlimPipeline` stages, plans, cost models, graph metadata,
IO, Studio lowering, Probe, and samples.

The user-facing module is [src/StackProcessing/StackProcessing.fs](/Users/jrh630/repositories/stackProcessing/src/StackProcessing/StackProcessing.fs:1).
Most implementation lives in `src/StackProcessing.Core`.

## Role

`Chunk.Core` owns chunk payloads and chunk-local algorithms.

`SlimPipeline` owns generic asynchronous streaming, stages, plans, cost models,
graphs, resource protocols, and deferred execution.

`StackProcessing` binds them:

```text
Chunk<'T>
    +
SlimPipeline.Stage / SlimPipeline.Plan
    =
StackProcessing image stream DSL
```

Its job is to make image processing feel like a high-level functional DSL while
preserving:

- bounded memory use
- explicit pooled-buffer release
- stream cardinality
- cost and memory models
- source metadata
- graph/debug information
- IO-aware read/write stages

## Public Shape

The public module re-exports the main stream and chunk concepts:

```fsharp
type Stage<'S,'T> = SlimPipeline.Stage<'S,'T>
type Window<'T> = SlimPipeline.Window<'T>
type Chunk<'T> = Chunk.Chunk<'T>
```

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
>=> gaussianFilter<float32> 1.5 (Some 7u)
>=> cast<_, uint8>
>=> write "output" ".tiff"
|> sink
```

Nothing is processed until `sink` or `drain` runs.

## Chunk Binding

The central binding is in [StackCore.fs](/Users/jrh630/repositories/stackProcessing/src/StackProcessing.Core/StackCore.fs:1).

StackCore supplies chunk resource operations:

```fsharp
ResourceOps<Chunk<'T>>
```

These tell SlimPipeline how to retain a chunk, release a chunk, and estimate
chunk memory. The key lifecycle rule is:

> A stage releases an input chunk after consuming it, unless the chunk was
> explicitly retained or copied first.

That rule lets StackProcessing stream larger-than-memory data while returning
`ArrayPool<byte>` buffers promptly instead of waiting for garbage collection.

## Stages As Lifted Chunk Operations

Most image algorithms start as ordinary functions over materialized chunks:

```fsharp
Chunk<'S> -> Chunk<'T>
```

StackProcessing lifts them into stages:

```fsharp
Stage<Chunk<'S>, Chunk<'T>>
```

The lifting adds:

- input release after successful computation
- memory estimates
- cost model labels
- element-size transformation
- graph nodes
- compatibility with `>=>` and `-->`

Examples include casts, arithmetic, comparisons, thresholding, intensity
transforms, smoothing, morphology, FFT wrappers, connected components, signed
distance bands, statistics, histograms, keypoints, structure tensor, PCA,
marching cubes, affine resampling, and object workflows.

## Chunk Algorithms

`ChunkFunctions` owns algorithmic image-processing code. The StackProcessing
Core modules arrange those functions into streaming stages:

- `StackChunk`: small Chunk stage helpers and public aliases.
- `StackConvolve`: fixed-kernel, single-axis, and separable convolution stages.
- `StackMedian`: median filters and native nth-element median stages.
- `StackBinaryMorphology`: zonohedral binary morphology, binary contour,
  top-hat, and gradient stages.
- `StackConnectedComponents`: Chunk SAUF connected components and relabelling.
- `StackFFT`: Chunk FFT, inverse FFT, complex helpers, and fftshift stages.
- `StackAffineResampler`: affine and serial-section resampling stages.
- `StackBias`: bias model fitting and correction.
- `StackObjects`, `StackPoints`, `StackRegistration`, `StackSerialSections`,
  and `StackMarchingCubes`: higher-level workflows over Chunk streams.

The preferred storage types are `uint8`, `uint16`, `float32`, and
complex64-interleaved `float32` where the algorithm supports them.

## IO As Plan Sources And Stage Sinks

IO is handled mostly by [StackIO.fs](/Users/jrh630/repositories/stackProcessing/src/StackProcessing.Core/StackIO.fs:1).

Read functions usually create source plans:

```fsharp
Plan<unit, unit> -> Plan<unit, Chunk<'T>>
```

They establish source length, shape metadata, elements per slice, memory peak,
IO cost terms, and pixel type context.

Write functions are usually stages or reducers:

```fsharp
Stage<Chunk<'T>, Chunk<'T>>
Stage<Chunk<'T>, unit>
```

The regular stack path reads TIFF scanlines directly into Chunk buffers and
writes Chunk buffers back to TIFF. Chunk-native scalar OME-Zarr range reads and
slice writes cover `uint8`, `uint16`, `float32`, and `float`; complex64 uses
the interleaved `float32` Chunk path. Random/range TIFF reads, repeats, object
streaming, color stack IO, histogram data, and chart/show policies also lower
through Chunk stages.

## Windows And Parallel Collection

`Window<'T>` is generic in SlimPipeline. StackProcessing mostly uses:

```fsharp
Window<Chunk<'T>>
```

This represents a local z-neighbourhood of Chunk slices. It supports
one-dimensional streaming over the z-axis while expressing 3D algorithms that
need halos.

Example structure:

```text
chunk slice stream
  -> window of adjacent chunks
  -> local operation
  -> emitted chunk slice stream
```

Padding, emit ranges, and release counts are carried by the window machinery so
stages can release consumed chunks correctly.

`Stage.parallelCollect` is the workhorse for Chunk neighbourhood stages. It
groups slices into bounded windows, lets worker-local functions produce zero,
one, or many output chunks, and emits results in a controlled order. Histogram
and statistics reducers use `Stage.parallelReduce` with worker-local
accumulators and explicit merge functions.

## Streaming Binary Morphology

Binary morphology uses the zonohedral approximation path. It approximates
spherical structuring elements by composing one-dimensional line erosions and
dilations using Jensen/Gorpho coefficients and van Herk/Gil-Werman style line
passes.

The stages keep only the slices required by the current halo, emit valid center
slices, and release consumed chunks through the window resource rules.
`opening`, `closing`, white top-hat, black top-hat, morphological gradient, and
binary contour are stage compositions over erosion and dilation.

## Convolution And Derivatives

Single-axis native convolution is available for `UInt8`, `Int8`, `UInt16`,
`Int32`, and `Float32` chunks with `float32` kernels. Separable box, Gaussian,
Sobel, finite-difference, gradient-vector, Hessian-upper, Laplacian, gradient
magnitude, Sobel magnitude, structure tensor, and PCA-gradient stages compose
those primitives.

Vector chunks use a component dimension in the chunk payload. Structure tensor
produces a 12-component Float32 vector chunk containing eigenvalues and three
eigenvectors. Component-wise Gaussian smoothing is native-backed for Float32
vector chunks.

## Connected Components

Connected components uses Chunk SAUF labelling for `UInt8` input and `UInt32`
labels. The streaming path labels chunks, records boundary equivalences, and
emits relabelled slices directly. It avoids dense all-label translation tables
and does not require a whole labelled volume to be resident.

## FFT

The Chunk FFT path uses complex64-interleaved `float32` chunks. The current
slice-local `fft` stage is an XY transform. Full-volume Chunk FFT work uses
the separable 3D stages:

- `fft3DComplexXY`
- `fft3DRealXY`
- `invFft3DRealXY`

The real-XY path stores Hermitian-packed complex64 spectra and is the preferred
route for real-valued convolution-style round trips. Zarr-backed z-axis passes,
subchunked spectral workspaces, 3D `fftshift`, and complex helpers are exposed
as explicit layout-changing stages because FFT is a global transform rather
than a local Chunk-window operator.

## Cost And Memory Binding

StackProcessing supplies domain-specific cost models to SlimPipeline:

- reads and writes are tagged by format and pixel type
- casts are tagged by source and target type
- native calls carry native cost units
- chunk maps and reducers carry pixel counts and worker counts
- windowed operations carry halo, emit range, and slice-count context

Probe collects measurements, fit estimates calibration coefficients, and
inspect highlights model gaps. Runtime plan composition uses those coefficients
for estimated time and memory checks.

## Studio Binding

Studio builds user-facing graphs and generates StackProcessing DSL code. The
compiler now lowers the regular image boxes to Chunk-backed stages. Box IDs can
remain stable while the generated DSL uses Chunk sources, stages, reducers, and
sinks.

The boundary is:

- Studio expresses user intent and UI graph structure.
- StackProcessing provides typed DSL functions.
- SlimPipeline executes deferred plans.
- Optimization works from structured stage and plan metadata.

## Important Modules

- `StackCore`: resource operations, composition helpers, windows, sink/drain.
- `StackIO`: TIFF, Zarr, file-info, source, sink, and Chunk write stages.
- `StackChunk`: common Chunk stage aliases and wrappers.
- `StackConvolve`, `StackMedian`, `StackBinaryMorphology`,
  `StackConnectedComponents`, `StackFFT`: focused algorithm stage families.
- `StackBias`, `StackAffineResampler`, `StackObjects`, `StackPoints`,
  `StackRegistration`, `StackSerialSections`, `StackMarchingCubes`: workflow
  stages over Chunk streams.
- `StackCharts`: Chunk-backed image show, histogram data, and chart helpers.
- `StackOptimizer`: candidate selection helpers over costed stages.

## Mental Model

```text
Chunk<'T>
    typed image payload backed by pooled bytes

Window<Chunk<'T>>
    bounded adjacent z-neighbourhood

Stage<Chunk<'S>, Chunk<'T>>
    reusable image stream operation

Plan<unit, Chunk<'T>>
    deferred image stream computation

sink/drain
    execute the plan and release owned resources
```

StackProcessing is the layer that makes those pieces work together.
