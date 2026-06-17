# StackProcessing

StackProcessing is a toolkit for processing very large 3D image volumes without loading the whole volume into memory at once.

It is designed for image stacks from microscopy, medical imaging, synchrotron imaging, and similar volume-imaging workflows. The main idea is simple: describe a workflow, then let StackProcessing stream typed `Chunk<'T>` data through that workflow with bounded memory.

You do not need to know F# to use StackProcessing. The easiest entry point is **Studio**, a visual graph editor for building and running image-processing workflows.

## Who This README Is For

This README is for general users who want to install StackProcessing, run Studio, try examples, and understand what the project does.

More technical notes are in:

- [notes/Concepts.md](notes/Concepts.md): current vocabulary and high-level architecture.
- [notes/SlimPipeline.md](notes/SlimPipeline.md): the streaming and deferred execution engine.
- [notes/StackProcessing.md](notes/StackProcessing.md): how Chunk operations are bound to the streaming engine.
- [notes/ChunkBackboneStageGaps.md](notes/ChunkBackboneStageGaps.md): current Chunk runtime status.
- [notes/supportingSoftware.md](notes/supportingSoftware.md): build, test, calibration, and benchmark support tools.
- [notes/dsl-stage-graph-enrichment.md](notes/dsl-stage-graph-enrichment.md): possible future optimizer/DSL graph work.

Those notes are written for programmers, maintainers, and theoretically minded readers.

## What StackProcessing Is Good For

StackProcessing is useful when:

- your 3D image is too large to comfortably fit in memory,
- you want to process image stacks as repeatable workflows,
- you need local 3D filters, morphology, statistics, connected components, or object measurements,
- you want a visual way to build workflows but still keep a generated program for reproducibility,
- you want to compare memory and timing behaviour across image-processing plans.

Examples of supported workflow pieces include:

- read and write TIFF stacks, OME-Zarr, and NeXus/HDF5 where supported,
- create synthetic sources such as zeros, coordinate images, and noise images,
- cast between pixel types,
- threshold, smooth, convolve, and apply intensity transforms,
- run binary morphology,
- compute histograms, quantiles, and image statistics,
- label connected components and measure streamed objects,
- work with vector chunks, point sets, meshes, registration, manifests, serial sections, FFT experiments, and keypoints.

Some algorithms are naturally outside the streaming model, especially algorithms that require repeated unrestricted access to the whole volume. StackProcessing focuses on algorithms that can be expressed with bounded local context or carefully staged passes.

## Install And Run Studio

StackProcessing is a .NET 10 F# solution.

1. Install the .NET 10 SDK.
2. Clone the repository.
3. Build the solution.
4. Run Studio.

```bash
git clone https://github.com/sporring/Zarr.NET.git
git clone https://github.com/sporring/stackProcessing.git
cd stackProcessing
dotnet build StackProcessing.sln
dotnet run --project src/Studio/Studio.fsproj
```

The current developer build expects the `Zarr.NET` checkout to sit next to
`stackProcessing`. If it is elsewhere, pass
`/p:ZarrNetProject=/absolute/path/to/Zarr.NET.csproj` to `dotnet build`.

Some high-performance Chunk stages use native helper libraries built from the
`native/` sources. Build or restore those native binaries before running
samples that use median filters, convolution, FFT, resampling, signed distance
bands, or connected components.

## Studio

Studio is the visual way to build StackProcessing programs.

![Studio graph editor showing a StackProcessing workflow](images/StudioExample.png)

A Studio graph is a recipe:

- boxes describe data sources, processing steps, statistics, and outputs,
- lines connect compatible inputs and outputs,
- parameters control values such as filenames, thresholds, filter radii, and output paths,
- pressing `Run` turns the graph into a StackProcessing program and executes it.

The main screen contains:

| Area | Purpose |
| ---- | ------- |
| Palette | Search and drag available boxes. |
| Graph | Connect boxes into a workflow. |
| Parameters | Edit settings for the selected box. |
| Output | See generated code, build logs, run output, and errors. |
| Overview | Small minimap of the graph canvas. |

### Basic Studio Workflow

1. Drag boxes from the Palette into the Graph.
2. Connect compatible pins.
3. Select boxes and adjust their Parameters.
4. Save the graph as JSON.
5. Press `Run`.

Studio checks common graph mistakes while you work, such as incompatible connections and cycles.

### Pins

Studio has two main kinds of pins:

| Pin kind | Meaning |
| -------- | ------- |
| White side pins | Streaming data, usually images or tuples of images. |
| Brown top/bottom pins | Scalar parameters and reducer outputs, such as numbers, paths, thresholds, statistics, or tables. |

If a parameter is connected from another box, Studio disables the text field and uses the connected value instead.

### Common Box Categories

| Category | Examples |
| -------- | -------- |
| Sources and sinks | read, zero, coordinate images, noise sources, write, print, charts |
| Arithmetic | image-image and image-scalar operations |
| Filters | Gaussian smoothing, median, finite differences, gradients, FFT |
| Segmentation | threshold, morphology, connected components |
| Statistics | histograms, quantiles, image stats, object measurements |
| Points and meshes | keypoints, point sets, registration, marching cubes |
| Serial sections | slice-wise correction and registration workflows |
| Debug | tap and print boxes |

Many boxes have type or operation dropdowns so the graph stays compact.

## Optional F# DSL

Studio generates F# code behind the scenes. Programmers can also write the same workflows directly.

```fsharp
open StackProcessing

let availableMemory = 2UL * 1024UL * 1024UL * 1024UL

source availableMemory
|> read<float32> "../data/volume" ".tiff"
>=> gaussianFilter<float32> 1.0 3 4
>=> cast<float32,uint8>
>=> write "../tmp/smoothedVolume" ".tiff"
|> sink
```

The important user-level idea is that the workflow is built first and executed only at `sink`. StackProcessing can then check memory estimates and stream the data through the workflow.

## Samples

The `samples/` folder contains example workflows. Most sample folders include:

- a small F# program,
- a matching Studio JSON graph,
- paths that read from `samples/data` and write to `samples/tmp`.

Run a sample from its folder:

```bash
cd samples/copy
dotnet run
```

Many samples accept a debug flag:

```bash
dotnet run -- -d 1
```

Good samples to start with:

| Sample | Demonstrates |
| ------ | ------------ |
| `copy` | simplest read/write pipeline |
| `chunk` | TIFF Chunk slices to OME-Zarr and back |
| `smoothWGauss` | local 3D Gaussian smoothing |
| `threshold` | binary thresholding |
| `dilate`, `erode`, `opening`, `closing` | morphology |
| `objectsSizeHistogram` | connected objects, measurements, histogram output |
| `objectsMarchingCubes` | object surfaces and mesh writing |
| `structureTensor` | vector-image output and visualization |
| `sumProjection` | projection reducer |
| `quantileClamp` | histogram sampling, quantiles, and intensity stretch |
| `siftKeypoints`, `harris3DKeypoints`, `hessianKeypoints` | keypoint detection |

Samples are also useful for learning how Studio graphs correspond to direct F# workflows.

## Working With Large Images

StackProcessing's usual pattern is:

```text
read Chunk slices
  -> process with bounded local context
  -> write results
```

For local 3D operations, StackProcessing can keep only the neighbouring slices needed by the algorithm. For example, a smoothing filter with a 10-slice depth does not need the whole volume in memory; it needs the local z-window and a small amount of bookkeeping.

This is the main larger-than-memory advantage: the workflow can be written as a 3D image-processing task, while execution stays bounded by the local context of the operations.

## F# Interactive

Programmers can experiment in F# Interactive after building the solution.

On macOS:

```bash
cd <root of StackProcessing>
dotnet build
DYLD_LIBRARY_PATH="$(pwd)/src/StackProcessing/bin/Debug/net10.0:$(pwd)/lib" dotnet fsi
```

On Linux use `LD_LIBRARY_PATH` similarly. On Windows, ensure the build output and
native helper directory are on `PATH`.

Inside `dotnet fsi`:

```fsharp
#load "scripts/stackprocessing.fsx";;
open StackProcessing;;

let availableMemory = 2UL * 1024UL * 1024UL * 1024UL;;

source availableMemory
|> zero<uint8> 64u 64u 8u
>=> write "tmp/fsi-zero" ".tiff"
|> sink;;
```

## Repository Layout

| Path | Purpose |
| ---- | ------- |
| `src/Chunk` | Chunk payloads, typed spans, ownership helpers, and ChunkFunctions. |
| `src/SlimPipeline` | Generic streaming, stages, plans, deferred execution, and cost metadata. |
| `src/StackProcessing.Core` | Chunk IO, algorithms, windows, objects, points, registration, manifests, and more. |
| `src/StackProcessing` | Public F# DSL re-exporting the StackProcessing API. |
| `src/Studio.Graph` | Studio graph model and box catalog. |
| `src/Studio.Compiler` | Compiler from Studio graph JSON to StackProcessing F# code. |
| `src/Studio` | Visual graph editor. |
| `src/StackProcessing.Probe` | Measurement, fitting, and inspection tooling. |
| `samples` | Example F# programs and Studio graphs. |
| `tests` | Unit and integration tests. |
| `notes` | Programmer/theory notes and future-development design notes. |
| `benchmarks` | Benchmark side project for comparing read-process-write workflows. |

Build, test, calibration, and benchmark support commands are collected in [notes/supportingSoftware.md](notes/supportingSoftware.md).

## Design Principles

- Keep workflows visual and reproducible.
- Stream large volumes instead of materializing them when possible.
- Make memory use explicit.
- Prefer local 3D algorithms that can run with bounded z-neighbourhoods.
- Let advanced users write the same workflows directly in F#.
- Keep the technical architecture documented separately from the user guide.

## Acknowledgements

StackProcessing builds on excellent open-source work, including:

- FSharp.Control.AsyncSeq for asynchronous streams.
- Avalonia, NodeEditorAvalonia, PanAndZoom, and CommunityToolkit.Mvvm for Studio.
- Plotly.NET for charts and reports.
- PureHDF and ZarrNET for HDF5/NeXus and Zarr-style array storage.
- DIKU.Graph for graph algorithms used in the core.

## How To Cite

Sporring, J. and Stansby, D. Larger than memory image processing, January 2026, [https://arxiv.org/abs/2601.18407](https://arxiv.org/abs/2601.18407)
