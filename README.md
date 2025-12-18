
# FSharp.StackProcessing

> **Namespace:** `FSharp`
> **Module:** `StackProcessing`
> **Purpose:** Memory-efficient, sequence-based 3D image processing pipelines built on top of `SlimPipeline`, connected seamlessly with the `Image` module.

 

## Overview

`StackProcessing` provides a **functional, streaming abstraction** for constructing large-scale image processing pipelines.

It allows users to define a flow like:

```fsharp
source availableMemory
|> read<float> "input" ".tiff"
>=> discreteGaussian 1.0 None None (Some 15u)
>=> cast<float,uint8>
>=> write "output" ".tiff"
|> sink
```

Each pipeline consists of:

* A **source** of image data (typically a stack of TIFF slices),
* A sequence of **plans** (`>=>` operator) transforming data,
* A final **sink** that consumes or writes the processed results.

 

## Key Concepts

### `Plan<'S, 'T>`

A **plan** represents one processing step, transforming a stream of items of type `'S` into `'T`.
Plans are composable, enabling streaming dataflow pipelines that minimize memory usage.

```fsharp
type Plan<'S,'T> = SlimPipeline.Plan<'S,'T>
```

### `Pipeline<'In,'Out>`

A complete dataflow, created from a source and extended via the composition operators:

* `>=>`:  append a plan to a pipeline
* `-->`:  compose two plans
* `|>`:   attach a source or a sink to a pipeline

 

## Pipeline Composition Operators

| Operator | Description |
| -------- | ----------- |
| `>=>`    | Append a plan to an existing pipeline. (`Pipeline<'a,'b> -> Plan<'b,'c> -> Pipeline<'a,'c>`)                                           |
| `-->`    | Compose two plans directly. (`Plan<'a,'b> -> Plan<'b,'c> -> Plan<'a,'c>`)                                                            |
| `>=>>`   | Parallelize two branches from the same pipeline source. (`Pipeline<'In,'S> -> (Plan<'S,'U> * Plan<'S,'V>) -> Pipeline<'In,('U * 'V)>`) |
| `>>=>`   | Combine paired results using a binary function. (`Pipeline<'a,('b * 'c)> -> ('b -> 'c -> 'd) -> Pipeline<'a,'d>`)                        |
| `>>=>>`  | Combine a function producing pairs with another plan transforming pairs.                                                                |

These operators let users declaratively build complex, memory-aware workflows.

 

## Core Pipeline Components

### `source`

```fsharp
val source : (uint64 -> SlimPipeline.Pipeline<unit, unit>)
```

Creates a pipeline source with the specified **memory budget** (in bytes).

Example:

```fsharp
let src = source (2UL * 1024UL * 1024UL * 1024UL) // 2 GB
```

 

### `sink`

```fsharp
val sink : SlimPipeline.Pipeline<unit,unit> -> unit
```

Executes the pipeline, pulling data from the source through all connected plans.
It is the terminal operator. Nothing is processed until the sink runs.

Example:

```fsharp
pipeline |> sink
```

 

### `map`

```fsharp
val map : f:('a -> 'b) -> Plan<'a,'b>
```

Applies a pure function to each element in the stream.
Acts as the functional `map` lifted to a streaming context.

Example:

```fsharp
>=> map (fun img -> img * 2.0)
```

 

### `window`

```fsharp
val window :
    windowSize:uint ->
    pad:uint ->
    stride:uint ->
    Plan<Image<'a>, Image<'a> list>
```

Converts a stream of 2D image slices into a **sliding window** of 3D image chunks.
Useful for localized processing (e.g., convolution or denoising across slices).

Example:

```fsharp
>=> window 5u 1u 1u
```

 

### `flatten`

```fsharp
val flatten : unit -> Plan<'a list,'a>
```

Reverses `window` - flattens a list of items back into a single stream.

 

### `releaseAfter` and `releaseAfter2`

Automatically free image memory once a processing function completes, enabling efficient out-of-core computation.

| Function                    | Description                                             |
| --------------------------- | ------------------------------------------------------- |
| `releaseAfter f img`        | Executes `f` on `img` and releases `img` afterward.     |
| `releaseAfter2 f img1 img2` | Executes `f` on two images and releases both afterward. |

 

### `zip`

```fsharp
val zip :
  (SlimPipeline.Pipeline<'a,'b> ->
   SlimPipeline.Pipeline<'a,'c> ->
   SlimPipeline.Pipeline<'a,('b * 'c)>)
```

Combines two pipelines elementwise, yielding paired outputs - analogous to `Seq.zip` but for streaming image data.

 

### `promoteStreamingToSliding`

```fsharp
val promoteStreamingToSliding :
  name:string ->
  winSz:uint -> pad:uint -> stride:uint ->
  emitStart:uint -> emitCount:uint ->
  plan:Plan<'T,'S> -> Plan<'T,'S>
```

Transforms a **streaming** plan into a **sliding-window** plan,
enabling localized operations over streaming image stacks.

 

## Utility Plans

| Function        | Description                                                                |
| --------------- | -------------------------------------------------------------------------- |
| `tap name`      | Inject a debugging/logging plan that prints or inspects elements.         |
| `tapIt f`       | Tap plan with a custom element-to-string converter.                       |
| `ignoreSingles` | Discard single-image elements in the stream.                               |
| `ignorePairs`   | Discard paired tuples in the stream.                                       |
| `zeroMaker`     | Create a zero-filled image of the same size and type as a reference image. |
| `idOp`          | Identity plan with a name — useful for profiling or marking plans.       |

 

## Example: Building a Gaussian Filter Pipeline

```fsharp
open FSharp.StackProcessing

let availableMemory = 2UL * 1024UL * 1024UL * 1024UL // 2 GB
let sigma = 1.0
let input, output = "image18", "result18"

source availableMemory
|> read<float> input ".tiff"
>=> discreteGaussian sigma None None (Some 15u)
>=> cast<float,uint8>
>=> write output ".tiff"
|> sink
```

In this example:

* The input image is streamed slice by slice from disk.
* Each slice is processed through a Gaussian filter.
* The result is written back as a TIFF stack.
* Memory never exceeds the specified budget.

 

## Advanced: Parallel and Combined Processing

You can branch and recombine pipelines:

```fsharp
source availableMemory
|> read<float> input ".tiff"
>=>> (discreteGaussian sigma None None (Some 15u),
      medianFilter 3u)
>>=> (fun gauss med -> gauss - med)
>=> write output ".tiff"
|> sink
```

Here:

* The input stream is split into two branches (Gaussian and Median filtering),
* Their results are combined pixel-wise,
* The final image is written to disk - all streaming safely.

 

## Design Philosophy

* **Functional composition:** pipelines are pure transformations of streams.
* **Controlled memory footprint:** each plan releases data as soon as possible.
* **Seamless IO:** TIFF stack reading/writing acts as the natural boundary between memory and disk.
* **Hidden complexity:** the underlying `SlimPipeline` machinery is fully abstracted from the user.
