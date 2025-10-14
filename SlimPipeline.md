# SlimPipeline - Stream-Oriented Stack Processing

`SlimPipeline` provides a composable, memory-aware streaming abstraction for large, stack-structured datasets (e.g. 3D image volumes represented as 2D slices) written in F#.
It wraps asynchronous sequence processing (`AsyncSeq<'T>`) into a typed and profiled pipeline system suitable for **out-of-core** workloads, where both **memory** and **I/O** are limiting factors.

This module is **type-agnostic** - it doesn't depend on image types, and can process any streamed stack of elements. It is intended to serve as the backend for domain-specific modules such as `StackProcessing` and `Image`.

## Overview

A `SlimPipeline` consists of one or more **stages**, `Stage<'S,'T>`, which wraps **pipes**, `Pipe<'S,'T>`, which in turn wraps **asynchronous sequences**, `FSharp.Control.AsyncSeq`. Pipe has the option of explicit memory release via callback functions. A stage:

* defines how a stream of elements is processed;
* carries a **memory usage profile** describing its streaming behavior;
* can be **composed** using intuitive operator syntax;
* can be fanned in and out in a stream-synchronized manner

Future versions will introduce **resource planning** through the `Stage` layer to automatically manage memory and buffering policies.


## Core Types

### `SingleOrPair`
```fsharp
type SingleOrPair =
    | Single of uint64
    | Pair of uint64 * uint64
```

Represents one or two related numerical quantities.


### `Profile`
The Profile type describes **how the elements in the connection between stages is streamed**. In F#, it is defined as:

```fsharp
type Profile =
    | Unit
    | Constant
    | Streaming
    | Sliding of uint * uint * uint * uint * uint
```

which has the following meaning:

| Case        | Description |
| ----------- | ----------- |
| `Unit`      | Stateless, no retained elements. |
| `Constant`  | End result terminating the stream. |
| `Streaming` | Streaming single elements. |
| `Sliding`   | Streaming list of elements with parametrized list size, stride, and initial and ending padding: `window`, `stride`, `pad`, `emitStart`, `emitCount`. |

### `Pipe<'S,'T>`

Encapsulates one transformation stage in the pipeline and wraps `FSharp.Core.AsyncSeq`. `AsyncSeq` is an asynchronous sequence of potentially (possibly infinite in length) with a promise of of evaluation in the future. A Pipe is sequence-function such as init, iter, map, and fold which creates, iterates over, maps functions on, and folds an accumulator over the asynchronous sequence. It's defined as:

```fsharp
type Pipe<'S,'T> =
  { Name: string
    Apply: bool -> AsyncSeq<'S> -> AsyncSeq<'T>
    Profile: Profile }
```
where

* **`Name`** - logical label for debugging and introspection.
* **`Apply`** - core asynchronous function preceded with a debugging flag.
* **`Profile`** - declares the output memory model used.

Each `Pipe` is a composable unit. They can be chained into full pipelines using functional composition operators. Key functions are constructers and transformers:

#### Constructers

| Function                   | Description |
| ------------------ | ----------- |
| `Pipe.create`      | Directly constructs a `Pipe` from name, function, and profile. |
| `Pipe.run`         | Execute a pipeline. |
| `Pipe.empty`       | No-op stage (identity on `unit`). |
| `Pipe.lift`        | Lifts a pure mapping function (`'S -> 'T`) into a `Pipe`. |
| `Pipe.liftRelease` | Lifts a mapping function to a pipe and releases the input after its application via the callback function given. |
| `Pipe.compose`     | Composes two pipes into a new pipe. |
| `Pipe.map2Sync`    | Synchronously maps a pure function `U -> 'V -> 'W` and fans out the result to two synchronized streams. |
| `Pipe.window`      | Create a sliding window of elements from a stream of given size, padding, and stride. |
| `Pipe.init`        | Initializes a new stream from a generator (`int -> 'T`).  |
| `Pipe.skip`        | Skips the first n elements in the sequence. |
| `Pipe.take`        | Takes the first n elements in the sequence. |

#### Transformers
In contrast to construction helpers, transformators call Pipe.Apply on the elements first in order for a possible sequence of compositions to be evaluated.

| Function                   | Description |
| ------------------ | ----------- |
| `Pipe.map`         | Map a pure function a sequence. |
| `Pipe.reduce`      | Reduce an AsyncSeq to an Async (the promise of a single value instead of a sequence). |
| `Pipe.fold`        | Accumulates an iterative processing of a sequence. |
| `Pipe.collect`     | Map a pure function (`'S -> 'T list`) and concatenate the result. |

### `Stage<'S,'T>`

The `Stage` abstraction wraps `Pipe` and will evolve into a **resource management layer** coordinating:

* Memory budgeting and preallocation;
* Concurrency and buffer scheduling;
* Streaming policy enforcement.

At present, `Stage` acts as a **lightweight wrapper**, maintaining a clean interface boundary for future extensions without impacting the existing processing model.

Absolutely — here’s a **Markdown documentation section** you can drop directly into your `SlimPipeline.md` file.
It matches the tone and structure of the earlier sections, focusing on the conceptual and practical aspects of the `Pipeline` type as built on `Pipe`.

### `Pipeline<'S,'T>`

A `Pipeline` defines **how** data moves from source to sink:

* It binds multiple `Pipe` stages into a coherent flow.
* It exposes a single entry point for execution (`Apply`).
* It aggregates **profiling information** from all constituent stages.

Conceptually:

```
AsyncSeq<'S>
   |
   v
 [ Pipe<'S,'A> ] --> [ Pipe<'A,'B> ] --> [ Pipe<'B,'T> ]
   |
   v
AsyncSeq<'T>
```

The `Pipeline` type represents a **fully composed processing chain** of `Pipe` stages operating on asynchronous streams. While `Pipe<'S,'T>` describes a *single* transformation from input stream to output stream, `Pipeline<'S,'T>` expresses the **end-to-end flow** of data through multiple stages - the executable form of a processing graph.

```fsharp
type Pipeline<'S,'T> = { 
    stage      : Stage<'S,'T> option
    nElems     : SingleOrPair
    length     : uint64
    memAvail   : uint64
    memPeak    : uint64
    debug      : bool }
```

where

* **`stage`** - the function to be applied, when the pipeline is run.
* **`nElems`** - the number of elments before transformation - this could be single or pair.
* **`length`** - the length of the sequence, the pipeline is applied to.
* **`memAvail`** - the memory available for the pipeline.
* **`memPeak`** - the pipeline's estimated peak memory consumption.
* **`debug`** - the debug flag.


Each `Pipeline` is a composable unit. They can be chained into full pipelines using functional composition operators. Key functions are sources, sinks and composition:

#### Source and sinks

| Function           | Description |
| ------------------ | ----------- |
| `Pipeline.create`  | Create a pipeline. |
| `Pipeline.source`  | The initial point of a pipeline which informs the pipeline about the memory budget. |
| `Pipeline.debug`  | The initial point of a pipeline which informs the pipeline about the memory budget and turns on debugging output. |
| `Pipeline.sink`    | The end-point of a pipeline which executes the pipeline and ignores the result. |
| `Pipe.drainLast`   | The endpoint of a pipeline which executes it and returns the last resulting value in the `AsyncSeq`. |


#### Composition
`Pipeline` defines operator-based combinators for building dataflows concisely.

| Operator | Description |
| -------- | ----------- |
| `-->`    | Functionally compose two stages, hence Pipe.Apply's. |
| `>=>`    | Functionally compose a pipeline and a stage returning a Pipeline. |
| `>=>>`   | Fan out a stream into a stream pair with Pipe.map2Sync synchronization. |
| `>>=>`   | Fan in a stream of pairs into a stream of singles. |
| `>>=>>`  | Functionally compose a Pipeline of pairs with a Stage for pairs. |
