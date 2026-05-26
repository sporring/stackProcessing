# SlimPipeline

`SlimPipeline` is the typed streaming layer underneath StackProcessing. It wraps `AsyncSeq<'T>` pipelines in a richer set of types that describe execution, memory behaviour, cost estimates, graph structure, slice cardinality, debug telemetry, and deferred execution.

The implementation lives in [src/SlimPipeline/SlimPipeline.fs](/Users/jrh630/repositories/stackProcessing/src/SlimPipeline/SlimPipeline.fs:1).

## Purpose

SlimPipeline provides the general, type-agnostic machinery for processing streams of elements. In StackProcessing those elements are usually image slices, windows of slices, slabs, point sets, or scalar reducer values, but SlimPipeline itself does not depend on image types.

It provides:

- typed stream transformations through `Pipe<'S,'T>`
- executable and modelled stages through `Stage<'S,'T>`
- user-facing deferred plans through `Plan<'S,'T>`
- memory and time cost models for stages
- stream profiles, including windowed streaming
- slice-cardinality tracking
- graph tracking for composed stages
- fan-out, fan-in, zip, branch, and sink/drain helpers
- runtime memory/time measurement and cost-discrepancy reporting

The public StackProcessing DSL is built on top of this layer.

## Layering

SlimPipeline is organized in three main layers:

```text
AsyncSeq<'T>
  ^
  |
Pipe<'S,'T>       executable stream function
  ^
  |
Stage<'S,'T>      executable function plus metadata, cost, memory, graph
  ^
  |
Plan<'S,'T>       deferred user-facing composition, checked and executed at sink/drain
```

The key distinction is:

- `Pipe` describes how to transform a stream.
- `Stage` describes what a reusable processing step is, including execution and modelling metadata.
- `Plan` is the deferred computation assembled by the DSL before the stream is actually run.

## Deferred Computation

SlimPipeline's implemented deferred computation is not a separate lazy IR. It is a typed `Plan<'S,'T>` that accumulates a composed `Stage<'S,'T>` and all its metadata while the user writes DSL code. Execution is delayed until a terminal operation such as `sink`, `drainSingle`, `drainList`, or `drainLast`.

For example:

```fsharp
source availableMemory
>=> readStage
>=> processStage
>=> writeStage
|> sink
```

The calls before `sink` do not process image data. They build a `Plan`.

During composition, the plan records:

- the composed executable stage
- the graph of stage nodes and edges
- current estimated elements per slice
- current estimated sequence length
- accumulated memory peak
- per-stage cost observations
- per-stage cost terms scaled by cardinality
- debug and optimisation flags
- optional source metadata
- optional discrepancy-reporting settings

Only when the plan is sunk or drained is the composed stage lowered to a `Pipe`, connected to an `AsyncSeq`, and evaluated.

This gives StackProcessing a practical deferred-computation model:

1. Build the pipeline in a typed way.
2. Accumulate memory/time/cardinality information while building.
3. Reject obviously impossible plans before running.
4. Lower to executable `AsyncSeq` only at the boundary.
5. Measure actual runtime and memory during execution when debug is enabled.

It is more modest than a full query optimizer, but it is already enough to keep memory checking, cost accounting, graph construction, and execution separated.

## Pipe

`Pipe<'S,'T>` is the executable stream function:

```fsharp
type Pipe<'S,'T> =
    { Name: string
      Apply: bool -> AsyncSeq<'S> -> AsyncSeq<'T>
      Profile: Profile }
```

`Apply` is a function from an input async sequence to an output async sequence. The `bool` argument is the debug flag.

Important pipe constructors include:

- `Pipe.init`: create a stream from an index-based generator.
- `Pipe.lift`: lift a pure element function.
- `Pipe.liftRelease`: lift a function and release the consumed input afterward.
- `Pipe.compose`: compose two pipes.
- `Pipe.window`: create streaming windows with pre/post padding.
- `Pipe.flatten`: flatten `AsyncSeq<'T list>` to `AsyncSeq<'T>`.
- `Pipe.flattenWindow`: emit a `Window<'T>` according to its `EmitRange`.
- `Pipe.reduce` and `Pipe.fold`: reduce a stream to a scalar-like value.
- `Pipe.ignore`: consume a stream and emit `unit`.
- `Pipe.tee`, `Pipe.map2`, and `Pipe.map2Sync`: support branch/fan-out patterns.

Pipes are intentionally close to `AsyncSeq`. They execute streams, but they do not by themselves carry the complete cost and graph information used by plans.

## Profile

`Profile` describes the broad streaming shape:

```fsharp
type Profile =
    | Unit
    | Constant
    | Streaming
    | Window of uint * uint * uint * uint * uint
```

The cases mean:

- `Unit`: no meaningful stream payload.
- `Constant`: a constant or scalar-like value.
- `Streaming`: one input element yields one output element in ordinary streaming form.
- `Window`: a windowed stream with size, stride, padding, emit start, and emit count.

Profiles are used for compatibility checks, memory estimation, and debug/explanation output. They are deliberately coarse. Detailed memory and time estimates live in `StageMemoryModel` and `StageTimeCostModel`.

## Window

Windowed streaming uses:

```fsharp
type Window<'T> =
    { Items: 'T list
      EmitRange: uint * uint
      ReleaseCount: uint }
```

`Items` is the current window payload. `EmitRange` describes which subset of the window should be emitted when the window is flattened. `ReleaseCount` records how many consumed inputs should be released when a resource-aware stage consumes the window.

The current `Pipe.window` implementation handles z-padding directly. It:

- rejects zero window size and zero stride
- peeks the first element so padding can be generated safely
- prepends and appends padding through the supplied `zeroMaker`
- emits full windows where possible
- emits a final partial window when needed
- records `EmitRange` and `ReleaseCount` so later stages can flatten/release correctly

In StackProcessing, this window handling is what allows one-dimensional streaming over the z-axis while still expressing local 3D halo operations.

## Stage

`Stage<'S,'T>` is the main reusable processing unit:

```fsharp
type Stage<'S,'T> =
    { Name: string
      Build: unit -> Pipe<'S,'T>
      Transition: ProfileTransition
      MemoryNeed: MemoryNeedWrapped
      MemoryModel: StageMemoryModel
      CostModel: StageCostModel
      ElementTransformation: ElementTransformation
      SliceCardinality: SliceCardinality
      Graph: PipelineGraph
      Cleaning: (unit -> unit) list }
```

A stage is still deferred: it stores `Build: unit -> Pipe<'S,'T>`, not a running stream. That means stages can be composed, costed, graphed, and checked before the actual `AsyncSeq` is created.

Stage metadata has several jobs:

- `Transition` describes input/output profiles.
- `MemoryNeed` and `MemoryModel` estimate peak memory.
- `CostModel` estimates memory and time together.
- `ElementTransformation` updates element-size estimates.
- `SliceCardinality` tracks sequence length changes.
- `Graph` records the composed stage graph.
- `Cleaning` stores cleanup actions to run after sink/drain.

### Stage Composition

Internal stage composition uses:

```fsharp
let (-->) = Stage.compose
```

`Stage.compose` combines:

- executable `Pipe` builders
- profile transitions
- cost models
- element transformations
- slice cardinalities
- cleanup actions
- pipeline graphs

This is the composition used inside Core helper functions such as window/slab scaffolding. It is ergonomic for library authors, but because it remains inside a single stage from the user's point of view, future optimizer work may need richer graph semantics to see these internal pieces.

## Plan

`Plan<'S,'T>` is the user-facing deferred computation:

```fsharp
type Plan<'S,'T> =
    { stage: Stage<'S,'T> option
      graph: PipelineGraph
      sourcePeek: SourcePeek option
      costPeak: StageCostEstimate option
      costObservations: StageCostEstimate list
      costTerms: PipelineCostTerm list
      nElemsPerSlice: SingleOrPair
      length: uint64
      memAvail: uint64
      memPeak: uint64
      debug: bool
      debugLevel: uint
      optimize: bool
      costDiscrepancy: bool
      costFlagPath: string option }
```

The plan is created by:

- `Plan.source`
- `Plan.sourceWithOptimizer`
- `Plan.debug`

It is composed with:

- `>=>`: append a stage to a plan
- `>=>>`: synchronize fan-out from one stream into two branches
- `zip`: combine two plans over compatible streams
- `>>=>>`: map two branches of a paired stream
- `>>=>`: combine paired stream values

It is executed by:

- `sink`
- `sinkList`
- `drainSingle`
- `drainList`
- `drainLast`

## Plan Composition

The central function is `Plan.composePlan`. When a stage is appended, it:

1. Logs the stage name in debug mode.
2. Composes the executable stage with the current plan stage.
3. Estimates the appended stage's memory and cost.
4. Updates the plan's peak memory estimate.
5. Updates cost observations and cost terms.
6. Updates sequence length using `SliceCardinality`.
7. Checks the memory budget unless debug mode is allowing exploration.
8. Updates the estimated element size using `ElementTransformation`.
9. Returns a new deferred plan.

This is the heart of implemented deferred computation: every composition step enriches the plan, but no stream data is processed yet.

## Slice Cardinality

SlimPipeline tracks how stages change sequence length:

```fsharp
type SliceCardinality =
    | Domain of SliceDomain
    | ReduceTo of uint64
    | UnknownCardinality
```

This supports:

- preserving length
- skip/take/trim-like domains
- reducers that collapse to a fixed number of outputs
- unknown fallback when a stage cannot express its cardinality

Cardinality is used to scale cost terms. For example, a per-slice stage over 128 slices should contribute differently from a reducer or a strided windowed stage.

## Cost And Memory Models

Stages carry both memory and time models:

```fsharp
type StageCostModel =
    { Memory: StageMemoryModel
      Time: StageTimeCostModel }
```

Memory estimates are structured as:

```fsharp
type StageMemoryEstimate =
    { InputLive: uint64
      OutputLive: uint64
      WorkLive: uint64
      RetainedLive: uint64
      Peak: uint64 }
```

Time estimates include:

- CPU cost units
- native cost units
- IO read/write bytes
- IO read/write operation counts
- optional calibration key
- tags such as stride or operation context

Calibration coefficients can be loaded through `StageTimeCalibration.loadJson`. If a calibration key is known, SlimPipeline can estimate milliseconds. If not, it can still report an uncalibrated cost score.

The cost model is intentionally generic. StackProcessing and the probe tools supply the domain-specific calibration data.

## Pipeline Graph

`PipelineGraph` records stage structure:

```fsharp
type PipelineGraph =
    { Nodes: PipelineGraphNode list
      Edges: PipelineGraphEdge list
      Entries: int list
      Exits: int list }
```

It is currently a lightweight graph with node names and profile transitions. It is useful for debug and explanation, and is a likely future extension point for DSL/stage graph enrichment.

Current graph operations include:

- singleton graph creation
- graph composition
- graph connection
- parallel joins
- append-node helpers

## Resource Operations

`ResourceOps<'T>` abstracts retain/release/memory operations for streamed resources:

```fsharp
type ResourceOps<'T> =
    { Retain: 'T -> unit
      Release: 'T -> unit
      MemoryOf: 'T -> uint64 option }
```

SlimPipeline itself does not know about images, but StackProcessing.Core supplies image-specific resource operations. This keeps the generic pipeline layer independent while still allowing image stages to release native resources predictably.

## Runtime Measurement

When debug mode is enabled, sinks and drains can measure actual runtime and RSS memory:

- `MemoryProbe.currentRssBytes`
- `MemoryProbe.measure`
- `Plan.runMeasured`

Debug output can include:

- estimated peak memory
- available memory
- measured peak RSS delta
- estimated runtime or uncalibrated cost score
- actual runtime
- pipeline cost terms

This is the runtime side of the feedback loop used by the probe and fitting workflow.

## Cost Discrepancy Reporting

Plans can enable discrepancy reporting:

- `Plan.withCostDiscrepancyReporting`
- `Plan.withCostDiscrepancyFlagPath`

When enabled, sink/drain compares predicted and observed memory/time. Large discrepancies are appended to CSV, including:

- timestamp
- label and discrepancy kind
- expected and actual values
- ratio
- estimated time
- actual time
- estimated and actual memory
- source metadata
- graph node names
- calibration keys
- cost tags
- per-stage cost terms

This is not the same as optimization, but it is important instrumentation for improving the model.

## What Is Implemented Versus Planned

Implemented now:

- deferred construction of typed plans
- deferred lowering to `Pipe`
- memory-budget checks before execution
- graph accumulation
- cost-term accumulation
- calibrated time estimates when coefficients are available
- runtime memory/time measurement
- cost discrepancy logging
- windowed streaming with padding and release metadata
- branch, zip, fan-out, and fan-in combinators

Not yet implemented as a general solver:

- automatic rewrite of DSL graphs
- search over alternative stage implementations
- global memory/time optimization across alternative plans
- rich semantic graph nodes for optimizer rewrites
- automatic insertion of buffers or prefetch policies
- automatic retry with smaller windows after memory pressure

So the current system is a deferred, typed, model-aware execution planner, not yet a full query optimizer.

## Relationship To StackProcessing

StackProcessing.Core builds image-specific stages on top of SlimPipeline:

- read/write stages
- casts
- image operators
- window/slab conversions
- reducers
- connected components
- object measurement
- registration and geometry operations

Those Core stages provide image-aware memory and cost models, and SlimPipeline provides the generic composition, deferral, and execution machinery.

## Design Guidance

Use `Pipe` when implementing the raw stream transformation.

Use `Stage` when creating a reusable operation that should carry memory, cost, graph, cardinality, and cleanup metadata.

Use `Plan` when composing user-facing DSL operations that should be checked, costed, and executed later.

Keep domain concepts such as images, slabs, TIFF, OME-Zarr, and SimpleITK outside SlimPipeline unless they are expressible as generic stream concepts.

## Open Direction

The most useful future development is likely graph enrichment at the `Stage` level. Internal `-->` compositions already preserve graph structure, but graph nodes are not yet semantic enough for reliable rewrites such as:

- `cast<T,T> -> identity`
- safe cast-chain fusion
- slab identity elimination
- read/cast alternative generation
- window/slab pattern recognition

That work should enrich the existing graph rather than adding a separate shadow IR.

