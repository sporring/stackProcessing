# DSL Stage Graph Enrichment

This note summarizes a possible future direction for making the Optimiser more aware of internal DSL structure without making the public StackProcessing DSL less ergonomic.

## Motivation

The Optimiser can currently reason most naturally at `Plan` composition boundaries, where user-facing stages are connected with `>=>`. Many useful implementation details, however, live inside stages composed with `-->`, for example:

```fsharp
window win pad stride
--> requireWindowSize ksz
--> parallelCollectChunk stage
--> emitValidRange outputStart outputCount
--> flattenList ()
```

Pulling these internal pieces up into user-visible `>=>` composition would improve optimiser visibility, but it would make the DSL much less pleasant. Users should be able to write compact Chunk stages such as `gaussianFilter`, `thresholdRange`, `read`, and `cast` without manually spelling out all windowed scaffolding.

## Rejected Direction: Shadow IR

One option would be to add a separate "shadow IR" to every stage. That IR would describe the internal structure of the executable stage for cost modelling and rewrites.

This is not attractive because it creates two sources of truth:

- the executable `Stage`
- the shadow representation used by the Optimiser

Over time, these can drift. Bugs then become hard to diagnose: the execution may be correct while the shadow plan is stale, or vice versa.

## Preferred Direction

Instead of adding a separate IR, enrich the existing stage graph and make internal `-->` compositions visible there.

Conceptually:

```text
Plan graph
  contains user-facing stages
    each stage may contain an internal --> graph
```

The public DSL remains ergonomic, while compound stages carry a real graph of their internal structure. The graph should be the same representation used for explanation, cost modelling, and possible rewrites.

## Current Gap

`Stage` already carries useful modelling fields:

- `Graph`
- `CostModel`
- `MemoryModel`
- `Transition`
- `ElementTransformation`
- `SliceCardinality`

The missing piece is that graph nodes are mostly name/transition records. They are not semantic enough to reliably distinguish operations such as:

- `Cast Float32 UInt8`
- `Read TIFF UInt8 as Float32`
- `Window`
- `WindowedCollect Threshold`
- `EmitValidRange`
- `Flatten`

String matching on stage names would be too fragile.

## Candidate Enrichment

Add semantic tags to stage graph nodes. For example:

```fsharp
type StageGraphOp =
    | Identity
    | Read of format: string * diskType: string option * outputType: string
    | Write of format: string * pixelType: string
    | Cast of sourceType: string * targetType: string
    | Window of size: uint * pad: uint * stride: uint
    | RequireWindowSize of depth: uint
    | WindowedCollect of name: string
    | EmitValidRange of start: uint * count: uint
    | WindowSkipTakeM of start: uint * count: uint
    | Flatten
    | Operator of name: string
    | Other of name: string
```

This should be treated as the graph's semantic identity, not as a second execution plan.

## Useful First Rewrites

Start with local, safe rules:

```text
identity --> x => x
x --> identity => x
cast<T,T> => identity
parallelCollect identity => identity
parallelCollect (cast<T,T>) => identity
```

Potentially useful but guarded rules:

```text
cast<S,T> --> cast<T,U> => cast<S,U>
```

This is only safe when the intermediate type `T` is not semantically lossy. For example, `UInt8 -> UInt16 -> Float32` is safe in spirit, while `Float32 -> UInt8 -> Float32` is not equivalent to direct identity.

For reads:

```text
read<diskT> --> cast<diskT,T>
```

should probably become an optimiser candidate rather than an unconditional rewrite:

```text
Candidate A: read diskT, then explicit cast to T
Candidate B: read directly as T
```

The measurements showed that implicit read-cast and explicit cast can have different timing behaviour. Therefore the Optimiser should choose between them using cost evidence rather than blindly canonicalising one into the other.

## Windowed Chunk Equivalents

The same idea should apply to windowed Chunk stages:

```text
parallelCollect identity => identity
parallelCollect (cast<T,T>) => identity
parallelCollect (safe cast chain) => parallelCollect fusedCast
```

Windowed Chunk scaffolding could also be recognized as a structured compound pattern:

```text
Window
RequireWindowSize
WindowedCollect op
WindowSkipTakeM
Flatten
```

This would let the Optimiser understand that the stage is a windowed Chunk computation without forcing users to write that scaffolding explicitly.

## Practical Roadmap

1. Add a semantic operation tag to `PipelineGraphNode`.
2. Update basic constructors such as `identityStage`, `cast`, `window`, `parallelCollect`, `emitValidRange`, and `flattenList` to set tags.
3. Keep execution unchanged.
4. Add graph-level simplification for the simplest identity/cast rules.
5. Teach `Plan.composePlan` or the Optimiser to inspect the enriched graph when constructing candidates and cost explanations.
6. Only later consider larger rewrites such as read/cast alternatives or windowed Chunk fusion.

## Design Principle

The public DSL should stay high-level and ergonomic. Internal `-->` structure should be visible to the Optimiser through enriched stage graphs, not by forcing users or Studio-generated code to spell out implementation scaffolding.
