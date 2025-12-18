
# A plan

## 1) Separate *description* from *execution*

Create a first-class **PlanSpec** (parametrisable description) and make `Plan` carry params but *not* an executable `Pipe` yet.

```fsharp
type PlanKind =
  | Source
  | Transform
  | Sink

type WindowParams = { Window: int; Stride: int; Pad: int; EmitStart: int; EmitCount: int }

type PlanParams =
  | None
  | Windowed of WindowParams
  | Custom of Map<string, obj> // extensibility

type ResourceModel =
  { // Functions must be PURE and monotone if possible
    MinStackIn     : int                      // minimal #input slices needed at once
    MemoryPressure : int list -> uint64       // est. peak bytes for a candidate slice batch
    CpuCost        : int list -> float        // est. time (ms) for that batch (rough)
    IoCost         : int list -> float        // est. IO time (ms) for that batch
  }

type PlanSpec<'S,'T> =
  { Name       : string
    Kind       : PlanKind
    // Declares the “shape” relationship and allowable parameter ranges
    ParamSpace : PlanParams -> bool                  // validates params
    ArityMap   : int -> int                           // slices out -> in mapping
    Resource   : PlanParams -> ResourceModel         // param → resource model
    // Lowering hook: given *fixed* params, build a Pipe on demand
    Build      : PlanParams -> Pipe<'S,'T>
  }

type Plan<'S,'T> =
  { Spec   : PlanSpec<'S,'T>
    Params : PlanParams }  // chosen (initially defaults), but not yet executed
```

> Why: this lets you reason about memory/time with `Resource(Spec, Params)` *before* any `AsyncSeq` is constructed.

## 2) Keep the compositional Domain-Specific Language (DSL), but build a **graph/plan** instead of running

Your `source availableMemory |> Plan >=> Plan |> sink` should return a **Plan** (linear chain is fine now; a DAG later if you add forks/joins).

```fsharp
type Plan<'S,'T> =
  { Plans      : obj list    // keep as heterogeneous chain internally
    AvailableMB : uint64
    Debug       : bool }

module Plan =
  let empty mem = { Plans = []; AvailableMB = mem; Debug = false }
  let add (s: Plan<'a,'b>) (p: Plan<'a,'z>) : Plan<'b,'z> = { p with Plans = p.Plans @ [ box s ] }
```

Provide combinators that *append* `Plan`s, not compose `Pipe`s:

```fsharp
let (>=>) (s1: Plan<'S,'T>) (s2: Plan<'T,'U>) : Plan<'S,'U> =
  // purely build a composite Plan wrapper OR just return s2 to be added next; the plan holds the sequence
  s2

let (-->) = (>=>) // your existing alias
```

`source` builds a `Plan`, `sink` will consume it (see Step 6).

## 3) Add **shape inference & validation** pass

Before optimisation, validate that the plan is consistent and infer end-to-end slice counts.

```fsharp
type Inferred =
  { SlicesIn   : int
    SlicesOut  : int
    MinStack   : int             // worst-case across Plans given current params
    ParamHints : PlanParams list }

// run once post-build
val infer : Plan<'S,'T> -> Result<Inferred, string>
```

* Compute `SlicesOut` by chaining each `Spec.ArityMap`.
* Compute `MinStack` = max of `Resource(params).MinStackIn` across the plan.
* Collect `ParamHints` (e.g., minimal window ≥ MinStack of downstream sliding Plans).

## 4) Define the **optimisation problem**

You want to **maximise memory utilisation without exceeding memory**, while also considering throughput. A practical objective:

* Primary: keep **peak** ≤ `AvailableMB`
* Secondary: **minimise total cost** = Σ max(IO, CPU) per Plan (pipeline throughput)
* Tertiary: push params up (e.g. bigger windows) to reduce IO seeks / amortise kernel overhead

Formalise the evaluation interface:

```fsharp
type EvalPoint =
  { ParamsPerPlan : PlanParams list
    PeakBytes      : uint64
    TotalCostMs    : float }

val evaluate : Plan<'S,'T> * PlanParams list -> EvalPoint
```

`evaluate` walks the plan:

* For each Plan, build `Resource(params)` and estimate memory for “coexisting” slices (respect upstream/downstream buffering; for first pass assume a bounded queue size like 2–3).
* Peak = max over accumulated “in-flight” footprint plus Plan’s `MemoryPressure`.
* TotalCost = Σ max(CPU, IO) (or a queuing model later).

## 5) Provide **parameter search** (heuristic + monotone assumptions)

Start simple:

* For each Plan, define a **finite candidate set** of parameter values (e.g., windows ∈ {1, 2, 3, 4, 6, 8, 12, 16}, strides coherent).
* Do a greedy forward search with backtracking:

  1. Start with minimal feasible params (safest window sizes).
  2. Try to “grow” params (increase window / batch) at the Plan with the best Δ(cost)/Δ(peak) improvement while staying ≤ memory.
  3. Stop when no beneficial move exists.

Provide API:

```fsharp
type OptimiserSettings =
  { MaxIters : int
    GreedyNeighborhood : int } // how many candidates ahead to try per Plan

val optimise : Plan<'S,'T> * OptimiserSettings -> Result<PlanParams list, string>
```

> Later you can plug in integer programming or simulated annealing, but greedy with good heuristics gets you far.

## 6) **Lowering**: build the executable `Pipe` pipeline once params are fixed

After `optimise` returns the chosen `PlanParams list`:

```fsharp
val lower : Plan<'S,'T> * PlanParams list -> Pipe<'S,'T>

let lower (plan, params) =
  let pipes =
    (plan.Plans, params)
    ||> List.zip
    |> List.map (fun (boxedPlan, prms) ->
         let s = unbox<Plan<obj,obj>> boxedPlan // internal detail, usually keep typed chain
         s.Spec.Build prms |> box)
  pipes |> composeAll  // your Pipe.compose folded left
```

> Implementation detail: to keep typing pleasant, hold the chain as a typed HList or encode with a small internal DU; the sketch uses `obj` to convey the idea.

## 7) **Execution planner**: bounded buffers, prefetch, and IO throttling

While building Pipes:

* Insert **bounded AsyncSeq buffers** between windowed Plans to cap residency.
* Add `prefetch`/`read-ahead` for sources constrained by IO latency.
* Respect a global **concurrency budget** (e.g., `Environment.ProcessorCount`) and an **IO queue depth**.
* Provide defaults and let Plans expose hints (e.g., `PreferredDegreeOfParallelism`).

```fsharp
type RuntimeHints = { BufferSlices: int; DegreeOfParallelism: int; IoQueueDepth: int }
```

Expose a single “runner”:

```fsharp
val run : availableMemory:uint64 -> (Plan<'S,'T> -> Async<unit>)
```

Internally: `plan |> infer |> optimise |> lower |> Pipe.run`.

## 8) **Integrate into your DSL**

Users keep writing:

```fsharp
let plan =
  source availableMemory
  |> Plan "read" readTiffsSpec
  >=> Plan "gauss" (convGaussSpec sigmaCandidates)
  >=> Plan "hist" histogramSpec
  |> sink sinkSpec // see Step 10
```

`sink` now becomes **driver** (not an immediate consumer). It should:

1. Build the plan
2. `infer` + `optimise` with `AvailableMB`
3. `lower` to Pipes
4. Run the pipeline and materialise the sink

## 9) **Resource model contracts** per Plan (make it easy to implement)

For each builtin Plan, provide a small helper to fill `ResourceModel`:

```fsharp
module Resource =
  let windowed (elemBytes:int64) (overlap:int) =
    fun (xs:int list) ->
      // xs = candidate batch sizes or “co-resident” slices; simple model:
      let slices = xs |> List.sum
      uint64 (slices |> int64 |> (*) elemBytes)

  let cpuLinear (k:float) (b:float) =
    fun (xs:int list) -> k * float (List.sum xs) + b
```

Then a Plan like Gaussian conv:

```fsharp
let convGaussSpec (sigmaCandidates:int list) : PlanSpec<float[], float[]> =
  let paramSpace = function
    | Windowed p -> p.Window >= 1 && p.Stride >= 1 && sigmaCandidates |> List.contains p.Window
    | _ -> false
  let build = function
    | Windowed p -> convGaussPipe p.Window p.Stride p.Pad p.EmitStart p.EmitCount
    | _ -> failwith "bad params"
  let arity inSlices = inSlices // emits one per valid center by emit policy
  let resource = function
    | Windowed p ->
        { MinStackIn     = p.Window
          MemoryPressure = Resource.windowed elemBytes p.Window
          CpuCost        = Resource.cpuLinear k b
          IoCost         = fun _ -> 0.0 }
    | _ -> failwith "bad params"
  { Name = "convGauss"; Kind = Transform; ParamSpace = paramSpace; ArityMap = arity; Resource = resource; Build = build }
```

## 10) **Sinks as specs** (so they participate in planning)

Do the same for sinks (e.g., “write tiffs”, “histogram plot”). Some sinks also affect memory (buffers) and IO.

```fsharp
type SinkSpec<'T> =
  { Name   : string
    Build  : PlanParams -> Pipe<'T,unit> 
    Resource : PlanParams -> ResourceModel
    ParamSpace : PlanParams -> bool }
```

`sink` operator finalises the plan and triggers the optimise→lower→run flow.

## 11) **Runtime telemetry + feedback loop**

* Record observed **peak RSS**, **per-Plan timings**, **IO throughput**.
* Feed back into the **cost model** (simple EMA update of `k,b` parameters).
* On OOM or backpressure anomalies, **auto-shrink** windows and retry once.

## 12) **Migration path**

* Keep `Pipe` API unchanged.
* Wrap each existing `Pipe` into a trivial `PlanSpec` with `Params=None`.
* Introduce windowed specs for the Plans that benefit (conv, morphology, resample).

## 13) **Testing strategy**

* Unit tests for `ArityMap` and `Resource` invariants.
* Property tests: increasing window size should be non-decreasing peak memory.
* Golden tests for optimiser: given `AvailableMB`, chosen params are stable.
* Integration tests on synthetic stacks to verify peak never exceeds cap.

## Minimal slice of work (suggested order)

1. Introduce `PlanSpec`, `Plan`, `Plan`, new `sink`.
2. Wrap two Plans (`readAs`, `convGauss`) and one sink.
3. Implement `infer`, a simple greedy `optimise`, and `lower`.
4. Insert bounded buffers in `lower` and run a real dataset.
5. Add telemetry; tighten cost model; iterate on optimiser.

--------
# Minimal recipe with scafold but no optimization

The idea is:

* Add a new `SlimPipeline.Planning` module that exports the **same DSL names/operators** (`source`, `Plan`, `>=>`, `>=>>`, `>>=>`, `sink`, `-->`) but they work on a lightweight **Plan** instead of immediately composing `Pipe`s.
* In this first step there’s **no optimisation**: `sink` simply **lowers the Plan → Pipe** and runs it.
* In StackProcessing.fs you only change which module you open (and ideally remove a few explicit type annotations if present). The plan code itself stays the same.

Here’s the minimal refactor recipe.

# 1) Add a thin Plan type

```fsharp
type Stage<'S,'T> =
  { Build : unit -> Pipe<'S,'T>
    Name  : string option }
```

# 2) Lightweight lifters

```fsharp
module Stage =
  let ofPipe  (p: Pipe<'S,'T>)  : Stage<'S,'T> = { Build = (fun () -> p); Name = Some p.Name }
  let ofStage (s: Stage<'S,'T>) : Stage<'S,'T> = { Build = (fun () -> s.Pipe); Name = Some s.Name }
```

# 3) Planning operators mirroring the existing DSL

Keep the exact operator names so StackProcessing code doesn’t change.

```fsharp
module SlimPipeline.Planning =
  // aliases to make migration effortless
  let Stage  (s: Stage<'S,'T>)  : Stage<'S,'T> = Stage.ofStage s
  let source (p: Pipe<'S,'T>)   : Stage<'S,'T> = Stage.ofPipe  p
  let sink   (sinkPipe: Pipe<'T,unit>) (pl: Stage<'S,'T>) = async {
      // Phase 1: no optimisation — just lower and run
      let exec = Pipe.compose (pl.Build()) sinkPipe
      do! Pipe.run exec
    }

  // linear comp
  let (>=>) (a: Stage<'S,'T>) (b: Stage<'T,'U>) : Stage<'S,'U> =
    { Build = fun () -> Pipe.compose (a.Build()) (b.Build())
      Name  = Some $"{b.Name |> Option.defaultValue "b"} ∘ {a.Name |> Option.defaultValue "a"}" }

  let (-->) = (>=>)

  // fan-out / fan-in simply delegate to your existing pipe-level impls
  let (>=>>) (p: Stage<'S,'T>) (lr: Stage<'T,'U> * Stage<'T,'V>) : Stage<'S,'U * 'V> =
    let (l,r) = lr
    { Build = fun () -> Pipe.fanOut (p.Build()) (l.Build(), r.Build())
      Name  = Some "fanOut" }

  let (>>=>) (pr: Stage<'S,'T * 'U>) (j: Stage<'T * 'U,'V>) : Stage<'S,'V> =
    { Build = fun () -> Pipe.fanIn (pr.Build()) (j.Build())
      Name  = Some "fanIn" }
```

> Internally you’re composing **builders**, not `AsyncSeq`s. Later you’ll insert the optimiser right before `Build()` is called.

# 4) Migration with (almost) zero edits in StackProcessing.fs

**Best case** (no explicit type annotations):

* Keep plan code the same:

  ```fsharp
  let readMaker = source mem |> Stage (readAs<uint8> "image" ".tiff")
  let plotHist  = Stage map2pairs --> Stage pairs2floats --> Stage (plot plt)

  readMaker
  >=>> ( Stage histogram --> plotHist
       , Stage (cast<uint8,float>) --> Stage (convGauss 1.0 None) --> Stage (cast<float,uint8>)
         --> Stage histogram --> plotHist )
  >>=> Stage combineIgnore
  |> sink sinkPipe
  ```

**If you have explicit types** like `let p: Pipe<_,_> = ...`, remove or relax them (let inference infer `Stage<_,_>`), or flip them to `Stage<_,_>` in a handful of places.

# 5) Why this minimizes churn

* **No API break** for callers: the names and operator shapes are the same.
* StackProcessing doesn’t need to know about DAGs or optimisation yet.
* You can later evolve `Stage` into a true DAG (nodes/edges, resource models) **without touching StackProcessing again**:

  * Enrich `Stage`’s internals to record a graph instead of a single `Build`.
  * Make `sink` do: `analyse → optimise → lower → run`.
  * Keep the same operators/DSL at call sites.

# 6) When you’re ready to optimise

Inside `SlimPipeline.Stagening.sink` change:

```fsharp
let stageGraph = /* materialise DAG from the composed stages */
let params    = Optimiser.solve stageGraph availableMemory
let pipe      = Lowering.toPipe stageGraph params
do! Pipe.run (Pipe.compose pipe sinkPipe)
```

No changes to StackProcessing.
