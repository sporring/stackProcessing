# Problem Summary

We want to:

1. **Defer pipeline application** until the optimal memory configuration (e.g., sliding window sizes) has been determined.
2. **Reason statically** about memory consumption **before actual execution**.
3. Be able to **recompute derived values like `MemoryNeed`, `Profile`, etc.**, as a function of optimisable parameters.

Currently, `Stage` has already "baked in" the concrete `Pipe<'S,'T>`, so the opportunity to modify `Stage`'s behaviour based on later optimisations is lost.

 

## Option 1: Turn `Stage` into a deferred computation description

Introduce a new abstraction (e.g., `StageSpec`) that represents **a parametrised stage** rather than a concrete applied one.

```fsharp
type StageSpec<'S,'T> = {
    Name: string
    ProfileFn: Profile
    TransitionFn: ProfileTransition
    MemoryNeedFn: uint64 -> SingleOrPair -> SingleOrPair
    NElemsTransform: uint64 -> uint64
    Build: (Profile -> Pipe<'S,'T>) // this allows regeneration of Pipe based on parameters
}
```

Then use this to **delay building the actual `Stage` until after analysis and optimisation.** For example:

```fsharp
type Stage<'S,'T> = {
    Spec: StageSpec<'S,'T>
    Params: { WindowSize: int; Stride: int; ... } option // optional
}
```

Now, instead of composing `Stage` by composing actual `Pipe`s, you compose `StageSpec`, and only convert to `Pipe` at the very end when all the parameters have been decided.

This allows:

* Static reasoning
* Late binding of window parameters
* Memory and profile estimation before execution

 

## Option 2: Keep `Stage` but abstract over memory and transformation computations

If you'd rather not split out `StageSpec`, you can instead modify `Stage` like this:

```fsharp
type Stage<'S,'T> =
    { Name       : string
      BuildPipe  : unit -> Pipe<'S,'T> // Delay building
      Transition : ProfileTransition
      MemoryNeed : SingleOrPair -> SingleOrPair
      NElemsTransformation : uint64 -> uint64
    }
```

Now your pipeline composition does **not** create a `Pipe<'S,'U>` immediately, only builds it lazily. This still gives you:

* Deferred execution
* Ability to "recompile" after optimisation
* Preservation of composition logic

This is a smaller change, but less flexible than `StageSpec`.

 

## Add Sliding Window Parameter Search

Now, to support **window optimisation**, you can:

1. Walk the `Pipeline` of `Stage`s.
2. Detect `Sliding` profiles.
3. Consider window size/stride as **hyperparameters**.
4. Simulate `MemoryNeed` and check it against `memAvail`.

You can define:

```fsharp
type StageParameter =
    | SlidingWindow of window: int * stride: int * padding: int
    | NoParams
```

And implement a pipeline optimiser:

```fsharp
let optimisePipeline (pipeline: Stage<_,'_> list) (memAvail: uint64) : Stage<_,'_> list =
    // heuristic search or grid search over Sliding params
    // evaluate using Stage.MemoryNeed and estimated mem usage
    ...
```

Then after optimisation, build the final pipeline:

```fsharp
let realisePipeline (stages: Stage<_,'_> list) : Pipe<_,_> =
    stages
    |> List.map (fun s -> s.BuildPipe())
    |> List.reduce Pipe.compose
```

 

## Bonus: Add Profiling Metadata to `Pipeline`

This is where your `Pipeline<'S,'T>` type can shine. It can hold:

* A tree or list of `Stage`s
* Memory simulation
* Parameter candidates
* Evaluation metrics (estimated latency, IO cost, etc.)

```fsharp
type Pipeline<'S,'T> = {
    Stages: Stage<'S,'T> list
    memAvail: uint64
    memPeak: uint64
    ...
}
```

 

## Summary

| Task                                 | Recommendation                                        |
| ------------------------------------ | ----------------------------------------------------- |
| Decouple execution from construction | Introduce `StageSpec` or make `Pipe` creation lazy    |
| Support window optimisation          | Add parameterisable `Stage`s and simulation of memory |
| Preserve composability               | Keep `Stage --> Stage` composition at spec level      |
| Final pipeline realisation           | Only build `Pipe` once optimisation is complete       |

