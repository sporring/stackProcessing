module Routing

open FSharp.Control
open AsyncSeqExtensions
open Core
open Slice

(*
/// zipWith two Pipes<'In, _> into one by zipping their outputs:
///   • applies both processors to the same input stream
///   • pairs each output and combines using the given function
///   • assumes both sides produce values in lockstep
let internal zipWithOp (f: 'A -> 'B -> 'C)
              (op1: Stage<'In, 'A, 'Shape>)
              (op2: Stage<'In, 'B, 'Shape>) : Stage<'In, 'C, 'Shape> =
    let name = $"zipWith({op1.Name}, {op2.Name})"
    let profile = MemoryProfile.combine op1.Pipe.Profile op2.Pipe.Profile
    let pipe =
        {
            Name = name
            Profile = profile
            Apply = fun input ->
                let a = op1.Pipe.Apply input
                let b = op2.Pipe.Apply input
                match op1.Pipe.Profile, op2.Pipe.Profile with
                | Full, Streaming | Streaming, Full ->
                    failwithf "[zipWith] Mixing Full and Streaming not supported: %s, %s"
                              (op1.Pipe.Profile.ToString()) (op2.Pipe.Profile.ToString())
                | Constant, _ ->
                    printfn "[Runtime analysis: zipWith sequential]"
                    asyncSeq {
                        let! constant = 
                            a 
                            |> AsyncSeq.tryLast 
                            |> Async.map (Option.defaultWith (fun () -> failwith $"No constant result from {op1.Name}"))
                        yield! b |> AsyncSeq.map (fun b -> f constant b)
                    }
                | _, Constant ->
                    printfn "[Runtime analysis: zipWith sequential]"
                    asyncSeq {
                        let! constant = 
                            b 
                            |> AsyncSeq.tryLast 
                            |> Async.map (Option.defaultWith (fun () -> failwith $"No constant result from {op2.Name}"))
                        yield! a |> AsyncSeq.map (fun a -> f a constant)
                    }
                | _ ->
                    printfn "[Runtime analysis: zipWith parallel]"
                    AsyncSeq.zip a b |> AsyncSeq.map (fun (x, y) -> f x y)
        }

    {
        Name = name
        Transition = Stage.transition op1.Transition.From op2.Transition.To
        Pipe = pipe
    }

let zipWith (f: 'A -> 'B -> 'C) (p1: Pipeline<'In, 'A>) (p2: Pipeline<'In, 'B>) : Pipeline<'In, 'C> =
    let flow (mem: uint64) (shape: SliceShape option) =
        let op1, mem1, shape1 = p1.flow mem shape
        let op2, mem2, shape2 = p2.flow mem1 shape1
        let zipped = zipWithOp f op1 op2
        zipped, mem2, shape2

    {
        flow = flow
        mem = min p1.mem p2.mem
        shape = p1.shape |> Option.orElse p2.shape
    }
*)

let runToScalar name (reducer: AsyncSeq<'T> -> Async<'R>) (pl: Pipeline<'In, 'T,'Shape>) : 'R =
    let op, _, _ = pl.flow pl.mem pl.shape pl.context
    let pipe = op.Pipe
    let input = AsyncSeq.singleton Unchecked.defaultof<'In>
    pipe.Apply input |> reducer |> Async.RunSynchronously

let drainSingle name pl =
    runToScalar name AsyncSeq.toListAsync pl
    |> function
        | [x] -> x
        | []  -> failwith $"[drainSingle] No result from {name}"
        | _   -> failwith $"[drainSingle] Multiple results from {name}, expected one."

let drainList name pl =
    runToScalar name AsyncSeq.toListAsync pl

let drainLast name pl =
    runToScalar name AsyncSeq.tryLast pl
    |> function
        | Some x -> x
        | None -> failwith $"[drainLast] No result from {name}"

/// Represents a pipeline that has been shared (split into synchronized branches)
type SharedPipeline<'T, 'U , 'V, 'Shape> = {
    flow: MemFlow<'T,'V,'Shape>
    branches: Stage<'T,'U,'Shape> * Stage<'T,'V,'Shape>
    mem: uint64
    shape: 'Shape option
}

/// parallel fanout with synchronization
/// Synchronously split the shared stream into two parallel pipelines
let (>=>>) 
    (pl: Pipeline<'In, 'T, 'Shape>) 
    (op1: Stage<'T, 'U, 'Shape>, op2: Stage<'T, 'V, 'Shape>) 
    : SharedPipeline<'In, 'U, 'V, 'Shape> =

    match pl.flow pl.mem pl.shape pl.context with
    | baseOp, mem', shape' when baseOp.Transition.To = Streaming ->

        let flow mem shape context =
            let opBase, mem', shape' = pl.flow mem shape context
            let pipe1, pipe2 = Pipe.tee opBase.Pipe
            let op1' = Stage.compose { opBase with Pipe = pipe1 } op1
            let op2' = Stage.compose { opBase with Pipe = pipe2 } op2
            ((op1', op2'), mem', shape')

        // Construct branches using same logic to fill .branches field
        let (op1b, op2b), _, _ = flow pl.mem pl.shape pl.context
        {
            flow = flow
            branches = (op1b, op2b)
            mem = pl.mem
            shape = pl.shape
        }

    | baseOp, mem', shape' when baseOp.Transition.To = Constant ->

        let cached = lazy (
            let result =
                AsyncSeq.singleton Unchecked.defaultof<'In>
                |> baseOp.Pipe.Apply
                |> AsyncSeq.tryLast
                |> Async.RunSynchronously
            result |> Option.defaultWith (fun () -> failwithf "No constant result from %s" baseOp.Name)
        )

        let applyOp (op: Stage<'T, 'X, 'Shape>) (value: 'T) : 'X =
            AsyncSeq.singleton value
            |> op.Pipe.Apply
            |> AsyncSeq.tryLast
            |> Async.RunSynchronously
            |> Option.defaultWith (fun () -> failwithf "applyOp: No output from %s" op.Name)

        let makeConstOp (op: Stage<'T, 'X, 'Shape>) label : Stage<'In, 'X, 'Shape> =
            {
                Name = $"shared-const:{label}"
                Transition = Stage.transition Constant Constant
                Pipe = Pipe.lift label Constant (fun _ -> async { return applyOp op (cached.Value) })
                ShapeUpdate = fun s -> s // I don't know what this should be!!!!
            }

        let op1' = makeConstOp op1 "left"
        let op2' = makeConstOp op2 "right"

        {
            flow = fun _ _ -> ((op1', op2'), mem', shape')
            branches = (op1', op2')
            mem = pl.mem
            shape = pl.shape
        }

    | _ -> failwith "Unsupported transition kind in >=>>"

let (>>=>)
    (shared: SharedPipeline<'In, 'U, 'V, 'Shape>)
    (combine: Stage<'In, 'U, 'Shape> * Stage<'In, 'V, 'Shape> -> Stage<'In, 'W, 'Shape>)
    : Pipeline<'In, 'W, 'Shape> =

    let opU, opV = shared.branches
    let op = combine (opU, opV)
    { flow = MemFlow.returnM op; mem = shared.mem; shape = shared.shape }

let combineIgnore : Stage<'In, 'U, 'Shape> * Stage<'In, 'V, 'Shape> -> Stage<'In, unit, 'Shape> =
    fun (op1, op2) ->
        {
            Name = $"combineIgnore({op1.Name}, {op2.Name})"
            Transition = Stage.transition op1.Transition.From op2.Transition.To
            Pipe =
                {
                    Name = "combineIgnore"
                    Profile = MemoryProfile.combine op1.Pipe.Profile op2.Pipe.Profile
                    Apply = fun input ->
                        let out1 = op1.Pipe.Apply input |> AsyncSeq.iterAsync (fun _ -> async.Return())
                        let out2 = op2.Pipe.Apply input |> AsyncSeq.iterAsync (fun _ -> async.Return())
                        asyncSeq {
                            do! Async.Parallel [ out1; out2 ] |> Async.Ignore
                            yield ()
                        }
                }
        }
