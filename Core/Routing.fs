module Routing

open FSharp.Control
open AsyncSeqExtensions
open Core
open Slice

/// Split a Pipe<'In,'T> into two branches that
///   • read the upstream only once
///   • keep at most one item in memory
///   • terminate correctly when both sides finish
type private Request<'T> =
    | Left of AsyncReplyChannel<Option<'T>>
    | Right of AsyncReplyChannel<Option<'T>>

let internal tee (p : Pipe<'In,'T>): Pipe<'In, 'T> * Pipe<'In, 'T> =
    printfn "[tee]"

    // ---------- 1. Create the broadcaster exactly *once* per pipeline run ----------
    // We store it in a mutable cell that is initialised on first Apply().
    let mutable shared : Lazy<AsyncSeq<'T> * AsyncSeq<'T>> option = None
    let syncRoot = obj()

    let makeShared (input : AsyncSeq<'In>) =
        let src = p.Apply input                        // drives upstream only ONCE
        let agent = MailboxProcessor.Start(fun inbox ->
            async {
                let enum = (AsyncSeq.toAsyncEnum src).GetAsyncEnumerator()
                let mutable current   : 'T option = None
                let mutable consumed  = (true, true)
                let mutable finished  = false

                let rec loop () = async {
                    // Pull next slice if needed
                    if current.IsNone && not finished then
                        let! hasNext = enum.MoveNextAsync().AsTask() |> Async.AwaitTask
                        if hasNext then
                            current  <- Some enum.Current
                            consumed <- (false, false)
                        else
                            finished <- true

                    // Serve whichever side is asking
                    let! msg = inbox.Receive()
                    match msg, current with
                    | Left ch, Some v when not (fst consumed) ->
                        ch.Reply(Some v)
                        consumed <- (true, snd consumed)
                    | Right ch, Some v when not (snd consumed) ->
                        ch.Reply(Some v)
                        consumed <- (fst consumed, true)
                    | Left ch, None when finished ->
                        ch.Reply(None)
                    | Right ch, None when finished ->
                        ch.Reply(None)
                    | _ -> ()

                    // Release the slice when both sides have seen it
                    if consumed = (true, true) then current <- None
                    return! loop ()
                }
                do! loop ()
            })

        // Helper to build one of the two consumer streams
        let makeStream tag =
            asyncSeq {
                let mutable done_ = false
                while not done_ do
                    let! vOpt = agent.PostAndAsyncReply(tag)
                    match vOpt with
                    | Some v -> yield v
                    | None   -> done_ <- true
            }

        makeStream Left, makeStream Right

    // Thread‑safe initialisation-on-first-use
    let getShared input =
        match shared with
        | Some lazyStreams -> lazyStreams.Value
        | None ->
            lock syncRoot (fun () ->
                match shared with
                | Some lazyStreams -> lazyStreams.Value
                | None ->
                    let lazyStreams = lazy (makeShared input)
                    shared <- Some lazyStreams
                    lazyStreams.Value)

    // ---------- 2. Build the two outward‑facing Pipes ----------
    let mkPipe name pick =
        { 
        Name  = $"{p.Name} - {name}"
        Profile = p.Profile
        Apply   = 
            fun input ->
                let left, right = getShared input
                pick (left, right) 
        }

    mkPipe "left" fst, mkPipe "right" snd

let internal teeOp (op: Operation<'In, 'T>) : Operation<'In, 'T> * Operation<'In, 'T> =
    let leftPipe, rightPipe = tee op.Pipe
    let mk name pipe = 
        {Name = $"{op.Name} - {name}";  Transition = op.Transition; Pipe = pipe }
    mk "left" leftPipe,
    mk "right" rightPipe

let teePipeline (pl: Pipeline<'In, 'T>) : Pipeline<'In, 'T> * Pipeline<'In, 'T> =
    let op, mem, shape = pl.flow pl.mem pl.shape
    let leftOp, rightOp = teeOp op
    let plLeft: Pipeline<'In, 'T>  = 
        { flow = returnM leftOp; mem = mem; shape = shape }
    let plRight: Pipeline<'In, 'T> = 
        { flow = returnM rightOp; mem = mem; shape = shape }
    plLeft, plRight

/// zipWith two Pipes<'In, _> into one by zipping their outputs:
///   • applies both processors to the same input stream
///   • pairs each output and combines using the given function
///   • assumes both sides produce values in lockstep
let internal zipWithOp (f: 'A -> 'B -> 'C)
              (op1: Operation<'In, 'A>)
              (op2: Operation<'In, 'B>) : Operation<'In, 'C> =
    let name = $"zipWith({op1.Name}, {op2.Name})"
    let profile = op1.Pipe.Profile.combineProfile op2.Pipe.Profile
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
        Transition = transition op1.Transition.From op2.Transition.To
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

let runToScalar name (reducer: AsyncSeq<'T> -> Async<'R>) (pl: Pipeline<'In, 'T>) : 'R =
    let op, _, _ = pl.flow pl.mem pl.shape
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

let tap label : Pipe<'T, 'T> =
    printfn "[tap]"
    lift $"tap: {label}" Streaming (fun x ->
        printfn "[%s] %A" label x
        async.Return x)

/// quick constructor for Streaming→Streaming unary ops
let liftUnaryOp name (f: Slice<'T> -> Slice<'T>) 
    : Operation<Slice<'T>,Slice<'T>> =
    { 
        Name = name
        Transition = transition Streaming Streaming
        Pipe = map name Streaming f
    }

let tapOp (label: string) : Operation<'T, 'T> =
    let _print x = 
        printfn "[%s] %A" label x
        x // return x such that Pipe below gets type of x
    { 
        Name = $"tap: {label}"
        Transition = transition Streaming Streaming
        Pipe = map label Streaming _print
    }

(*
/////////////////////////////////////
// Not yet transformed to Pipeline-operation version. Are they needed?

let run (p: Pipe<unit,'T>) : AsyncSeq<'T> =
    printfn "[run]"
    (AsyncSeq.singleton ()) |> p.Apply

let sequentialJoin (p1: Pipe<'S, 'T>) (p2: Pipe<'S, 'T>) : Pipe<'S, 'T> =
    {
        Name = p1.Name + " ++ " + p2.Name
        Profile = p1.Profile.combineProfile p2.Profile
        Apply = fun input ->
            asyncSeq {
                yield! p1.Apply input
                yield! p2.Apply input
            }
    }
*)

/// Represents a pipeline that has been shared (split into synchronized branches)
type SharedPipeline<'T, 'U , 'V> = {
    flow: uint64 -> SliceShape option -> (Operation<'T, 'U> * Operation<'T, 'V>) * uint64 * SliceShape option
    branches: Operation<'T,'U> * Operation<'T,'V>
    mem: uint64
    shape: SliceShape option
}

(*
/// Share a pipeline by exposing its underlying stream source
let share (pl: Pipeline<'In, 'T>) : SharedPipeline<'In, 'T> =
    let op, mem', shape' = pl.flow pl.mem pl.shape
    { source = op.Pipe }
*)

/// parallel fanout with synchronization
/// Synchronously split the shared stream into two parallel pipelines
let (>=>>) (pl: Pipeline<'In, 'T>) (op1: Operation<'T, 'U>, op2: Operation<'T, 'V>) : SharedPipeline<'In, 'U, 'V> =
    let flow mem shape =
        let baseOp, mem', shape' = pl.flow mem shape
        let pipe1, pipe2 = tee baseOp.Pipe
        let op1' = composeOp { baseOp with Pipe = pipe1 } op1
        let op2' = composeOp { baseOp with Pipe = pipe2 } op2
        ((op1', op2'), mem', shape')

    {
        flow = flow
        branches = flow pl.mem pl.shape |> (fun ((a, b), _, _) -> (a, b))
        mem = pl.mem
        shape = pl.shape
    }

let (>>=>>) 
    (pl: SharedPipeline<'In, 'U, 'V>) 
    (combine: (Pipeline<'In, 'U> * Pipeline<'In, 'V>) -> (Pipeline<'In, 'M> * Pipeline<'In, 'N>)) 
    : SharedPipeline<'In, 'M, 'N> =

    // Turn the current operations into full pipelines
    let toPipeline op =
        { flow = returnM op; mem = pl.mem; shape = pl.shape }

    let p1 = toPipeline (fst pl.branches)
    let p2 = toPipeline (snd pl.branches)

    let p1', p2' = combine (p1, p2)

    // Run the new pipelines to get composed operations
    let composed1, _, _ = p1'.flow pl.mem pl.shape
    let composed2, _, _ = p2'.flow pl.mem pl.shape

    {
        flow = pl.flow
        branches = (composed1, composed2)
        mem = pl.mem
        shape = pl.shape
    }

let (>>=>)
    (shared: SharedPipeline<'In, 'U, 'V>)
    (combine: (Pipeline<'In, 'U> * Pipeline<'In, 'V>) -> Pipeline<'In, 'W>)
    : Pipeline<'In, 'W> =

    let opU, opV = shared.branches

    let leftPipeline : Pipeline<'In, 'U> =
        { flow = returnM opU; mem = shared.mem; shape = shared.shape }

    let rightPipeline : Pipeline<'In, 'V> =
        { flow = returnM opV; mem = shared.mem; shape = shared.shape }

    combine (leftPipeline, rightPipeline)

let unitPipeline<'T> () : Pipeline<'T, unit> =
    let op : Operation<'T, unit> =
        {
            Name = "unit"
            Transition = transition Streaming Streaming
            Pipe = lift "unit" Streaming (fun _ -> async.Return ())
        }
    { flow = returnM op; mem = 0UL; shape = None }

let combineIgnore (left: Pipeline<'In, 'U>, right: Pipeline<'In, 'V>) : Pipeline<'In, unit> =
    let runBothAndIgnore =
        zipWith (fun _ _ -> ()) left right
    runBothAndIgnore
