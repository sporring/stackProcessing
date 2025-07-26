module Routing

open FSharp.Control
open AsyncSeqExtensions
open Core
open Core.Helpers
open Slice

let run (p: Pipe<unit,'T>) : AsyncSeq<'T> =
    printfn "[run]"
    (AsyncSeq.singleton ()) |> p.Apply

/// Split a Pipe<'In,'T> into two branches that
///   • read the upstream only once
///   • keep at most one item in memory
///   • terminate correctly when both sides finish
type private Request<'T> = Left of AsyncReplyChannel<Option<'T>> | Right of AsyncReplyChannel<Option<'T>>
let tee (p : Pipe<'In,'T>): Pipe<'In,'T> * Pipe<'In,'T> =
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
    let mkSide name pick =
        { 
        Name  = $"{p.Name}-{name}"
        Profile = p.Profile
        Apply   = fun input ->
                        let left, right = getShared input
                        pick (left, right) }

    mkSide "left"  fst,
    mkSide "right" snd

let teeOp (op: Operation<'In, 'T>) : Operation<'In, 'T> * Operation<'In, 'T> =
    let leftPipe, rightPipe = tee op.Pipe

    let mk name pipe =
        {
            Name = $"{op.Name} ⊣ {name}"
            Transition = op.Transition  // preserve transition
            Pipe = pipe
        }

    mk "left" leftPipe,
    mk "right" rightPipe

let teePipeline (pl: Builder.Pipeline<'In, 'T>) : Builder.Pipeline<'In, 'T> * Builder.Pipeline<'In, 'T> =
    let op, mem', shape = pl.flow pl.mem pl.shape
    let leftOp, rightOp = teeOp op
    let plLeft: Builder.Pipeline<'In, 'T>  = { flow = Builder.returnM leftOp; mem = mem'; shape = shape }
    let plRight: Builder.Pipeline<'In, 'T> = { flow = Builder.returnM rightOp; mem = mem'; shape = shape }
    plLeft, plRight

/// zipWith two Pipes<'In, _> into one by zipping their outputs:
///   • applies both processors to the same input stream
///   • pairs each output and combines using the given function
///   • assumes both sides produce values in lockstep
let zipWithOld (f: 'A -> 'B -> 'C) (p1: Pipe<'In, 'A>) (p2: Pipe<'In, 'B>) : Pipe<'In, 'C> =
    printfn "[zipWith]"
    let name = $"zipWith({p1.Name}, {p2.Name})"
    let profile = p1.Profile.combineProfile p2.Profile
    {
        Name = name
        Profile = profile
        Apply = fun input ->
            let a = p1.Apply input
            let b = p2.Apply input
            match p1.Profile, p2.Profile with
            | Full, Streaming | Streaming, Full ->
                failwithf "[zipWith] Mixing Full and Streaming is not supported: %s, %s"
                          (p1.Profile.ToString()) (p2.Profile.ToString())
            | Constant, _ ->
                printfn "[Runtime analysis: zipWith sequential]"
                asyncSeq {
                    let! constant =
                        a
                        |> AsyncSeq.tryLast
                        |> Async.map (function
                            | Some v -> v
                            | None -> failwithf "[zipWith] Constant pipe '%s' returned no result." p1.Name)
                    yield! b |> AsyncSeq.map (fun b -> f constant b)
                }
            | _, Constant ->
                printfn "[Runtime analysis: zipWith sequential]"
                asyncSeq {
                    let! constant =
                        b
                        |> AsyncSeq.tryLast
                        |> Async.map (function
                            | Some v -> v
                            | None -> failwithf "[zipWith] Constant pipe '%s' returned no result." p2.Name)
                    yield! a |> AsyncSeq.map (fun a -> f a constant)
               }
            | _ ->
                printfn "[Runtime analysis: zipWith parallel]"
                AsyncSeq.zip a b |> AsyncSeq.map (fun (x, y) -> f x y)
    }

let zipWithPipeOld
    (f: 'A -> 'B -> Pipe<'In, 'C>)
    (pa: Pipe<'In, 'A>)
    (pb: Pipe<'In, 'B>) : Pipe<'In, 'C> =

    zipWithOld f pa pb |> bindPipe

let zipWithOp (f: 'A -> 'B -> 'C)
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
                    asyncSeq {
                        let! constant = a |> AsyncSeq.tryLast |> Async.map (Option.defaultWith (fun () -> failwith $"No constant result from {op1.Name}"))
                        yield! b |> AsyncSeq.map (fun b -> f constant b)
                    }
                | _, Constant ->
                    asyncSeq {
                        let! constant = b |> AsyncSeq.tryLast |> Async.map (Option.defaultWith (fun () -> failwith $"No constant result from {op2.Name}"))
                        yield! a |> AsyncSeq.map (fun a -> f a constant)
                    }
                | _ ->
                    AsyncSeq.zip a b |> AsyncSeq.map (fun (x, y) -> f x y)
        }

    {
        Name = name
        Transition = transition op1.Transition.From op2.Transition.To
        Pipe = pipe
    }

let zipWith (f: 'A -> 'B -> 'C)
            (p1: Builder.Pipeline<'In, 'A>)
            (p2: Builder.Pipeline<'In, 'B>) : Builder.Pipeline<'In, 'C> =
    let flow (mem: uint64) (shape: SliceShape option) =
        let op1, mem1, shape1 = p1.flow mem shape
        let op2, mem2, shape2 = p2.flow mem1 shape1
        let zipped = zipWithOp f op1 op2
        zipped, mem2, shape2

    {
        mem = min p1.mem p2.mem
        shape = p1.shape |> Option.orElse p2.shape
        flow = flow
    }

// Needs to be updated for *Constant MemoryProfiles
let cacheScalar (name: string) (p: Pipe<unit, 'T>) : Pipe<'In, 'T> =
    printfn "[cacheScalar]"
    let result =
        run p
        |> AsyncSeq.tryLast
        |> Async.RunSynchronously
        |> function
           | Some x -> x
           | None -> failwithf "[cacheScalar] No output from pipeline '%s'" p.Name

    lift $"cacheScalar: {name}" Constant (fun _ -> async.Return result)

let cacheScalarOp (name: string) (pl: Builder.Pipeline<'In, 'T>) : Operation<'In, 'T> =
    let op, _, _ = pl.flow pl.mem pl.shape
    let pipe = op.Pipe

    // Run the pipe with dummy input to extract a single value
    let dummyInput = AsyncSeq.singleton Unchecked.defaultof<'In>

    let result =
        pipe.Apply dummyInput
        |> AsyncSeq.tryLast
        |> Async.RunSynchronously
        |> function
           | Some value -> value
           | None -> failwithf "[cacheScalar] No result from pipeline '%s'" pipe.Name

    // Return new constant-producing operation
    {
        Name = $"cacheScalar({name})"
        Transition = transition Constant Constant
        Pipe = lift $"cacheScalar({name})" Constant (fun _ -> async.Return result)
    }

let runToScalar name (reducer: AsyncSeq<'T> -> Async<'R>) (pl: Builder.Pipeline<'In, 'T>) : 'R =
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
let map (label: string) (profile: MemoryProfile) (f: 'S -> 'T) 
    : Pipe<'S,'T> =
    {
        Name = label; 
        Profile = profile
        Apply = fun input -> input |> AsyncSeq.map f
    }

let liftUnaryOp name (f: Slice<'T> -> Slice<'T>) 
    : Operation<Slice<'T>,Slice<'T>> =
    { 
        Name = name
        Transition = transition Streaming Streaming
        Pipe = map name Streaming f
    }

let tapOp (label: string) : Operation<'T, 'T> =
    liftUnaryOp $"tap: {label}" (fun x ->
        printfn "[%s] %A" label x
        x)

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