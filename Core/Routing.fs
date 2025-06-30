module Routing

open FSharp.Control
open AsyncSeqExtensions
open Core
open Core.Helpers
open Slice

let private run (p: Pipe<unit,'T>) : AsyncSeq<'T> =
    (AsyncSeq.singleton ()) |> p.Apply


let sinkLst (processors: Pipe<unit, unit> list) : unit =
    processors
    |> List.map (fun p -> run p |> AsyncSeq.iterAsync (fun () -> async.Return()))
    |> Async.Parallel
    |> Async.Ignore
    |> Async.RunSynchronously

let sink (p: Pipe<unit, unit>) : unit = 
    sinkLst [p]

let sourceLst 
    (availableMemory: uint64)
    (width: uint)
    (height: uint)
    (depth: uint)
    (processors: Pipe<unit,'T> list) 
    : Pipe<unit,'T> list =
    processors |>
    List.map (fun p -> 
        pipeline availableMemory width height depth {return p}
    )

let source
    (availableMemory: uint64)
    (width: uint)
    (height: uint)
    (depth: uint)
    (p: Pipe<unit,'T>) 
    : Pipe<unit,'T> =
    let lst = sourceLst availableMemory width height depth [p]
    List.head lst

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

/// zipWith two Pipes<'In, _> into one by zipping their outputs:
///   • applies both processors to the same input stream
///   • pairs each output and combines using the given function
///   • assumes both sides produce values in lockstep
let zipWith (f: 'A -> 'B -> 'C) (p1: Pipe<'In, 'A>) (p2: Pipe<'In, 'B>) : Pipe<'In, 'C> =
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
            | Constant, _ 
            | StreamingConstant, _
            | SlidingConstant _, _
            | FullConstant, _ ->
                printfn "[Sequential zipWith]"
                asyncSeq {
                    let! constant =
                        a
                        |> AsyncSeq.tryLast
                        |> Async.map (function
                            | Some v -> v
                            | None -> failwithf "[zipWith] Constant pipe '%s' returned no result." p1.Name)
                    yield! b |> AsyncSeq.map (fun b -> f constant b)
                }
            | _, Constant
            | _, StreamingConstant
            | _, SlidingConstant _
            | _, FullConstant ->
                printfn "[Sequential zipWith]"
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
                printfn "[parallel zipWith]"
                AsyncSeq.zip a b |> AsyncSeq.map (fun (x, y) -> f x y)
    }

let zipWithPipe
    (f: 'A -> 'B -> Pipe<'In, 'C>)
    (pa: Pipe<'In, 'A>)
    (pb: Pipe<'In, 'B>) : Pipe<'In, 'C> =

    zipWith f pa pb |> bindPipe

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

/// Combine two <c>Pipe</c> instances into one by composing their memory profiles and transformation functions.
let composePipe (p1: Pipe<'S,'T>) (p2: Pipe<'T,'U>) : Pipe<'S,'U> =
    printfn "[composePipe]"
    {
        Name = $"{p2.Name} {p1.Name}"; 
        Profile = p1.Profile.combineProfile p2.Profile
        Apply = fun input -> input |> p1.Apply |> p2.Apply
    }

let (>=>) p1 p2 = composePipe p1 p2
let (<=<) p1 p2 = composePipe p2 p1

let tap label : Pipe<'T, 'T> =
    printfn "[tap]"
    lift $"tap: {label}" Streaming (fun x ->
        printfn "[%s] %A" label x
        async.Return x)

let validate op1 op2 =
    if op1.Transition.To = op2.Transition.From then
        true
    else
        failwithf "Memory transition mismatch: %A → %A" op1.Transition.To op2.Transition.From

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