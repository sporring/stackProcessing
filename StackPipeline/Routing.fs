module Routing

open FSharp.Control
open AsyncSeqExtensions
open Slice
open Core

let private runWith (input: AsyncSeq<'In>) (p: Pipe<'In,'T>) : AsyncSeq<'T> =
    p.Apply input

let private run (p: Pipe<unit,'T>) : AsyncSeq<'T> =
    runWith (AsyncSeq.singleton ()) p

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
    (processors: Pipe<unit,Slice.Slice<'T>> list) 
    : Pipe<unit,Slice<'T>> list =
    processors |>
    List.map (fun p -> 
        pipeline availableMemory width height depth {return p}
    )

let source
    (availableMemory: uint64)
    (width: uint)
    (height: uint)
    (depth: uint)
    (p: Pipe<unit,Slice.Slice<'T>>) 
    : Pipe<unit,Slice<'T>> =
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

/// Fan out a Pipe<'In,'T> to two branches:
///   • processes input once using tee
///   • applies separate processors to each branch
///   • zips outputs into a tuple
let fanOut (p: Pipe<'In,'T>) (f1: Pipe<'T,'U>) (f2: Pipe<'T,'V>) : Pipe<'In, 'U * 'V> =
    printfn "[fanOut]"

    let left, right = tee p
    {
        Name = $"fanout2 ({f1.Name}, {f2.Name})"
        Profile = Buffered
        Apply = fun input ->
            let lStream = left.Apply input |> f1.Apply
            let rStream = right.Apply input |> f2.Apply
            AsyncSeq.zip lStream rStream
    }
/// zipWith two Pipes<'In, _> into one by zipping their outputs:
///   • applies both processors to the same input stream
///   • pairs each output and combines using the given function
///   • assumes both sides produce values in lockstep
let zipWith (f: 'A -> 'B -> 'C) (p1: Pipe<'In, 'A>) (p2: Pipe<'In, 'B>) : Pipe<'In, 'C> =
    printfn "[zipWith]"
    {
        Name = $"zipWith({p1.Name}, {p2.Name})"
        Profile = p1.Profile.combineProfile p2.Profile
        Apply = fun input ->
            let a = p1.Apply input
            let b = p2.Apply input
            AsyncSeq.zip a b |> AsyncSeq.map (fun (x, y) -> f x y)
    }

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

let inject
    (f: 'A -> 'B -> 'C)
    (scalarProc: Pipe<'In, 'A>)
    (streamProc: Pipe<'In, 'B>)
    : Pipe<'In, 'C> =
    printfn "[inject]"
    {
        Name = $"inject({streamProc.Name}, {scalarProc.Name})"
        Profile = streamProc.Profile.combineProfile scalarProc.Profile
        Apply = fun input -> asyncSeq {
            // Evaluate the scalar processor first
            let! scalar =
                scalarProc.Apply input
                |> AsyncSeq.tryLast // could also use head, if only one expected
                |> Async.map (function
                    | Some v -> v
                    | None   -> failwithf "[inject] No value from scalar processor '%s'" scalarProc.Name)

            // Now stream the input through streamProc
            let stream = streamProc.Apply input
            yield! stream |> AsyncSeq.map (fun s -> f scalar s)
        }
    }

let (>>~>) s (f, a) = inject f a s

/// Combine two <c>Pipe</c> instances into one by composing their memory profiles and transformation functions.
let composePipe (p1: Pipe<'S,'T>) (p2: Pipe<'T,'U>) : Pipe<'S,'U> =
    printfn "[composePipe]"
    {
        Name = $"{p2.Name} {p1.Name}"; 
        Profile = p1.Profile.combineProfile p2.Profile
        Apply = fun input -> input |> p1.Apply |> p2.Apply
    }

let (>>=>) p1 p2 = composePipe p1 p2

let injectPipe
    (streamProc: Pipe<'In, 'A>)
    (f: 'B -> 'A -> 'C, reducerProc: Pipe<'In, 'B>)
    : Pipe<'In, 'C> =
    inject f reducerProc streamProc

let tap label : Pipe<'T, 'T> =
    printfn "[tap]"
    lift $"tap: {label}" Streaming (fun x ->
        printfn "[%s] %A" label x
        async.Return x)
