module Routing

open FSharp.Control
open AsyncSeqExtensions
open Core
open Core.Helpers
open Slice

module internal Builder =

    // ───── 1. the state monad ──────────────────────────────────────────────
    type MemFlow<'S,'T> =
         uint64 -> SliceShape option -> Operation<'S,'T> * uint64 * SliceShape option

    // How many bytes does this pipe claim for one full slice?
    let private memNeed (shape: uint list) (p : Pipe<_,_>) =
        p.Profile.EstimateUsage shape[0] shape[1] shape[2]

    /// Try to shrink a too‑hungry pipe to a cheaper profile.
    /// *You* control the downgrade policy here.
    let private shrinkProfile avail shape (p : Pipe<'S,'T>) =
        let needed = memNeed shape p
        if needed <= avail then p                              // fits as is
        else
            match p.Profile with
            | FullConstant -> { p with Profile = StreamingConstant }
            | Full         -> { p with Profile = Streaming }
            | SlidingConstant w when w > 1u ->
                { p with Profile = SlidingConstant (w/2u) }
            | Sliding w when w > 1u ->
                { p with Profile = Sliding (w/2u) }
            | _ ->
                failwith
                    $"Stage “{p.Name}” needs {needed}s B but only {avail}s B are left."

    let returnM (op : Operation<'S,'T>) : MemFlow<'S,'T> =
        fun bytes shapeOpt ->
            match shapeOpt with
            | None      -> op, bytes, shapeOpt       // can’t check memory yet
            | Some sh   ->
                let p'     = shrinkProfile bytes sh op.Pipe
                let need   = memNeed sh p'
                { op with Pipe = p' }, bytes - need, shapeOpt

    let composeOp (op1 : Operation<'S,'T>) (op2 : Operation<'T,'U>) : Operation<'S,'U> =
        {
            Name       = $"{op2.Name} ∘ {op1.Name}"
            Transition = transition op1.Transition.From op2.Transition.To
            Pipe       = composePipe (asPipe op1) (asPipe op2)
        }

    let bindM m k =
        fun bytes shapeOpt ->
            let op1, bytes', shapeOpt' = m bytes shapeOpt
            let m2 = k op1
            let op2, bytes'', shapeOpt'' = m2 bytes' shapeOpt'

            // validate memory transition, if shape is known
            shapeOpt
            |> Option.iter (fun shape ->
                if op1.Transition.To <> op2.Transition.From || not (op2.Transition.Check shape) then 
                    failwith $"Invalid memory transition: {op1.Name} → {op2.Name}")

            composeOp op1 op2, bytes'', shapeOpt''

    // ───── 2. the builder wrapper ──────────────────────────────────────────
    type Pipeline<'S,'T> =
        { flow  : MemFlow<'S,'T>
          mem   : uint64
          shape : SliceShape option }

    // source: only memory budget, shape = None
    let source memBudget =
        { flow  = fun _ _ -> failwith "pipeline not started yet"
          mem   = memBudget
          shape = None }

    // attach the first stage that *discovers* the shape
    let attachFirst ((op, probeShape) : Operation<unit,'T> * (unit -> SliceShape))
                    (pl : Pipeline<unit,'T>) =
        let shape = probeShape ()
        { flow  = returnM op
          mem   = pl.mem
          shape = Some shape }

    // later compositions Pipeline composition
    let (>>=>) pl next =
        { pl with flow = bindM pl.flow (fun _ -> returnM next) }

    // sink
    let sink (pl : Pipeline<'S,'T>) =
        match pl.shape with
        | None      -> failwith "No stage provided slice dimensions."
        | Some sh ->
            let op, rest, _ = pl.flow pl.mem pl.shape
            printfn $"Pipeline built – {rest} B still free."
            Helpers.asPipe op


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
                printfn "[Runtine analysis: zipWith sequential]"
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
                printfn "[Runtine analysis: zipWith sequential]"
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
                printfn "[Runtine analysis: zipWith parallel]"
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

let tap label : Pipe<'T, 'T> =
    printfn "[tap]"
    lift $"tap: {label}" Streaming (fun x ->
        printfn "[%s] %A" label x
        async.Return x)

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