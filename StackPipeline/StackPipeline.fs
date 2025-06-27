module StackPipeline

open System
open FSharp.Control
open System.Collections.Generic
open AsyncSeqExtensions
open System.IO
open Slice
open Plotly.NET

/// The memory usage strategies during image processing.
type MemoryProfile =
    | Streaming // Slice by slice independently
    | Sliding of uint // Sliding window of slices of depth
    | Buffered // All slices of depth

    member this.EstimateUsage (width: uint) (height: uint) (depth: uint) : uint64 =
        let pixelSize = 1UL // Assume 1 byte per pixel for UInt8
        let sliceBytes = (uint64 width) * (uint64 height) * pixelSize
        match this with
            | Streaming -> sliceBytes
            | Sliding windowSize -> sliceBytes * uint64 windowSize
            | Buffered -> sliceBytes * uint64 depth

    member this.RequiresBuffering (availableMemory: uint64) (width: uint) (height: uint) (depth: uint) : bool =
        let usage = this.EstimateUsage width height depth
        usage > availableMemory

/// A configurable image processing step that operates on image slices.
type StackProcessor<'S,'T> = {
    Name: string // Name of the process
    Profile: MemoryProfile
    Apply: AsyncSeq<'S> -> AsyncSeq<'T>
}

let private fromReducer (name: string) (profile: MemoryProfile) (reducer: AsyncSeq<'In> -> Async<'Out>) : StackProcessor<'In, 'Out> =
    {
        Name = name
        Profile = profile
        Apply = fun input ->
            reducer input |> ofAsync
    }

let private fromConsumer
        (name    : string)
        (profile : MemoryProfile)
        (consume : AsyncSeq<'T> -> Async<unit>)
        : StackProcessor<'T, unit> =

    let reducer (s : AsyncSeq<'T>) = consume s          // Async<unit>
    fromReducer name profile reducer                    // gives AsyncSeq<unit>

let fromMapper
    (name: string)
    (profile: MemoryProfile)
    (f: 'In -> Async<'Out>)
    : StackProcessor<'In, 'Out> =
    {
        Name = name
        Profile = profile
        Apply = fun input ->
            input |> AsyncSeq.mapAsync f
    }

// https://plotly.net/#For-applications-and-libraries
let private plotListAsync (plt: (float list)->(float list)->unit) (vectorSeq: AsyncSeq<(float*float) list>) =
    vectorSeq
    |> AsyncSeq.iterAsync (fun points ->
        async {
            let x,y = points |> List.unzip
            plt x y
        })

let showSliceAsync (plt: (Slice<'T>->unit)) (slices : AsyncSeq<Slice<'T>>) =
    slices
    |> AsyncSeq.iterAsync (fun slice ->
        async {
            let width = slice |> GetWidth |> int
            let height = slice |>GetHeight |> int
            plt slice
        })

let private printAsync (slices: AsyncSeq<'T>) =
    slices
    |> AsyncSeq.iterAsync (fun data ->
        async {
            printfn "[Print] %A" data
        })

let private writeSlicesAsync (outputDir: string) (suffix: string) (slices: AsyncSeq<Slice<'T>>) =
    if not (Directory.Exists(outputDir)) then
        Directory.CreateDirectory(outputDir) |> ignore
    slices
    |> AsyncSeq.iterAsync (fun slice ->
        async {
            let fileName = Path.Combine(outputDir, sprintf "slice_%03d%s" slice.Index suffix)
            slice.Image.toFile(fileName)
            printfn "[Write] Saved slice %d to %s" slice.Index fileName
        })

let private readSlices<'T when 'T: equality> (inputDir: string) (suffix: string) : AsyncSeq<Slice<'T>> =
    Directory.GetFiles(inputDir, "*"+suffix) |> Array.sort
    |> Array.mapi (fun i fileName ->
        async {
            printfn "[Read] Reading slice %d to %s" (uint i) fileName
            return Slice.readSlice (uint i) fileName
        })
    |> Seq.ofArray
    |> AsyncSeq.ofSeqAsync

/// Pipeline computation expression
type PipelineBuilder(availableMemory: uint64, width: uint, height: uint, depth: uint) =
    /// Chain two <c>StackProcessor</c> instances, optionally inserting intermediate disk I/O
    member _.Bind(p: StackProcessor<'S,'T>, f: StackProcessor<'S,'T> -> StackProcessor<'S,'T>) : StackProcessor<'S,'T> =
        let composed = f p
        (*
        let combinedProfile = composed.Profile
        if combinedProfile.RequiresBuffering availableMemory  width  height depth then
            printfn "[Memory] Exceeded memory limits. Splitting pipeline."
            let tempDir = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName())
            Directory.CreateDirectory(tempDir) |> ignore
            let intermediate = fun input ->
                // Step 1: Write intermediate slices to disk
                writeSlicesAsync tempDir ".tif" input
                |> Async.RunSynchronously
                // Step 2: Read them back for next stage
                readSlices tempDir ".tif"

            { Name = $"{composed.Name} {p.Name}"; Profile = Streaming; Apply = composed.Apply << intermediate } // The profile needs to be reset here. How to do that?
        else *)
        composed

    member _.Bind(p: StackProcessor<'S, 'T>, f: 'T -> StackProcessor<'S, 'U>) : StackProcessor<'S, 'U> =
        {
            Name = $"bindResult({p.Name})"
            Profile = p.Profile // You could also recompute based on `f` if needed
            Apply = fun input ->
                asyncSeq {
                    let! result = p.Apply input |> AsyncSeq.toListAsync
                    match result with
                    | [single] ->
                        let nextProcessor = f single
                        yield! nextProcessor.Apply input
                    | _ ->
                        failwithf "Expected reducer output to produce a single result, but got %d values." result.Length
                }
        }

    /// Wraps a processor value for use in the pipeline computation expression.
    member _.Return(p: StackProcessor<'S,'T>) = p

    /// Allows returning a processor directly from another computation expression.
    member _.ReturnFrom(p: StackProcessor<'S,'T>) = p

    /// Provides a default identity processor using streaming as the memory profile.
    member _.Zero() = { Name=""; Profile = Streaming; Apply = id }

/// Combine two <c>StackProcessor</c> instances into one by composing their memory profiles and transformation functions.
let (>>=>) (p1: StackProcessor<'S,'T>) (p2: StackProcessor<'T,'U>) : StackProcessor<'S,'U> =
    {
        Name = $"{p2.Name} {p1.Name}"; 
        Profile = 
            match p1.Profile, p2.Profile with
                | Streaming, Streaming -> Streaming
                | Sliding sz, Streaming
                | Streaming, Sliding sz -> Sliding sz
                | Sliding sz1, Sliding sz2 -> Sliding (max sz1 sz2)
                | Streaming, Buffered
                | Buffered, Streaming
                | Sliding _, Buffered
                | Buffered, Sliding _
                | Buffered, Buffered -> Buffered
        Apply = fun input -> input |> p1.Apply |> p2.Apply
    }

/// A memory-aware pipeline builder with the specified processing constraints.
let pipeline availableMemory width height depth = PipelineBuilder(availableMemory, width, height, depth)

/// Pipeline helper functions
let singleton (x: 'In) : StackProcessor<'In, 'In> =
    {
        Name = "[singleton]"
        Profile = Streaming
        Apply = fun _ -> AsyncSeq.singleton x
    }

let private runWith (input: AsyncSeq<'In>) (p: StackProcessor<'In,'T>) : AsyncSeq<'T> =
    p.Apply input

let private run (p: StackProcessor<unit,'T>) : AsyncSeq<'T> =
    runWith (AsyncSeq.singleton ()) p

(*
let runNWriteSlices path suffix maker =
    printfn "[runNWriteSlices]"
    let stream = run maker
    writeSlicesAsync path suffix stream |> Async.RunSynchronously

let runNShowSlice<'T> maker =
    printfn "[runNShowSlice]"
    let stream = run maker
    showSliceAsync stream |> Async.RunSynchronously

let runNPrint maker =
    printfn "[runNPrint]"
    let stream = run maker
    printAsync stream |> Async.RunSynchronously

let runNPlotList plt maker =
    printfn "[runNPlotList]"
    let stream = run maker
    plotListAsync plt stream |> Async.RunSynchronously
*)

//let print p = printfn "[print]"; run p |> printAsync
//let plot plt p = printfn "[plot]"; run p |> plotListAsync plt
//let show plt p = printfn "[show]"; run p |> showSliceAsync plt

let print<'T> : StackProcessor<'T, unit> =
    fromConsumer "print" Streaming (fun stream ->
        async {
            printfn "[print]"
            do! printAsync stream 
        })

let plot plt : StackProcessor<(float*float) list, unit> =
    fromConsumer "plot" Streaming (fun stream ->
        async {
            printfn "[plot]"
            do! (plotListAsync plt) stream 
        })

let show plt : StackProcessor<Slice<'a>, unit> =
    fromConsumer "show" Streaming (fun stream ->
        async {
            printfn "[show]"
            do! (showSliceAsync plt) stream
        })

let writeSlices path suffix : StackProcessor<Slice<'a>, unit> =
    fromConsumer "write" Streaming (fun stream ->
        async {
            printfn "[show]"
            do! (writeSlicesAsync path suffix) stream
        })

let ignore<'T> : StackProcessor<'T, unit> =
    fromConsumer "ignore" Streaming (fun stream ->
        async {
            printfn "[ignore]"
            do! stream |> AsyncSeq.iterAsync (fun _ -> async.Return())
        })
        
/// Join two StackProcessors<'In, _> into one by zipping their outputs:
///   • applies both processors to the same input stream
///   • pairs each output and combines using the given function
///   • assumes both sides produce values in lockstep
let join (f: 'A -> 'B -> 'C) (p1: StackProcessor<'In, 'A>) (p2: StackProcessor<'In, 'B>) : StackProcessor<'In, 'C> =
    printfn "[join]"
    {
        Name = $"zipJoin({p1.Name}, {p2.Name})"
        Profile = 
            match p1.Profile, p2.Profile with
            | Streaming, Streaming -> Streaming
            | Sliding sz1, Sliding sz2 -> Sliding (max sz1 sz2)
            | _ -> Buffered // conservative fallback

        Apply = fun input ->
            let a = p1.Apply input
            let b = p2.Apply input
            AsyncSeq.zip a b |> AsyncSeq.map (fun (x, y) -> f x y)
    }

let joinScalar
    (f: 'A -> 'B -> 'C)
    (streamProc: StackProcessor<'In, 'A>)
    (scalarProc: StackProcessor<'In, 'B>)
    : StackProcessor<'In, 'C> =
    printfn "[joinScalar]"
    {
        Name = $"joinScalar({streamProc.Name}, {scalarProc.Name})"
        Profile =
            match streamProc.Profile, scalarProc.Profile with
            | Streaming, Streaming -> Streaming
            | Sliding s1, Sliding s2 -> Sliding (max s1 s2)
            | _ -> Buffered

        Apply = fun input -> asyncSeq {
            // Evaluate the scalar processor first
            let! scalarValue =
                scalarProc.Apply input
                |> AsyncSeq.tryLast // could also use head, if only one expected
                |> Async.map (function
                    | Some v -> v
                    | None   -> failwithf "[joinScalar] No value from scalar processor '%s'" scalarProc.Name)

            // Now stream the input through streamProc
            let stream = streamProc.Apply input
            yield! stream |> AsyncSeq.map (fun a -> f a scalarValue)
        }
    }

/// Split a StackProcessor<'In,'T> into two branches that
///   • read the upstream only once
///   • keep at most one item in memory
///   • terminate correctly when both sides finish
type private Request<'T> = Left of AsyncReplyChannel<Option<'T>> | Right of AsyncReplyChannel<Option<'T>>
let tee (p : StackProcessor<'In,'T>): StackProcessor<'In,'T> * StackProcessor<'In,'T> =
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

    // ---------- 2. Build the two outward‑facing StackProcessors ----------
    let mkSide name pick =
        { Name    = $"{p.Name}-{name}"
          Profile = p.Profile
          Apply   = fun input ->
                        let left, right = getShared input
                        pick (left, right) }

    mkSide "left"  fst,
    mkSide "right" snd

let tap label : StackProcessor<'T, 'T> =
    fromMapper $"tap: {label}" Streaming (fun x ->
        printfn "[%s] %A" label x
        async.Return x)

/// Fan out a StackProcessor<'In,'T> to two branches:
///   • processes input once using tee
///   • applies separate processors to each branch
///   • zips outputs into a tuple
let fanOut (p: StackProcessor<'In,'T>) (f1: StackProcessor<'T,'U>) (f2: StackProcessor<'T,'V>) : StackProcessor<'In, 'U * 'V> =
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

/// Run two StackProcessors<unit, _> in parallel:
///   • executes each with its own consumer function
///   • waits for both to finish before returning
let runBoth2 
    (p1: StackProcessor<unit, 'T1>) (f1: StackProcessor<unit, 'T1> -> Async<unit>)
    (p2: StackProcessor<unit, 'T2>) (f2: StackProcessor<unit, 'T2> -> Async<unit>) 
    : unit =
    async {
        let! _ =
            Async.Parallel [
                async { do! f1 p1 }
                async { do! f2 p2 }
            ]
        return ()
    } |> Async.RunSynchronously

let sinkLst (processors: StackProcessor<unit, unit> list) : unit =
    processors
    |> List.map (fun p -> run p |> AsyncSeq.iterAsync (fun () -> async.Return()))
    |> Async.Parallel
    |> Async.Ignore
    |> Async.RunSynchronously

let sink (p: StackProcessor<unit, unit>) : unit = 
    sinkLst [p]

let sourceLst 
    (availableMemory: uint64)
    (width: uint)
    (height: uint)
    (depth: uint)
    (processors: StackProcessor<unit,Slice.Slice<'T>> list) 
    : StackProcessor<unit,Slice<'T>> list =
    processors |>
    List.map (fun p -> 
        pipeline availableMemory width height depth {return p}
    )

let source
    (availableMemory: uint64)
    (width: uint)
    (height: uint)
    (depth: uint)
    (p: StackProcessor<unit,Slice.Slice<'T>>) 
    : StackProcessor<unit,Slice<'T>> =
    let lst = sourceLst availableMemory width height depth [p]
    List.head lst
