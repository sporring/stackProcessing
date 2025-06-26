module StackPipeline

open System
open FSharp.Control
open System.Collections.Generic
open AsyncSeqExtensions
open System.IO
open Slice
open Plotly.NET

// https://plotly.net/#For-applications-and-libraries
type PlotStyle = Line | Points
let plotListAsync (style: PlotStyle) (vectorSeq: AsyncSeq<(float*float) list>) =
    vectorSeq
    |> AsyncSeq.iterAsync (fun points ->
        async {
            let x,y = points |> Array.ofList |> Array.unzip
            let chart = 
                match style with
                    Line -> Chart.Line(x,y) 
                    | Points -> Chart.Point(x,y)
            Chart.show chart
        })

let inline showSliceAsync<^T when ^T: (static member op_Explicit: ^T -> float) and ^T: equality> (slices: AsyncSeq<Slice<^T >>) : Async<unit> =
    async {
        let! maybeSlice =
            slices
            |> AsyncSeqExtensions.tryItem 0

        match maybeSlice with
        | Some slice ->
            let image = slice.Image
            let width = image.GetWidth() |> int
            let height = image.GetHeight() |> int
            printfn "[showSliceAsync] Showing first slice (%d x %d)" width height
            let data = image.toArray2D()
            let data =
                Seq.init height (fun y ->
                    Seq.init width (fun x ->
                        image[x,y] |> float))
            data |> Chart.Heatmap |> Chart.show
        | None ->
            printfn "No slice to show"
    }

let printAsync (slices: AsyncSeq<'T>) =
    slices
    |> AsyncSeq.iterAsync (fun data ->
        async {
            printfn "[Print] %A" data
        })

let writeSlicesAsync (outputDir: string) (suffix: string) (slices: AsyncSeq<Slice<'T>>) =
    if not (Directory.Exists(outputDir)) then
        Directory.CreateDirectory(outputDir) |> ignore
    slices
    |> AsyncSeq.iterAsync (fun slice ->
        async {
            let fileName = Path.Combine(outputDir, sprintf "slice_%03d%s" slice.Index suffix)
            slice.Image.toFile(fileName)
            printfn "[Write] Saved slice %d to %s" slice.Index fileName
        })

let writeSlices path suffix stream =
    printfn "[writeSlices]"
    writeSlicesAsync path suffix stream |> Async.RunSynchronously

let readSlices<'T when 'T: equality> (inputDir: string) (suffix: string) : AsyncSeq<Slice<'T>> =
    Directory.GetFiles(inputDir, "*"+suffix) |> Array.sort
    |> Array.mapi (fun i fileName ->
        async {
            printfn "[Read] Reading slice %d to %s" (uint i) fileName
            return Slice.readSlice (uint i) fileName
        })
    |> Seq.ofArray
    |> AsyncSeq.ofSeqAsync

// --- Types ---
/// <summary>
/// Represents memory usage strategies during image processing.
/// </summary>
type MemoryProfile =
    | Streaming // Slice by slice independently
    | Sliding of uint // Sliding window of slices of depth
    | Buffered // All slices of depth

    /// <summary>
    /// Estimates the amount of memory required for the given dimensions.
    /// </summary>
    /// <param name="width">Image width in pixels</param>
    /// <param name="height">Image height in pixels</param>
    /// <param name="depth">Number of image slices</param>
    member this.EstimateUsage (width: uint) (height: uint) (depth: uint) : uint64 =
        let pixelSize = 1UL // Assume 1 byte per pixel for UInt8
        let sliceBytes = (uint64 width) * (uint64 height) * pixelSize
        match this with
            | Streaming -> sliceBytes
            | Sliding windowSize -> sliceBytes * uint64 windowSize
            | Buffered -> sliceBytes * uint64 depth

    /// <summary>
    /// Determines whether buffering is required based on available memory and image size.
    /// </summary>
    /// <param name="availableMemory">Memory available for processing</param>
    /// <param name="width">Image width</param>
    /// <param name="height">Image height</param>
    /// <param name="depth">Image depth</param>
    member this.RequiresBuffering (availableMemory: uint64) (width: uint) (height: uint) (depth: uint) : bool =
        let usage = this.EstimateUsage width height depth
        usage > availableMemory

/// <summary>
/// Represents a configurable image processing step that operates on image slices.
/// </summary>
type StackProcessor<'S,'T> = {
    /// <summary>
    /// A name of this processor (e.g., "Thresholding").
    /// </summary>
    Name: string

    /// <summary>
    /// Defines the memory strategy used when applying this processor (e.g., streaming, buffered).
    /// </summary>
    Profile: MemoryProfile

    /// <summary>
    /// The function that processes a stream of image slices and returns a transformed stream.
    /// </summary>
    Apply: AsyncSeq<'S> -> AsyncSeq<'T>
}

let singleton (x: 'In) : StackProcessor<'In, 'In> =
    {
        Name = "[singleton]"
        Profile = Streaming
        Apply = fun _ -> AsyncSeq.singleton x
    }

let join 
    (f: 'A -> 'B -> 'C) 
    (p1: StackProcessor<'In, 'A>) 
    (p2: StackProcessor<'In, 'B>) 
    (txt: string option) 
    : StackProcessor<'In, 'C> =
    match txt with 
        Some t -> printfn "%s" t 
        | None -> ()
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

let runWith (input: AsyncSeq<'In>) (p: StackProcessor<'In,'T>) : AsyncSeq<'T> =
    p.Apply input

let run (p: StackProcessor<unit,'T>) : AsyncSeq<'T> =
    runWith (AsyncSeq.singleton ()) p

let runNWriteSlices path suffix maker =
    printfn "[runNWriteSlices]"
    let stream = run maker
    writeSlicesAsync path suffix stream |> Async.RunSynchronously

let inline runNShowSlice<^T when ^T: (static member op_Explicit: ^T -> float) and ^T: equality> maker =
    printfn "[runNShowSlice]"
    let stream = run maker
    showSliceAsync<'T> stream |> Async.RunSynchronously

let runNPrint maker =
    printfn "[runNPrint]"
    let stream = run maker
    printAsync stream |> Async.RunSynchronously

let inline runNPlotList style maker =
    printfn "[runNPlotList]"
    let stream = run maker
    plotListAsync style stream |> Async.RunSynchronously

// --- Pipeline computation expression ---
/// <summary>
/// Provides computation expression support for building memory-aware image processing pipelines.
/// Automatically inserts disk-based buffering when memory limits are exceeded.
/// </summary>
/// <param name="availableMemory">Maximum memory available for processing.</param>
/// <param name="width">Width of each image slice.</param>
/// <param name="height">Height of each image slice.</param>
/// <param name="depth">Total number of image slices.</param>
type PipelineBuilder(availableMemory: uint64, width: uint, height: uint, depth: uint) =
    /// <summary>
    /// Chains two <c>StackProcessor</c> instances, optionally inserting intermediate disk I/O
    /// if the combined profile exceeds available memory.
    /// </summary>
    /// <param name="p">The input processor.</param>
    /// <param name="f">A function that transforms the processor.</param>
    /// <returns>A new <c>StackProcessor</c> that includes buffering if needed.</returns>
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

    /// <summary>
    /// Wraps a processor value for use in the pipeline computation expression.
    /// </summary>
    member _.Return(p: StackProcessor<'S,'T>) = p

    /// <summary>
    /// Allows returning a processor directly from another computation expression.
    /// </summary>
    member _.ReturnFrom(p: StackProcessor<'S,'T>) = p

    /// <summary>
    /// Provides a default identity processor using streaming as the memory profile.
    /// </summary>
    member _.Zero() = { Name=""; Profile = Streaming; Apply = id }

/// <summary>
/// Combines two <c>StackProcessor</c> instances into one by composing their memory profiles and transformation functions.
/// </summary>
/// <param name="p1">The first processor to apply.</param>
/// <param name="p2">The second processor to apply.</param>
/// <returns>
/// A new <c>StackProcessor</c> whose memory profile is the more restrictive of the two,
/// and whose apply function is the composition of the two processors' apply functions.
/// </returns>
/// <remarks>
/// Memory profile resolution follows these rules:
/// <list type="bullet">
/// <item><description><c>Streaming + Streaming = Streaming</c></description></item>
/// <item><description><c>Streaming + Sliding(sz) = Sliding(sz)</c></description></item>
/// <item><description><c>Sliding(sz1) + Sliding(sz2) = Sliding(max sz1 sz2)</c></description></item>
/// <item><description>Any combination involving <c>Buffered</c> results in <c>Buffered</c>.</description></item>
/// </list>
/// </remarks>
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

/// <summary>
/// Initializes a memory-aware pipeline builder with the specified processing constraints.
/// </summary>
/// <param name="availableMemory">The total memory available for the pipeline in bytes.</param>
/// <param name="width">The width of each image slice.</param>
/// <param name="height">The height of each image slice.</param>
/// <param name="depth">The total number of image slices (volume depth).</param>
/// <returns>
/// A <c>PipelineBuilder</c> instance that supports memory-constrained pipeline composition using computation expressions.
/// </returns>
let pipeline availableMemory width height depth = PipelineBuilder(availableMemory, width, height, depth)

type Request<'T> = Left of AsyncReplyChannel<Option<'T>> | Right of AsyncReplyChannel<Option<'T>>
let tee (p: StackProcessor<unit, 'T>) : StackProcessor<unit, 'T> * StackProcessor<unit, 'T> =
    let leftRightStreams (input: AsyncSeq<unit>) : AsyncSeq<'T> * AsyncSeq<'T> =
        let src = p.Apply input
        let agent = MailboxProcessor.Start(fun inbox ->
            async {
                let enumerator = (AsyncSeq.toAsyncEnum src).GetAsyncEnumerator()
                let mutable current : Option<'T> = None
                let mutable consumed = (true, true)
                let mutable finished = false
                let rec loop () = async {
                    if current.IsNone && not finished then
                        let! hasNext = enumerator.MoveNextAsync().AsTask() |> Async.AwaitTask
                        if hasNext then
                            current <- Some enumerator.Current
                            consumed <- (false, false)
                        else
                            finished <- true
                    let! msg = inbox.Receive()
                    match msg, current with
                    | Left ch, Some v when not (fst consumed) ->
                        ch.Reply(Some v)
                        consumed <- (true, snd consumed)
                    | Right ch, Some v when not (snd consumed) ->
                        ch.Reply(Some v)
                        consumed <- (fst consumed, true)
                    | (Left ch | Right ch), None when finished ->
                        ch.Reply(None)
                    | _ -> ()
                    if consumed = (true, true) then
                        current <- None
                    return! loop ()
                }
                do! loop ()
            })
        let mkTaggedStream tag =
            asyncSeq {
                let mutable done_ = false
                while not done_ do
                    let! vOpt = agent.PostAndAsyncReply(tag)
                    match vOpt with
                    | Some v -> yield v
                    | None -> done_ <- true
            }
        mkTaggedStream Left, mkTaggedStream Right
    let lazyStreams = lazy (leftRightStreams (AsyncSeq.singleton ()))
    let mkSide name get =
        {
            Name = $"{p.Name}-{name}"
            Profile = p.Profile
            Apply = fun _ -> get lazyStreams.Value
        }
    mkSide "left" fst, mkSide "right" snd

/// Split a StackProcessor<'In,'T> into two branches that
///   • read the upstream only once
///   • keep at most one item in memory
///   • terminate correctly when both sides finish
let tee2 (p : StackProcessor<'In,'T>)
        : StackProcessor<'In,'T> * StackProcessor<'In,'T> =

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

let fanout
    (p: StackProcessor<'In,'T>) 
    (f1: StackProcessor<'T,'U>) 
    (f2: StackProcessor<'T,'V>) : StackProcessor<'In, 'U * 'V> =
    
    let left, right = tee p
    {
        Name = $"fanout2 ({f1.Name}, {f2.Name})"
        Profile = Buffered
        Apply = fun input ->
            let lStream = left.Apply input |> f1.Apply
            let rStream = right.Apply input |> f2.Apply
            AsyncSeq.zip lStream rStream
    }

let runBoth (p1: StackProcessor<unit, 'T1>) (f1: StackProcessor<unit, 'T1> -> Async<unit>)
            (p2: StackProcessor<unit, 'T2>) (f2: StackProcessor<unit, 'T2> -> Async<unit>) : unit =
    async {
        let! _ =
            Async.Parallel [
                async { do! f1 p1 }
                async { do! f2 p2 }
            ]
        return ()
    } |> Async.RunSynchronously

let print p = run p |> printAsync
let plot style p = run p |> plotListAsync style
