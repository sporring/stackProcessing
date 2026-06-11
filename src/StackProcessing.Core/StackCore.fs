module StackCore

open SlimPipeline // Core processing model
open System

module ChunkPrimitive = Chunk

// Whole-slice stages should do their pixel work in managed arrays and cross
// the ITK boundary once per slice. Per-pixel Image.Get/setter calls are kept
// for sparse/random access paths where bulk transport would be the wrong cost.

type Stage<'S,'T> = SlimPipeline.Stage<'S,'T>
type Profile = SlimPipeline.Profile
type ProfileTransition = SlimPipeline.ProfileTransition
type ResourceOps<'T> = SlimPipeline.ResourceOps<'T>
type Window<'T> = SlimPipeline.Window<'T>
//type Slice<'S when 'S: equality> = Slice.Slice<'S>
type Image<'S when 'S: equality> = Image.Image<'S>

type Slab<'T when 'T: equality> =
    { Image: Image<'T>
      EmitRange: uint * uint }

type ChunkLayout = ChunkPrimitive.ChunkLayout
type ChunkIndex = ChunkPrimitive.ChunkIndex
type Chunk<'T when 'T: equality> = ChunkPrimitive.Chunk<'T>
type VectorChunk<'T when 'T: equality> = ChunkPrimitive.VectorChunk<'T>
type DenseUInt32UnionFind = ChunkPrimitive.DenseUInt32UnionFind

type HistogramBinning =
    | FixedEdges of firstLeftEdge: float * lastLeftEdge: float * bins: uint32
    | FixedWidth of binWidth: uint64

type Histogram<'T when 'T: comparison> =
    { Counts: Map<'T, uint64>
      Binning: HistogramBinning option }

module NativeSp =
    let ensureAvailable () = ChunkPrimitive.NativeSp.ensureAvailable ()
    let fftwfComplexXYInplace(interleaved, width, height, inverse) =
        ChunkPrimitive.NativeSp.fftwfComplexXYInplace(interleaved, width, height, inverse)
    let fftwfComplexZInplace(interleaved, width, height, depth, inverse) =
        ChunkPrimitive.NativeSp.fftwfComplexZInplace(interleaved, width, height, depth, inverse)
    let checkStatus operation status = ChunkPrimitive.NativeSp.checkStatus operation status

module Histogram =
    let ofMap counts =
        { Counts = counts
          Binning = None }

    let withFixedEdges firstLeftEdge lastLeftEdge bins counts =
        { Counts = counts
          Binning = Some(FixedEdges(firstLeftEdge, lastLeftEdge, bins)) }

    let withFixedWidth binWidth counts =
        { Counts = counts
          Binning = Some(FixedWidth binWidth) }

module Chunk =
    let create<'T when 'T: equality> = ChunkPrimitive.create<'T>
    let span<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (chunk: Chunk<'T>) =
        ChunkPrimitive.span<'T> chunk
    let incRef = ChunkPrimitive.incRef
    let decRef = ChunkPrimitive.decRef
    let vectorSpan<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (vector: VectorChunk<'T>) =
        ChunkPrimitive.vectorSpan<'T> vector
    let toVectorImage<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkPrimitive.toVectorImage<'T>
    let vectorElement<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkPrimitive.vectorElement<'T>
    let appendVectorElement<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkPrimitive.appendVectorElement<'T>
    let mapVectorElements = ChunkPrimitive.mapVectorElements
    let vectorDot = ChunkPrimitive.vectorDot
    let vectorMagnitude = ChunkPrimitive.vectorMagnitude
    let vectorCross3D = ChunkPrimitive.vectorCross3D
    let vectorAngleTo = ChunkPrimitive.vectorAngleTo
    let mapVectorElementsFloat32 = ChunkPrimitive.mapVectorElementsFloat32
    let vectorDotFloat32 = ChunkPrimitive.vectorDotFloat32
    let vectorMagnitudeFloat32 = ChunkPrimitive.vectorMagnitudeFloat32
    let vectorAngleToFloat32 = ChunkPrimitive.vectorAngleToFloat32
    let inline toIndex width height x y z = ChunkPrimitive.toIndex width height x y z
    let inline ofIndex width height index = ChunkPrimitive.ofIndex width height index
    let iter f chunk = ChunkPrimitive.iter f chunk
    let iteri f chunk = ChunkPrimitive.iteri f chunk
    let mapInto f input output = ChunkPrimitive.mapInto f input output
    let map f chunk = ChunkPrimitive.map f chunk
    let mapi f chunk = ChunkPrimitive.mapi f chunk
    let fold folder state chunk = ChunkPrimitive.fold folder state chunk
    let foldi folder state chunk = ChunkPrimitive.foldi folder state chunk

    let ofImage<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (image: Image<'T>) =
        let chunkSize =
            match image.GetSize() with
            | [ width; height ] -> uint64 width, uint64 height, 1UL
            | [ width; height; depth ] -> uint64 width, uint64 height, uint64 depth
            | size -> invalidArg "image" $"Chunk.ofImage supports 2D and 3D scalar images, got size {size}."

        let chunk = create<'T> chunkSize
        try
            let pixels = image.toFlatArray()
            pixels.CopyTo(span<'T> chunk)
            chunk
        with
        | _ ->
            decRef chunk
            reraise()

    let toImageWith<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        (name: string)
        (index: int)
        (chunk: Chunk<'T>)
        : Image<'T> =
        let width, height, depth = chunk.Size
        let size =
            if depth = 1UL then
                [ uint width; uint height ]
            else
                [ uint width; uint height; uint depth ]

        Image<'T>.ofFlatArray(size, (span<'T> chunk).ToArray(), name, index)

    let toImage<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (chunk: Chunk<'T>) =
        toImageWith "chunk.toImage" 0 chunk

    let toSlabWith<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
        (name: string)
        (window: Window<Chunk<'T>>)
        : Slab<'T> =
        match window.Items with
        | [] ->
            invalidArg "window" "Chunk.toSlab requires a non-empty chunk window."
        | chunks ->
            let images =
                chunks
                |> List.mapi (fun index chunk -> toImageWith $"{name}.slice{index}" index chunk)

            try
                { Image = ImageFunctions.stack images
                  EmitRange = window.EmitRange }
            finally
                images |> List.iter (fun image -> image.decRefCount())

    let toSlab<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (window: Window<Chunk<'T>>) =
        toSlabWith "chunk.toSlab" window

    let ofSlab<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (slab: Slab<'T>) =
        let start, count = slab.EmitRange
        if count = 0u then
            []
        elif slab.Image.GetDimensions() = 2u then
            if start <> 0u || count <> 1u then
                invalidArg "slab" $"Chunk.ofSlab can emit only slice 0 from a 2D slab, got range ({start}, {count})."
            [ ofImage slab.Image ]
        elif slab.Image.GetDimensions() = 3u then
            let depth = slab.Image.GetDepth()
            if start + count > depth then
                invalidArg "slab" $"Chunk.ofSlab emit range ({start}, {count}) exceeds slab depth {depth}."

            [ for z in int start .. int (start + count) - 1 do
                let slice = ImageFunctions.extractSlice 2u z slab.Image
                try
                    yield ofImage slice
                finally
                    slice.decRefCount() ]
        else
            invalidArg "slab" $"Chunk.ofSlab supports 2D and 3D scalar slab images, got {slab.Image.GetDimensions()}D."

type Point2D =
    { X: float
      Y: float }

type Polygon2D = Point2D list

let imageResourceOps<'S when 'S: equality> : ResourceOps<Image<'S>> =
    { Retain = fun image -> image.incRefCount()
      Release = fun image -> image.decRefCount()
      MemoryOf = fun image -> Image<'S>.memoryEstimateSItk image.Image |> uint64 |> Some }

let releaseAfterWith (ops: ResourceOps<'S>) (f: 'S -> 'T) (value: 'S) =
    try
        f value
    finally
        ops.Release value

let liftUnary name =
    Stage.liftReleaseUnary name ignore

let liftUnaryReleaseAfter (name: string) (f: Image<'S> -> Image<'T>) (memoryNeed: MemoryNeed) (elementTransformation: ElementTransformation) =
    Stage.liftResourceUnary name imageResourceOps f memoryNeed elementTransformation

let identityStage name =
    Stage.map name (fun _ value -> value) id id

let incIfImage x =
    match box x with
    | :? Image<uint8> as im -> im.incRefCount()   // or img.incNRefCount(1) if it takes an int
    | :? Image<int8> as im -> im.incRefCount()   // or img.incNRefCount(1) if it takes an int
    | :? Image<uint16> as im -> im.incRefCount()   // or img.incNRefCount(1) if it takes an int
    | :? Image<int16> as im -> im.incRefCount()   // or img.incNRefCount(1) if it takes an int
    | :? Image<uint> as im -> im.incRefCount()   // or img.incNRefCount(1) if it takes an int
    | :? Image<int> as im -> im.incRefCount()   // or img.incNRefCount(1) if it takes an int
    | :? Image<uint64> as im -> im.incRefCount()   // or img.incNRefCount(1) if it takes an int
    | :? Image<int64> as im -> im.incRefCount()   // or img.incNRefCount(1) if it takes an int
    | :? Image<float32> as im -> im.incRefCount()   // or img.incNRefCount(1) if it takes an int
    | :? Image<float> as im -> im.incRefCount()   // or img.incNRefCount(1) if it takes an int
    | :? Image<System.Numerics.Complex> as im -> im.incRefCount()
    | _ -> ()
    x

let private copyIfImageForPadding index x =
    let copyImage (image: Image<'T>) =
        image.copy($"padding[{index}]", index) :> obj |> unbox

    match box x with
    | :? Image<uint8> as im -> copyImage im
    | :? Image<int8> as im -> copyImage im
    | :? Image<uint16> as im -> copyImage im
    | :? Image<int16> as im -> copyImage im
    | :? Image<uint> as im -> copyImage im
    | :? Image<int> as im -> copyImage im
    | :? Image<uint64> as im -> copyImage im
    | :? Image<int64> as im -> copyImage im
    | :? Image<float32> as im -> copyImage im
    | :? Image<float> as im -> copyImage im
    | :? Image<System.Numerics.Complex> as im -> copyImage im
    | _ -> x

let incRef () =
    Stage.map "incRefCountOp" (fun _ -> incIfImage) id id
let decIfImage x =
    match box x with
    | :? Image<uint8> as im -> im.decRefCount()   // or img.incNRefCount(1) if it takes an int
    | :? Image<int8> as im -> im.decRefCount()   // or img.incNRefCount(1) if it takes an int
    | :? Image<uint16> as im -> im.decRefCount()   // or img.incNRefCount(1) if it takes an int
    | :? Image<int16> as im -> im.decRefCount()   // or img.incNRefCount(1) if it takes an int
    | :? Image<uint> as im -> im.decRefCount()   // or img.incNRefCount(1) if it takes an int
    | :? Image<int> as im -> im.decRefCount()   // or img.incNRefCount(1) if it takes an int
    | :? Image<uint64> as im -> im.decRefCount()   // or img.incNRefCount(1) if it takes an int
    | :? Image<int64> as im -> im.decRefCount()   // or img.incNRefCount(1) if it takes an int
    | :? Image<float32> as im -> im.decRefCount()   // or img.incNRefCount(1) if it takes an int
    | :? Image<float> as im -> im.decRefCount()   // or img.incNRefCount(1) if it takes an int
    | :? Image<System.Numerics.Complex> as im -> im.decRefCount()
    | _ -> ()
    x
let decRef () =
    Stage.map "decRefCountOp" (fun _ -> decIfImage) id id
let releaseAfter (f: Image<'S>->'T) (I: Image<'S>) = 
    releaseAfterWith imageResourceOps f I
let releaseAfter2 (f: Image<'S>->Image<'S>->'T) (I: Image<'S>) (J: Image<'S>) = 
    try
        f I J
    finally
        imageResourceOps.Release I
        imageResourceOps.Release J

let liftRelease2 f I J =
    releaseAfter2 (fun a b -> f a b) I J

let memNeeded<'T> nTimes nElems =
    3UL * nElems * nTimes * (typeof<'T> |> Image.getBytesPerComponent |> uint64)

let failTypeMismatch<'T> name supportedTypes =
    let t = typeof<'T>
    if supportedTypes |> List.exists ((=) t) |> not then
        let names = List.map (fun (t: Type) -> t.Name) supportedTypes
        failwith $"[{name}] wrong type. Type {t} must be one of {names}"

let isInteractiveProcess () =
    AppDomain.CurrentDomain.FriendlyName.IndexOf("fsi", StringComparison.OrdinalIgnoreCase) >= 0
    || Environment.GetCommandLineArgs()
       |> Array.exists (fun arg -> arg.IndexOf("fsi", StringComparison.OrdinalIgnoreCase) >= 0)

let stopWithConfigurationError message =
    if isInteractiveProcess () then
        invalidOp message
    else
        Console.Error.WriteLine(message)
        Environment.ExitCode <- 2
        Environment.Exit 2
        failwith message

let trd (_, _, c) = c
(*
let releaseNAfter (n: int) (f: Image<'S> list->'T list) (sLst: Image<'S> list) : 'T list =
    let tLst = f sLst;
    sLst |> List.take (int n) |> List.map (decIfImage >> ignore) |> ignore
    tLst 
*)

let private rssKb () =
    MemoryProbe.currentRssBytes() / 1024UL

let private sampleVolRssProbe enabled label startKb previousKb =
    if enabled then
        let currentKb = rssKb()
        let stepDelta = int64 currentKb - int64 previousKb
        let totalDelta = int64 currentKb - int64 startKb
        printfn $"[rss:vol] {label}: RSS {currentKb} KB, step %+d{stepDelta} KB, total %+d{totalDelta} KB"
        currentKb, stepDelta
    else
        previousKb, 0L

let private printVolRssSummary enabled startKb finalKb stackDelta releaseInputsDelta volumeFunctionDelta disposeStackDelta unstackDelta disposeVolumeDelta =
    if enabled then
        let totalDelta = int64 finalKb - int64 startKb
        printfn $"[rss:vol-summary] stack %+d{stackDelta} KB, releaseInputs %+d{releaseInputsDelta} KB, volumeFunction %+d{volumeFunctionDelta} KB, disposeStack %+d{disposeStackDelta} KB, unstack %+d{unstackDelta} KB, disposeVolume %+d{disposeVolumeDelta} KB, total %+d{totalDelta} KB"

let private releaseAllImages (images: Image<'S> list) =
    images |> List.iter (fun image -> image.decRefCount())

let private releaseConsumedImages (window: Window<Image<'S>>) =
    window.Items
    |> List.take (min (int window.ReleaseCount) window.Items.Length)
    |> List.iter (fun image -> image.decRefCount())

let volFctToWindowFctReleaseAfterDebug
        (debug: bool)
        (f: Image<'S> -> Image<'T>)
        (requiredInputDepth: uint)
        (outputStart: uint)
        (outputCount: uint)
        (window: Window<Image<'S>>)
        : Image<'T> list =
    let _, windowEmitCount = window.EmitRange
    let effectiveOutputCount = min outputCount windowEmitCount

    if uint window.Items.Length < requiredInputDepth then
        releaseAllImages window.Items
        []
    else
        if effectiveOutputCount = 0u then
            releaseAllImages window.Items
            []
        else
        match window.Items with
        | [ image ] when requiredInputDepth <= 1u ->
            if outputStart = 0u then
                let result =
                    try
                        f image
                    finally
                        releaseConsumedImages window
                [ result ]
            else
                releaseConsumedImages window
                []
        | _ ->
            let rssDebug = debug && DebugLevel.rssEnabled()
            let startKb = if rssDebug then rssKb() else 0UL
            let mutable previousKb, _ = sampleVolRssProbe rssDebug "start" startKb startKb
            let stack = ImageFunctions.stack window.Items
            let currentKb, stackDelta = sampleVolRssProbe rssDebug "after stack" startKb previousKb
            previousKb <- currentKb
            releaseConsumedImages window
            let currentKb, releaseInputsDelta = sampleVolRssProbe rssDebug "after release input slices" startKb previousKb
            previousKb <- currentKb
            let vol = f stack
            let currentKb, volumeFunctionDelta = sampleVolRssProbe rssDebug "after volume function" startKb previousKb
            previousKb <- currentKb
            stack.decRefCount ()
            let currentKb, disposeStackDelta = sampleVolRssProbe rssDebug "after dispose stack" startKb previousKb
            previousKb <- currentKb
            let depth = vol.GetDepth()
            let result =
                if outputStart >= depth then
                    []
                else
                    let count = min effectiveOutputCount (depth - outputStart)
                    ImageFunctions.unstackSkipNTakeM outputStart count vol
            let currentKb, unstackDelta = sampleVolRssProbe rssDebug "after unstack" startKb previousKb
            previousKb <- currentKb
            vol.decRefCount ()
            let currentKb, disposeVolumeDelta = sampleVolRssProbe rssDebug "after dispose volume" startKb previousKb
            previousKb <- currentKb
            printVolRssSummary rssDebug startKb previousKb stackDelta releaseInputsDelta volumeFunctionDelta disposeStackDelta unstackDelta disposeVolumeDelta
            result

let volFctToWindowFctReleaseAfter (f: Image<'S> -> Image<'T>) requiredInputDepth outputStart outputCount window =
    volFctToWindowFctReleaseAfterDebug false f requiredInputDepth outputStart outputCount window

let volFctToLstFctReleaseAfterDebug (debug: bool) (f: Image<'S> -> Image<'T>) (pad: uint) (stride: uint) (images: Image<'S> list) : Image<'T> list =
    let requiredInputDepth =
        if pad = 0u then 1u else 2u * pad + 1u
    let window = Window.create pad stride images
    volFctToWindowFctReleaseAfterDebug debug f requiredInputDepth pad stride window

let volFctToLstFctReleaseAfter (f: Image<'S>->Image<'T>) pad stride images =
    volFctToLstFctReleaseAfterDebug false f pad stride images

let (>=>) = Plan.(>=>)
let (-->) = Stage.(-->)
let private tryParseBool value =
    match value |> string |> fun v -> v.Trim().ToLowerInvariant() with
    | "1" | "true" | "yes" | "y" | "on" -> Some true
    | "0" | "false" | "no" | "n" | "off" -> Some false
    | _ -> None

let optimizerEnabled () =
    match System.Environment.GetEnvironmentVariable "STACKPROCESSING_OPTIMIZE" with
    | null | "" -> true
    | value ->
        tryParseBool value
        |> Option.defaultWith (fun () ->
            failwith $"STACKPROCESSING_OPTIMIZE must be true/false, 1/0, yes/no, or on/off; got '{value}'")

let sourceWithOptimizer optimize availableMemory =
    Plan.sourceWithOptimizer optimize availableMemory

let source availableMemory =
    sourceWithOptimizer (optimizerEnabled ()) availableMemory

let debug level optimize availableMemory =
    let level = max 1u level
    Image.Image<_>.setDebugLevel (if level > 1u then level - 1u else 0u)
    Plan.debug level optimize availableMemory

let debugDefault level availableMemory =
    debug level (optimizerEnabled ()) availableMemory

let commandLineSource availableMemory (args: string array) =
    let rec parse debugLevel optimize costDiscrepancies costFlags costModel remaining kept =
        match remaining with
        | [] -> debugLevel, optimize, costDiscrepancies, costFlags, costModel, kept |> List.rev |> List.toArray
        | "-d" :: value :: rest
        | "--debug-level" :: value :: rest ->
            match System.UInt32.TryParse value with
            | true, level -> parse (Some level) optimize costDiscrepancies costFlags costModel rest kept
            | false, _ -> failwith $"Expected unsigned integer debug level after -d, got '{value}'"
        | ("--no-optimize" | "--no-optimizer") :: rest ->
            parse debugLevel false costDiscrepancies costFlags costModel rest kept
        | "--optimize" :: value :: rest
        | "--optimizer" :: value :: rest ->
            match tryParseBool value with
            | Some enabled -> parse debugLevel enabled costDiscrepancies costFlags costModel rest kept
            | None -> failwith $"Expected boolean optimizer value after --optimize, got '{value}'"
        | ("--cost-discrepancies" | "--cost-discrepancy-report") :: rest ->
            parse (debugLevel |> Option.orElse (Some 1u)) optimize true costFlags costModel rest kept
        | ("--no-cost-discrepancies" | "--no-cost-discrepancy-report") :: rest ->
            parse debugLevel optimize false costFlags costModel rest kept
        | ("--cost-flags" | "--cost-discrepancy-path" | "--cost-discrepancy-file") :: value :: rest ->
            parse (debugLevel |> Option.orElse (Some 1u)) optimize true (Some value) costModel rest kept
        | "--cost-model" :: value :: rest ->
            parse debugLevel optimize costDiscrepancies costFlags (Some value) rest kept
        | arg :: rest ->
            parse debugLevel optimize costDiscrepancies costFlags costModel rest (arg :: kept)

    let debugLevel, optimize, costDiscrepancies, costFlags, costModel, rest =
        parse None (optimizerEnabled ()) false None None (args |> Array.toList) []

    match costModel with
    | Some path -> StackProcessingCost.Fitting.OperatorCostRuntime.load path |> ignore
    | None -> StackProcessingCost.Fitting.OperatorCostRuntime.ensureLoaded ()

    let plan =
        match debugLevel with
        | Some level -> debug level optimize availableMemory
        | None -> sourceWithOptimizer optimize availableMemory

    let plan =
        match costFlags with
        | Some path -> plan |> Plan.withCostDiscrepancyReporting true |> Plan.withCostDiscrepancyFlagPath path
        | None -> plan |> Plan.withCostDiscrepancyReporting costDiscrepancies

    plan, rest
 
let zip = Plan.zip

(*
let inline isExactlyImage<'T> () =
    let t = typeof<'T>
    t.IsGenericType && t.GetGenericTypeDefinition() = typedefof<Image<_>>
*)
let (>=>>) (pl: Plan<'In, 'S>) (stage1: Stage<'S, 'U>, stage2: Stage<'S, 'V>) : Plan<'In, 'U * 'V> = 
    let stream2Window winSz pad stride (stg: Stage<'T,'Out>) : Stage<'T,'Out> =
        let zeroMaker index item =
            copyIfImageForPadding index item
        Stage.window "makeWindow: window" winSz pad zeroMaker stride
        --> Stage.map "makeWindow: select delayed emit range"
                (fun _ window ->
                    let start = 0
                    let count = 1
                    let result =
                        window.Items
                        |> List.skip start
                        |> List.take (min count (max 0 (window.Items.Length - start)))
                    result |> List.iter (incIfImage >> ignore)
                    window.Items |> List.take (min (int stride) window.Items.Length) |> List.iter (decIfImage >> ignore)
                    result)
                id
                id
        --> Stage.flatten "makeWindow: flatten"
        --> stg

    let stg1,stg2 =
        match stage1.Transition.From, stage2.Transition.From with
        | Streaming, Streaming -> stage1, stage2
        | Window (a1,b1,c1,d1,e1), Window (a2,b2,c2,d2,e2) when a1=a2 && b1=b2 && c1=c2 && d1=d2 && e1=e2 -> stage1, stage2
        | Streaming, Window (winSz, stride, pad, emitStart, emitCount) -> 
            if pl.debug && DebugLevel.current() >= 2u then printfn "left is promoted"
            stream2Window winSz pad stride stage1, stage2 
        | Window (winSz, stride, pad, emitStart, emitCount), Streaming -> 
            if pl.debug && DebugLevel.current() >= 2u then printfn "right is promoted"
            stage1, stream2Window winSz pad stride stage2
        | _,_ -> failwith $"[>=>>] does not know how to combine the stage-profiles: {stage1.Transition.From} vs {stage2.Transition.From}"

    Plan.(>=>>) (pl >=> incRef ()) (stg1, stg2)

let private alignForkStages name debug (stage1: Stage<'S, 'U>) (stage2: Stage<'S, 'V>) =
    let stream2Window winSz pad stride (stg: Stage<'T,'Out>) : Stage<'T,'Out> =
        let zeroMaker index item =
            copyIfImageForPadding index item
        Stage.window "makeWindow: window" winSz pad zeroMaker stride
        --> Stage.map "makeWindow: select delayed emit range"
                (fun _ window ->
                    let start = 0
                    let count = 1
                    let result =
                        window.Items
                        |> List.skip start
                        |> List.take (min count (max 0 (window.Items.Length - start)))
                    result |> List.iter (incIfImage >> ignore)
                    window.Items |> List.take (min (int stride) window.Items.Length) |> List.iter (decIfImage >> ignore)
                    result)
                id
                id
        --> Stage.flatten "makeWindow: flatten"
        --> stg

    match stage1.Transition.From, stage2.Transition.From with
    | Streaming, Streaming -> stage1, stage2
    | Window (a1,b1,c1,d1,e1), Window (a2,b2,c2,d2,e2) when a1=a2 && b1=b2 && c1=c2 && d1=d2 && e1=e2 -> stage1, stage2
    | Streaming, Window (winSz, stride, pad, _, _) ->
        if debug && DebugLevel.current() >= 2u then printfn $"{name}: left is promoted"
        stream2Window winSz pad stride stage1, stage2
    | Window (winSz, stride, pad, _, _), Streaming ->
        if debug && DebugLevel.current() >= 2u then printfn $"{name}: right is promoted"
        stage1, stream2Window winSz pad stride stage2
    | _,_ -> failwith $"[{name}] does not know how to combine the stage-profiles: {stage1.Transition.From} vs {stage2.Transition.From}"

let fork (stage1: Stage<'S, 'U>, stage2: Stage<'S, 'V>) : Stage<'S, 'U * 'V> =
    let stg1, stg2 = alignForkStages "fork" false stage1 stage2
    incRef () --> Stage.fork (stg1, stg2)

let (-->>) (stage: Stage<'S, 'T>) (stage1: Stage<'T, 'U>, stage2: Stage<'T, 'V>) : Stage<'S, 'U * 'V> =
    stage --> fork (stage1, stage2)

let (>>=>) = Plan.(>>=>)
let (>>=>>) = Plan.(>>=>>)
let teeFst = Stage.teeFst
let teeSnd = Stage.teeSnd
let ignoreSingles () : Stage<_,unit> = Stage.ignore (decIfImage>>ignore)
let ignorePairs () : Stage<_,unit> = Stage.ignorePairs ((decIfImage>>ignore),(decIfImage>>ignore))
let zeroMaker (index: int) (ex: Image<'S>) : Image<'S> =
    new Image<'S>(ex.GetSize(), ex.GetNumberOfComponentsPerPixel(), "padding", index)

let window windowSize pad stride = Stage.window "window" windowSize pad zeroMaker stride
let flatten () = Stage.flattenWindow "flatten"
let flattenList () = Stage.flatten "flatten"
let releaseWindowItems (items: Image<'T> list) =
    items |> List.iter (fun image -> image.decRefCount())

let releaseConsumedWindowItems (window: Window<Image<'T>>) =
    window.Items
    |> List.take (min (int window.ReleaseCount) window.Items.Length)
    |> releaseWindowItems

let windowToSlab<'T when 'T: equality> : Stage<Window<Image<'T>>, Image<'T>> =
    let mapper (_debug: bool) (window: Window<Image<'T>>) =
        match window.Items with
        | [] ->
            invalidOp "windowToSlab requires a non-empty window."
        | images ->
            let slab = ImageFunctions.stack images
            releaseConsumedWindowItems window
            slab

    let memoryNeed nPixels =
        2UL * nPixels * (typeof<'T> |> Image.getBytesPerComponent |> uint64)

    Stage.map "windowToSlab" mapper memoryNeed id

let windowToSlabWithRange<'T when 'T: equality> : Stage<Window<Image<'T>>, Slab<'T>> =
    let mapper (_debug: bool) (window: Window<Image<'T>>) =
        match window.Items with
        | [] ->
            invalidOp "windowToSlabWithRange requires a non-empty window."
        | images ->
            let slab = ImageFunctions.stack images
            releaseConsumedWindowItems window
            { Image = slab
              EmitRange = window.EmitRange }

    let memoryNeed nPixels =
        2UL * nPixels * (typeof<'T> |> Image.getBytesPerComponent |> uint64)

    Stage.map "windowToSlabWithRange" mapper memoryNeed id

let mapSlabWithStage<'S, 'T when 'S: equality and 'T: equality> (stage: Stage<Image<'S>, Image<'T>>) : Stage<Slab<'S>, Slab<'T>> =
    let mapper debug (slab: Slab<'S>) =
        match Stage.runSingletonToList debug slab.Image stage with
        | [ image ] ->
            { Image = image
              EmitRange = slab.EmitRange }
        | [] ->
            invalidOp $"mapSlabWithStage expected stage '{stage.Name}' to emit one slab image, but it emitted none."
        | many ->
            many |> List.iter (fun image -> image.decRefCount())
            invalidOp $"mapSlabWithStage expected stage '{stage.Name}' to emit one slab image, but it emitted {many.Length}."

    let memoryNeed nPixels =
        stage.MemoryNeed (SlimPipeline.Single nPixels)
        |> SingleOrPair.sum
        |> SingleOrPair.fst

    Stage.map $"mapSlabWithStage ({stage.Name})" mapper memoryNeed id

let slabToWindowAlong<'T when 'T: equality> axis : Stage<Image<'T>, Window<Image<'T>>> =
    let mapper (_debug: bool) (slab: Image<'T>) =
        let items =
            try
                if slab.GetDimensions() <= axis then
                    slab.incRefCount()
                    [ slab ]
                else
                    ImageFunctions.unstack axis slab
            finally
                slab.decRefCount()

        items
        |> List.iteri (fun offset item -> item.index <- slab.index + offset)

        Window.create 0u (uint items.Length) items

    let memoryNeed nPixels = 2UL * nPixels * (typeof<'T> |> Image.getBytesPerComponent |> uint64)
    Stage.map "slabToWindow" mapper memoryNeed id

let slabToWindow<'T when 'T: equality> : Stage<Image<'T>, Window<Image<'T>>> =
    slabToWindowAlong<'T> 2u

let slabWithRangeToWindowAlong<'T when 'T: equality> axis : Stage<Slab<'T>, Window<Image<'T>>> =
    let mapper (_debug: bool) (slab: Slab<'T>) =
        let items =
            try
                if slab.Image.GetDimensions() <= axis then
                    slab.Image.incRefCount()
                    [ slab.Image ]
                else
                    ImageFunctions.unstack axis slab.Image
            finally
                slab.Image.decRefCount()

        items
        |> List.iteri (fun offset item -> item.index <- slab.Image.index + offset)

        let _, emitCount = slab.EmitRange
        let boundedEmitCount = min emitCount (uint items.Length)
        Window.create (fst slab.EmitRange) boundedEmitCount items

    let memoryNeed nPixels = 2UL * nPixels * (typeof<'T> |> Image.getBytesPerComponent |> uint64)
    Stage.map "slabWithRangeToWindow" mapper memoryNeed id

let slabWithRangeToWindow<'T when 'T: equality> : Stage<Slab<'T>, Window<Image<'T>>> =
    slabWithRangeToWindowAlong<'T> 2u

let windowSkipTakeMReleaseAfterWithMemory outputStart outputCount releaseItem memoryNeed : Stage<Window<'T>, 'T list> =
    let mapper (_debug: bool) (window: Window<'T>) =
        let emitStart, emitCount = window.EmitRange
        let requestedStart = outputStart
        let requestedEnd = outputStart + outputCount
        let emitEnd = emitStart + emitCount
        let effectiveStart = max requestedStart emitStart
        let effectiveEnd = min requestedEnd emitEnd
        let effectiveCount = if effectiveEnd > effectiveStart then effectiveEnd - effectiveStart else 0u

        if effectiveCount = 0u || int effectiveStart >= window.Items.Length then
            window.Items |> List.iter releaseItem
            []
        else
            let selected =
                window.Items
                |> List.skip (int effectiveStart)
                |> List.take (min (int effectiveCount) (max 0 (window.Items.Length - int effectiveStart)))

            window.Items
            |> List.filter (fun image ->
                selected
                |> List.exists (fun selectedImage -> Object.ReferenceEquals(selectedImage, image))
                |> not)
            |> List.iter releaseItem

            selected

    Stage.map "windowSkipTakeM" mapper memoryNeed id

let windowSkipTakeMReleaseAfter outputStart outputCount releaseItem : Stage<Window<'T>, 'T list> =
    let memoryNeed nElements = nElements * uint64 (max 1u outputCount)
    windowSkipTakeMReleaseAfterWithMemory outputStart outputCount releaseItem memoryNeed

let windowSkipTakeM outputStart outputCount : Stage<Window<'T>, 'T list> =
    windowSkipTakeMReleaseAfter outputStart outputCount ignore

let windowItems () : Stage<Window<'T>, 'T list> =
    Stage.map "windowItems" (fun _ (window: Window<'T>) -> window.Items) id id

let slabSkipTakeM<'T when 'T: equality> outputStart outputCount : Stage<Window<Image<'T>>, Image<'T> list> =
    let memoryNeed nPixels =
        2UL * nPixels * uint64 (max 1u outputCount) * (typeof<'T> |> Image.getBytesPerComponent |> uint64)
    windowSkipTakeMReleaseAfterWithMemory outputStart outputCount (fun (image: Image<'T>) -> image.decRefCount()) memoryNeed

let mapWindow (name: string) (f: bool -> Window<'T> -> 'S) memoryNeed elementTransformation =
    Stage.map name (fun debug (window: Window<'T>) -> f debug window) memoryNeed elementTransformation
let mapWindowItems (name: string) (f: bool -> 'T list -> 'S) memoryNeed elementTransformation =
    Stage.map name (fun debug (window: Window<'T>) -> f debug window.Items) memoryNeed elementTransformation

let liftWindowedUnaryReleaseAfter
    (name: string)
    (windowSize: uint)
    (f: Image<'S> -> Image<'T>)
    (memoryNeed: MemoryNeed)
    (elementTransformation: ElementTransformation)
    : Stage<Image<'S>, Image<'T>> =
    let win = max 1u windowSize
    let mapper debug =
        volFctToWindowFctReleaseAfterDebug debug f 1u 0u win
    let stg = mapWindow name mapper memoryNeed elementTransformation
    (window win 0u win) --> stg --> flattenList ()

let requireWindowSize<'T when 'T: equality> requiredInputDepth : Stage<Window<Image<'T>>, Window<Image<'T>>> =
    Stage.filter
        "requireWindowSize"
        (fun _ window -> uint window.Items.Length >= requiredInputDepth)
        (fun window -> releaseWindowItems window.Items)

let liftSlabReleaseAfter<'S, 'T when 'S: equality and 'T: equality> name f memoryNeed elementTransformation : Stage<Image<'S>, Image<'T>> =
    liftUnaryReleaseAfter name f memoryNeed elementTransformation

let windowedViaSlabRequired<'S, 'T when 'S: equality and 'T: equality>
    (windowSize: uint)
    (pad: uint)
    (stride: uint)
    (requiredInputDepth: uint)
    (outputStart: uint)
    (outputCount: uint)
    (stage: Stage<Image<'S>, Image<'T>>)
    : Stage<Image<'S>, Image<'T>> =
    (window windowSize pad stride)
    --> requireWindowSize requiredInputDepth
    --> windowToSlabWithRange<'S>
    --> mapSlabWithStage stage
    --> slabWithRangeToWindow<'T>
    --> slabSkipTakeM outputStart outputCount
    --> flattenList ()

let map f = Stage.map "map" f id id
let sinkOp (pl: Plan<unit,unit>) : unit = 
    Plan.sink pl
let sink (pl: Plan<unit,'T>) : unit =
    pl >=> ignoreSingles () |> Plan.sink
let sinkList (plLst: Plan<unit,unit> list) : unit = Plan.sinkList plLst
//let combineIgnore = Plan.combineIgnore
let drain pl = Plan.drainSingle "drainSingle" pl
let drainList pl = Plan.drainList "drainList" pl
let drainLast pl = Plan.drainLast "drainLast" pl
//let tap str = incRefCountOp () --> (Stage.tap str)

let tap = Stage.tap
//let tap str = Stage.tap str --> incRef()// tap and tapIt neither realeases after nor increases number of references
let tapIt = Stage.tapIt
