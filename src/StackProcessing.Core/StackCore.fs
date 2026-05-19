module StackCore

open SlimPipeline // Core processing model

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

type HistogramBinning =
    | FixedEdges of firstLeftEdge: float * lastLeftEdge: float * bins: uint32
    | FixedWidth of binWidth: uint64

type Histogram<'T when 'T: comparison> =
    { Counts: Map<'T, uint64>
      Binning: HistogramBinning option }

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

let getMem () =
    System.GC.Collect()
    System.GC.WaitForPendingFinalizers()
    System.GC.Collect()

let imageResourceOps<'S when 'S: equality> : ResourceOps<Image<'S>> =
    { Retain = fun image -> image.incRefCount()
      Release = fun image -> image.decRefCount()
      MemoryOf = fun image -> Image<'S>.memoryEstimateSItk image.Image |> uint64 |> Some }

let releaseAfterWith (ops: ResourceOps<'S>) (f: 'S -> 'T) (value: 'S) =
    try
        f value
    finally
        ops.Release value

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
    let rec parse debugLevel optimize costDiscrepancies remaining kept =
        match remaining with
        | [] -> debugLevel, optimize, costDiscrepancies, kept |> List.rev |> List.toArray
        | "-d" :: value :: rest
        | "--debug-level" :: value :: rest ->
            match System.UInt32.TryParse value with
            | true, level -> parse (Some level) optimize costDiscrepancies rest kept
            | false, _ -> failwith $"Expected unsigned integer debug level after -d, got '{value}'"
        | ("--no-optimize" | "--no-optimizer") :: rest ->
            parse debugLevel false costDiscrepancies rest kept
        | "--optimize" :: value :: rest
        | "--optimizer" :: value :: rest ->
            match tryParseBool value with
            | Some enabled -> parse debugLevel enabled costDiscrepancies rest kept
            | None -> failwith $"Expected boolean optimizer value after --optimize, got '{value}'"
        | ("--cost-discrepancies" | "--cost-discrepancy-report") :: rest ->
            parse (debugLevel |> Option.orElse (Some 1u)) optimize true rest kept
        | ("--no-cost-discrepancies" | "--no-cost-discrepancy-report") :: rest ->
            parse debugLevel optimize false rest kept
        | arg :: rest ->
            parse debugLevel optimize costDiscrepancies rest (arg :: kept)

    let debugLevel, optimize, costDiscrepancies, rest =
        parse None (optimizerEnabled ()) false (args |> Array.toList) []

    let plan =
        match debugLevel with
        | Some level -> debug level optimize availableMemory
        | None -> sourceWithOptimizer optimize availableMemory

    Plan.withCostDiscrepancyReporting costDiscrepancies plan, rest
 
let zip = Plan.zip

(*
let inline isExactlyImage<'T> () =
    let t = typeof<'T>
    t.IsGenericType && t.GetGenericTypeDefinition() = typedefof<Image<_>>
*)
let (>=>>) (pl: Plan<'In, 'S>) (stage1: Stage<'S, 'U>, stage2: Stage<'S, 'V>) : Plan<'In, 'U * 'V> = 
    let stream2Window winSz pad stride (stg: Stage<'T,'Out>) : Stage<'T,'Out> =
        let zeroMaker _ = id
        Stage.window "makeWindow: window" winSz pad zeroMaker stride
        --> Stage.map "makeWindow: select delayed emit range"
                (fun _ window ->
                    let start = 0
                    let count = 1
                    let result =
                        window.Items
                        |> List.skip start
                        |> List.take (min count (max 0 (window.Items.Length - start)))
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
let (>>=>) = Plan.(>>=>)
let (>>=>>) = Plan.(>>=>>)
let teeFst = Stage.teeFst
let teeSnd = Stage.teeSnd
let ignoreSingles () : Stage<_,unit> = Stage.ignore (decIfImage>>ignore)
let ignorePairs () : Stage<_,unit> = Stage.ignorePairs<_,unit> ((decIfImage>>ignore),(decIfImage>>ignore))
let zeroMaker (index: int) (ex: Image<'S>) : Image<'S> =
    new Image<'S>(ex.GetSize(), ex.GetNumberOfComponentsPerPixel(), "padding", index)

let window windowSize pad stride = Stage.window "window" windowSize pad zeroMaker stride
let flatten () = Stage.flattenWindow "flatten"
let flattenList () = Stage.flatten "flatten"
let mapWindow (name: string) (f: bool -> Window<'T> -> 'S) memoryNeed elementTransformation =
    Stage.map name (fun debug (window: Window<'T>) -> f debug window) memoryNeed elementTransformation
let mapWindowItems (name: string) (f: bool -> 'T list -> 'S) memoryNeed elementTransformation =
    Stage.map name (fun debug (window: Window<'T>) -> f debug window.Items) memoryNeed elementTransformation
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
