module StackCore

open SlimPipeline // Core processing model

type Stage<'S,'T> = SlimPipeline.Stage<'S,'T>
type Profile = SlimPipeline.Profile
type ProfileTransition = SlimPipeline.ProfileTransition
type ResourceOps<'T> = SlimPipeline.ResourceOps<'T>
//type Slice<'S when 'S: equality> = Slice.Slice<'S>
type Image<'S when 'S: equality> = Image.Image<'S>

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

let volFctToLstFctReleaseAfterDebug debug (f: Image<'S>->Image<'T>) pad stride images =
    let rssDebug = debug && DebugLevel.rssEnabled()
    let startKb = if rssDebug then rssKb() else 0UL
    let mutable previousKb, _ = sampleVolRssProbe rssDebug "start" startKb startKb
    let stack = ImageFunctions.stack images 
    let currentKb, stackDelta = sampleVolRssProbe rssDebug "after stack" startKb previousKb
    previousKb <- currentKb
    images |> List.take (min (int stride) images.Length) |> List.iter (fun I -> I.decRefCount())
    let currentKb, releaseInputsDelta = sampleVolRssProbe rssDebug "after release input slices" startKb previousKb
    previousKb <- currentKb
    let vol = f stack
    let currentKb, volumeFunctionDelta = sampleVolRssProbe rssDebug "after volume function" startKb previousKb
    previousKb <- currentKb
    stack.decRefCount ()
    let currentKb, disposeStackDelta = sampleVolRssProbe rssDebug "after dispose stack" startKb previousKb
    previousKb <- currentKb
    let result = ImageFunctions.unstackSkipNTakeM pad stride vol
    let currentKb, unstackDelta = sampleVolRssProbe rssDebug "after unstack" startKb previousKb
    previousKb <- currentKb
    vol.decRefCount ()
    let currentKb, disposeVolumeDelta = sampleVolRssProbe rssDebug "after dispose volume" startKb previousKb
    previousKb <- currentKb
    printVolRssSummary rssDebug startKb previousKb stackDelta releaseInputsDelta volumeFunctionDelta disposeStackDelta unstackDelta disposeVolumeDelta
    result

let volFctToLstFctReleaseAfter (f: Image<'S>->Image<'T>) pad stride images =
    volFctToLstFctReleaseAfterDebug false f pad stride images

let (>=>) = Plan.(>=>)
let (-->) = Stage.(-->)
let source = Plan.source 
let debug level availableMemory = 
    let level = max 1u level
    Image.Image<_>.setDebugLevel level
    Plan.debug level availableMemory
let commandLineSource availableMemory (args: string array) =
    if args.Length >= 2 && args[0] = "-d" then
        let level =
            match System.UInt32.TryParse(args[1]) with
            | true, value -> value
            | false, _ -> failwith $"Expected unsigned integer debug level after -d, got '{args[1]}'"
        debug level availableMemory, args |> Array.skip 2
    else
        source availableMemory, args
 
let zip = Plan.zip

(*
let inline isExactlyImage<'T> () =
    let t = typeof<'T>
    t.IsGenericType && t.GetGenericTypeDefinition() = typedefof<Image<_>>
*)
let promoteStreamingToWindow (name: string) (winSz: uint) (pad: uint) (stride: uint) (emitStart: uint) (emitCount: uint) (stage: Stage<'T,'S>) : Stage<'T, 'S> = // Does not change shape
        let zeroMaker i = id
        (Stage.window $"{name}: window" winSz pad zeroMaker stride) 
        --> (Stage.map $"{name}: skip and take" (fun _ lst ->
                let result = lst |> List.skip (int stride) |> List.take 1
                lst |> List.take (int stride) |> List.map decIfImage |> ignore
                result
            ) id id )
        --> Stage.flatten $"{name}: flatten"
        --> stage

let (>=>>) (pl: Plan<'In, 'S>) (stage1: Stage<'S, 'U>, stage2: Stage<'S, 'V>) : Plan<'In, 'U * 'V> = 
    let stream2Window winSz pad stride stg = 
        stg |> promoteStreamingToWindow "makeWindow" winSz pad stride 0u 1u

    let stg1,stg2 =
        match stage1.Transition.From, stage2.Transition.From with
        | Streaming, Streaming -> stage1, stage2
        | Window (a1,b1,c1,d1,e1), Window (a2,b2,c2,d2,e2) when a1=a2 && b1=b2 && c1=c2 && d1=d2 && e1=e2 -> stage1, stage2
        | Streaming, Window (winSz, stride, pad, emitStart, emitCount) -> 
            printfn "left is promoted"
            stream2Window winSz pad stride stage1, stage2 
        | Window (winSz, stride, pad, emitStart, emitCount), Streaming -> 
            printfn "right is promoted"
            stage1, stream2Window winSz pad stride stage2
        | _,_ -> failwith $"[>=>>] does not know how to combine the stage-profiles: {stage1.Transition.From} vs {stage2.Transition.From}"

    Plan.(>=>>) (pl >=> incRef ()) (stg1, stg2)
let (>>=>) = Plan.(>>=>)
let (>>=>>) = Plan.(>>=>>)
let teeFst = Stage.teeFst
let teeSnd = Stage.teeSnd
let ignoreSingles () : Stage<_,unit> = Stage.ignore (decIfImage>>ignore)
let ignorePairs () : Stage<_,unit> = Stage.ignorePairs<_,unit> ((decIfImage>>ignore),(decIfImage>>ignore))
let zeroMaker (index: int) (ex: Image<'S>) : Image<'S> = new Image<'S>(ex.GetSize(), 1u, "padding", index)
let window windowSize pad stride = Stage.window "window" windowSize pad zeroMaker stride
let flatten () = Stage.flatten "flatten"
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
let idStage<'T> = Stage.idStage<'T>
