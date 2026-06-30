module StackCore

open SlimPipeline
open System
open System.Runtime.InteropServices

module ChunkPrimitive = Chunk

type Stage<'S,'T> = SlimPipeline.Stage<'S,'T>
type Profile = SlimPipeline.Profile
type ProfileTransition = SlimPipeline.ProfileTransition
type ResourceOps<'T> = SlimPipeline.ResourceOps<'T>
type Window<'T> = SlimPipeline.Window<'T>

type ChunkLayout = ChunkPrimitive.ChunkLayout
type ChunkIndex = ChunkPrimitive.ChunkIndex
type Chunk<'T when 'T: equality> = ChunkPrimitive.Chunk<'T>
type LocatedChunk<'T when 'T: equality> = ChunkPrimitive.LocatedChunk<'T>
type EncodedLocatedChunk = ChunkPrimitive.EncodedLocatedChunk
type VectorChunk<'T when 'T: equality> = ChunkPrimitive.VectorChunk<'T>
type SpectralLayout = ChunkPrimitive.SpectralLayout
type SpectralChunk = ChunkPrimitive.SpectralChunk
type DenseUInt32UnionFind = ChunkPrimitive.DenseUInt32UnionFind
type ChunkStats = ChunkPrimitive.ChunkStats

type HistogramBinning =
    | FixedEdges of firstLeftEdge: float * lastLeftEdge: float * bins: uint32
    | FixedWidth of binWidth: uint64

type Histogram<'T when 'T: comparison> =
    { Counts: Map<'T, uint64>
      Binning: HistogramBinning option }

type VectorizedMatrix =
    { Rows: uint
      Columns: uint
      Values: float list }

type ImageStats =
    { NumPixels: uint64
      Mean: float
      Std: float
      Min: float
      Max: float
      Sum: float
      Var: float }

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
    let withIndex = ChunkPrimitive.withIndex
    let withSourceDepth = ChunkPrimitive.withSourceDepth
    let withIndexOption = ChunkPrimitive.withIndexOption
    let withSameIndex = ChunkPrimitive.withSameIndex
    let setDebugLevel = ChunkPrimitive.ChunkStats.setDebugLevel
    let resetStats = ChunkPrimitive.ChunkStats.reset
    let stats = ChunkPrimitive.ChunkStats.snapshot
    let formatStats = ChunkPrimitive.ChunkStats.format
    let span<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (chunk: Chunk<'T>) =
        ChunkPrimitive.span<'T> chunk
    let incRef = ChunkPrimitive.incRef
    let decRef = ChunkPrimitive.decRef
    let incRefVector = ChunkPrimitive.incRefVector
    let decRefVector = ChunkPrimitive.decRefVector
    let vectorComponentCount = ChunkPrimitive.vectorComponentCount
    let toChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkPrimitive.toChunk<'T>
    let ofChunk = ChunkPrimitive.ofChunk
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
    let vectorMagnitudeSquaredFloat32 = ChunkPrimitive.vectorMagnitudeSquaredFloat32
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

type Point2D =
    { X: float
      Y: float }

type Polygon2D = Point2D list

let (>=>) = Plan.(>=>)
let (-->) = Stage.(-->)
let (>>=>) = Plan.(>>=>)
let (>>=>>) = Plan.(>>=>>)
let teeFst = Stage.teeFst
let teeSnd = Stage.teeSnd

let identityStage name =
    Stage.map name (fun _ value -> value) id id

let map f = Stage.map "map" f id id
let flattenList () = Stage.flatten "flatten"

let mapWindow (name: string) (f: bool -> Window<'T> -> 'S) memoryNeed elementTransformation =
    Stage.map name (fun debug (window: Window<'T>) -> f debug window) memoryNeed elementTransformation

let mapWindowItems (name: string) (f: bool -> 'T list -> 'S) memoryNeed elementTransformation =
    Stage.map name (fun debug (window: Window<'T>) -> f debug window.Items) memoryNeed elementTransformation

let windowSkipTakeMReleaseAfterWithMemory outputStart outputCount releaseItem memoryNeed : Stage<Window<'T>, 'T list> =
    let mapper (_debug: bool) (window: Window<'T>) =
        let emitStart, emitCount = window.EmitRange
        let requestedEnd = outputStart + outputCount
        let emitEnd = emitStart + emitCount
        let effectiveStart = max outputStart emitStart
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
            |> List.filter (fun item ->
                selected
                |> List.exists (fun selectedItem -> Object.ReferenceEquals(box selectedItem, box item))
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

let zeroMaker (_index: int) (_example: 'T) : 'T =
    Unchecked.defaultof<'T>

let inline flatIndex2 width x y =
    y * width + x

let inline flatIndex3 width height x y z =
    (z * height + y) * width + x

let window windowSize pad stride =
    Stage.window "window" windowSize pad zeroMaker stride

let flatten () = Stage.flattenWindow "flatten"

let chunkResourceOps<'T when 'T: equality> : ResourceOps<Chunk<'T>> =
    { Retain = fun chunk -> Chunk.incRef chunk |> ignore
      Release = Chunk.decRef
      MemoryOf = fun chunk -> Some(uint64 chunk.ByteLength) }

let fork (stage1: Stage<'S, 'U>, stage2: Stage<'S, 'V>) : Stage<'S, 'U * 'V> =
    Stage.fork (stage1, stage2)

let forkChunk<'T, 'U, 'V when 'T: equality> (stage1: Stage<Chunk<'T>, 'U>, stage2: Stage<Chunk<'T>, 'V>) : Stage<Chunk<'T>, 'U * 'V> =
    Stage.retainWith "forkChunk.retain" chunkResourceOps<'T>
    --> Stage.fork (stage1, stage2)

let (-->>) (stage: Stage<'S, 'T>) (stage1: Stage<'T, 'U>, stage2: Stage<'T, 'V>) : Stage<'S, 'U * 'V> =
    stage --> fork (stage1, stage2)

let (>=>>) (pl: Plan<'In, 'S>) (stage1: Stage<'S, 'U>, stage2: Stage<'S, 'V>) : Plan<'In, 'U * 'V> =
    Plan.(>=>>) pl (stage1, stage2)

let ignoreSingles () : Stage<_,unit> = Stage.ignore ignore
let ignorePairs () : Stage<_,unit> = Stage.ignorePairs (ignore, ignore)

let sinkOp (pl: Plan<unit,unit>) : unit =
    Plan.sink pl
    if pl.debug && pl.debugLevel >= 2u then
        printfn $"[chunkStats] {Chunk.stats() |> Chunk.formatStats}"

let sink (pl: Plan<unit,'T>) : unit =
    let pl = pl >=> ignoreSingles ()
    Plan.sink pl
    if pl.debug && pl.debugLevel >= 2u then
        printfn $"[chunkStats] {Chunk.stats() |> Chunk.formatStats}"

let sinkList (plLst: Plan<unit,unit> list) : unit = Plan.sinkList plLst
let drain pl = Plan.drainSingle "drainSingle" pl

let private tryParseBool value =
    match value |> string |> fun v -> v.Trim().ToLowerInvariant() with
    | "1" | "true" | "yes" | "y" | "on" -> Some true
    | "0" | "false" | "no" | "n" | "off" -> Some false
    | _ -> None

let optimizerEnabled () =
    match Environment.GetEnvironmentVariable "STACKPROCESSING_OPTIMIZE" with
    | null | "" -> true
    | value ->
        tryParseBool value
        |> Option.defaultWith (fun () ->
            failwith $"STACKPROCESSING_OPTIMIZE must be true/false, 1/0, yes/no, or on/off; got '{value}'")

let sourceWithOptimizer optimize availableMemory =
    Chunk.setDebugLevel 0u
    Plan.sourceWithOptimizer optimize availableMemory

let source availableMemory =
    sourceWithOptimizer (optimizerEnabled ()) availableMemory

let debug level optimize availableMemory =
    let level = max 1u level
    Chunk.setDebugLevel level
    if level >= 2u then
        Chunk.resetStats()
    Plan.debug level optimize availableMemory

let debugDefault level availableMemory =
    debug level (optimizerEnabled ()) availableMemory

let commandLineSource availableMemory (args: string array) =
    let rec parse debugLevel optimize costDiscrepancies costFlags costModel remaining kept =
        match remaining with
        | [] -> debugLevel, optimize, costDiscrepancies, costFlags, costModel, kept |> List.rev |> List.toArray
        | "-d" :: value :: rest
        | "--debug-level" :: value :: rest ->
            match UInt32.TryParse value with
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

let memNeeded<'T> nTimes nElems =
    nElems * nTimes * uint64 (Marshal.SizeOf(typeof<'T>))

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
