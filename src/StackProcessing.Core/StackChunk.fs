module ChunkFunctions

open System
open System.Collections.Generic
open System.Numerics
open System.IO
open System.Runtime.InteropServices
open System.Threading.Tasks
open FSharp.Control
open SlimPipeline
open StackCore
open TinyLinAlg

module ChunkKernel = ChunkCore.ChunkFunctions

let private binaryBackground = 0uy
let private binaryForeground = 1uy

let chunkElementBytes<'T> =
    Marshal.SizeOf<'T>()

let chunkMemoryNeed<'T> nPixels =
    nPixels * uint64 (chunkElementBytes<'T>)

let private chunkSourceStage<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    name
    (width: uint)
    (height: uint)
    (depth: uint)
    (mapper: int -> Chunk<'T>)
    =
    let nPixels = uint64 width * uint64 height
    let memoryNeed _ = chunkMemoryNeed<'T> nPixels
    let transition = ProfileTransition.create Unit Streaming
    Stage.init name depth mapper transition memoryNeed (fun _ -> nPixels)

let private chunkSourcePlan<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (pl: Plan<unit, unit>)
    (width: uint)
    (height: uint)
    (depth: uint)
    (stage: Stage<unit, Chunk<'T>> option)
    =
    let nPixels = uint64 width * uint64 height
    let memPeak = chunkMemoryNeed<'T> nPixels
    Plan.createWithOptimizer stage pl.memAvail memPeak nPixels (uint64 depth) pl.debug pl.optimize
    |> Plan.withRuntimeOptionsFrom pl

let private convertFloat<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> value =
    Convert.ChangeType(value, typeof<'T>) :?> 'T

let chunkZero<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (width: uint)
    (height: uint)
    (depth: uint)
    (pl: Plan<unit, unit>)
    : Plan<unit, Chunk<'T>> =
    let mapper (i: int) =
        let chunk = Chunk.create<'T> (uint64 width, uint64 height, 1UL)
        (Chunk.span chunk).Fill Unchecked.defaultof<'T>
        if pl.debug && DebugLevel.current() >= 1u then printfn "[chunkZero] Created slice %A" i
        chunk

    let stage = chunkSourceStage $"chunkZero.{typeof<'T>.Name}" width height depth mapper |> Some
    chunkSourcePlan pl width height depth stage

let private coordinateChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    width
    height
    z
    coordinate
    =
    let chunk = Chunk.create<'T> (uint64 width, uint64 height, 1UL)
    let pixels = Chunk.span chunk
    let widthI = int width
    let heightI = int height

    for y in 0 .. heightI - 1 do
        for x in 0 .. widthI - 1 do
            pixels[Chunk.toIndex widthI heightI x y 0] <- convertFloat<'T> (coordinate x y z)

    chunk

let chunkCoordinateX<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (width: uint)
    (height: uint)
    (depth: uint)
    (pl: Plan<unit, unit>)
    : Plan<unit, Chunk<'T>> =
    let mapper i = coordinateChunk<'T> width height i (fun x _ _ -> float x)
    let stage = chunkSourceStage $"chunkCoordinateX.{typeof<'T>.Name}" width height depth mapper |> Some
    chunkSourcePlan pl width height depth stage

let chunkCoordinateY<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (width: uint)
    (height: uint)
    (depth: uint)
    (pl: Plan<unit, unit>)
    : Plan<unit, Chunk<'T>> =
    let mapper i = coordinateChunk<'T> width height i (fun _ y _ -> float y)
    let stage = chunkSourceStage $"chunkCoordinateY.{typeof<'T>.Name}" width height depth mapper |> Some
    chunkSourcePlan pl width height depth stage

let chunkCoordinateZ<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (width: uint)
    (height: uint)
    (depth: uint)
    (pl: Plan<unit, unit>)
    : Plan<unit, Chunk<'T>> =
    let mapper i = coordinateChunk<'T> width height i (fun _ _ z -> float z)
    let stage = chunkSourceStage $"chunkCoordinateZ.{typeof<'T>.Name}" width height depth mapper |> Some
    chunkSourcePlan pl width height depth stage

let private pointInPolygon (polygon: Polygon2D) x y =
    let px = float x + 0.5
    let py = float y + 0.5
    let vertices = polygon |> List.toArray
    let mutable inside = false

    if vertices.Length >= 3 then
        let mutable j = vertices.Length - 1
        for i in 0 .. vertices.Length - 1 do
            let xi = vertices[i].X
            let yi = vertices[i].Y
            let xj = vertices[j].X
            let yj = vertices[j].Y
            let crosses = (yi > py) <> (yj > py)
            if crosses then
                let xIntersect = (xj - xi) * (py - yi) / (yj - yi) + xi
                if px < xIntersect then
                    inside <- not inside
            j <- i

    inside

let chunkPolygonMask
    (width: uint)
    (height: uint)
    (polygon: Polygon2D)
    : unit -> Chunk<uint8> =
    if width = 0u then invalidArg "width" "chunkPolygonMask width must be positive."
    if height = 0u then invalidArg "height" "chunkPolygonMask height must be positive."

    fun () ->
        let chunk = Chunk.create<uint8> (uint64 width, uint64 height, 1UL)
        chunk.Bytes.AsSpan(0, chunk.ByteLength).Clear()
        let pixels = Chunk.span chunk
        let widthI = int width
        let heightI = int height

        for y in 0 .. heightI - 1 do
            for x in 0 .. widthI - 1 do
                if pointInPolygon polygon x y then
                    pixels[Chunk.toIndex widthI heightI x y 0] <- 1uy

        chunk

let chunkRepeat<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (chunkSeed: unit -> Chunk<'T>)
    (depth: uint)
    (pl: Plan<unit, unit>)
    : Plan<unit, Chunk<'T>> =
    if depth = 0u then invalidArg "depth" "chunkRepeat requires a positive depth."
    let chunk = chunkSeed()
    let width, height, _ = chunk.Size
    let widthU = uint width
    let heightU = uint height

    let mapper (i: int) =
        let copy = ChunkCore.ChunkFunctions.copyChunk chunk
        if pl.debug && DebugLevel.current() >= 1u then printfn "[chunkRepeat] Created slice %A" i
        copy

    let stage =
        { chunkSourceStage $"chunkRepeat.{typeof<'T>.Name}" widthU heightU depth mapper with
            Cleaning = [ fun () -> Chunk.decRef chunk ] }
        |> Some

    chunkSourcePlan pl widthU heightU depth stage

let chunkRepeatStage<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    depth
    : Stage<Chunk<'T>, Chunk<'T>> =
    if depth = 0u then invalidArg "depth" "chunkRepeatStage requires a positive depth."

    let copySlice (chunk: Chunk<'T>) =
        try
            [ for _ in 0 .. int depth - 1 -> ChunkCore.ChunkFunctions.copyChunk chunk ]
        finally
            Chunk.decRef chunk

    Stage.map $"chunkRepeatStage {depth}" (fun _ chunk -> copySlice chunk) (fun n -> n * uint64 depth) id
    --> flattenList ()
    |> Stage.withSliceCardinality SliceCardinality.unknown

let releaseUnaryChunk name f memoryNeed : Stage<Chunk<'T>, Chunk<'U>> =
    let mapper _debug chunk =
        try
            f chunk
        finally
            Chunk.decRef chunk

    Stage.map name mapper memoryNeed id

let mapi<'T, 'U when 'T: equality and 'U: equality
                  and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType
                  and 'U: (new: unit -> 'U) and 'U: struct and 'U :> ValueType>
    (f: uint -> Chunk<'T> -> Chunk<'U>)
    : Stage<Chunk<'T>, Chunk<'U>> =
    let mapper _debug index chunk =
        try
            f (uint index) chunk
        finally
            Chunk.decRef chunk

    Stage.mapi
        $"chunkMapi.{typeof<'T>.Name}To{typeof<'U>.Name}"
        mapper
        (fun n -> n * uint64 (chunkElementBytes<'T> + chunkElementBytes<'U>))
        id

let validateSameSize name (a: Chunk<'T>) (b: Chunk<'U>) =
    ChunkKernel.validateSameSize name a b

let inline map2Chunk<'T, 'U, 'V when 'T: equality and 'U: equality and 'V: equality
                                       and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType
                                       and 'U: (new: unit -> 'U) and 'U: struct and 'U :> ValueType
                                       and 'V: (new: unit -> 'V) and 'V: struct and 'V :> ValueType>
    name
    f
    (a: Chunk<'T>)
    (b: Chunk<'U>) =
    ChunkKernel.map2Chunk name f a b

let releaseBinaryChunk name f memoryNeed : Stage<Chunk<'T> * Chunk<'U>, Chunk<'V>> =
    let mapper _debug (a, b) =
        try
            f a b
        finally
            Chunk.decRef a
            Chunk.decRef b

    Stage.map name mapper memoryNeed id

let releaseUnaryVectorChunk name f memoryNeed : Stage<VectorChunk<'T>, Chunk<'U>> =
    let mapper _debug (vector: VectorChunk<'T>) =
        try
            f vector
        finally
            Chunk.decRefVector vector

    Stage.map name mapper memoryNeed id

let releaseUnaryVectorToVectorChunk name f memoryNeed : Stage<VectorChunk<'T>, VectorChunk<'U>> =
    let mapper _debug (vector: VectorChunk<'T>) =
        try
            f vector
        finally
            Chunk.decRefVector vector

    Stage.map name mapper memoryNeed id

let releaseUnaryVectorToVectorChunkParallel name workers f memoryNeed : Stage<VectorChunk<'T>, VectorChunk<'U>> =
    if workers < 1 then
        invalidArg "workers" $"{name} expects at least one worker, got {workers}."

    let mapper _debug (window: Window<VectorChunk<'T>>) =
        match window.Items with
        | [ vector ] ->
            try
                [ f vector ]
            finally
                Chunk.decRefVector vector
        | items ->
            items |> List.iter Chunk.decRefVector
            invalidArg "window" $"{name} expects singleton windows, got {items.Length} items."

    Stage.parallelCollect
        $"{name}.parallelCollect.workers{workers}"
        1
        workers
        1
        0
        (fun _ vector -> vector)
        mapper
        memoryNeed
        id

let releaseBinaryVectorChunk name f memoryNeed : Stage<VectorChunk<'T> * VectorChunk<'T>, Chunk<'U>> =
    let mapper _debug ((a, b): VectorChunk<'T> * VectorChunk<'T>) =
        try
            f a b
        finally
            Chunk.decRefVector a
            Chunk.decRefVector b

    Stage.map name mapper memoryNeed id

let copy<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> : Stage<Chunk<'T>, Chunk<'T>> =
    releaseUnaryChunk $"chunkCopy.{typeof<'T>.Name}" ChunkKernel.copyChunk<'T> (fun n -> 2UL * chunkMemoryNeed<'T> n)

let pad<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    lowerX upperX lowerY upperY lowerZ upperZ value
    : Stage<Chunk<'T>, Chunk<'T>> =
    releaseUnaryChunk
        $"chunkPad.{typeof<'T>.Name}"
        (ChunkKernel.padChunk<'T> lowerX upperX lowerY upperY lowerZ upperZ value)
        (fun n -> 3UL * chunkMemoryNeed<'T> n)

let crop<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    lowerX upperX lowerY upperY lowerZ upperZ
    : Stage<Chunk<'T>, Chunk<'T>> =
    releaseUnaryChunk
        $"chunkCrop.{typeof<'T>.Name}"
        (ChunkKernel.cropChunk<'T> lowerX upperX lowerY upperY lowerZ upperZ)
        (fun n -> 2UL * chunkMemoryNeed<'T> n)

let squeeze<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    : Stage<Chunk<'T>, Chunk<'T>> =
    releaseUnaryChunk $"chunkSqueeze.{typeof<'T>.Name}" ChunkKernel.squeezeChunk<'T> (fun n -> 2UL * chunkMemoryNeed<'T> n)

let concatenateAlong<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    axis
    : Stage<Chunk<'T> * Chunk<'T>, Chunk<'T>> =
    let mapper a b = ChunkKernel.concatenateChunk<'T> axis a b
    releaseBinaryChunk $"chunkConcatenateAlong{axis}.{typeof<'T>.Name}" mapper (fun n -> 3UL * chunkMemoryNeed<'T> n)

let permuteAxes<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    order
    : Stage<Chunk<'T>, Chunk<'T>> =
    releaseUnaryChunk
        $"chunkPermuteAxes.{typeof<'T>.Name}"
        (ChunkKernel.permuteAxesChunk<'T> (order |> Seq.toArray))
        (fun n -> 2UL * chunkMemoryNeed<'T> n)

let resample2DNative<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    interpolationName
    outputWidth
    outputHeight
    spacingX
    spacingY
    : Stage<Chunk<'T>, Chunk<'T>> =
    let interpolation = ChunkKernel.ResampleInterpolation.parse interpolationName
    releaseUnaryChunk
        $"chunkResample2DNative.{typeof<'T>.Name}"
        (ChunkKernel.resample2DNativeChunk<'T> interpolation outputWidth outputHeight spacingX spacingY)
        (fun n -> 3UL * chunkMemoryNeed<'T> n)

let private roundPositiveToUInt (value: float) =
    Math.Round(value, MidpointRounding.AwayFromZero)
    |> max 1.0
    |> uint

let private outputSpacingForSize inputSize outputSize =
    if inputSize <= 1u || outputSize <= 1u then
        1.0
    else
        float (inputSize - 1u) / float (outputSize - 1u)

let private trySourcePeekUInt key (pl: Plan<unit, Chunk<'T>>) =
    pl.sourcePeek
    |> Option.bind (fun peek -> peek.Shape |> Map.tryFind key)
    |> Option.bind (fun text ->
        match UInt32.TryParse text with
        | true, value -> Some value
        | _ -> None)

let private validateResizeOutput name value =
    if value = 0u then
        invalidArg name $"{name} must be positive."
    value

let private resize3DNativeStage<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    outputWidth
    outputHeight
    outputDepth
    spacingX
    spacingY
    spacingZ
    interpolationName
    : Stage<Chunk<'T>, Chunk<'T>> =
    let outputWidth = validateResizeOutput "outputWidth" outputWidth
    let outputHeight = validateResizeOutput "outputHeight" outputHeight
    let outputDepth = validateResizeOutput "outputDepth" outputDepth
    if spacingX <= 0.0 || spacingY <= 0.0 || spacingZ <= 0.0 then
        invalidArg "spacing" $"chunkResize expects positive spacing, got ({spacingX}, {spacingY}, {spacingZ})."

    let interpolation = ChunkKernel.ResampleInterpolation.parse interpolationName
    let name = $"chunkResize3DNative.{typeof<'T>.Name}.{outputWidth}x{outputHeight}x{outputDepth}.{interpolationName}"

    let apply debug (input: AsyncSeq<Chunk<'T>>) =
        asyncSeq {
            let mutable previous: (int * Chunk<'T>) option = None
            let mutable currentIndex = -1
            let mutable outputIndex = 0
            let mutable completed = false

            let releasePrevious () =
                match previous with
                | Some(_, chunk) -> Chunk.decRef chunk
                | None -> ()
                previous <- None

            let emitFromPair lowerIndex lower upperIndex upper =
                seq {
                    let mutable keepEmitting = true
                    while keepEmitting && outputIndex < int outputDepth do
                        let sourceZ = float outputIndex * spacingZ
                        if sourceZ > float upperIndex + 1.0e-9 then
                            keepEmitting <- false
                        else
                            let z0 = int (Math.Floor sourceZ)
                            let z1 = int (Math.Ceiling sourceZ)
                            if z0 < lowerIndex || z1 > upperIndex then
                                keepEmitting <- false
                            else
                                let zFraction =
                                    if z1 = z0 then 0.0
                                    else sourceZ - float z0

                                let lowerSlice = if z0 = lowerIndex then lower else upper
                                let upperSlice = if z1 = upperIndex then upper else lower
                                let output =
                                    ChunkKernel.resize3DPairSliceNativeChunk<'T>
                                        interpolation
                                        outputWidth
                                        outputHeight
                                        spacingX
                                        spacingY
                                        zFraction
                                        lowerSlice
                                        upperSlice
                                outputIndex <- outputIndex + 1
                                yield output
                }

            let emitRemainingFromLast lastChunk =
                seq {
                    while outputIndex < int outputDepth do
                        let output =
                            ChunkKernel.resize3DPairSliceNativeChunk<'T>
                                interpolation
                                outputWidth
                                outputHeight
                                spacingX
                                spacingY
                                0.0
                                lastChunk
                                lastChunk
                        outputIndex <- outputIndex + 1
                        yield output
                }

            try
                for chunk in input do
                    currentIndex <- currentIndex + 1
                    match previous with
                    | None ->
                        previous <- Some(currentIndex, chunk)
                        if currentIndex = 0 then
                            for output in emitFromPair 0 chunk 0 chunk do
                                yield output
                    | Some(prevIndex, prevChunk) ->
                        for output in emitFromPair prevIndex prevChunk currentIndex chunk do
                            yield output
                        Chunk.decRef prevChunk
                        previous <- Some(currentIndex, chunk)

                match previous with
                | Some(lastIndex, lastChunk) ->
                    for output in emitFromPair lastIndex lastChunk lastIndex lastChunk do
                        yield output
                    for output in emitRemainingFromLast lastChunk do
                        yield output
                    releasePrevious ()
                | None -> ()

                completed <- true
            finally
                if not completed then
                    releasePrevious ()
        }

    let memoryNeed n =
        let inputBytes = chunkMemoryNeed<'T> n
        let outputBytes = chunkMemoryNeed<'T> (uint64 outputWidth * uint64 outputHeight)
        2UL * inputBytes + outputBytes

    Stage.fromAsyncSeq
        name
        apply
        (ProfileTransition.create Streaming Streaming)
        (StageMemoryModel.fromSinglePeak Map memoryNeed)
        (fun _ -> uint64 outputWidth * uint64 outputHeight)
    |> Stage.withSliceCardinality (SliceCardinality.reduceTo (uint64 outputDepth))

let chunkResize<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    outputWidth
    outputHeight
    outputDepth
    interpolationName
    (pl: Plan<unit, Chunk<'T>>)
    : Plan<unit, Chunk<'T>> =
    let outputWidth = validateResizeOutput "outputWidth" outputWidth
    let outputHeight = validateResizeOutput "outputHeight" outputHeight
    let outputDepth = validateResizeOutput "outputDepth" outputDepth
    let inputWidth = trySourcePeekUInt "width" pl
    let inputHeight = trySourcePeekUInt "height" pl
    let inputDepth =
        trySourcePeekUInt "depth" pl
        |> Option.defaultValue (uint pl.length)

    let spacingX = inputWidth |> Option.map (fun width -> outputSpacingForSize width outputWidth) |> Option.defaultValue 1.0
    let spacingY = inputHeight |> Option.map (fun height -> outputSpacingForSize height outputHeight) |> Option.defaultValue 1.0
    let spacingZ = outputSpacingForSize inputDepth outputDepth

    pl >=> resize3DNativeStage<'T> outputWidth outputHeight outputDepth spacingX spacingY spacingZ interpolationName

let chunkResample<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    factorX
    factorY
    factorZ
    interpolationName
    (pl: Plan<unit, Chunk<'T>>)
    : Plan<unit, Chunk<'T>> =
    if factorX <= 0.0 || factorY <= 0.0 || factorZ <= 0.0 then
        invalidArg "factor" "chunkResample factors must be positive."

    let inputWidth = trySourcePeekUInt "width" pl
    let inputHeight = trySourcePeekUInt "height" pl
    let inputDepth =
        trySourcePeekUInt "depth" pl
        |> Option.defaultValue (uint pl.length)

    let outputWidth =
        inputWidth
        |> Option.map (fun width -> roundPositiveToUInt (float width * factorX))
        |> Option.defaultValue (roundPositiveToUInt (Math.Sqrt(float (SingleOrPair.fst pl.nElemsPerSlice)) * factorX))
    let outputHeight =
        inputHeight
        |> Option.map (fun height -> roundPositiveToUInt (float height * factorY))
        |> Option.defaultValue (roundPositiveToUInt (Math.Sqrt(float (SingleOrPair.fst pl.nElemsPerSlice)) * factorY))
    let outputDepth = roundPositiveToUInt (float inputDepth * factorZ)

    pl >=> resize3DNativeStage<'T> outputWidth outputHeight outputDepth (1.0 / factorX) (1.0 / factorY) (1.0 / factorZ) interpolationName

let euler2DTransformNative<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    rotation
    translation
    : Stage<Chunk<'T>, Chunk<'T>> =
    releaseUnaryChunk
        $"chunkEuler2DTransformNative.{typeof<'T>.Name}"
        (ChunkKernel.euler2DTransformNativeChunk<'T> rotation translation)
        (fun n -> 2UL * chunkMemoryNeed<'T> n)

let euler2DRotateNative<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    center
    angle
    : Stage<Chunk<'T>, Chunk<'T>> =
    releaseUnaryChunk
        $"chunkEuler2DRotateNative.{typeof<'T>.Name}"
        (ChunkKernel.euler2DRotateNativeChunk<'T> center angle)
        (fun n -> 2UL * chunkMemoryNeed<'T> n)

let show<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (plt: Chunk<'T> -> unit)
    : Stage<Chunk<'T>, unit> =
    let consumer (debug: bool) (idx: int) (chunk: Chunk<'T>) =
        try
            if debug && DebugLevel.current() >= 2u then printfn "[chunkShow] Showing chunk %d" idx
            plt chunk
        finally
            Chunk.decRef chunk

    Stage.consumeWith "chunkShow" consumer id

let private zeroUInt8ChunkLike _index (source: Chunk<uint8>) =
    let width, height, depth = source.Size
    if depth <> 1UL then
        invalidArg "source" $"Chunk signed distance band expects 2D slice chunks with depth 1, got {source.Size}."
    let chunk = Chunk.create<uint8> (width, height, 1UL)
    chunk.Bytes.AsSpan(0, chunk.ByteLength).Clear()
    chunk

let private releaseConsumedUInt8Window (window: Window<Chunk<uint8>>) =
    let _emitStart, emitCount = window.EmitRange
    if emitCount = 0u then
        window.Items |> List.iter Chunk.decRef
    else
        window.Items
        |> List.truncate (int window.ReleaseCount)
        |> List.iter Chunk.decRef

let signedDistanceBandNativeParallelCollect workers bandRadius stride : Stage<Chunk<uint8>, Chunk<float32>> =
    if bandRadius = 0u then
        invalidArg "bandRadius" "Chunk signed distance band requires a positive band radius."
    if stride = 0u then
        invalidArg "stride" "Chunk signed distance band requires a positive stride."
    if workers < 1 then
        invalidArg "workers" $"Chunk signed distance band expects at least one worker, got {workers}."
    let winSz = stride + 2u * bandRadius
    if winSz > uint Int32.MaxValue || stride > uint Int32.MaxValue || bandRadius > uint Int32.MaxValue then
        invalidArg "bandRadius" $"Chunk signed distance band window parameters must fit in Int32, got bandRadius={bandRadius}, stride={stride}."

    let mapper _debug (window: Window<Chunk<uint8>>) =
        let items = window.Items |> List.toArray
        let outputStart, outputCount = window.EmitRange
        try
            if outputCount = 0u then
                []
            else
                ChunkKernel.signedDistanceBandNativeUInt8 bandRadius (int outputStart) (int outputCount) items
        finally
            releaseConsumedUInt8Window window

    Stage.parallelCollect
        $"chunkSignedDistanceBandNative.parallelCollect.UInt8.band{bandRadius}.stride{stride}.workers{workers}"
        (int winSz)
        workers
        (int stride)
        (int bandRadius)
        zeroUInt8ChunkLike
        mapper
        (fun n -> uint64 winSz * n * uint64 (chunkElementBytes<uint8> + chunkElementBytes<float32>))
        id

let connectedComponentsSauf3DUInt8UInt32ArrayUf () = StackConnectedComponents.connectedComponentsSauf3DUInt8UInt32ArrayUf ()
let connectedComponentsSauf3DUInt8UInt32 () = StackConnectedComponents.connectedComponentsSauf3DUInt8UInt32 ()
let connectedComponentsSauf3DUInt8UInt32ParallelCollect workers windowSize = StackConnectedComponents.connectedComponentsSauf3DUInt8UInt32ParallelCollect windowSize workers
let connectedComponentsSauf3DUInt8 () = StackConnectedComponents.connectedComponentsSauf3DUInt8 ()

let fftXYFloat32ToComplex64Interleaved = StackFFT.fftXYFloat32ToComplex64Interleaved
let fftRealXYFloat32ToHermitianPackedComplex64Interleaved = StackFFT.fftRealXYFloat32ToHermitianPackedComplex64Interleaved
let fftXYFloat32ToComplex64InterleavedParallelCollect workers = StackFFT.fftXYFloat32ToComplex64InterleavedParallelCollect workers
let fftXYThenZFloat32ToComplex64InterleavedPlanned windowLength = StackFFT.fftXYThenZFloat32ToComplex64InterleavedPlanned windowLength
let fft3DFloat32ToComplex64Interleaved windowLength = StackFFT.fft3DFloat32ToComplex64Interleaved windowLength
let fft3DRealXYFloat32ToComplex64Interleaved windowLength = StackFFT.fft3DRealXYFloat32ToComplex64Interleaved windowLength
let invFft3DRealXYComplex64InterleavedToFloat32 windowLength = StackFFT.invFft3DRealXYComplex64InterleavedToFloat32 windowLength
let invFftXYComplex64InterleavedToFloat32 = StackFFT.invFftXYComplex64InterleavedToFloat32
let invFftXYHermitianPackedComplex64InterleavedToFloat32 = StackFFT.invFftXYHermitianPackedComplex64InterleavedToFloat32
let invFftXYComplex64InterleavedToFloat32ParallelCollect workers = StackFFT.invFftXYComplex64InterleavedToFloat32ParallelCollect workers
let fftShift3DComplex64Interleaved = StackFFT.fftShift3DComplex64Interleaved
let toComplex64 = StackFFT.toComplex64
let polarToComplex64 = StackFFT.polarToComplex64
let toComplex128 = StackFFT.toComplex128
let polarToComplex128 = StackFFT.polarToComplex128
let complex64Real = StackFFT.complex64Real
let complex64Imag = StackFFT.complex64Imag
let complex64Modulus = StackFFT.complex64Modulus
let complex64Argument = StackFFT.complex64Argument
let complex128Real = StackFFT.complex128Real
let complex128Imag = StackFFT.complex128Imag
let complex128Modulus = StackFFT.complex128Modulus
let complex128Argument = StackFFT.complex128Argument
let complex64Conjugate = StackFFT.complex64Conjugate
let complex128Conjugate = StackFFT.complex128Conjugate

let toVectorImage<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    : Stage<Chunk<'T> * Chunk<'T>, VectorChunk<'T>> =
    let mapper (a: Chunk<'T>) (b: Chunk<'T>) =
        try
            Chunk.toVectorImage [ a; b ]
        finally
            Chunk.decRef a
            Chunk.decRef b

    Stage.map $"chunkToVectorImage.{typeof<'T>.Name}" (fun _ (a, b) -> mapper a b) (fun n -> 4UL * chunkMemoryNeed<'T> n) id

let vectorElement<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (componentId: uint)
    : Stage<VectorChunk<'T>, Chunk<'T>> =
    releaseUnaryVectorChunk
        $"chunkVectorElement.{typeof<'T>.Name}.{componentId}"
        (Chunk.vectorElement<'T> componentId)
        (fun n -> 2UL * chunkMemoryNeed<'T> n)

let vectorRange<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (firstComponent: uint)
    (componentCount: uint)
    : Stage<VectorChunk<'T>, VectorChunk<'T>> =
    let stageName = $"chunkVectorRange.{typeof<'T>.Name}.{firstComponent}.{componentCount}"
    releaseUnaryVectorToVectorChunk stageName (Chunk.vectorRange<'T> firstComponent componentCount) (fun n -> 2UL * chunkMemoryNeed<'T> n)

let appendVectorElement<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    : Stage<VectorChunk<'T> * Chunk<'T>, VectorChunk<'T>> =
    let mapper _debug ((vector, element): VectorChunk<'T> * Chunk<'T>) =
        try
            Chunk.appendVectorElement vector element
        finally
            Chunk.decRefVector vector
            Chunk.decRef element

    Stage.map $"chunkAppendVectorElement.{typeof<'T>.Name}" mapper (fun n -> 3UL * chunkMemoryNeed<'T> n) id

let private vectorElementFunction functionName =
    match functionName with
    | "sqrt" -> sqrt
    | "square" -> fun x -> x * x
    | "abs" -> abs
    | "log" -> log
    | "exp" -> exp
    | other -> invalidArg "functionName" $"Unsupported vector element function '{other}'."

let vectorMapElements functionName : Stage<VectorChunk<float>, VectorChunk<float>> =
    let f = vectorElementFunction functionName
    releaseUnaryVectorToVectorChunk
        $"chunkVectorMapElements.{functionName}"
        (Chunk.mapVectorElements f)
        (fun n -> 2UL * chunkMemoryNeed<float> n)

let intensityStretchVector<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    inputMinimum
    inputMaximum
    outputMinimum
    outputMaximum
    : Stage<VectorChunk<'T>, VectorChunk<'T>> =
    releaseUnaryVectorToVectorChunk
        $"chunkIntensityStretchVector.{typeof<'T>.Name}.{inputMinimum}.{inputMaximum}.{outputMinimum}.{outputMaximum}"
        (Chunk.intensityStretchVector<'T> inputMinimum inputMaximum outputMinimum outputMaximum)
        (fun n -> 2UL * chunkMemoryNeed<'T> n)

let colorToVector3 outputMinimum outputMaximum : Stage<VectorChunk<uint8>, VectorChunk<float>> =
    releaseUnaryVectorToVectorChunk "chunkColorToVector3" (Chunk.colorToVector3 outputMinimum outputMaximum) (fun n -> n * uint64 (chunkElementBytes<uint8> + chunkElementBytes<float>))

let vectorDot : Stage<VectorChunk<float> * VectorChunk<float>, Chunk<float>> =
    releaseBinaryVectorChunk
        "chunkVectorDot"
        Chunk.vectorDot
        (fun n -> 3UL * chunkMemoryNeed<float> n)

let vectorMagnitude : Stage<VectorChunk<float>, Chunk<float>> =
    releaseUnaryVectorChunk
        "chunkVectorMagnitude"
        Chunk.vectorMagnitude
        (fun n -> 2UL * chunkMemoryNeed<float> n)

let vectorCross3D : Stage<VectorChunk<float> * VectorChunk<float>, VectorChunk<float>> =
    let mapper _debug ((a, b): VectorChunk<float> * VectorChunk<float>) =
        try
            Chunk.vectorCross3D a b
        finally
            Chunk.decRefVector a
            Chunk.decRefVector b

    Stage.map "chunkVectorCross3D" mapper (fun n -> 3UL * chunkMemoryNeed<float> n) id

let vectorAngleTo reference : Stage<VectorChunk<float>, Chunk<float>> =
    releaseUnaryVectorChunk
        "chunkVectorAngleTo"
        (Chunk.vectorAngleTo reference)
        (fun n -> 2UL * chunkMemoryNeed<float> n)

let PCA components : Stage<VectorChunk<float>, VectorChunk<float>> =
    if components < 2u then invalidArg "components" "Chunk PCA needs at least two vector components."
    let componentsI = int components

    let outputVector (values: float list) : VectorChunk<float> =
        let valuesArray = values |> List.toArray
        let chunks =
            valuesArray
            |> Array.map (fun value ->
                let chunk = Chunk.create<float> (1UL, 1UL, 1UL)
                let pixels = Chunk.span chunk
                pixels[0] <- value
                chunk)
        { SpatialSize = (1UL, 1UL, 1UL)
          Components = chunks }

    let reducer (_debug: bool) (input: AsyncSeq<VectorChunk<float>>) =
        async {
            let! state =
                input
                |> AsyncSeq.foldAsync
                    (fun state vector ->
                        async {
                            try
                                if Chunk.vectorComponentCount vector <> uint32 componentsI then
                                    invalidArg "vector" $"Chunk PCA expected {componentsI}-component vector chunks, got {Chunk.vectorComponentCount vector} components."

                                let spatialCount =
                                    let first = Chunk.span<float> vector.Components[0]
                                    first.Length
                                let mutable state = state

                                for i in 0 .. spatialCount - 1 do
                                    let values =
                                        [ for c in 0 .. componentsI - 1 ->
                                            let pixels = Chunk.span<float> vector.Components[c]
                                            pixels[i] ]
                                    state <- addPcaVector state values

                                return state
                            finally
                                Chunk.decRefVector vector
                        })
                    (zeroPcaAccumulator componentsI)

            let eigen = pcaEigenSystem state
            let eigenvalues = eigen |> List.map fst
            return
                [ yield outputVector eigenvalues
                  for _, vector in eigen do
                      yield outputVector vector ]
        }

    Stage.reduce "chunkPCA" reducer Streaming (fun _ -> uint64 ((componentsI + 1) * componentsI * sizeof<float>)) (fun _ -> uint64 (componentsI + 1))
    --> flattenList ()

let selectGroupedVectorOutput<'T when 'T: equality>
    (groupSize: uint)
    (part: uint)
    : Stage<VectorChunk<'T>, VectorChunk<'T>> =
    if groupSize = 0u then
        invalidArg "groupSize" "chunkSelectGroupedVectorOutput: groupSize must be positive."
    if part >= groupSize then
        invalidArg "part" $"chunkSelectGroupedVectorOutput: part must be smaller than groupSize ({groupSize})."

    Stage.mapi
        "chunkSelectGroupedVectorOutput"
        (fun _ index vector ->
            if uint (index % int64 groupSize) = part then
                [ vector ]
            else
                Chunk.decRefVector vector
                [])
        id
        (fun slices -> (slices + uint64 groupSize - 1UL) / uint64 groupSize)
    --> flattenList ()

let covarianceMatrix : Stage<VectorChunk<float32>, VectorizedMatrix> =
    let reducer (_debug: bool) (input: AsyncSeq<VectorChunk<float32>>) =
        async {
            let mutable count = 0UL
            let mutable sx = 0.0
            let mutable sy = 0.0
            let mutable sz = 0.0
            let mutable sxx = 0.0
            let mutable sxy = 0.0
            let mutable sxz = 0.0
            let mutable syy = 0.0
            let mutable syz = 0.0
            let mutable szz = 0.0

            let accumulate (vector: VectorChunk<float32>) =
                if Chunk.vectorComponentCount vector <> 3u then
                    invalidArg "vector" $"chunkCovarianceMatrix expected 3-component vector chunks, got {Chunk.vectorComponentCount vector}."

                let xPixels = Chunk.span<float32> vector.Components[0]
                let yPixels = Chunk.span<float32> vector.Components[1]
                let zPixels = Chunk.span<float32> vector.Components[2]
                let spatialCount = xPixels.Length

                if yPixels.Length <> spatialCount || zPixels.Length <> spatialCount then
                    invalidArg "vector" "chunkCovarianceMatrix expected all vector components to have the same spatial size."

                let mutable i = 0
                while i < spatialCount do
                    let x = float xPixels[i]
                    let y = float yPixels[i]
                    let z = float zPixels[i]
                    sx <- sx + x
                    sy <- sy + y
                    sz <- sz + z
                    sxx <- sxx + x * x
                    sxy <- sxy + x * y
                    sxz <- sxz + x * z
                    syy <- syy + y * y
                    syz <- syz + y * z
                    szz <- szz + z * z
                    i <- i + 1

                count <- count + uint64 spatialCount

            do!
                input
                |> AsyncSeq.iterAsync (fun vector ->
                    async {
                        try
                            accumulate vector
                        finally
                            Chunk.decRefVector vector
                    })

            if count = 0UL then
                invalidOp "chunkCovarianceMatrix cannot reduce an empty vector sequence."

            let n = float count
            let mx = sx / n
            let my = sy / n
            let mz = sz / n
            let xx = sxx / n - mx * mx
            let xy = sxy / n - mx * my
            let xz = sxz / n - mx * mz
            let yy = syy / n - my * my
            let yz = syz / n - my * mz
            let zz = szz / n - mz * mz

            return
                [ { Rows = 3u
                    Columns = 3u
                    Values =
                        [ xx; xy; xz
                          xy; yy; yz
                          xz; yz; zz ] } ]
        }

    Stage.reduce
        "chunkCovarianceMatrix"
        reducer
        Streaming
        (fun _ -> 6UL * chunkMemoryNeed<float32> 1UL)
        (fun _ -> 1UL)
    --> flattenList ()

let symmetricMatrixEigenbasis (matrix: VectorizedMatrix) : VectorizedMatrix =
    if matrix.Rows <> matrix.Columns then
        invalidArg "matrix" $"symmetricMatrixEigenbasis expects a square matrix, got {matrix.Rows}x{matrix.Columns}."
    let n = int matrix.Rows
    if n < 2 then
        invalidArg "matrix" $"symmetricMatrixEigenbasis expects at least two dimensions, got {n}."
    if matrix.Values.Length <> n * n then
        invalidArg "matrix" $"symmetricMatrixEigenbasis expected {n * n} values, got {matrix.Values.Length}."

    let values = matrix.Values |> List.toArray
    let dense = Array2D.zeroCreate<float> n n
    for row in 0 .. n - 1 do
        for column in 0 .. n - 1 do
            dense[row, column] <- values[row * n + column]

    let eigen = symmetricEigenN dense
    { Rows = uint eigen.Length
      Columns = matrix.Columns
      Values =
        [ for _, vector in eigen do
            yield! vector ] }

let symmetricMatrixEigenvalues (matrix: VectorizedMatrix) : float list =
    if matrix.Rows <> matrix.Columns then
        invalidArg "matrix" $"symmetricMatrixEigenvalues expects a square matrix, got {matrix.Rows}x{matrix.Columns}."
    let n = int matrix.Rows
    if matrix.Values.Length <> n * n then
        invalidArg "matrix" $"symmetricMatrixEigenvalues expected {n * n} values, got {matrix.Values.Length}."

    let values = matrix.Values |> List.toArray
    let dense = Array2D.zeroCreate<float> n n
    for row in 0 .. n - 1 do
        for column in 0 .. n - 1 do
            dense[row, column] <- values[row * n + column]

    symmetricEigenN dense |> List.map fst

let projectVectorBasisFloat32 (basis: VectorizedMatrix) : Stage<VectorChunk<float32>, VectorChunk<float32>> =
    let rows = int basis.Rows
    let columns = int basis.Columns
    if rows <> 3 || columns <> 3 then
        invalidArg "basis" $"projectVectorBasisFloat32 currently expects a 3x3 basis matrix, got {basis.Rows}x{basis.Columns}."
    if basis.Values.Length <> 9 then
        invalidArg "basis" $"projectVectorBasisFloat32 expected 9 basis values, got {basis.Values.Length}."
    let weights = basis.Values |> List.map float32 |> List.toArray

    let mapper (vector: VectorChunk<float32>) =
        let components = int (Chunk.vectorComponentCount vector)
        if components <> 3 then
            invalidArg "vector" $"projectVectorBasisFloat32 expected 3-component vector chunks, got {components}."

        let xPixels = Chunk.span<float32> vector.Components[0]
        let yPixels = Chunk.span<float32> vector.Components[1]
        let zPixels = Chunk.span<float32> vector.Components[2]
        let spatialCount = xPixels.Length
        if yPixels.Length <> spatialCount || zPixels.Length <> spatialCount then
            invalidArg "vector" "projectVectorBasisFloat32 expected all vector components to have the same spatial size."

        let output = Array.init 3 (fun _ -> Chunk.create<float32> vector.SpatialSize)
        try
            let out0 = Chunk.span<float32> output[0]
            let out1 = Chunk.span<float32> output[1]
            let out2 = Chunk.span<float32> output[2]
            let mutable i = 0
            while i < spatialCount do
                let x = xPixels[i]
                let y = yPixels[i]
                let z = zPixels[i]
                out0[i] <- weights[0] * x + weights[1] * y + weights[2] * z
                out1[i] <- weights[3] * x + weights[4] * y + weights[5] * z
                out2[i] <- weights[6] * x + weights[7] * y + weights[8] * z
                i <- i + 1

            let outputVector: VectorChunk<float32> =
                { SpatialSize = vector.SpatialSize
                  Components = output }
            outputVector
        with
        | _ ->
            output |> Array.iter Chunk.decRef
            reraise()

    releaseUnaryVectorToVectorChunk
        "chunkProjectVectorBasisFloat32"
        mapper
        (fun n -> 6UL * chunkMemoryNeed<float32> n)

let vectorDotFloat32 : Stage<VectorChunk<float32> * VectorChunk<float32>, Chunk<float32>> =
    releaseBinaryVectorChunk
        "chunkVectorDotFloat32"
        Chunk.vectorDotFloat32
        (fun n -> 3UL * chunkMemoryNeed<float32> n)

let vectorMagnitudeFloat32 : Stage<VectorChunk<float32>, Chunk<float32>> =
    releaseUnaryVectorChunk
        "chunkVectorMagnitudeFloat32"
        Chunk.vectorMagnitudeFloat32
        (fun n -> 2UL * chunkMemoryNeed<float32> n)

let vectorAngleToFloat32 reference : Stage<VectorChunk<float32>, Chunk<float32>> =
    releaseUnaryVectorChunk
        "chunkVectorAngleToFloat32"
        (Chunk.vectorAngleToFloat32 reference)
        (fun n -> 2UL * chunkMemoryNeed<float32> n)

let PCAFloat32 components : Stage<VectorChunk<float32>, VectorChunk<float32>> =
    if components < 2u then invalidArg "components" "Chunk PCA needs at least two vector components."
    let componentsI = int components

    let outputVector (values: float list) : VectorChunk<float32> =
        let valuesArray = values |> List.toArray
        let chunks =
            valuesArray
            |> Array.map (fun value ->
                let chunk = Chunk.create<float32> (1UL, 1UL, 1UL)
                let pixels = Chunk.span chunk
                pixels[0] <- float32 value
                chunk)
        { SpatialSize = (1UL, 1UL, 1UL)
          Components = chunks }

    let reducer (_debug: bool) (input: AsyncSeq<VectorChunk<float32>>) =
        async {
            let! state =
                input
                |> AsyncSeq.foldAsync
                    (fun state vector ->
                        async {
                            try
                                if Chunk.vectorComponentCount vector <> uint32 componentsI then
                                    invalidArg "vector" $"Chunk PCA expected {componentsI}-component vector chunks, got {Chunk.vectorComponentCount vector} components."

                                let spatialCount =
                                    let first = Chunk.span<float32> vector.Components[0]
                                    first.Length
                                let mutable state = state

                                for i in 0 .. spatialCount - 1 do
                                    let values =
                                        [ for c in 0 .. componentsI - 1 ->
                                            let pixels = Chunk.span<float32> vector.Components[c]
                                            float pixels[i] ]
                                    state <- addPcaVector state values

                                return state
                            finally
                                Chunk.decRefVector vector
                        })
                    (zeroPcaAccumulator componentsI)

            let eigen = pcaEigenSystem state
            let eigenvalues = eigen |> List.map fst
            return
                [ yield outputVector eigenvalues
                  for _, vector in eigen do
                      yield outputVector vector ]
        }

    Stage.reduce "chunkPCAFloat32" reducer Streaming (fun _ -> uint64 ((componentsI + 1) * componentsI * sizeof<float32>)) (fun _ -> uint64 (componentsI + 1))
    --> flattenList ()

let private structureTensorOuterProductFloat32 : Stage<VectorChunk<float32>, VectorChunk<float32>> =
    let mapper (vector: VectorChunk<float32>) =
        if Chunk.vectorComponentCount vector <> 3u then
            invalidArg "vector" $"chunkStructureTensorOuterProductFloat32 expected a 3-component gradient, got {Chunk.vectorComponentCount vector}."

        let output = Array.init 6 (fun _ -> Chunk.create<float32> vector.SpatialSize)
        try
            let outputVector: VectorChunk<float32> =
                { SpatialSize = vector.SpatialSize
                  Components = output }
            let gxPixels = Chunk.span<float32> vector.Components[0]
            let gyPixels = Chunk.span<float32> vector.Components[1]
            let gzPixels = Chunk.span<float32> vector.Components[2]
            let xxPixels = Chunk.span<float32> output[0]
            let xyPixels = Chunk.span<float32> output[1]
            let xzPixels = Chunk.span<float32> output[2]
            let yyPixels = Chunk.span<float32> output[3]
            let yzPixels = Chunk.span<float32> output[4]
            let zzPixels = Chunk.span<float32> output[5]
            let spatialCount = gxPixels.Length
            let width = Vector<float32>.Count
            let vectorEnd = spatialCount - spatialCount % width
            let mutable i = 0
            while i < vectorEnd do
                let gx = Vector<float32>(gxPixels.Slice(i, width))
                let gy = Vector<float32>(gyPixels.Slice(i, width))
                let gz = Vector<float32>(gzPixels.Slice(i, width))
                (gx * gx).CopyTo(xxPixels.Slice(i, width))
                (gx * gy).CopyTo(xyPixels.Slice(i, width))
                (gx * gz).CopyTo(xzPixels.Slice(i, width))
                (gy * gy).CopyTo(yyPixels.Slice(i, width))
                (gy * gz).CopyTo(yzPixels.Slice(i, width))
                (gz * gz).CopyTo(zzPixels.Slice(i, width))
                i <- i + width
            while i < spatialCount do
                let gx = gxPixels[i]
                let gy = gyPixels[i]
                let gz = gzPixels[i]
                xxPixels[i] <- gx * gx
                xyPixels[i] <- gx * gy
                xzPixels[i] <- gx * gz
                yyPixels[i] <- gy * gy
                yzPixels[i] <- gy * gz
                zzPixels[i] <- gz * gz
                i <- i + 1
            outputVector
        with
        | _ ->
            output |> Array.iter Chunk.decRef
            reraise()

    releaseUnaryVectorToVectorChunk
        "chunkStructureTensorOuterProductFloat32"
        mapper
        (fun n -> n * uint64 (chunkElementBytes<float32> * (3 + 6)))

let private zeroVectorFloat32Like (_index: int) (source: VectorChunk<float32>) : VectorChunk<float32> =
    let width, height, depth = source.SpatialSize
    if depth <> 1UL then
        invalidArg "source" $"Chunk vector convolution stages expect 2D slice vector chunks with depth 1, got {source.SpatialSize}."

    let components =
        Array.init
            (int (Chunk.vectorComponentCount source))
            (fun _ ->
                let chunk = Chunk.create<float32> (width, height, 1UL)
                chunk.Bytes.AsSpan(0, chunk.ByteLength).Clear()
                chunk)
    { SpatialSize = source.SpatialSize
      Components = components }

let private releaseConsumedVectorWindow (window: Window<VectorChunk<float32>>) =
    let _emitStart, emitCount = window.EmitRange
    let releaseCount =
        if emitCount = 0u then
            window.Items.Length
        else
            min (int window.ReleaseCount) window.Items.Length

    window.Items
    |> List.truncate releaseCount
    |> List.iter Chunk.decRefVector

let private vectorComponentMemoryNeed componentBudget liveChunks n =
    uint64 liveChunks * n * uint64 (chunkElementBytes<float32> * componentBudget)

let private convolveVectorComponentsSingleSliceFloat32
    axisName
    (kernel: float32[])
    workers
    (convolve: float32[] -> VectorChunk<float32> -> VectorChunk<float32>)
    : Stage<VectorChunk<float32>, VectorChunk<float32>> =
    if isNull kernel then
        nullArg "kernel"
    if kernel.Length = 0 || kernel.Length % 2 = 0 then
        invalidArg "kernel" $"chunkConvolveVectorComponents{axisName}Float32 expects a non-empty odd-length kernel, got {kernel.Length}."
    if workers < 1 then
        invalidArg "workers" $"chunkConvolveVectorComponents{axisName}Float32 expects at least one worker, got {workers}."

    let mapper _debug (window: Window<VectorChunk<float32>>) =
        match window.Items with
        | [ vector ] ->
            try
                [ convolve kernel vector ]
            finally
                Chunk.decRefVector vector
        | items ->
            items |> List.iter Chunk.decRefVector
            invalidArg "window" $"chunkConvolveVectorComponents{axisName}Float32 expects singleton windows, got {items.Length}."

    Stage.parallelCollect
        $"chunkConvolveVectorComponents{axisName}Float32.k{kernel.Length}.workers{workers}"
        1
        workers
        1
        0
        zeroVectorFloat32Like
        mapper
        (vectorComponentMemoryNeed 6 (2 * workers))
        id

let private convolveVectorComponentsXFloat32 (kernel: float32[]) workers : Stage<VectorChunk<float32>, VectorChunk<float32>> =
    convolveVectorComponentsSingleSliceFloat32
        "X"
        kernel
        workers
        ChunkKernel.convolveVectorComponentsNativeXFloat32

let private convolveVectorComponentsYFloat32 (kernel: float32[]) workers : Stage<VectorChunk<float32>, VectorChunk<float32>> =
    convolveVectorComponentsSingleSliceFloat32
        "Y"
        kernel
        workers
        ChunkKernel.convolveVectorComponentsNativeYFloat32

let private convolveVectorComponentsZChunkFloat32 (kernel: float32[]) (items: VectorChunk<float32>[]) =
    if items.Length <> kernel.Length then
        invalidArg "items" $"chunkConvolveVectorComponentsZFloat32 expects {kernel.Length} slices, got {items.Length}."
    ChunkKernel.convolveVectorComponentsNativeZFloat32 kernel items

let private convolveVectorComponentsZFloat32 (kernel: float32[]) (workers: int) : Stage<VectorChunk<float32>, VectorChunk<float32>> =
    if isNull kernel then
        nullArg "kernel"
    if kernel.Length = 0 || kernel.Length % 2 = 0 then
        invalidArg "kernel" $"chunkConvolveVectorComponentsZFloat32 expects a non-empty odd-length kernel, got {kernel.Length}."
    let radius = kernel.Length / 2
    if workers < 1 then
        invalidArg "workers" $"chunkConvolveVectorComponentsZFloat32 expects at least one worker, got {workers}."

    let mapper _debug (window: Window<VectorChunk<float32>>) =
        let items = window.Items |> List.toArray
        let _emitStart, emitCount = window.EmitRange
        try
            if emitCount = 0u then
                []
            else
                [ convolveVectorComponentsZChunkFloat32 kernel items ]
        finally
            releaseConsumedVectorWindow window

    Stage.parallelCollect
        $"chunkConvolveVectorComponentsZFloat32.k{kernel.Length}.workers{workers}"
        kernel.Length
        workers
        1
        radius
        zeroVectorFloat32Like
        mapper
        (vectorComponentMemoryNeed 6 (kernel.Length + 2 * workers - 1))
        id

let convolveVectorComponentsFloat32NativeParallelCollect (xKernel: float32[]) (yKernel: float32[]) (zKernel: float32[]) (workers: int) : Stage<VectorChunk<float32>, VectorChunk<float32>> =
    convolveVectorComponentsXFloat32 xKernel workers
    --> convolveVectorComponentsYFloat32 yKernel workers
    --> convolveVectorComponentsZFloat32 zKernel workers

let private gaussianSmoothVectorComponentsFloat32 (sigma: float) (radius: int) (workers: int) =
    if sigma <= 0.0 then
        Stage.map "chunkGaussianSmoothVectorComponentsFloat32.identity" (fun _ vector -> vector) id id
    else
        let radius =
            if radius > 0 then
                radius
            else
                StackConvolve.defaultGaussianRadius sigma
        let kernel = StackConvolve.gaussianKernel sigma radius
        convolveVectorComponentsFloat32NativeParallelCollect kernel kernel kernel workers

let private sortValuesDescending3 a b c =
    let mutable x = a
    let mutable y = b
    let mutable z = c
    if x < y then
        let t = x
        x <- y
        y <- t
    if y < z then
        let t = y
        y <- z
        z <- t
    if x < y then
        let t = x
        x <- y
        y <- t
    struct (x, y, z)

let inline private normalize3 x y z =
    let n = sqrt (x * x + y * y + z * z)
    if n <= 1e-18 then
        struct (1.0, 0.0, 0.0)
    else
        let inv = 1.0 / n
        let mutable x = x * inv
        let mutable y = y * inv
        let mutable z = z * inv
        let ax = abs x
        let ay = abs y
        let az = abs z
        let sign =
            if ax >= ay && ax >= az then x
            elif ay >= az then y
            else z
        if sign < 0.0 then
            x <- -x
            y <- -y
            z <- -z
        struct (x, y, z)

let inline private cross3 ax ay az bx by bz =
    struct (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)

let inline private trySymmetricEigenvector3 xx xy xz yy yz zz lambda tolerance =
    let r00 = xx - lambda
    let r01 = xy
    let r02 = xz
    let r10 = xy
    let r11 = yy - lambda
    let r12 = yz
    let r20 = xz
    let r21 = yz
    let r22 = zz - lambda
    let struct (c01x, c01y, c01z) = cross3 r00 r01 r02 r10 r11 r12
    let struct (c02x, c02y, c02z) = cross3 r00 r01 r02 r20 r21 r22
    let struct (c12x, c12y, c12z) = cross3 r10 r11 r12 r20 r21 r22
    let n01 = c01x * c01x + c01y * c01y + c01z * c01z
    let n02 = c02x * c02x + c02y * c02y + c02z * c02z
    let n12 = c12x * c12x + c12y * c12y + c12z * c12z
    let struct (cx, cy, cz, n) =
        if n01 >= n02 && n01 >= n12 then struct (c01x, c01y, c01z, n01)
        elif n02 >= n12 then struct (c02x, c02y, c02z, n02)
        else struct (c12x, c12y, c12z, n12)
    if n <= tolerance * tolerance then
        struct (false, 1.0, 0.0, 0.0)
    else
        let struct (x, y, z) = normalize3 cx cy cz
        struct (true, x, y, z)

let inline private orthonormalComplement3 ax ay az =
    let aax = abs ax
    let aay = abs ay
    let aaz = abs az
    let struct (rx, ry, rz) =
        if aax <= aay && aax <= aaz then struct (1.0, 0.0, 0.0)
        elif aay <= aaz then struct (0.0, 1.0, 0.0)
        else struct (0.0, 0.0, 1.0)
    let struct (u0, u1, u2) = cross3 ax ay az rx ry rz
    let struct (u0, u1, u2) = normalize3 u0 u1 u2
    let struct (v0, v1, v2) = cross3 ax ay az u0 u1 u2
    let struct (v0, v1, v2) = normalize3 v0 v1 v2
    struct (u0, u1, u2, v0, v1, v2)

let inline private symmetricEigenvalues3Raw xx xy xz yy yz zz =
    let p1 = xy * xy + xz * xz + yz * yz
    if p1 = 0.0 then
        sortValuesDescending3 xx yy zz
    else
        let q = (xx + yy + zz) / 3.0
        let axx = xx - q
        let ayy = yy - q
        let azz = zz - q
        let p2 = axx * axx + ayy * ayy + azz * azz + 2.0 * p1
        let p = sqrt (p2 / 6.0)
        let bxx = axx / p
        let bxy = xy / p
        let bxz = xz / p
        let byy = ayy / p
        let byz = yz / p
        let bzz = azz / p
        let detB =
            bxx * (byy * bzz - byz * byz)
            - bxy * (bxy * bzz - byz * bxz)
            + bxz * (bxy * byz - byy * bxz)
        let r = detB / 2.0
        let phi =
            if r <= -1.0 then Math.PI / 3.0
            elif r >= 1.0 then 0.0
            else Math.Acos(r) / 3.0
        let e0 = q + 2.0 * p * Math.Cos(phi)
        let e2 = q + 2.0 * p * Math.Cos(phi + 2.0 * Math.PI / 3.0)
        let e1 = 3.0 * q - e0 - e2
        sortValuesDescending3 e0 e1 e2

let inline private symmetricEigenvectors3Raw xx xy xz yy yz zz e0 e1 e2 scale =
    let gap = 1e-10 * scale
    let vectorTol = 1e-12 * scale
    let mutable x0 = 1.0
    let mutable y0 = 0.0
    let mutable z0 = 0.0
    let mutable x1 = 0.0
    let mutable y1 = 1.0
    let mutable z1 = 0.0
    let mutable x2 = 0.0
    let mutable y2 = 0.0
    let mutable z2 = 1.0
    if abs (e0 - e1) <= gap && abs (e1 - e2) <= gap then
        ()
    elif abs (e0 - e1) <= gap then
        let struct (ok, tx, ty, tz) = trySymmetricEigenvector3 xx xy xz yy yz zz e2 vectorTol
        if ok then
            x2 <- tx; y2 <- ty; z2 <- tz
            let struct (u0, u1, u2, w0, w1, w2) = orthonormalComplement3 x2 y2 z2
            x0 <- u0; y0 <- u1; z0 <- u2
            x1 <- w0; y1 <- w1; z1 <- w2
    elif abs (e1 - e2) <= gap then
        let struct (ok, tx, ty, tz) = trySymmetricEigenvector3 xx xy xz yy yz zz e0 vectorTol
        if ok then
            x0 <- tx; y0 <- ty; z0 <- tz
            let struct (u0, u1, u2, w0, w1, w2) = orthonormalComplement3 x0 y0 z0
            x1 <- u0; y1 <- u1; z1 <- u2
            x2 <- w0; y2 <- w1; z2 <- w2
    else
        let struct (ok0, tx0, ty0, tz0) = trySymmetricEigenvector3 xx xy xz yy yz zz e0 vectorTol
        if ok0 then
            x0 <- tx0; y0 <- ty0; z0 <- tz0
        let struct (ok1, tx1, ty1, tz1) = trySymmetricEigenvector3 xx xy xz yy yz zz e1 vectorTol
        if ok1 then
            x1 <- tx1; y1 <- ty1; z1 <- tz1
        let struct (ok2, tx2, ty2, tz2) = trySymmetricEigenvector3 xx xy xz yy yz zz e2 vectorTol
        if ok2 then
            x2 <- tx2; y2 <- ty2; z2 <- tz2
    struct (x0, y0, z0, x1, y1, z1, x2, y2, z2)

let private symmetricMatrixEigensystemFloat32Stage workers : Stage<VectorChunk<float32>, VectorChunk<float32>> =
    let mapper (tensor: VectorChunk<float32>) =
        if Chunk.vectorComponentCount tensor <> 6u then
            invalidArg "matrix" $"chunkSymmetricMatrixEigensystemFloat32 expected a 6-component symmetric matrix, got {Chunk.vectorComponentCount tensor}."

        let output = Array.init 12 (fun _ -> Chunk.create<float32> tensor.SpatialSize)
        try
            let outputVector: VectorChunk<float32> =
                { SpatialSize = tensor.SpatialSize
                  Components = output }
            let xxPixels = Chunk.span<float32> tensor.Components[0]
            let xyPixels = Chunk.span<float32> tensor.Components[1]
            let xzPixels = Chunk.span<float32> tensor.Components[2]
            let yyPixels = Chunk.span<float32> tensor.Components[3]
            let yzPixels = Chunk.span<float32> tensor.Components[4]
            let zzPixels = Chunk.span<float32> tensor.Components[5]
            let out0 = Chunk.span<float32> output[0]
            let out1 = Chunk.span<float32> output[1]
            let out2 = Chunk.span<float32> output[2]
            let out3 = Chunk.span<float32> output[3]
            let out4 = Chunk.span<float32> output[4]
            let out5 = Chunk.span<float32> output[5]
            let out6 = Chunk.span<float32> output[6]
            let out7 = Chunk.span<float32> output[7]
            let out8 = Chunk.span<float32> output[8]
            let out9 = Chunk.span<float32> output[9]
            let out10 = Chunk.span<float32> output[10]
            let out11 = Chunk.span<float32> output[11]
            let spatialCount = xxPixels.Length
            let mutable i = 0
            while i < spatialCount do
                let xx = float xxPixels[i]
                let xy = float xyPixels[i]
                let xz = float xzPixels[i]
                let yy = float yyPixels[i]
                let yz = float yzPixels[i]
                let zz = float zzPixels[i]
                let scale =
                    max 1.0
                        (max (abs xx)
                            (max (abs xy)
                                (max (abs xz)
                                    (max (abs yy)
                                        (max (abs yz) (abs zz))))))
                let struct (e0, e1, e2) = symmetricEigenvalues3Raw xx xy xz yy yz zz
                let struct (x0, y0, z0, x1, y1, z1, x2, y2, z2) =
                    symmetricEigenvectors3Raw xx xy xz yy yz zz e0 e1 e2 scale
                out0[i] <- float32 e0
                out1[i] <- float32 e1
                out2[i] <- float32 e2
                out3[i] <- float32 x0
                out4[i] <- float32 y0
                out5[i] <- float32 z0
                out6[i] <- float32 x1
                out7[i] <- float32 y1
                out8[i] <- float32 z1
                out9[i] <- float32 x2
                out10[i] <- float32 y2
                out11[i] <- float32 z2
                i <- i + 1
            outputVector
        with
        | _ ->
            output |> Array.iter Chunk.decRef
            reraise()

    releaseUnaryVectorToVectorChunkParallel
        "chunkSymmetricMatrixEigensystemFloat32"
        workers
        mapper
        (fun n -> uint64 (6 + workers * 12) * chunkMemoryNeed<float32> n)

let symmetricMatrixEigenvaluesFloat32 workers : Stage<VectorChunk<float32>, VectorChunk<float32>> =
    let mapper (matrix: VectorChunk<float32>) =
        if Chunk.vectorComponentCount matrix <> 6u then
            invalidArg "matrix" $"chunkSymmetricMatrixEigenvaluesFloat32 expected a 6-component symmetric matrix, got {Chunk.vectorComponentCount matrix}."

        let output = Array.init 3 (fun _ -> Chunk.create<float32> matrix.SpatialSize)
        try
            let outputVector: VectorChunk<float32> =
                { SpatialSize = matrix.SpatialSize
                  Components = output }
            let xxPixels = Chunk.span<float32> matrix.Components[0]
            let xyPixels = Chunk.span<float32> matrix.Components[1]
            let xzPixels = Chunk.span<float32> matrix.Components[2]
            let yyPixels = Chunk.span<float32> matrix.Components[3]
            let yzPixels = Chunk.span<float32> matrix.Components[4]
            let zzPixels = Chunk.span<float32> matrix.Components[5]
            let out0 = Chunk.span<float32> output[0]
            let out1 = Chunk.span<float32> output[1]
            let out2 = Chunk.span<float32> output[2]
            let spatialCount = xxPixels.Length
            let mutable i = 0
            while i < spatialCount do
                let struct (e0, e1, e2) =
                    symmetricEigenvalues3Raw
                        (float xxPixels[i])
                        (float xyPixels[i])
                        (float xzPixels[i])
                        (float yyPixels[i])
                        (float yzPixels[i])
                        (float zzPixels[i])
                out0[i] <- float32 e0
                out1[i] <- float32 e1
                out2[i] <- float32 e2
                i <- i + 1
            outputVector
        with
        | _ ->
            output |> Array.iter Chunk.decRef
            reraise()

    releaseUnaryVectorToVectorChunkParallel
        "chunkSymmetricMatrixEigenvaluesFloat32"
        workers
        mapper
        (fun n -> uint64 (6 + workers * 3) * chunkMemoryNeed<float32> n)

let symmetricMatrixEigenvectorFloat32 workers eigenIndex : Stage<VectorChunk<float32>, VectorChunk<float32>> =
    if eigenIndex > 2u then
        invalidArg "eigenIndex" $"chunkSymmetricMatrixEigenvectorFloat32 expects eigenIndex 0, 1, or 2, got {eigenIndex}."

    let mapper (tensor: VectorChunk<float32>) =
        if Chunk.vectorComponentCount tensor <> 6u then
            invalidArg "matrix" $"chunkSymmetricMatrixEigenvectorFloat32 expected a 6-component symmetric matrix, got {Chunk.vectorComponentCount tensor}."

        let output = Array.init 3 (fun _ -> Chunk.create<float32> tensor.SpatialSize)
        try
            let outputVector: VectorChunk<float32> =
                { SpatialSize = tensor.SpatialSize
                  Components = output }
            let xxPixels = Chunk.span<float32> tensor.Components[0]
            let xyPixels = Chunk.span<float32> tensor.Components[1]
            let xzPixels = Chunk.span<float32> tensor.Components[2]
            let yyPixels = Chunk.span<float32> tensor.Components[3]
            let yzPixels = Chunk.span<float32> tensor.Components[4]
            let zzPixels = Chunk.span<float32> tensor.Components[5]
            let outX = Chunk.span<float32> output[0]
            let outY = Chunk.span<float32> output[1]
            let outZ = Chunk.span<float32> output[2]
            let spatialCount = xxPixels.Length
            let mutable i = 0
            while i < spatialCount do
                let xx = float xxPixels[i]
                let xy = float xyPixels[i]
                let xz = float xzPixels[i]
                let yy = float yyPixels[i]
                let yz = float yzPixels[i]
                let zz = float zzPixels[i]
                let scale =
                    max 1.0
                        (max (abs xx)
                            (max (abs xy)
                                (max (abs xz)
                                    (max (abs yy)
                                        (max (abs yz) (abs zz))))))
                let struct (e0, e1, e2) = symmetricEigenvalues3Raw xx xy xz yy yz zz
                let struct (x0, y0, z0, x1, y1, z1, x2, y2, z2) =
                    symmetricEigenvectors3Raw xx xy xz yy yz zz e0 e1 e2 scale
                let struct (x, y, z) =
                    match eigenIndex with
                    | 0u -> struct (x0, y0, z0)
                    | 1u -> struct (x1, y1, z1)
                    | _ -> struct (x2, y2, z2)
                outX[i] <- float32 x
                outY[i] <- float32 y
                outZ[i] <- float32 z
                i <- i + 1
            outputVector
        with
        | _ ->
            output |> Array.iter Chunk.decRef
            reraise()

    releaseUnaryVectorToVectorChunkParallel
        $"chunkSymmetricMatrixEigenvectorFloat32.{eigenIndex}"
        workers
        mapper
        (fun n -> uint64 (6 + workers * 3) * chunkMemoryNeed<float32> n)

let private zeroFloat32ChunkLike _index (source: Chunk<float32>) =
    let width, height, depth = source.Size
    if depth <> 1UL then
        invalidArg "source" $"Chunk derivative stages expect 2D slice chunks with depth 1, got {source.Size}."
    let chunk = Chunk.create<float32> (width, height, 1UL)
    chunk.Bytes.AsSpan(0, chunk.ByteLength).Clear()
    chunk

let private releaseConsumedFloat32Window (window: Window<Chunk<float32>>) =
    let _emitStart, emitCount = window.EmitRange
    if emitCount = 0u then
        window.Items |> List.iter Chunk.decRef
    else
        window.Items
        |> List.truncate (int window.ReleaseCount)
        |> List.iter Chunk.decRef

let private derivativeVectorFromSmoothedWindow name components workers f : Stage<Chunk<float32>, VectorChunk<float32>> =
    if workers < 1 then
        invalidArg "workers" $"{name} expects at least one worker, got {workers}."

    let mapper _debug (window: Window<Chunk<float32>>) =
        let items = window.Items |> List.toArray
        let _emitStart, emitCount = window.EmitRange
        try
            if emitCount = 0u then
                []
            else
                [ f items ]
        finally
            releaseConsumedFloat32Window window

    Stage.parallelCollect
        $"{name}.parallelCollect.Float32.components{components}.workers{workers}"
        3
        workers
        1
        1
        zeroFloat32ChunkLike
        mapper
        (fun n -> uint64 (3 + workers * components) * chunkMemoryNeed<float32> n)
        id

let private derivativeScalarFromSmoothedWindow name workers f : Stage<Chunk<float32>, Chunk<float32>> =
    if workers < 1 then
        invalidArg "workers" $"{name} expects at least one worker, got {workers}."

    let mapper _debug (window: Window<Chunk<float32>>) =
        let items = window.Items |> List.toArray
        let _emitStart, emitCount = window.EmitRange
        try
            if emitCount = 0u then
                []
            else
                [ f items ]
        finally
            releaseConsumedFloat32Window window

    Stage.parallelCollect
        $"{name}.parallelCollect.Float32.workers{workers}"
        3
        workers
        1
        1
        zeroFloat32ChunkLike
        mapper
        (fun n -> uint64 (3 + workers) * chunkMemoryNeed<float32> n)
        id

let private gaussianSmoothFloat32XYZ sigmaX radiusX sigmaY radiusY sigmaZ radiusZ workers =
    if sigmaX <= 0.0 && sigmaY <= 0.0 && sigmaZ <= 0.0 then
        Stage.map "chunkGaussianSmoothFloat32XYZ.identity" (fun _ chunk -> chunk) id id
    else
        let xKernel = if sigmaX <= 0.0 then [| 1.0f |] else StackConvolve.gaussianKernel sigmaX radiusX
        let yKernel = if sigmaY <= 0.0 then [| 1.0f |] else StackConvolve.gaussianKernel sigmaY radiusY
        let zKernel = if sigmaZ <= 0.0 then [| 1.0f |] else StackConvolve.gaussianKernel sigmaZ radiusZ
        StackConvolve.separableConvolveNativeParallelCollect<float32> xKernel yKernel zKernel workers

let gradientVectorNativeParallelCollectXYZ workers sigmaX radiusX sigmaY radiusY sigmaZ radiusZ : Stage<Chunk<float32>, VectorChunk<float32>> =
    let smooth = gaussianSmoothFloat32XYZ sigmaX radiusX sigmaY radiusY sigmaZ radiusZ workers
    let derivatives =
        derivativeVectorFromSmoothedWindow
            "chunkGradientVectorNative"
            3
            workers
            ChunkKernel.gradientVectorFromSmoothedNative
    smooth --> derivatives

let gradientVectorNativeParallelCollect workers sigma radius : Stage<Chunk<float32>, VectorChunk<float32>> =
    gradientVectorNativeParallelCollectXYZ workers sigma radius sigma radius sigma radius

let gradientMagnitudeNativeParallelCollectXYZ sigmaX radiusX sigmaY radiusY sigmaZ radiusZ workers : Stage<Chunk<float32>, Chunk<float32>> =
    gradientVectorNativeParallelCollectXYZ workers sigmaX radiusX sigmaY radiusY sigmaZ radiusZ
    --> vectorMagnitudeFloat32

let gradientMagnitudeNativeParallelCollect sigma radius workers : Stage<Chunk<float32>, Chunk<float32>> =
    gradientMagnitudeNativeParallelCollectXYZ sigma radius sigma radius sigma radius workers

let structureTensorNativeParallelCollect workers sigma radius rho rhoRadius : Stage<Chunk<float32>, VectorChunk<float32>> =
    let outerProduct =
        gradientVectorNativeParallelCollect workers sigma radius
        --> structureTensorOuterProductFloat32

    outerProduct
    --> gaussianSmoothVectorComponentsFloat32 rho rhoRadius workers

let symmetricMatrixEigensystemFloat32 workers : Stage<VectorChunk<float32>, VectorChunk<float32>> =
    symmetricMatrixEigensystemFloat32Stage workers

let hessianUpperNativeParallelCollectXYZ sigmaX radiusX sigmaY radiusY sigmaZ radiusZ workers : Stage<Chunk<float32>, VectorChunk<float32>> =
    let smooth = gaussianSmoothFloat32XYZ sigmaX radiusX sigmaY radiusY sigmaZ radiusZ workers
    let derivatives =
        derivativeVectorFromSmoothedWindow
            "chunkHessianUpperNative"
            6
            workers
            ChunkKernel.hessianUpperFromSmoothedNative
    smooth --> derivatives

let hessianUpperNativeParallelCollect sigma radius workers : Stage<Chunk<float32>, VectorChunk<float32>> =
    hessianUpperNativeParallelCollectXYZ sigma radius sigma radius sigma radius workers

let laplacianNativeParallelCollectXYZ sigmaX radiusX sigmaY radiusY sigmaZ radiusZ workers : Stage<Chunk<float32>, Chunk<float32>> =
    let smooth = gaussianSmoothFloat32XYZ sigmaX radiusX sigmaY radiusY sigmaZ radiusZ workers
    let derivatives =
        derivativeScalarFromSmoothedWindow
            "chunkLaplacianNative"
            workers
            ChunkKernel.laplacianFromSmoothedNative
    smooth --> derivatives

let laplacianNativeParallelCollect sigma radius workers : Stage<Chunk<float32>, Chunk<float32>> =
    laplacianNativeParallelCollectXYZ sigma radius sigma radius sigma radius workers

let sobelMagnitudeNativeParallelCollect workers : Stage<Chunk<float32>, Chunk<float32>> =
    derivativeScalarFromSmoothedWindow
        "chunkSobelMagnitudeNative"
        workers
        ChunkKernel.sobelMagnitudeFromNativeFloat32

let inline map<'T, 'U when 'T: equality and 'U: equality
                         and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType
                         and 'U: (new: unit -> 'U) and 'U: struct and 'U :> ValueType>
    name
    f
    : Stage<Chunk<'T>, Chunk<'U>> =
    releaseUnaryChunk name (Chunk.map f) (fun n -> n * uint64 (chunkElementBytes<'T> + chunkElementBytes<'U>))

let inline map2<'T, 'U, 'V when 'T: equality and 'U: equality and 'V: equality
                              and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType
                              and 'U: (new: unit -> 'U) and 'U: struct and 'U :> ValueType
                              and 'V: (new: unit -> 'V) and 'V: struct and 'V :> ValueType>
    name
    f
    : Stage<Chunk<'T> * Chunk<'U>, Chunk<'V>> =
    let mapper (a: Chunk<'T>) (b: Chunk<'U>) : Chunk<'V> =
        ChunkKernel.map2Chunk name f a b
    releaseBinaryChunk name mapper (fun n -> n * uint64 (chunkElementBytes<'T> + chunkElementBytes<'U> + chunkElementBytes<'V>))

let inline sum<'T when 'T: equality
                    and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType
                    and 'T: (static member ( + ) : 'T * 'T -> 'T)
                    and 'T: (static member Zero : 'T)> (chunk: Chunk<'T>) : 'T =
    ChunkKernel.sum chunk

let inline prod<'T when 'T: equality
                     and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType
                     and 'T: (static member ( * ) : 'T * 'T -> 'T)
                     and 'T: (static member One : 'T)> (chunk: Chunk<'T>) : 'T =
    ChunkKernel.prod chunk

let inline minMax<'T when 'T: equality and 'T: comparison
                       and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (chunk: Chunk<'T>) =
    ChunkKernel.minMax chunk

let inline getMinMax chunk = minMax chunk

let mapFloat32Vector name (scalarOp: float32 -> float32) (vectorOp: Vector<float32> -> Vector<float32>) (chunk: Chunk<float32>) =
    ChunkKernel.mapFloat32Vector name scalarOp vectorOp chunk

let map2Float32Vector name (scalarOp: float32 -> float32 -> float32) (vectorOp: Vector<float32> -> Vector<float32> -> Vector<float32>) (a: Chunk<float32>) (b: Chunk<float32>) =
    ChunkKernel.map2Float32Vector name scalarOp vectorOp a b

let float32UnaryStage name (scalarOp: float32 -> float32) (vectorOp: Vector<float32> -> Vector<float32>) =
    releaseUnaryChunk name (mapFloat32Vector name scalarOp vectorOp) (fun n -> 2UL * chunkMemoryNeed<float32> n)

let float32BinaryStage name (scalarOp: float32 -> float32 -> float32) (vectorOp: Vector<float32> -> Vector<float32> -> Vector<float32>) =
    releaseBinaryChunk name (map2Float32Vector name scalarOp vectorOp) (fun n -> 3UL * chunkMemoryNeed<float32> n)

let addScalarFloat32 (value: float32) =
    let valueV = Vector<float32>(value)
    float32UnaryStage $"chunkAddScalar.Float32.{value}" (fun x -> x + value) (fun v -> v + valueV)

let subScalarFloat32 (value: float32) =
    let valueV = Vector<float32>(value)
    float32UnaryStage $"chunkSubScalar.Float32.{value}" (fun x -> x - value) (fun v -> v - valueV)

let scalarSubFloat32 (value: float32) =
    let valueV = Vector<float32>(value)
    float32UnaryStage $"chunkScalarSub.Float32.{value}" (fun x -> value - x) (fun v -> valueV - v)

let mulScalarFloat32 (value: float32) =
    let valueV = Vector<float32>(value)
    float32UnaryStage $"chunkMulScalar.Float32.{value}" (fun x -> x * value) (fun v -> v * valueV)

let divScalarFloat32 (value: float32) =
    let valueV = Vector<float32>(value)
    float32UnaryStage $"chunkDivScalar.Float32.{value}" (fun x -> x / value) (fun v -> v / valueV)

let scalarDivFloat32 (value: float32) =
    let valueV = Vector<float32>(value)
    float32UnaryStage $"chunkScalarDiv.Float32.{value}" (fun x -> value / x) (fun v -> valueV / v)

let addFloat32 =
    float32BinaryStage "chunkAdd.Float32" (fun a b -> a + b) (fun a b -> a + b)

let subtractFloat32 =
    float32BinaryStage "chunkSubtract.Float32" (fun a b -> a - b) (fun a b -> a - b)

let multiplyFloat32 =
    float32BinaryStage "chunkMultiply.Float32" (fun a b -> a * b) (fun a b -> a * b)

let divideFloat32 =
    float32BinaryStage "chunkDivide.Float32" (fun a b -> a / b) (fun a b -> a / b)

let inline addScalar value =
    if typeof<'T> = typeof<float32> then
        unbox (box (addScalarFloat32 (unbox<float32> (box value))))
    else
        map $"chunkAddScalar.{typeof<'T>.Name}" (fun (x: 'T) -> x + value)

let inline scalarAdd value = addScalar value

let inline subScalar value =
    if typeof<'T> = typeof<float32> then
        unbox (box (subScalarFloat32 (unbox<float32> (box value))))
    else
        map $"chunkSubScalar.{typeof<'T>.Name}" (fun (x: 'T) -> x - value)

let inline scalarSub value =
    if typeof<'T> = typeof<float32> then
        unbox (box (scalarSubFloat32 (unbox<float32> (box value))))
    else
        map $"chunkScalarSub.{typeof<'T>.Name}" (fun (x: 'T) -> value - x)

let inline mulScalar value =
    if typeof<'T> = typeof<float32> then
        unbox (box (mulScalarFloat32 (unbox<float32> (box value))))
    else
        map $"chunkMulScalar.{typeof<'T>.Name}" (fun (x: 'T) -> x * value)

let inline scalarMul value = mulScalar value

let inline divScalar value =
    if typeof<'T> = typeof<float32> then
        unbox (box (divScalarFloat32 (unbox<float32> (box value))))
    else
        map $"chunkDivScalar.{typeof<'T>.Name}" (fun (x: 'T) -> x / value)

let inline scalarDiv value =
    if typeof<'T> = typeof<float32> then
        unbox (box (scalarDivFloat32 (unbox<float32> (box value))))
    else
        map $"chunkScalarDiv.{typeof<'T>.Name}" (fun (x: 'T) -> value / x)

let inline add<'T when 'T: equality
                    and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType
                    and 'T: (static member ( + ) : 'T * 'T -> 'T)> =
    if typeof<'T> = typeof<float32> then
        unbox (box addFloat32)
    else
        map2< 'T, 'T, 'T> $"chunkAdd.{typeof<'T>.Name}" (fun a b -> a + b)

let inline subtract<'T when 'T: equality
                         and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType
                         and 'T: (static member ( - ) : 'T * 'T -> 'T)> =
    if typeof<'T> = typeof<float32> then
        unbox (box subtractFloat32)
    else
        map2< 'T, 'T, 'T> $"chunkSubtract.{typeof<'T>.Name}" (fun a b -> a - b)

let inline multiply<'T when 'T: equality
                         and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType
                         and 'T: (static member ( * ) : 'T * 'T -> 'T)> =
    if typeof<'T> = typeof<float32> then
        unbox (box multiplyFloat32)
    else
        map2< 'T, 'T, 'T> $"chunkMultiply.{typeof<'T>.Name}" (fun a b -> a * b)

let inline divide<'T when 'T: equality
                       and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType
                       and 'T: (static member ( / ) : 'T * 'T -> 'T)> =
    if typeof<'T> = typeof<float32> then
        unbox (box divideFloat32)
    else
        map2< 'T, 'T, 'T> $"chunkDivide.{typeof<'T>.Name}" (fun a b -> a / b)

let inline maximum<'T when 'T: equality and 'T: comparison
                        and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> =
    map2< 'T, 'T, 'T> $"chunkMaximum.{typeof<'T>.Name}" (fun a b -> if a > b then a else b)

let inline minimum<'T when 'T: equality and 'T: comparison
                        and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> =
    map2< 'T, 'T, 'T> $"chunkMinimum.{typeof<'T>.Name}" (fun a b -> if a < b then a else b)

let inline equal<'T when 'T: equality
                      and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> =
    map2< 'T, 'T, uint8> $"chunkEqual.{typeof<'T>.Name}" (fun a b -> if a = b then 1uy else 0uy)

let inline notEqual<'T when 'T: equality
                         and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> =
    map2< 'T, 'T, uint8> $"chunkNotEqual.{typeof<'T>.Name}" (fun a b -> if a <> b then 1uy else 0uy)

let inline greater<'T when 'T: equality and 'T: comparison
                        and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> =
    map2< 'T, 'T, uint8> $"chunkGreater.{typeof<'T>.Name}" (fun a b -> if a > b then 1uy else 0uy)

let inline greaterEqual<'T when 'T: equality and 'T: comparison
                             and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> =
    map2< 'T, 'T, uint8> $"chunkGreaterEqual.{typeof<'T>.Name}" (fun a b -> if a >= b then 1uy else 0uy)

let inline less<'T when 'T: equality and 'T: comparison
                     and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> =
    map2< 'T, 'T, uint8> $"chunkLess.{typeof<'T>.Name}" (fun a b -> if a < b then 1uy else 0uy)

let inline lessEqual<'T when 'T: equality and 'T: comparison
                          and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> =
    map2< 'T, 'T, uint8> $"chunkLessEqual.{typeof<'T>.Name}" (fun a b -> if a <= b then 1uy else 0uy)

let private maskBinaryByte name op =
    releaseBinaryChunk name (fun a b -> map2Chunk name op a b) (fun n -> 3UL * n)

let maskAnd : Stage<Chunk<uint8> * Chunk<uint8>, Chunk<uint8>> =
    maskBinaryByte "chunkMaskAnd" (fun a b -> a &&& b)

let maskOr : Stage<Chunk<uint8> * Chunk<uint8>, Chunk<uint8>> =
    maskBinaryByte "chunkMaskOr" (fun a b -> a ||| b)

let maskXor : Stage<Chunk<uint8> * Chunk<uint8>, Chunk<uint8>> =
    maskBinaryByte "chunkMaskXor" (fun a b -> a ^^^ b)

let maskNot : Stage<Chunk<uint8>, Chunk<uint8>> =
    map "chunkMaskNot" (fun value -> if value = binaryBackground then binaryForeground else binaryBackground)

let mask<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (outsideValue: 'T)
    : Stage<Chunk<'T> * Chunk<uint8>, Chunk<'T>> =
    map2 $"chunkMask.{typeof<'T>.Name}" (fun value maskValue -> if maskValue = binaryBackground then outsideValue else value)

let private projectionTransform (transformName: string) =
    match transformName.Trim().ToLowerInvariant().Replace("-", "").Replace("_", "").Replace(" ", "") with
    | ""
    | "identity" -> id
    | "abs"
    | "absolute" -> abs
    | "squared"
    | "square" -> fun value -> value * value
    | "sqrtabs"
    | "sqrt"
    | "squareroot" -> fun value -> Math.Sqrt(Math.Abs value)
    | "log1pabs"
    | "log"
    | "logabs" -> fun value -> Math.Log(1.0 + Math.Abs value)
    | _ ->
        invalidArg "transformName" $"Unknown projection transform '{transformName}'."

let inline private accumulateSumProjectionTyped (transform: float -> float) (pixels: Span< ^T>) (values: float32[]) =
    let mutable i = 0
    while i < pixels.Length do
        values[i] <- values[i] + float32 (transform (float pixels[i]))
        i <- i + 1

let chunkSumProjection<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    transformName
    : Stage<Chunk<'T>, Chunk<float32>> =
    let transform = projectionTransform transformName

    let reducer (_debug: bool) (input: AsyncSeq<Chunk<'T>>) =
        async {
            let! state =
                input
                |> AsyncSeq.foldAsync
                    (fun state chunk ->
                        async {
                            try
                                let widthU, heightU, depthU = chunk.Size
                                if depthU <> 1UL then
                                    invalidArg "chunk" $"chunkSumProjection expects 2D slice chunks with depth 1, got {chunk.Size}."
                                let width = int widthU
                                let height = int heightU
                                let accumulator =
                                    match state with
                                    | None ->
                                        Some(width, height, Array.zeroCreate<float32> (width * height))
                                    | Some(expectedWidth, expectedHeight, values) ->
                                        if expectedWidth <> width || expectedHeight <> height then
                                            invalidOp $"chunkSumProjection requires constant x-y slice size; got {width}x{height}, expected {expectedWidth}x{expectedHeight}."
                                        Some(expectedWidth, expectedHeight, values)

                                match accumulator with
                                | None -> return None
                                | Some(_, _, values) ->
                                    let t = typeof<'T>
                                    if t = typeof<uint8> then
                                        accumulateSumProjectionTyped transform (MemoryMarshal.Cast<byte, uint8>(chunk.Bytes.AsSpan(0, chunk.ByteLength))) values
                                    elif t = typeof<int8> then
                                        accumulateSumProjectionTyped transform (MemoryMarshal.Cast<byte, int8>(chunk.Bytes.AsSpan(0, chunk.ByteLength))) values
                                    elif t = typeof<uint16> then
                                        accumulateSumProjectionTyped transform (MemoryMarshal.Cast<byte, uint16>(chunk.Bytes.AsSpan(0, chunk.ByteLength))) values
                                    elif t = typeof<int16> then
                                        accumulateSumProjectionTyped transform (MemoryMarshal.Cast<byte, int16>(chunk.Bytes.AsSpan(0, chunk.ByteLength))) values
                                    elif t = typeof<uint32> then
                                        accumulateSumProjectionTyped transform (MemoryMarshal.Cast<byte, uint32>(chunk.Bytes.AsSpan(0, chunk.ByteLength))) values
                                    elif t = typeof<int32> then
                                        accumulateSumProjectionTyped transform (MemoryMarshal.Cast<byte, int32>(chunk.Bytes.AsSpan(0, chunk.ByteLength))) values
                                    elif t = typeof<uint64> then
                                        accumulateSumProjectionTyped transform (MemoryMarshal.Cast<byte, uint64>(chunk.Bytes.AsSpan(0, chunk.ByteLength))) values
                                    elif t = typeof<int64> then
                                        accumulateSumProjectionTyped transform (MemoryMarshal.Cast<byte, int64>(chunk.Bytes.AsSpan(0, chunk.ByteLength))) values
                                    elif t = typeof<float32> then
                                        accumulateSumProjectionTyped transform (MemoryMarshal.Cast<byte, float32>(chunk.Bytes.AsSpan(0, chunk.ByteLength))) values
                                    elif t = typeof<float> then
                                        accumulateSumProjectionTyped transform (MemoryMarshal.Cast<byte, float>(chunk.Bytes.AsSpan(0, chunk.ByteLength))) values
                                    else
                                        invalidArg "T" $"chunkSumProjection supports real numeric chunks, got {t.Name}."
                                    return accumulator
                            finally
                                Chunk.decRef chunk
                        })
                    None

            match state with
            | None ->
                return raise (InvalidOperationException "chunkSumProjection cannot reduce an empty chunk stream.")
            | Some(width, height, values) ->
                let output = Chunk.create<float32> (uint64 width, uint64 height, 1UL)
                let outputPixels = Chunk.span output
                values.AsSpan().CopyTo outputPixels
                return output
        }

    Stage.reduce $"chunkSumProjection {transformName}" reducer Streaming (fun n -> n * uint64 (chunkElementBytes<'T> + chunkElementBytes<float32>)) id

let absFloat32 : Stage<Chunk<float32>, Chunk<float32>> =
    float32UnaryStage "chunkAbsFloat32" abs (fun v -> Vector.Abs(v))

let absFloat : Stage<Chunk<float>, Chunk<float>> =
    map "chunkAbsFloat" Math.Abs

let sqrtFloat32 : Stage<Chunk<float32>, Chunk<float32>> =
    float32UnaryStage "chunkSqrtFloat32" sqrt (fun v -> Vector.SquareRoot(v))

let sqrtFloat : Stage<Chunk<float>, Chunk<float>> =
    map "chunkSqrtFloat" Math.Sqrt

let logFloat32 : Stage<Chunk<float32>, Chunk<float32>> =
    map "chunkLogFloat32" MathF.Log

let logFloat : Stage<Chunk<float>, Chunk<float>> =
    map "chunkLogFloat" Math.Log

let squareFloat32 : Stage<Chunk<float32>, Chunk<float32>> =
    float32UnaryStage "chunkSquareFloat32" (fun x -> x * x) (fun (v: Vector<float32>) -> v * v)

let shiftScaleFloat32 (shift: double) (scale: double) : Stage<Chunk<float32>, Chunk<float32>> =
    let shiftV = Vector<float32>(float32 shift)
    let scaleV = Vector<float32>(float32 scale)
    float32UnaryStage $"chunkShiftScaleFloat32.{shift}.{scale}" (fun x -> (x + float32 shift) * float32 scale) (fun (v: Vector<float32>) -> (v + shiftV) * scaleV)

let clampFloat32 (lower: double) (upper: double) : Stage<Chunk<float32>, Chunk<float32>> =
    let lowerF = float32 lower
    let upperF = float32 upper
    let lowerV = Vector<float32>(lowerF)
    let upperV = Vector<float32>(upperF)
    float32UnaryStage $"chunkClampFloat32.{lower}.{upper}" (fun x -> min upperF (max lowerF x)) (fun (v: Vector<float32>) -> Vector.Min(upperV, Vector.Max(lowerV, v)))

let intensityStretchFloat32 (windowMinimum: double) (windowMaximum: double) (outputMinimum: double) (outputMaximum: double) : Stage<Chunk<float32>, Chunk<float32>> =
    if windowMaximum = windowMinimum then
        invalidArg "windowMaximum" "ChunkFunctions.intensityStretchFloat32 requires a non-zero input window width."
    let scale = (outputMaximum - outputMinimum) / (windowMaximum - windowMinimum)
    let scalar x =
        if x <= float32 windowMinimum then float32 outputMinimum
        elif x >= float32 windowMaximum then float32 outputMaximum
        else float32 outputMinimum + (x - float32 windowMinimum) * float32 scale

    let minV = Vector<float32>(float32 windowMinimum)
    let maxV = Vector<float32>(float32 windowMaximum)
    let outMinV = Vector<float32>(float32 outputMinimum)
    let outMaxV = Vector<float32>(float32 outputMaximum)
    let scaleV = Vector<float32>(float32 scale)
    let vector (v: Vector<float32>) =
        Vector.Min(outMaxV, Vector.Max(outMinV, outMinV + (v - minV) * scaleV))

    float32UnaryStage $"chunkIntensityStretchFloat32.{windowMinimum}.{windowMaximum}.{outputMinimum}.{outputMaximum}" scalar vector

let invertIntensityFloat32 (maximum: double) : Stage<Chunk<float32>, Chunk<float32>> =
    let maximumV = Vector<float32>(float32 maximum)
    float32UnaryStage $"chunkInvertIntensityFloat32.{maximum}" (fun x -> float32 maximum - x) (fun (v: Vector<float32>) -> maximumV - v)

let thresholdBinary (threshold: uint8) : Stage<Chunk<uint8>, Chunk<uint8>> =
    let mapper _debug chunk =
        try
            Chunk.map (fun value -> if value >= threshold then binaryForeground else binaryBackground) chunk
        finally
            Chunk.decRef chunk

    Stage.map $"chunkThresholdBinary.{threshold}" mapper id id

let private thresholdNativeChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (threshold: double) (chunk: Chunk<'T>) =
    ChunkKernel.thresholdNativeChunk threshold chunk

let thresholdNative<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (threshold: double)
    : Stage<Chunk<'T>, Chunk<'T>> =
    let mapper _debug chunk =
        try
            thresholdNativeChunk threshold chunk
        finally
            Chunk.decRef chunk

    Stage.map $"chunkThresholdNative.{typeof<'T>.Name}.{threshold}" mapper id id

let thresholdLocatedNativeUInt8
    (threshold: double)
    : Stage<LocatedChunk<uint8>, LocatedChunk<uint8>> =
    let mapper _debug (located: LocatedChunk<uint8>) : LocatedChunk<uint8> =
        try
            let output = thresholdNativeChunk threshold located.Chunk
            { Index = located.Index
              Layout = { located.Layout with PixelType = "uint8" }
              Chunk = output }
        finally
            Chunk.decRef located.Chunk

    Stage.map $"chunkThresholdLocatedNative.UInt8.{threshold}" mapper id id

let thresholdNativeParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (threshold: double)
    (workers: int)
    : Stage<Chunk<'T>, Chunk<'T>> =
    if workers < 1 then
        invalidArg "workers" $"ChunkFunctions.thresholdNativeParallelCollect expects at least one worker, got {workers}."

    let mapper _debug (window: Window<Chunk<'T>>) =
        match window.Items with
        | [ chunk ] ->
            try
                [ thresholdNativeChunk threshold chunk ]
            finally
                Chunk.decRef chunk
        | items ->
            for chunk in items do
                Chunk.decRef chunk
            invalidArg "window" $"ChunkFunctions.thresholdNativeParallelCollect expects singleton windows, got {items.Length} items."

    Stage.parallelCollect
        $"chunkThresholdNative.parallelCollect.{typeof<'T>.Name}.{threshold}.workers{workers}"
        1
        workers
        1
        0
        (fun _ chunk -> chunk)
        mapper
        id
        id

let private castChunkToUInt8<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (chunk: Chunk<'T>) =
    ChunkKernel.castChunkToUInt8 chunk

let castToUInt8<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> : Stage<Chunk<'T>, Chunk<uint8>> =
    let mapper _debug chunk =
        try
            castChunkToUInt8 chunk |> Chunk.withSameIndex chunk
        finally
            Chunk.decRef chunk

    Stage.map $"chunkCastToUInt8.{typeof<'T>.Name}" mapper id id

let private castChunkToFloat32<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (chunk: Chunk<'T>) =
    ChunkKernel.castChunkToFloat32 chunk

let castToFloat32<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> : Stage<Chunk<'T>, Chunk<float32>> =
    let mapper _debug chunk =
        try
            castChunkToFloat32 chunk |> Chunk.withSameIndex chunk
        finally
            Chunk.decRef chunk

    Stage.map $"chunkCastToFloat32.{typeof<'T>.Name}" mapper id id

let private castFloat32ToChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (chunk: Chunk<float32>) =
    ChunkKernel.castFloat32ToChunk<'T> chunk

let castFromFloat32<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> : Stage<Chunk<float32>, Chunk<'T>> =
    let mapper _debug chunk =
        try
            castFloat32ToChunk<'T> chunk |> Chunk.withSameIndex chunk
        finally
            Chunk.decRef chunk

    Stage.map $"chunkCastFromFloat32.{typeof<'T>.Name}" mapper id id

let castChunk<'S, 'T when 'S: equality and 'S: (new: unit -> 'S) and 'S: struct and 'S :> ValueType
                          and 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> : Stage<Chunk<'S>, Chunk<'T>> =
    let mapper _debug chunk =
        try
            ChunkKernel.castChunk<'S, 'T> chunk |> Chunk.withSameIndex chunk
        finally
            Chunk.decRef chunk

    Stage.map $"chunkCast.{typeof<'S>.Name}.{typeof<'T>.Name}" mapper id id

let vectorCast<'S, 'T when 'S: equality and 'S: (new: unit -> 'S) and 'S: struct and 'S :> ValueType
                         and 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    : Stage<VectorChunk<'S>, VectorChunk<'T>> =
    let mapper _debug (vector: VectorChunk<'S>) =
        let components = Array.zeroCreate<Chunk<'T>> vector.Components.Length
        let mutable initialized = 0
        try
            try
                for c in 0 .. vector.Components.Length - 1 do
                    if vector.Components[c].Size <> vector.SpatialSize then
                        invalidArg "vector" $"vectorCast component {c} size {vector.Components[c].Size} does not match spatial size {vector.SpatialSize}."
                    components[c] <- ChunkKernel.castChunk<'S, 'T> vector.Components[c]
                    initialized <- initialized + 1

                let output: VectorChunk<'T> =
                    { SpatialSize = vector.SpatialSize
                      Components = components }
                output
            with
            | _ ->
                for c in 0 .. initialized - 1 do
                    Chunk.decRef components[c]
                reraise()
        finally
            Chunk.decRefVector vector

    Stage.map
        $"chunkVectorCast.{typeof<'S>.Name}.{typeof<'T>.Name}"
        mapper
        (fun n -> n * uint64 (chunkElementBytes<'S> + chunkElementBytes<'T>))
        id

let vectorCastTo<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = vectorCast<_, 'T>

let thresholdRange<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (lower: double)
    (upper: double)
    : Stage<Chunk<'T>, Chunk<uint8>> =
    let lowerF = float32 lower
    let upperF = float32 upper
    let thresholdFloat32 =
        map "chunkThresholdRange.Float32" (fun value -> if value >= lowerF && value <= upperF then 1uy else 0uy)

    if typeof<'T> = typeof<float32> then
        unbox (box thresholdFloat32)
    else
        castToFloat32<'T> --> thresholdFloat32

let shiftScale<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> shift scale : Stage<Chunk<'T>, Chunk<'T>> =
    if typeof<'T> = typeof<float32> then
        unbox (box (shiftScaleFloat32 shift scale))
    else
        Stage.compose (castToFloat32<'T>) (Stage.compose (shiftScaleFloat32 shift scale) (castFromFloat32<'T>))

let clamp<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> lower upper : Stage<Chunk<'T>, Chunk<'T>> =
    if typeof<'T> = typeof<float32> then
        unbox (box (clampFloat32 lower upper))
    else
        Stage.compose (castToFloat32<'T>) (Stage.compose (clampFloat32 lower upper) (castFromFloat32<'T>))

let private intensityStretchScalar<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    windowMinimum
    windowMaximum
    outputMinimum
    outputMaximum
    : Stage<Chunk<'T>, Chunk<'T>> =
    if typeof<'T> = typeof<float32> then
        unbox (box (intensityStretchFloat32 windowMinimum windowMaximum outputMinimum outputMaximum))
    else
        Stage.compose (castToFloat32<'T>) (Stage.compose (intensityStretchFloat32 windowMinimum windowMaximum outputMinimum outputMaximum) (castFromFloat32<'T>))

let intensityStretch windowMinimum windowMaximum outputMinimum outputMaximum : Stage<'Image, 'Image> =
    if typeof<'Image> = typeof<Chunk<uint8>> then
        unbox (box (intensityStretchScalar<uint8> windowMinimum windowMaximum outputMinimum outputMaximum))
    elif typeof<'Image> = typeof<Chunk<int8>> then
        unbox (box (intensityStretchScalar<int8> windowMinimum windowMaximum outputMinimum outputMaximum))
    elif typeof<'Image> = typeof<Chunk<uint16>> then
        unbox (box (intensityStretchScalar<uint16> windowMinimum windowMaximum outputMinimum outputMaximum))
    elif typeof<'Image> = typeof<Chunk<int16>> then
        unbox (box (intensityStretchScalar<int16> windowMinimum windowMaximum outputMinimum outputMaximum))
    elif typeof<'Image> = typeof<Chunk<uint32>> then
        unbox (box (intensityStretchScalar<uint32> windowMinimum windowMaximum outputMinimum outputMaximum))
    elif typeof<'Image> = typeof<Chunk<int32>> then
        unbox (box (intensityStretchScalar<int32> windowMinimum windowMaximum outputMinimum outputMaximum))
    elif typeof<'Image> = typeof<Chunk<uint64>> then
        unbox (box (intensityStretchScalar<uint64> windowMinimum windowMaximum outputMinimum outputMaximum))
    elif typeof<'Image> = typeof<Chunk<int64>> then
        unbox (box (intensityStretchScalar<int64> windowMinimum windowMaximum outputMinimum outputMaximum))
    elif typeof<'Image> = typeof<Chunk<float32>> then
        unbox (box (intensityStretchScalar<float32> windowMinimum windowMaximum outputMinimum outputMaximum))
    elif typeof<'Image> = typeof<Chunk<float>> then
        unbox (box (intensityStretchScalar<float> windowMinimum windowMaximum outputMinimum outputMaximum))
    elif typeof<'Image> = typeof<VectorChunk<float32>> then
        unbox (box (intensityStretchVector<float32> windowMinimum windowMaximum outputMinimum outputMaximum))
    elif typeof<'Image> = typeof<VectorChunk<float>> then
        unbox (box (intensityStretchVector<float> windowMinimum windowMaximum outputMinimum outputMaximum))
    else
        invalidArg "Image" $"intensityStretch supports real scalar chunks and float32/float vector chunks, got {typeof<'Image>.Name}."

let invertIntensity<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> maximum : Stage<Chunk<'T>, Chunk<'T>> =
    if typeof<'T> = typeof<float32> then
        unbox (box (invertIntensityFloat32 maximum))
    else
        Stage.compose (castToFloat32<'T>) (Stage.compose (invertIntensityFloat32 maximum) (castFromFloat32<'T>))

let addNormalNoise<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> mean stddev : Stage<Chunk<'T>, Chunk<'T>> =
    let memoryNeed n = 2UL * chunkMemoryNeed<'T> n
    releaseUnaryChunk $"chunkAddNormalNoise.{typeof<'T>.Name}" (ChunkKernel.addNormalNoiseChunk<'T> mean stddev) memoryNeed

let addSaltAndPepperNoise<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> probability pepper salt : Stage<Chunk<'T>, Chunk<'T>> =
    let memoryNeed n = 2UL * chunkMemoryNeed<'T> n
    releaseUnaryChunk $"chunkAddSaltAndPepperNoise.{typeof<'T>.Name}" (ChunkKernel.addSaltAndPepperNoiseChunk<'T> probability pepper salt) memoryNeed

let addPoissonNoise<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> lambda : Stage<Chunk<'T>, Chunk<'T>> =
    let memoryNeed n = 2UL * chunkMemoryNeed<'T> n
    releaseUnaryChunk $"chunkAddPoissonNoise.{typeof<'T>.Name}" (ChunkKernel.addPoissonNoiseChunk<'T> lambda) memoryNeed

type DenseHistogram = ChunkKernel.DenseHistogram
type LeftEdgeHistogram = ChunkKernel.LeftEdgeHistogram
type ChunkStats = ChunkKernel.ChunkStats

let addCountsInto target source = ChunkKernel.addCountsInto target source
let computeChunkStats<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> chunk = ChunkKernel.computeStats<'T> chunk
let addChunkStats = ChunkKernel.addStats
let histogramDictionaryBytes<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> bytes byteLength = ChunkKernel.histogramDictionaryBytes<'T> bytes byteLength
let addDictionaryInto<'T when 'T: equality> (target: Dictionary<'T, uint64>) (source: Dictionary<'T, uint64>) = ChunkKernel.addDictionaryInto target source
let dictionaryToMap<'T when 'T: comparison> (counts: Dictionary<'T, uint64>) = ChunkKernel.dictionaryToMap counts
let histogramBytes<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> bytes byteLength = ChunkKernel.histogramBytes<'T> bytes byteLength
let histogramDenseCountsBytes<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> bytes byteLength = ChunkKernel.histogramDenseCountsBytes<'T> bytes byteLength
let addDenseInto target source = ChunkKernel.addDenseInto target source
let emptyDenseCounts<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> () = ChunkKernel.emptyDenseCounts<'T> ()
let addDenseChunkInto<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> target chunk = ChunkKernel.addDenseChunkInto<'T> target chunk
let denseToMap<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> dense = ChunkKernel.denseToMap<'T> dense
let histogramDenseBytes<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> bytes byteLength = ChunkKernel.histogramDenseBytes<'T> bytes byteLength
let validateLeftEdges leftEdges = ChunkKernel.validateLeftEdges leftEdges
let histogramLeftEdgeCountsBytes<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> leftEdges bytes byteLength = ChunkKernel.histogramLeftEdgeCountsBytes<'T> leftEdges bytes byteLength
let addLeftEdgesInto target source = ChunkKernel.addLeftEdgesInto target source
let leftEdgesToMap leftEdgeHistogram = ChunkKernel.leftEdgesToMap leftEdgeHistogram
let histogramLeftEdgesBytes<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> leftEdges bytes byteLength = ChunkKernel.histogramLeftEdgesBytes<'T> leftEdges bytes byteLength
let histogramDictionary<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> chunk = ChunkKernel.histogramDictionary<'T> chunk
let addChunkIntoDictionary<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> counts chunk = ChunkKernel.addChunkIntoDictionary<'T> counts chunk
let histogram<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> chunk = ChunkKernel.histogram<'T> chunk
let histogramDenseCounts<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> chunk = ChunkKernel.histogramDenseCounts<'T> chunk
let histogramDense<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> chunk = ChunkKernel.histogramDense<'T> chunk
let histogramLeftEdgeCounts<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> leftEdges chunk = ChunkKernel.histogramLeftEdgeCounts<'T> leftEdges chunk
let histogramLeftEdges<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> leftEdges chunk = ChunkKernel.histogramLeftEdges<'T> leftEdges chunk

let private toImageStats (stats: ChunkStats) : StackCore.ImageStats =
    { NumPixels = stats.NumPixels
      Mean = stats.Mean
      Std = stats.Std
      Min = stats.Min
      Max = stats.Max
      Sum = stats.Sum
      Var = stats.Var }

let computeStats<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    ()
    : Stage<Chunk<'T>, StackCore.ImageStats> =
    let reducer _debug (input: AsyncSeq<Chunk<'T>>) =
        async {
            let mutable stats = ChunkKernel.zeroStats
            for chunk in input do
                try
                    stats <- ChunkKernel.addStats stats (ChunkKernel.computeStats<'T> chunk)
                finally
                    Chunk.decRef chunk
            return toImageStats stats
        }

    Stage.reduce $"chunkComputeStats.{typeof<'T>.Name}" reducer Streaming (fun _ -> 0UL) (fun _ -> 1UL)

let volume xUnit yUnit zUnit : Stage<Chunk<uint8>, float> =
    if xUnit <= 0.0 then invalidArg "xUnit" "xUnit must be positive."
    if yUnit <= 0.0 then invalidArg "yUnit" "yUnit must be positive."
    if zUnit <= 0.0 then invalidArg "zUnit" "zUnit must be positive."

    let voxelVolume = xUnit * yUnit * zUnit
    let reducer _debug (input: AsyncSeq<Chunk<uint8>>) =
        async {
            let mutable total = 0.0
            for chunk in input do
                try
                    let pixels = chunk.Bytes
                    let mutable foreground = 0UL
                    let mutable i = 0
                    while i < chunk.ByteLength do
                        match pixels[i] with
                        | 0uy -> ()
                        | 1uy -> foreground <- foreground + 1UL
                        | value -> invalidOp $"chunkVolume expects a UInt8 0-1 mask stream; got pixel value {value}."
                        i <- i + 1

                    total <- total + float foreground * voxelVolume
                finally
                    Chunk.decRef chunk

            return total
        }

    Stage.reduce "chunkVolume" reducer Streaming (fun _ -> 0UL) (fun _ -> 1UL)

let histogramReducer<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> () =
    let reducer _debug (input: AsyncSeq<Chunk<'T>>) =
        async {
            let counts = Dictionary<'T, uint64>()
            for chunk in input do
                try
                    addChunkIntoDictionary counts chunk
                finally
                    Chunk.decRef chunk
            return counts |> dictionaryToMap |> Histogram.ofMap
        }

    Stage.reduce $"chunkHistogram.{typeof<'T>.Name}" reducer Streaming (fun _ -> 0UL) (fun _ -> 1UL)

let histogramReducerParallel<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    windowSize
    =
    if windowSize <= 1 then
        histogramReducer<'T> ()
    else
        let folder: MonoidFolder<Chunk<'T>, Dictionary<'T, uint64>, Histogram<'T>> =
            { Create = fun () -> Dictionary<'T, uint64>()
              AddItemInto = fun counts chunk -> addChunkIntoDictionary counts chunk
              MergeInto = fun target source -> addDictionaryInto target source
              Finish = fun counts -> counts |> dictionaryToMap |> Histogram.ofMap
              ReleaseItem = Chunk.decRef }

        Stage.parallelReduce
            $"chunkHistogramParallel.{typeof<'T>.Name}.window{windowSize}"
            windowSize
            folder
            Streaming
            (fun _ -> 0UL)
            (fun _ -> 1UL)

let histogramDenseReducer<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> () =
    let reducer _debug (input: AsyncSeq<Chunk<'T>>) =
        async {
            let mutable accumulator: DenseHistogram option = None
            for chunk in input do
                try
                    match accumulator with
                    | None ->
                        let counts = emptyDenseCounts<'T> ()
                        addDenseChunkInto<'T> counts chunk
                        accumulator <- Some counts
                    | Some target ->
                        addDenseChunkInto<'T> target chunk
                finally
                    Chunk.decRef chunk

            let counts =
                accumulator
                |> Option.map (denseToMap<'T>)
                |> Option.defaultValue Map.empty<'T, uint64>

            return Histogram.ofMap counts
        }

    Stage.reduce $"chunkHistogramDense.{typeof<'T>.Name}" reducer Streaming (fun _ -> 524288UL) (fun _ -> 1UL)

let histogramDenseReducerParallel<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    windowSize
    =
    if windowSize <= 1 then
        histogramDenseReducer<'T> ()
    else
        let folder: MonoidFolder<Chunk<'T>, DenseHistogram, Histogram<'T>> =
            { Create = emptyDenseCounts<'T>
              AddItemInto = fun counts chunk -> addDenseChunkInto<'T> counts chunk
              MergeInto = addDenseInto
              Finish =
                fun counts ->
                    counts
                    |> denseToMap<'T>
                    |> Histogram.ofMap
              ReleaseItem = Chunk.decRef }

        Stage.parallelReduce
            $"chunkHistogramDenseParallel.{typeof<'T>.Name}.window{windowSize}"
            windowSize
            folder
            Streaming
            (fun _ -> uint64 windowSize * 524288UL)
            (fun _ -> 1UL)

let histogramLeftEdgesReducer<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> leftEdges =
    let edges = validateLeftEdges leftEdges
    let reducer _debug (input: AsyncSeq<Chunk<'T>>) =
        async {
            let mutable accumulator: LeftEdgeHistogram option = None
            for chunk in input do
                try
                    let chunkCounts = histogramLeftEdgeCounts edges chunk
                    match accumulator with
                    | None -> accumulator <- Some chunkCounts
                    | Some target -> addLeftEdgesInto target chunkCounts
                finally
                    Chunk.decRef chunk

            let counts =
                accumulator
                |> Option.map leftEdgesToMap
                |> Option.defaultValue (edges |> Array.map (fun edge -> edge, 0UL) |> Map.ofArray)

            return Histogram.withFixedEdges edges[0] edges[edges.Length - 1] (uint32 edges.Length) counts
        }

    Stage.reduce $"chunkHistogramLeftEdges.{typeof<'T>.Name}" reducer Streaming (fun _ -> uint64 edges.Length * 8UL) (fun _ -> 1UL)

let histogramFixedBinsReducer<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    firstLeftEdge
    lastLeftEdge
    bins =
    if bins = 0u then invalidArg "bins" "chunkHistogramFixedBins needs at least one bin."
    let edges =
        if bins = 1u then
            [| firstLeftEdge |]
        else
            let step = (lastLeftEdge - firstLeftEdge) / float (bins - 1u)
            [| for i in 0 .. int bins - 1 -> firstLeftEdge + float i * step |]
    histogramLeftEdgesReducer<'T> edges

let histogramEqualizationDense<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    dense
    : Stage<Chunk<'T>, Chunk<'T>> =
    let mapper _debug chunk =
        try
            ChunkKernel.histogramEqualizationDense<'T> dense chunk
        finally
            Chunk.decRef chunk

    Stage.map $"chunkHistogramEqualizationDense.{typeof<'T>.Name}" mapper (fun n -> 2UL * chunkMemoryNeed<'T> n) id

let histogramEqualizationLeftEdges<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    histogram
    : Stage<Chunk<'T>, Chunk<'T>> =
    let mapper _debug chunk =
        try
            ChunkKernel.histogramEqualizationLeftEdges<'T> histogram chunk
        finally
            Chunk.decRef chunk

    Stage.map $"chunkHistogramEqualizationLeftEdges.{typeof<'T>.Name}" mapper (fun n -> 2UL * chunkMemoryNeed<'T> n) id

let histogramEqualizationSparse<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    counts
    : Stage<Chunk<'T>, Chunk<'T>> =
    let mapper _debug chunk =
        try
            ChunkKernel.histogramEqualizationSparse<'T> counts chunk
        finally
            Chunk.decRef chunk

    Stage.map $"chunkHistogramEqualizationSparse.{typeof<'T>.Name}" mapper (fun n -> 2UL * chunkMemoryNeed<'T> n) id

let histogramEqualization<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (histogram: obj)
    : Stage<Chunk<'T>, Chunk<'T>> =
    match histogram with
    | :? DenseHistogram as dense ->
        histogramEqualizationDense<'T> dense
    | :? LeftEdgeHistogram as leftEdges ->
        histogramEqualizationLeftEdges<'T> leftEdges
    | :? Histogram<'T> as exact ->
        let counts: Map<'T, uint64> = exact.Counts
        histogramEqualizationSparse<'T> counts
    | :? Map<'T, uint64> as counts ->
        histogramEqualizationSparse<'T> counts
    | null ->
        nullArg "histogram"
    | other ->
        invalidArg "histogram" $"Chunk histogram equalization expected DenseHistogram, LeftEdgeHistogram, Histogram<{typeof<'T>.Name}>, or Map<{typeof<'T>.Name}, uint64>, got {other.GetType().Name}."

let private quantilesFromPairs quantileValues (pairs: (float * uint64) seq) =
    let ordered =
        pairs
        |> Seq.filter (fun (_, count) -> count > 0UL)
        |> Seq.sortBy fst
        |> Seq.toArray

    if ordered.Length = 0 then
        invalidArg "histogram" "Cannot estimate quantiles from an empty histogram."

    let total = ordered |> Array.sumBy snd
    if total = 0UL then
        invalidArg "histogram" "Cannot estimate quantiles from a histogram with zero total count."

    quantileValues
    |> List.map (fun quantile ->
        if quantile < 0.0 || quantile > 1.0 || Double.IsNaN quantile || Double.IsInfinity quantile then
            invalidArg "quantiles" "Quantiles must be finite numbers between 0 and 1."

        let target = uint64 (ceil (quantile * float total)) |> max 1UL
        let mutable cumulative = 0UL
        ordered
        |> Array.pick (fun (value, count) ->
            cumulative <- cumulative + count
            if cumulative >= target then Some value else None))

let private denseQuantilePairs = function
    | DenseHistogram.UInt8Counts counts ->
        counts |> Seq.mapi (fun i count -> float i, count)
    | DenseHistogram.Int8Counts counts ->
        counts |> Seq.mapi (fun i count -> float (i + int SByte.MinValue), count)
    | DenseHistogram.UInt16Counts counts ->
        counts |> Seq.mapi (fun i count -> float i, count)
    | DenseHistogram.Int16Counts counts ->
        counts |> Seq.mapi (fun i count -> float (i + int Int16.MinValue), count)

let quantiles (quantileValues: float list) (histogram: obj) =
    match histogram with
    | :? DenseHistogram as dense ->
        dense |> denseQuantilePairs |> quantilesFromPairs quantileValues
    | :? LeftEdgeHistogram as leftEdges ->
        if leftEdges.LeftEdges.Length <> leftEdges.Counts.Length then
            invalidArg "histogram" $"Left-edge histogram has {leftEdges.LeftEdges.Length} edges but {leftEdges.Counts.Length} counts."
        Seq.zip leftEdges.LeftEdges leftEdges.Counts
        |> quantilesFromPairs quantileValues
    | :? Histogram<uint8> as exact ->
        exact.Counts |> Seq.map (fun pair -> float pair.Key, pair.Value) |> quantilesFromPairs quantileValues
    | :? Histogram<int8> as exact ->
        exact.Counts |> Seq.map (fun pair -> float pair.Key, pair.Value) |> quantilesFromPairs quantileValues
    | :? Histogram<uint16> as exact ->
        exact.Counts |> Seq.map (fun pair -> float pair.Key, pair.Value) |> quantilesFromPairs quantileValues
    | :? Histogram<int16> as exact ->
        exact.Counts |> Seq.map (fun pair -> float pair.Key, pair.Value) |> quantilesFromPairs quantileValues
    | :? Histogram<int32> as exact ->
        exact.Counts |> Seq.map (fun pair -> float pair.Key, pair.Value) |> quantilesFromPairs quantileValues
    | :? Histogram<uint32> as exact ->
        exact.Counts |> Seq.map (fun pair -> float pair.Key, pair.Value) |> quantilesFromPairs quantileValues
    | :? Histogram<float32> as exact ->
        exact.Counts |> Seq.map (fun pair -> float pair.Key, pair.Value) |> quantilesFromPairs quantileValues
    | :? Histogram<float> as exact ->
        exact.Counts |> Seq.map (fun pair -> pair.Key, pair.Value) |> quantilesFromPairs quantileValues
    | :? Map<uint8, uint64> as counts ->
        counts |> Seq.map (fun pair -> float pair.Key, pair.Value) |> quantilesFromPairs quantileValues
    | :? Map<int8, uint64> as counts ->
        counts |> Seq.map (fun pair -> float pair.Key, pair.Value) |> quantilesFromPairs quantileValues
    | :? Map<uint16, uint64> as counts ->
        counts |> Seq.map (fun pair -> float pair.Key, pair.Value) |> quantilesFromPairs quantileValues
    | :? Map<int16, uint64> as counts ->
        counts |> Seq.map (fun pair -> float pair.Key, pair.Value) |> quantilesFromPairs quantileValues
    | :? Map<int32, uint64> as counts ->
        counts |> Seq.map (fun pair -> float pair.Key, pair.Value) |> quantilesFromPairs quantileValues
    | :? Map<uint32, uint64> as counts ->
        counts |> Seq.map (fun pair -> float pair.Key, pair.Value) |> quantilesFromPairs quantileValues
    | :? Map<float32, uint64> as counts ->
        counts |> Seq.map (fun pair -> float pair.Key, pair.Value) |> quantilesFromPairs quantileValues
    | :? Map<float, uint64> as counts ->
        counts |> Seq.map (fun pair -> pair.Key, pair.Value) |> quantilesFromPairs quantileValues
    | null ->
        nullArg "histogram"
    | other ->
        invalidArg "histogram" $"Chunk quantiles expected DenseHistogram, LeftEdgeHistogram, Histogram<T>, or Map<T, uint64>, got {other.GetType().Name}."

let convolveFixedKernel<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> kernel =
    StackConvolve.convolveFixedKernel<'T> kernel
let convolveFixedKernelParallel<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> kernel windowSize =
    StackConvolve.convolveFixedKernelParallel<'T> kernel windowSize
let convolveFixedKernelNativeFloat32 kernel = StackConvolve.convolveFixedKernelNativeFloat32 kernel
let convolveFixedKernelNativeFloat32Parallel kernel windowSize = StackConvolve.convolveFixedKernelNativeFloat32Parallel kernel windowSize
let convolveFixedKernelNativeUInt8 kernel = StackConvolve.convolveFixedKernelNativeUInt8 kernel
let convolveFixedKernelNativeUInt8Parallel kernel windowSize = StackConvolve.convolveFixedKernelNativeUInt8Parallel kernel windowSize
let convolveFixedKernelNative<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> kernel =
    StackConvolve.convolveFixedKernelNative<'T> kernel
let convolveFixedKernelNativeParallel<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> kernel windowSize =
    StackConvolve.convolveFixedKernelNativeParallel<'T> kernel windowSize
let convolveNativeXParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> kernel workers =
    StackConvolve.convolveNativeXParallelCollect<'T> kernel workers
let convolveNativeYParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> kernel workers =
    StackConvolve.convolveNativeYParallelCollect<'T> kernel workers
let convolveNativeZParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> kernel workers =
    StackConvolve.convolveNativeZParallelCollect<'T> kernel workers
let finiteDiffKernel1D order = StackConvolve.finiteDiffKernel1D order
let finiteDiffNativeXParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> order workers =
    StackConvolve.finiteDiffNativeXParallelCollect<'T> order workers
let finiteDiffNativeYParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> order workers =
    StackConvolve.finiteDiffNativeYParallelCollect<'T> order workers
let finiteDiffNativeZParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> order workers =
    StackConvolve.finiteDiffNativeZParallelCollect<'T> order workers
let separableConvolveNativeParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> xKernel yKernel zKernel workers =
    StackConvolve.separableConvolveNativeParallelCollect<'T> xKernel yKernel zKernel workers
let boxFilterNativeParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> radius workers =
    StackConvolve.boxFilterNativeParallelCollect<'T> radius workers
let boxFilterNativeParallelCollectXYZ<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> radiusX radiusY radiusZ workers =
    StackConvolve.boxFilterNativeParallelCollectXYZ<'T> radiusX radiusY radiusZ workers
let gaussianFilterNativeParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> sigma radius workers =
    StackConvolve.gaussianFilterNativeParallelCollect<'T> sigma radius workers
let gaussianFilterNativeParallelCollectXYZ<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> sigmaX radiusX sigmaY radiusY sigmaZ radiusZ workers =
    StackConvolve.gaussianFilterNativeParallelCollectXYZ<'T> sigmaX radiusX sigmaY radiusY sigmaZ radiusZ workers
let sobelXNativeParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> workers =
    StackConvolve.sobelXNativeParallelCollect<'T> workers
let sobelYNativeParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> workers =
    StackConvolve.sobelYNativeParallelCollect<'T> workers
let sobelZNativeParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> workers =
    StackConvolve.sobelZNativeParallelCollect<'T> workers

let medianPerreaultHebertUInt8Dense radius = StackMedian.medianPerreaultHebertUInt8Dense radius
let medianPerreaultHebertUInt8DenseRolling radius = StackMedian.medianPerreaultHebertUInt8DenseRolling radius
let medianNthElementUInt16 radius = StackMedian.medianNthElementUInt16 radius
let medianNthElementInt16 radius = StackMedian.medianNthElementInt16 radius
let medianNthElementFloat32 radius = StackMedian.medianNthElementFloat32 radius
let medianQuickselectUInt8 radius = StackMedian.medianQuickselectUInt8 radius
let medianQuickselectUInt16 radius = StackMedian.medianQuickselectUInt16 radius
let medianQuickselectUInt16ParallelCollect radius workers = StackMedian.medianQuickselectUInt16ParallelCollect radius workers
let medianSortUInt16 radius = StackMedian.medianSortUInt16 radius
let medianSortUInt16ParallelCollect radius workers = StackMedian.medianSortUInt16ParallelCollect radius workers
let medianNativeNthElementUInt8 radius = StackMedian.medianNativeNthElementUInt8 radius
let medianNativeNthElementUInt8ParallelCollect radius workers = StackMedian.medianNativeNthElementUInt8ParallelCollect radius workers
let medianNativeNthElementUInt16 radius = StackMedian.medianNativeNthElementUInt16 radius
let medianNativeNthElementUInt16ParallelCollect radius workers = StackMedian.medianNativeNthElementUInt16ParallelCollect radius workers
let medianNativeNthElementInt32 radius = StackMedian.medianNativeNthElementInt32 radius
let medianNativeNthElementInt32ParallelCollect radius workers = StackMedian.medianNativeNthElementInt32ParallelCollect radius workers
let medianNativeNthElementFloat32 radius = StackMedian.medianNativeNthElementFloat32 radius
let medianNativeNthElementFloat32ParallelCollect radius workers = StackMedian.medianNativeNthElementFloat32ParallelCollect radius workers
let medianQuickselectInt16 radius = StackMedian.medianQuickselectInt16 radius
let medianPerreaultHebertUInt8DenseRollingYBands radius workers = StackMedian.medianPerreaultHebertUInt8DenseRollingYBands radius workers
let medianPerreaultHebertUInt8DenseRollingTree radius = StackMedian.medianPerreaultHebertUInt8DenseRollingTree radius
let medianPerreaultHebertUInt8DenseRollingTransposedXBlock radius = StackMedian.medianPerreaultHebertUInt8DenseRollingTransposedXBlock radius
let medianPerreaultHebertUInt8DenseRollingBlockedZ radius = StackMedian.medianPerreaultHebertUInt8DenseRollingBlockedZ radius
let medianPerreaultHebertUInt8DenseXFirstMaterialized radius = StackMedian.medianPerreaultHebertUInt8DenseXFirstMaterialized radius
let medianPerreaultHebertUInt8DenseXBlock radius = StackMedian.medianPerreaultHebertUInt8DenseXBlock radius

let binaryDilateZonohedral radius = StackBinaryMorphology.binaryDilateZonohedral radius
let binaryDilateZonohedralParallel radius windowSize = StackBinaryMorphology.binaryDilateZonohedralParallel radius windowSize
let binaryErodeZonohedral radius = StackBinaryMorphology.binaryErodeZonohedral radius
let binaryErodeZonohedralParallel radius windowSize = StackBinaryMorphology.binaryErodeZonohedralParallel radius windowSize
let binaryOpeningZonohedral radius = StackBinaryMorphology.binaryOpeningZonohedral radius
let binaryOpeningZonohedralParallel radius windowSize = StackBinaryMorphology.binaryOpeningZonohedralParallel radius windowSize
let binaryClosingZonohedral radius = StackBinaryMorphology.binaryClosingZonohedral radius
let binaryClosingZonohedralParallel radius windowSize = StackBinaryMorphology.binaryClosingZonohedralParallel radius windowSize
let binaryWhiteTopHatZonohedral radius = StackBinaryMorphology.binaryWhiteTopHatZonohedral radius
let binaryWhiteTopHatZonohedralParallel radius windowSize = StackBinaryMorphology.binaryWhiteTopHatZonohedralParallel radius windowSize
let binaryBlackTopHatZonohedral radius = StackBinaryMorphology.binaryBlackTopHatZonohedral radius
let binaryBlackTopHatZonohedralParallel radius windowSize = StackBinaryMorphology.binaryBlackTopHatZonohedralParallel radius windowSize
let binaryGradientZonohedral radius = StackBinaryMorphology.binaryGradientZonohedral radius
let binaryGradientZonohedralParallel radius windowSize = StackBinaryMorphology.binaryGradientZonohedralParallel radius windowSize
let binaryContourZonohedral fullyConnected = StackBinaryMorphology.binaryContourZonohedral fullyConnected
let binaryContourZonohedralParallel fullyConnected windowSize = StackBinaryMorphology.binaryContourZonohedralParallel fullyConnected windowSize
