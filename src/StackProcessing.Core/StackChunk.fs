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

module ChunkKernel = ChunkCore.ChunkFunctions

let private binaryBackground = 0uy
let private binaryForeground = 1uy

let chunkElementBytes<'T> =
    Marshal.SizeOf<'T>()

let chunkMemoryNeed<'T> nPixels =
    nPixels * uint64 (chunkElementBytes<'T>)

let releaseUnaryChunk name f memoryNeed : Stage<Chunk<'T>, Chunk<'U>> =
    let mapper _debug chunk =
        try
            f chunk
        finally
            Chunk.decRef chunk

    Stage.map name mapper memoryNeed id

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

let copy<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> : Stage<Chunk<'T>, Chunk<'T>> =
    releaseUnaryChunk $"chunkCopy.{typeof<'T>.Name}" ChunkKernel.copyChunk<'T> (fun n -> 2UL * chunkMemoryNeed<'T> n)

let connectedComponentsSauf3DUInt8UInt32ArrayUf () = StackConnectedComponents.connectedComponentsSauf3DUInt8UInt32ArrayUf ()
let connectedComponentsSauf3DUInt8UInt32 () = StackConnectedComponents.connectedComponentsSauf3DUInt8UInt32 ()
let connectedComponentsSauf3DUInt8UInt32ParallelCollect windowSize workers = StackConnectedComponents.connectedComponentsSauf3DUInt8UInt32ParallelCollect windowSize workers
let connectedComponentsSauf3DUInt8 () = StackConnectedComponents.connectedComponentsSauf3DUInt8 ()

let fftXYFloat32ToComplex64Interleaved = StackFFT.fftXYFloat32ToComplex64Interleaved
let fftXYFloat32ToComplex64InterleavedParallelCollect workers = StackFFT.fftXYFloat32ToComplex64InterleavedParallelCollect workers

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

let inline addScalar value =
    map $"chunkAddScalar.{typeof<'T>.Name}" (fun (x: 'T) -> x + value)

let inline scalarAdd value = addScalar value

let inline subScalar value =
    map $"chunkSubScalar.{typeof<'T>.Name}" (fun (x: 'T) -> x - value)

let inline scalarSub value =
    map $"chunkScalarSub.{typeof<'T>.Name}" (fun (x: 'T) -> value - x)

let inline mulScalar value =
    map $"chunkMulScalar.{typeof<'T>.Name}" (fun (x: 'T) -> x * value)

let inline scalarMul value = mulScalar value

let inline divScalar value =
    map $"chunkDivScalar.{typeof<'T>.Name}" (fun (x: 'T) -> x / value)

let inline scalarDiv value =
    map $"chunkScalarDiv.{typeof<'T>.Name}" (fun (x: 'T) -> value / x)

let inline add<'T when 'T: equality
                    and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType
                    and 'T: (static member ( + ) : 'T * 'T -> 'T)> =
    map2< 'T, 'T, 'T> $"chunkAdd.{typeof<'T>.Name}" (fun a b -> a + b)

let inline subtract<'T when 'T: equality
                         and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType
                         and 'T: (static member ( - ) : 'T * 'T -> 'T)> =
    map2< 'T, 'T, 'T> $"chunkSubtract.{typeof<'T>.Name}" (fun a b -> a - b)

let inline multiply<'T when 'T: equality
                         and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType
                         and 'T: (static member ( * ) : 'T * 'T -> 'T)> =
    map2< 'T, 'T, 'T> $"chunkMultiply.{typeof<'T>.Name}" (fun a b -> a * b)

let inline divide<'T when 'T: equality
                       and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType
                       and 'T: (static member ( / ) : 'T * 'T -> 'T)> =
    map2< 'T, 'T, 'T> $"chunkDivide.{typeof<'T>.Name}" (fun a b -> a / b)

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

let private mapFloat32Vector name (scalarOp: float32 -> float32) (vectorOp: Vector<float32> -> Vector<float32>) (chunk: Chunk<float32>) =
    ChunkKernel.mapFloat32Vector name scalarOp vectorOp chunk

let private float32UnaryStage name (scalarOp: float32 -> float32) (vectorOp: Vector<float32> -> Vector<float32>) =
    releaseUnaryChunk name (mapFloat32Vector name scalarOp vectorOp) (fun n -> 2UL * chunkMemoryNeed<float32> n)

let absFloat32 : Stage<Chunk<float32>, Chunk<float32>> =
    float32UnaryStage "chunkAbsFloat32" abs (fun v -> Vector.Abs(v))

let sqrtFloat32 : Stage<Chunk<float32>, Chunk<float32>> =
    float32UnaryStage "chunkSqrtFloat32" sqrt (fun v -> Vector.SquareRoot(v))

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

let intensityWindowFloat32 (windowMinimum: double) (windowMaximum: double) (outputMinimum: double) (outputMaximum: double) : Stage<Chunk<float32>, Chunk<float32>> =
    if windowMaximum = windowMinimum then
        invalidArg "windowMaximum" "ChunkFunctions.intensityWindowFloat32 requires a non-zero input window width."
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

    float32UnaryStage $"chunkIntensityWindowFloat32.{windowMinimum}.{windowMaximum}.{outputMinimum}.{outputMaximum}" scalar vector

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
            castChunkToUInt8 chunk
        finally
            Chunk.decRef chunk

    Stage.map $"chunkCastToUInt8.{typeof<'T>.Name}" mapper id id

let private castChunkToFloat32<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (chunk: Chunk<'T>) =
    ChunkKernel.castChunkToFloat32 chunk

let castToFloat32<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> : Stage<Chunk<'T>, Chunk<float32>> =
    let mapper _debug chunk =
        try
            castChunkToFloat32 chunk
        finally
            Chunk.decRef chunk

    Stage.map $"chunkCastToFloat32.{typeof<'T>.Name}" mapper id id

let private castFloat32ToChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (chunk: Chunk<float32>) =
    ChunkKernel.castFloat32ToChunk<'T> chunk

let castFromFloat32<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> : Stage<Chunk<float32>, Chunk<'T>> =
    let mapper _debug chunk =
        try
            castFloat32ToChunk<'T> chunk
        finally
            Chunk.decRef chunk

    Stage.map $"chunkCastFromFloat32.{typeof<'T>.Name}" mapper id id

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

let intensityWindow<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    windowMinimum
    windowMaximum
    outputMinimum
    outputMaximum
    : Stage<Chunk<'T>, Chunk<'T>> =
    if typeof<'T> = typeof<float32> then
        unbox (box (intensityWindowFloat32 windowMinimum windowMaximum outputMinimum outputMaximum))
    else
        Stage.compose (castToFloat32<'T>) (Stage.compose (intensityWindowFloat32 windowMinimum windowMaximum outputMinimum outputMaximum) (castFromFloat32<'T>))

let invertIntensity<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> maximum : Stage<Chunk<'T>, Chunk<'T>> =
    if typeof<'T> = typeof<float32> then
        unbox (box (invertIntensityFloat32 maximum))
    else
        Stage.compose (castToFloat32<'T>) (Stage.compose (invertIntensityFloat32 maximum) (castFromFloat32<'T>))

type DenseHistogram = ChunkKernel.DenseHistogram
type LeftEdgeHistogram = ChunkKernel.LeftEdgeHistogram

let addCountsInto target source = ChunkKernel.addCountsInto target source
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

let convolveFixedKernel<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> kernel =
    StackConvolve.convolveFixedKernel<'T> kernel
let convolveFixedKernelParallel<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> kernel windowSize =
    StackConvolve.convolveFixedKernelParallel<'T> kernel windowSize
let convolveFixedKernelNativeFloat32 kernel = StackConvolve.convolveFixedKernelNativeFloat32 kernel
let convolveFixedKernelNativeFloat32Parallel kernel windowSize = StackConvolve.convolveFixedKernelNativeFloat32Parallel kernel windowSize
let convolveFixedKernelNativeUInt8 kernel = StackConvolve.convolveFixedKernelNativeUInt8 kernel
let convolveFixedKernelNativeUInt8Parallel kernel windowSize = StackConvolve.convolveFixedKernelNativeUInt8Parallel kernel windowSize
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
let medianItkWrappedParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> radius workers =
    StackMedian.medianItkWrappedParallelCollect<'T> radius workers
let medianItkWrapped<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> radius =
    StackMedian.medianItkWrapped<'T> radius
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
