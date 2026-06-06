module ChunkFunctions

open System
open System.Collections.Generic
open System.Numerics
open System.Runtime.InteropServices
open FSharp.Control
open SlimPipeline
open StackCore

type DenseHistogram =
    | UInt8Counts of uint64[]
    | Int8Counts of uint64[]
    | UInt16Counts of uint64[]
    | Int16Counts of uint64[]

type LeftEdgeHistogram =
    { LeftEdges: float[]
      Counts: uint64[] }

let addCountsInto (target: uint64[]) (source: uint64[]) =
    if target.Length <> source.Length then
        invalidArg "source" $"Cannot add count arrays with different lengths: target has {target.Length}, source has {source.Length}."

    let width = Vector<uint64>.Count
    let vectorEnd = target.Length - (target.Length % width)
    let mutable i = 0
    while i < vectorEnd do
        let targetVector = Vector<uint64>(target.AsSpan(i, width))
        let sourceVector = Vector<uint64>(source.AsSpan(i, width))
        (targetVector + sourceVector).CopyTo(target.AsSpan(i, width))
        i <- i + width
    while i < target.Length do
        target[i] <- target[i] + source[i]
        i <- i + 1

let histogramDictionaryBytes<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (bytes: byte[])
    byteLength
    =
    let values = MemoryMarshal.Cast<byte, 'T>(bytes.AsSpan(0, byteLength))
    let counts = Dictionary<'T, uint64>()
    let mutable i = 0
    while i < values.Length do
        let mutable exists = false
        let count = &CollectionsMarshal.GetValueRefOrAddDefault(counts, values[i], &exists)
        count <- count + 1UL
        i <- i + 1
    counts

let addDictionaryInto<'T when 'T: equality> (target: Dictionary<'T, uint64>) (source: Dictionary<'T, uint64>) =
    for pair in source do
        let mutable exists = false
        let count = &CollectionsMarshal.GetValueRefOrAddDefault(target, pair.Key, &exists)
        count <- count + pair.Value

let dictionaryToMap<'T when 'T: comparison> (counts: Dictionary<'T, uint64>) =
    counts
    |> Seq.map (fun pair -> pair.Key, pair.Value)
    |> Map.ofSeq

let histogramBytes<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (bytes: byte[])
    byteLength
    =
    histogramDictionaryBytes<'T> bytes byteLength
    |> dictionaryToMap

let private denseHistogramMap<'T when 'T: comparison> (counts: uint64[]) (keyOfIndex: int -> 'T) =
    let mutable histogram = Map.empty<'T, uint64>
    for index in 0 .. counts.Length - 1 do
        let count = counts[index]
        if count <> 0UL then
            histogram <- histogram.Add(keyOfIndex index, count)
    histogram

let private denseCountsFromBytes<'Raw when 'Raw: (new: unit -> 'Raw) and 'Raw: struct and 'Raw :> ValueType>
    (bytes: byte[])
    byteLength
    binCount
    (indexOf: 'Raw -> int)
    =
    let values = MemoryMarshal.Cast<byte, 'Raw>(bytes.AsSpan(0, byteLength))
    let counts = Array.zeroCreate<uint64> binCount
    let mutable i = 0
    while i < values.Length do
        let index = indexOf values[i]
        counts[index] <- counts[index] + 1UL
        i <- i + 1
    counts

let private addDenseCountsFromBytesInto<'Raw when 'Raw: (new: unit -> 'Raw) and 'Raw: struct and 'Raw :> ValueType>
    (counts: uint64[])
    (bytes: byte[])
    byteLength
    (indexOf: 'Raw -> int)
    =
    let values = MemoryMarshal.Cast<byte, 'Raw>(bytes.AsSpan(0, byteLength))
    let mutable i = 0
    while i < values.Length do
        let index = indexOf values[i]
        counts[index] <- counts[index] + 1UL
        i <- i + 1

let histogramDenseCountsBytes<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (bytes: byte[])
    byteLength
    =
    let t = typeof<'T>
    if t = typeof<uint8> then
        UInt8Counts(denseCountsFromBytes<uint8> bytes byteLength 256 int)
    elif t = typeof<int8> then
        let offset = -int SByte.MinValue
        Int8Counts(denseCountsFromBytes<int8> bytes byteLength 256 (fun value -> int value + offset))
    elif t = typeof<uint16> then
        UInt16Counts(denseCountsFromBytes<uint16> bytes byteLength 65536 int)
    elif t = typeof<int16> then
        let offset = -int Int16.MinValue
        Int16Counts(denseCountsFromBytes<int16> bytes byteLength 65536 (fun value -> int value + offset))
    else
        invalidArg "T" $"ChunkFunctions.histogramDense supports only 8-bit and 16-bit integer chunks, but got {t.Name}. Use ChunkFunctions.histogram or a binned histogram instead."

let addDenseInto target source =
    match target, source with
    | UInt8Counts targetCounts, UInt8Counts sourceCounts
    | Int8Counts targetCounts, Int8Counts sourceCounts
    | UInt16Counts targetCounts, UInt16Counts sourceCounts
    | Int16Counts targetCounts, Int16Counts sourceCounts ->
        addCountsInto targetCounts sourceCounts
    | _ ->
        invalidArg "source" "Cannot add dense histograms with different integer domains."

let emptyDenseCounts<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> () =
    let t = typeof<'T>
    if t = typeof<uint8> then
        UInt8Counts(Array.zeroCreate<uint64> 256)
    elif t = typeof<int8> then
        Int8Counts(Array.zeroCreate<uint64> 256)
    elif t = typeof<uint16> then
        UInt16Counts(Array.zeroCreate<uint64> 65536)
    elif t = typeof<int16> then
        Int16Counts(Array.zeroCreate<uint64> 65536)
    else
        invalidArg "T" $"ChunkFunctions.histogramDense supports only 8-bit and 16-bit integer chunks, but got {t.Name}. Use ChunkFunctions.histogram or a binned histogram instead."

let addDenseChunkInto<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> target (chunk: Chunk<'T>) =
    let t = typeof<'T>
    match target with
    | UInt8Counts counts when t = typeof<uint8> ->
        addDenseCountsFromBytesInto<uint8> counts chunk.Bytes chunk.ByteLength int
    | Int8Counts counts when t = typeof<int8> ->
        let offset = -int SByte.MinValue
        addDenseCountsFromBytesInto<int8> counts chunk.Bytes chunk.ByteLength (fun value -> int value + offset)
    | UInt16Counts counts when t = typeof<uint16> ->
        addDenseCountsFromBytesInto<uint16> counts chunk.Bytes chunk.ByteLength int
    | Int16Counts counts when t = typeof<int16> ->
        let offset = -int Int16.MinValue
        addDenseCountsFromBytesInto<int16> counts chunk.Bytes chunk.ByteLength (fun value -> int value + offset)
    | _ ->
        invalidArg "target" $"Dense histogram accumulator does not match chunk pixel type {t.Name}."

let denseToMap<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> dense =
    let t = typeof<'T>
    if t = typeof<uint8> then
        match dense with
        | UInt8Counts counts -> denseHistogramMap counts (fun index -> box (uint8 index) :?> 'T)
        | _ -> invalidArg "dense" $"Expected UInt8 dense histogram for {t.Name} map conversion."
    elif t = typeof<int8> then
        match dense with
        | Int8Counts counts ->
            let offset = -int SByte.MinValue
            denseHistogramMap counts (fun index -> box (int8 (index - offset)) :?> 'T)
        | _ -> invalidArg "dense" $"Expected Int8 dense histogram for {t.Name} map conversion."
    elif t = typeof<uint16> then
        match dense with
        | UInt16Counts counts -> denseHistogramMap counts (fun index -> box (uint16 index) :?> 'T)
        | _ -> invalidArg "dense" $"Expected UInt16 dense histogram for {t.Name} map conversion."
    elif t = typeof<int16> then
        match dense with
        | Int16Counts counts ->
            let offset = -int Int16.MinValue
            denseHistogramMap counts (fun index -> box (int16 (index - offset)) :?> 'T)
        | _ -> invalidArg "dense" $"Expected Int16 dense histogram for {t.Name} map conversion."
    else
        invalidArg "T" $"ChunkFunctions.denseToMap supports only 8-bit and 16-bit integer chunks, but got {t.Name}."

let histogramDenseBytes<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (bytes: byte[])
    byteLength
    =
    histogramDenseCountsBytes<'T> bytes byteLength
    |> denseToMap<'T>

let private leftEdgeBin (edges: float[]) value =
    let search = Array.BinarySearch(edges, value)
    if search >= 0 then
        search
    else
        let insertion = ~~~search
        if insertion = 0 then 0
        elif insertion >= edges.Length then edges.Length - 1
            else insertion - 1

let private histogramLeftEdgesFromBytes<'Raw when 'Raw: (new: unit -> 'Raw) and 'Raw: struct and 'Raw :> ValueType>
    (edges: float[])
    (bytes: byte[])
    byteLength
    (toFloat: 'Raw -> float)
    =
    let values = MemoryMarshal.Cast<byte, 'Raw>(bytes.AsSpan(0, byteLength))
    let counts = Array.zeroCreate<uint64> edges.Length
    let mutable i = 0
    while i < values.Length do
        let value = toFloat values[i]
        if not (Double.IsNaN value || Double.IsInfinity value) then
            let bin = leftEdgeBin edges value
            counts[bin] <- counts[bin] + 1UL
        i <- i + 1
    counts

let validateLeftEdges (leftEdges: float seq) =
    let edges = leftEdges |> Seq.toArray
    if edges.Length = 0 then
        invalidArg "leftEdges" "Histogram left edges must contain at least one edge."

    for i in 0 .. edges.Length - 1 do
        if Double.IsNaN edges[i] || Double.IsInfinity edges[i] then
            invalidArg "leftEdges" "Histogram left edges must be finite values."
        if i > 0 && edges[i] <= edges[i - 1] then
            invalidArg "leftEdges" "Histogram left edges must be strictly increasing."
    edges

let histogramLeftEdgeCountsBytes<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (leftEdges: float seq)
    (bytes: byte[])
    byteLength
    =
    let edges = validateLeftEdges leftEdges
    let t = typeof<'T>
    let counts =
        if t = typeof<uint8> then
            histogramLeftEdgesFromBytes<uint8> edges bytes byteLength float
        elif t = typeof<int8> then
            histogramLeftEdgesFromBytes<int8> edges bytes byteLength float
        elif t = typeof<uint16> then
            histogramLeftEdgesFromBytes<uint16> edges bytes byteLength float
        elif t = typeof<int16> then
            histogramLeftEdgesFromBytes<int16> edges bytes byteLength float
        elif t = typeof<uint32> then
            histogramLeftEdgesFromBytes<uint32> edges bytes byteLength float
        elif t = typeof<int32> then
            histogramLeftEdgesFromBytes<int32> edges bytes byteLength float
        elif t = typeof<float32> then
            histogramLeftEdgesFromBytes<float32> edges bytes byteLength float
        elif t = typeof<float> then
            histogramLeftEdgesFromBytes<float> edges bytes byteLength id
        else
            let values = MemoryMarshal.Cast<byte, 'T>(bytes.AsSpan(0, byteLength))
            let counts = Array.zeroCreate<uint64> edges.Length
            let mutable i = 0
            while i < values.Length do
                let value = Convert.ToDouble(box values[i])
                if not (Double.IsNaN value || Double.IsInfinity value) then
                    let bin = leftEdgeBin edges value
                    counts[bin] <- counts[bin] + 1UL
                i <- i + 1
            counts

    { LeftEdges = edges
      Counts = counts }

let private leftEdgesEqual (left: float[]) (right: float[]) =
    if left.Length <> right.Length then
        false
    else
        let mutable equal = true
        let mutable i = 0
        while equal && i < left.Length do
            equal <- left[i] = right[i]
            i <- i + 1
        equal

let addLeftEdgesInto target source =
    if not (leftEdgesEqual target.LeftEdges source.LeftEdges) then
        invalidArg "source" "Cannot add left-edge histograms with different bin edges."
    addCountsInto target.Counts source.Counts

let leftEdgesToMap leftEdgeHistogram =
    let mutable output = Map.empty<float, uint64>
    for i in 0 .. leftEdgeHistogram.LeftEdges.Length - 1 do
        output <- output.Add(leftEdgeHistogram.LeftEdges[i], leftEdgeHistogram.Counts[i])
    output

let histogramLeftEdgesBytes<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (leftEdges: float seq)
    (bytes: byte[])
    byteLength
    =
    histogramLeftEdgeCountsBytes<'T> leftEdges bytes byteLength
    |> leftEdgesToMap

let histogramDictionary<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (chunk: Chunk<'T>)
    =
    histogramDictionaryBytes<'T> chunk.Bytes chunk.ByteLength

let addChunkIntoDictionary<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (counts: Dictionary<'T, uint64>)
    (chunk: Chunk<'T>)
    =
    let values = MemoryMarshal.Cast<byte, 'T>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
    let mutable i = 0
    while i < values.Length do
        let mutable exists = false
        let count = &CollectionsMarshal.GetValueRefOrAddDefault(counts, values[i], &exists)
        count <- count + 1UL
        i <- i + 1

let histogram<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (chunk: Chunk<'T>)
    =
    histogramBytes<'T> chunk.Bytes chunk.ByteLength

let histogramDenseCounts<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (chunk: Chunk<'T>)
    =
    histogramDenseCountsBytes<'T> chunk.Bytes chunk.ByteLength

let histogramDense<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (chunk: Chunk<'T>)
    =
    histogramDenseBytes<'T> chunk.Bytes chunk.ByteLength

let histogramLeftEdgeCounts<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    leftEdges
    (chunk: Chunk<'T>)
    =
    histogramLeftEdgeCountsBytes<'T> leftEdges chunk.Bytes chunk.ByteLength

let histogramLeftEdges<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    leftEdges
    (chunk: Chunk<'T>)
    =
    histogramLeftEdgesBytes<'T> leftEdges chunk.Bytes chunk.ByteLength

let histogramReducer<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> () =
    let reducer _debug (input: AsyncSeq<Chunk<'T>>) =
        async {
            let counts = Dictionary<'T, uint64>()
            for chunk in input do
                try
                    let chunkCounts = histogramDictionary chunk
                    addDictionaryInto counts chunkCounts
                finally
                    Chunk.decRef chunk
            return counts |> dictionaryToMap |> Histogram.ofMap
        }

    Stage.reduce $"chunkHistogram.{typeof<'T>.Name}" reducer Streaming (fun _ -> 0UL) (fun _ -> 1UL)

let histogramReducerParallel<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    windowSize
    =
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
                    let chunkCounts = histogramDenseCounts chunk
                    match accumulator with
                    | None -> accumulator <- Some chunkCounts
                    | Some target -> addDenseInto target chunkCounts
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
