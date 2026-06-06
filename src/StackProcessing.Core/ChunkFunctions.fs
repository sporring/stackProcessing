module ChunkFunctions

open System
open System.Collections.Generic
open System.Numerics
open System.Runtime.InteropServices
open System.Threading.Tasks
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

type private ChunkSlice =
    { Index: int
      Chunk: Chunk<uint8> }

let private binaryBackground = 0uy
let private binaryForeground = 1uy

let private flatIndex2 width x y =
    y * width + x

let private lineHalo dz length =
    let left = length - length / 2 - 1
    let right = length / 2
    let a = -left * dz
    let b = right * dz
    max 0 (-min a b), max 0 (max a b)

let private clearChunk (chunk: Chunk<uint8>) =
    chunk.Bytes.AsSpan(0, chunk.ByteLength).Clear()

let private zeroChunk width height =
    let chunk = Chunk.create<uint8> (uint64 width, uint64 height, 1UL)
    clearChunk chunk
    chunk

let private validateSliceChunk width height (chunk: Chunk<uint8>) =
    let chunkWidth, chunkHeight, chunkDepth = chunk.Size
    if chunkDepth <> 1UL then
        invalidArg "chunk" $"Chunk binary dilation expects 2D slice chunks with depth 1, got {chunk.Size}."
    if chunkWidth <> uint64 width || chunkHeight <> uint64 height then
        invalidArg "chunk" $"Chunk binary dilation expects chunks of size {width}x{height}x1, got {chunk.Size}."

let private orSpanIntoRange (target: Span<byte>) targetStart (source: ReadOnlySpan<byte>) sourceStart count =
    let vectorWidth = Vector<byte>.Count
    let vectorEnd = count - (count % vectorWidth)
    let mutable i = 0
    while i < vectorEnd do
        let targetSlice: ReadOnlySpan<byte> = target.Slice(targetStart + i, vectorWidth)
        let sourceSlice: ReadOnlySpan<byte> = source.Slice(sourceStart + i, vectorWidth)
        let destination: Span<byte> = target.Slice(targetStart + i, vectorWidth)
        let targetVector = MemoryMarshal.Read<Vector<byte>>(targetSlice)
        let sourceVector = MemoryMarshal.Read<Vector<byte>>(sourceSlice)
        let mutable result: Vector<byte> = Vector.BitwiseOr(targetVector, sourceVector)
        MemoryMarshal.Write(destination, &result)
        i <- i + vectorWidth
    while i < count do
        target[targetStart + i] <- target[targetStart + i] ||| source[sourceStart + i]
        i <- i + 1

let private andSpanIntoRange (target: Span<byte>) targetStart (source: ReadOnlySpan<byte>) sourceStart count =
    let vectorWidth = Vector<byte>.Count
    let vectorEnd = count - (count % vectorWidth)
    let mutable i = 0
    while i < vectorEnd do
        let targetSlice: ReadOnlySpan<byte> = target.Slice(targetStart + i, vectorWidth)
        let sourceSlice: ReadOnlySpan<byte> = source.Slice(sourceStart + i, vectorWidth)
        let destination: Span<byte> = target.Slice(targetStart + i, vectorWidth)
        let targetVector = MemoryMarshal.Read<Vector<byte>>(targetSlice)
        let sourceVector = MemoryMarshal.Read<Vector<byte>>(sourceSlice)
        let mutable result: Vector<byte> = Vector.BitwiseAnd(targetVector, sourceVector)
        MemoryMarshal.Write(destination, &result)
        i <- i + vectorWidth
    while i < count do
        target[targetStart + i] <- target[targetStart + i] &&& source[sourceStart + i]
        i <- i + 1

let private tryDilateLineChunkSliceSimd width height (window: ChunkSlice[]) center dx dy dz left right (outputPixels: Span<byte>) =
    if dy = 0 && dz = 0 && abs dx = 1 then
        for y in 0 .. height - 1 do
            let row = y * width
            let mutable t = -left
            while t <= right do
                let shift = t * dx
                let xStart = max 0 (-shift)
                let xStop = min width (width - shift)
                if xStop > xStart then
                    let inputPixels = Chunk.span<uint8> window[center].Chunk
                    orSpanIntoRange outputPixels (row + xStart) inputPixels (row + xStart + shift) (xStop - xStart)
                t <- t + 1
        true
    elif dx = 0 && dz = 0 && abs dy = 1 then
        let mutable t = -left
        while t <= right do
            let shift = t * dy
            let yStart = max 0 (-shift)
            let yStop = min height (height - shift)
            if yStop > yStart then
                let inputPixels = Chunk.span<uint8> window[center].Chunk
                for y in yStart .. yStop - 1 do
                    orSpanIntoRange outputPixels (y * width) inputPixels ((y + shift) * width) width
            t <- t + 1
        true
    elif dx = 0 && dy = 0 && abs dz = 1 then
        let mutable t = -left
        while t <= right do
            let zz = center + t * dz
            if zz >= 0 && zz < window.Length then
                let inputPixels = Chunk.span<uint8> window[zz].Chunk
                orSpanIntoRange outputPixels 0 inputPixels 0 outputPixels.Length
            t <- t + 1
        true
    else
        false

let private tryErodeLineChunkSliceSimd width height (window: ChunkSlice[]) center dx dy dz left right (outputPixels: Span<byte>) =
    if dy = 0 && dz = 0 && abs dx = 1 then
        let mutable xStart = 0
        let mutable xStop = width
        let mutable t = -left
        while t <= right do
            let shift = t * dx
            xStart <- max xStart (max 0 (-shift))
            xStop <- min xStop (min width (width - shift))
            t <- t + 1

        if xStop > xStart then
            for y in 0 .. height - 1 do
                outputPixels.Slice(y * width + xStart, xStop - xStart).Fill(binaryForeground)

            let inputPixels = Chunk.span<uint8> window[center].Chunk
            t <- -left
            while t <= right do
                let shift = t * dx
                for y in 0 .. height - 1 do
                    let row = y * width
                    andSpanIntoRange outputPixels (row + xStart) inputPixels (row + xStart + shift) (xStop - xStart)
                t <- t + 1
        true
    elif dx = 0 && dz = 0 && abs dy = 1 then
        let mutable yStart = 0
        let mutable yStop = height
        let mutable t = -left
        while t <= right do
            let shift = t * dy
            yStart <- max yStart (max 0 (-shift))
            yStop <- min yStop (min height (height - shift))
            t <- t + 1

        if yStop > yStart then
            for y in yStart .. yStop - 1 do
                outputPixels.Slice(y * width, width).Fill(binaryForeground)

            let inputPixels = Chunk.span<uint8> window[center].Chunk
            t <- -left
            while t <= right do
                let shift = t * dy
                for y in yStart .. yStop - 1 do
                    andSpanIntoRange outputPixels (y * width) inputPixels ((y + shift) * width) width
                t <- t + 1
        true
    elif dx = 0 && dy = 0 && abs dz = 1 then
        outputPixels.Fill(binaryForeground)
        let mutable t = -left
        let mutable valid = true
        while valid && t <= right do
            let zz = center + t * dz
            if zz < 0 || zz >= window.Length then
                outputPixels.Clear()
                valid <- false
            else
                let inputPixels = Chunk.span<uint8> window[zz].Chunk
                andSpanIntoRange outputPixels 0 inputPixels 0 outputPixels.Length
            t <- t + 1
        true
    else
        false

let private dilateLineChunkSlice width height (window: ChunkSlice[]) center dx dy dz length =
    let output = Chunk.create<uint8> (uint64 width, uint64 height, 1UL)
    try
        clearChunk output
        let outputPixels = Chunk.span<uint8> output
        let left = length - length / 2 - 1
        let right = length / 2

        if tryDilateLineChunkSliceSimd width height window center dx dy dz left right outputPixels then
            output
        else

            for y in 0 .. height - 1 do
                let row = y * width
                for x in 0 .. width - 1 do
                    let mutable found = false
                    let mutable t = -left
                    while not found && t <= right do
                        let xx = x + t * dx
                        let yy = y + t * dy
                        let zz = center + t * dz
                        if xx >= 0 && xx < width && yy >= 0 && yy < height && zz >= 0 && zz < window.Length then
                            let inputPixels = Chunk.span<uint8> window[zz].Chunk
                            if inputPixels[flatIndex2 width xx yy] = binaryForeground then
                                found <- true
                        t <- t + 1
                    if found then
                        outputPixels[row + x] <- binaryForeground

            output
    with
    | _ ->
        Chunk.decRef output
        reraise()

let private erodeLineChunkSlice width height (window: ChunkSlice[]) center dx dy dz length =
    let output = Chunk.create<uint8> (uint64 width, uint64 height, 1UL)
    try
        clearChunk output
        let outputPixels = Chunk.span<uint8> output
        let left = length - length / 2 - 1
        let right = length / 2

        if tryErodeLineChunkSliceSimd width height window center dx dy dz left right outputPixels then
            output
        else

            for y in 0 .. height - 1 do
                let row = y * width
                for x in 0 .. width - 1 do
                    let mutable inside = true
                    let mutable t = -left
                    while inside && t <= right do
                        let xx = x + t * dx
                        let yy = y + t * dy
                        let zz = center + t * dz
                        if xx < 0 || xx >= width || yy < 0 || yy >= height || zz < 0 || zz >= window.Length then
                            inside <- false
                        else
                            let inputPixels = Chunk.span<uint8> window[zz].Chunk
                            if inputPixels[flatIndex2 width xx yy] <> binaryForeground then
                                inside <- false
                        t <- t + 1
                    if inside then
                        outputPixels[row + x] <- binaryForeground

            output
    with
    | _ ->
        Chunk.decRef output
        reraise()

let private retainWindow (queue: ResizeArray<ChunkSlice>) start length =
    let window = Array.zeroCreate<ChunkSlice> length
    let mutable retained = 0
    try
        for i in 0 .. length - 1 do
            let item = queue[start + i]
            Chunk.incRef item.Chunk |> ignore
            window[i] <- item
            retained <- retained + 1
        window
    with
    | _ ->
        for i in 0 .. retained - 1 do
            Chunk.decRef window[i].Chunk
        reraise()

let private releaseWindow (window: ChunkSlice[]) =
    for item in window do
        Chunk.decRef item.Chunk

let private chunkZonohedralLineStage
    operationName
    operatorName
    lineOperator
    radius
    batchSize
    (lineIndex: int)
    (dx: int, dy: int, dz: int, length: int)
    =
    if batchSize < 1 then
        invalidArg "batchSize" $"Chunk zonohedral {operationName} expects a positive batch size, got {batchSize}."

    let prePad, postPad = lineHalo dz length
    let lineWindowLength = prePad + 1 + postPad
    let memoryNeed nPixels =
        uint64 (lineWindowLength + batchSize) * nPixels

    let transition = ProfileTransition.create Streaming Streaming
    let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed
    let suffix = if batchSize = 1 then "" else $".parallel{batchSize}"

    let apply (_debug: bool) (input: AsyncSeq<Chunk<uint8>>) =
        asyncSeq {
            let queue = ResizeArray<ChunkSlice>()
            let mutable width = 0
            let mutable height = 0
            let mutable initialized = false
            let mutable realCount = 0
            let mutable emittedCount = 0
            let mutable lastIndex = -1

            let addPadding index =
                let chunk = zeroChunk width height
                queue.Add({ Index = index; Chunk = chunk })

            let ensureInitialized (chunk: Chunk<uint8>) =
                let chunkWidth, chunkHeight, chunkDepth = chunk.Size
                if chunkDepth <> 1UL then
                    invalidArg "chunk" $"Chunk zonohedral {operationName} expects 2D slice chunks with depth 1, got {chunk.Size}."

                if not initialized then
                    width <- int chunkWidth
                    height <- int chunkHeight
                    if width <= 0 || height <= 0 then
                        invalidArg "chunk" $"Chunk zonohedral {operationName} expects positive slice dimensions, got {chunk.Size}."
                    for i in prePad .. -1 .. 1 do
                        addPadding -i
                    initialized <- true
                else
                    validateSliceChunk width height chunk

            let releasePrefix count =
                for _ in 1 .. count do
                    let removed = queue[0]
                    queue.RemoveAt(0)
                    Chunk.decRef removed.Chunk

            let tryProcessBatch draining =
                if initialized then
                    let remainingRealOutputs = realCount - emittedCount
                    let availableWindows = queue.Count - lineWindowLength + 1
                    let availableOutputs = min remainingRealOutputs availableWindows
                    let batchCount =
                        if draining then
                            min batchSize availableOutputs
                        elif availableOutputs >= batchSize then
                            batchSize
                        else
                            0
                    if batchCount > 0 then
                        let windows = Array.zeroCreate<ChunkSlice[]> batchCount
                        let releasedWindows = Array.zeroCreate<bool> batchCount
                        let outputs = Array.zeroCreate<Chunk<uint8>> batchCount
                        try
                            for i in 0 .. batchCount - 1 do
                                windows[i] <- retainWindow queue i lineWindowLength

                            if batchCount = 1 then
                                try
                                    outputs[0] <- lineOperator width height windows[0] prePad dx dy dz length
                                finally
                                    releaseWindow windows[0]
                                    releasedWindows[0] <- true
                            else
                                Parallel.For(
                                    0,
                                    batchCount,
                                    fun i ->
                                        try
                                            outputs[i] <- lineOperator width height windows[i] prePad dx dy dz length
                                        finally
                                            releaseWindow windows[i]
                                            releasedWindows[i] <- true
                                )
                                |> ignore

                            releasePrefix batchCount
                            emittedCount <- emittedCount + batchCount
                            Some outputs
                        with
                        | _ ->
                            for i in 0 .. windows.Length - 1 do
                                if not releasedWindows[i] && not (isNull (box windows[i])) then
                                    releaseWindow windows[i]
                            for i in 0 .. outputs.Length - 1 do
                                if not (isNull (box outputs[i])) then
                                    Chunk.decRef outputs[i]
                            reraise()
                    else
                        None
                else
                    None

            let emitAvailable () =
                seq {
                    let mutable keepGoing = true
                    while keepGoing do
                        match tryProcessBatch false with
                        | Some outputs ->
                            for output in outputs do
                                yield output
                        | None ->
                            keepGoing <- false
                }

            let drainAvailable () =
                seq {
                    let mutable keepGoing = true
                    while keepGoing do
                        match tryProcessBatch true with
                        | Some outputs ->
                            for output in outputs do
                                yield output
                        | None ->
                            keepGoing <- false
                }

            try
                for chunk in input do
                    ensureInitialized chunk
                    queue.Add({ Index = realCount; Chunk = chunk })
                    lastIndex <- realCount
                    realCount <- realCount + 1

                    for output in emitAvailable () do
                        yield output

                if initialized then
                    for i in 1 .. postPad do
                        addPadding (lastIndex + i)
                        for output in emitAvailable () do
                            yield output
                    for output in drainAvailable () do
                        yield output
            finally
                for item in queue do
                    Chunk.decRef item.Chunk
                queue.Clear()
        }

    Stage.fromAsyncSeq
        $"chunkBinary{operatorName}Zonohedral{suffix}.line{lineIndex}"
        apply
        transition
        memoryModel
        id

let binaryDilateZonohedral radius : Stage<Chunk<uint8>, Chunk<uint8>> =
    let lines = ImageFunctions.zonohedralBestLines radius
    if lines.Length = 0 then
        Stage.map "chunkBinaryDilateZonohedral.identity" (fun _ chunk -> chunk) id id
    else
        lines
        |> Array.mapi (chunkZonohedralLineStage "dilation" "Dilate" dilateLineChunkSlice radius 1)
        |> Array.fold Stage.compose (Stage.map "chunkBinaryDilateZonohedral.start" (fun _ chunk -> chunk) id id)

let binaryDilateZonohedralParallel radius windowSize : Stage<Chunk<uint8>, Chunk<uint8>> =
    let lines = ImageFunctions.zonohedralBestLines radius
    if lines.Length = 0 then
        Stage.map "chunkBinaryDilateZonohedralParallel.identity" (fun _ chunk -> chunk) id id
    elif windowSize <= 1 then
        binaryDilateZonohedral radius
    else
        lines
        |> Array.mapi (chunkZonohedralLineStage "dilation" "Dilate" dilateLineChunkSlice radius windowSize)
        |> Array.fold Stage.compose (Stage.map $"chunkBinaryDilateZonohedral.parallel{windowSize}.start" (fun _ chunk -> chunk) id id)

let binaryErodeZonohedral radius : Stage<Chunk<uint8>, Chunk<uint8>> =
    let lines = ImageFunctions.zonohedralBestLines radius
    if lines.Length = 0 then
        Stage.map "chunkBinaryErodeZonohedral.identity" (fun _ chunk -> chunk) id id
    else
        lines
        |> Array.mapi (chunkZonohedralLineStage "erosion" "Erode" erodeLineChunkSlice radius 1)
        |> Array.fold Stage.compose (Stage.map "chunkBinaryErodeZonohedral.start" (fun _ chunk -> chunk) id id)

let binaryErodeZonohedralParallel radius windowSize : Stage<Chunk<uint8>, Chunk<uint8>> =
    let lines = ImageFunctions.zonohedralBestLines radius
    if lines.Length = 0 then
        Stage.map "chunkBinaryErodeZonohedralParallel.identity" (fun _ chunk -> chunk) id id
    elif windowSize <= 1 then
        binaryErodeZonohedral radius
    else
        lines
        |> Array.mapi (chunkZonohedralLineStage "erosion" "Erode" erodeLineChunkSlice radius windowSize)
        |> Array.fold Stage.compose (Stage.map $"chunkBinaryErodeZonohedral.parallel{windowSize}.start" (fun _ chunk -> chunk) id id)

let binaryOpeningZonohedral radius : Stage<Chunk<uint8>, Chunk<uint8>> =
    Stage.compose (binaryErodeZonohedral radius) (binaryDilateZonohedral radius)

let binaryOpeningZonohedralParallel radius windowSize : Stage<Chunk<uint8>, Chunk<uint8>> =
    Stage.compose (binaryErodeZonohedralParallel radius windowSize) (binaryDilateZonohedralParallel radius windowSize)

let binaryClosingZonohedral radius : Stage<Chunk<uint8>, Chunk<uint8>> =
    Stage.compose (binaryDilateZonohedral radius) (binaryErodeZonohedral radius)

let binaryClosingZonohedralParallel radius windowSize : Stage<Chunk<uint8>, Chunk<uint8>> =
    Stage.compose (binaryDilateZonohedralParallel radius windowSize) (binaryErodeZonohedralParallel radius windowSize)

let thresholdBinary (threshold: uint8) : Stage<Chunk<uint8>, Chunk<uint8>> =
    let mapper _debug chunk =
        try
            Chunk.map (fun value -> if value >= threshold then binaryForeground else binaryBackground) chunk
        finally
            Chunk.decRef chunk

    Stage.map $"chunkThresholdBinary.{threshold}" mapper id id

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
