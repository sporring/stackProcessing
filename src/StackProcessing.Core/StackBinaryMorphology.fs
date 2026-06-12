module StackBinaryMorphology

open System
open System.Collections.Generic
open System.Numerics
open System.IO
open System.Runtime.InteropServices
open System.Threading.Tasks
open FSharp.Control
open SlimPipeline
open StackCore

type private ChunkSlice =
    { Index: int
      Chunk: Chunk<uint8> }

type private LineSamplePlan =
    { Z: int
      XShift: int
      YShift: int
      XStart: int
      XStop: int
      YStart: int
      YStop: int }

type private LinePlan =
    { Left: int
      Right: int
      Samples: LineSamplePlan[]
      ErodeXStart: int
      ErodeXStop: int
      ErodeYStart: int
      ErodeYStop: int }

let private binaryBackground = 0uy
let private binaryForeground = 1uy

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

let private zeroChunkTyped<'T when 'T: equality> width height =
    let chunk = Chunk.create<'T> (uint64 width, uint64 height, 1UL)
    chunk.Bytes.AsSpan(0, chunk.ByteLength).Clear()
    chunk

let private validateSliceChunk width height (chunk: Chunk<uint8>) =
    let chunkWidth, chunkHeight, chunkDepth = chunk.Size
    if chunkDepth <> 1UL then
        invalidArg "chunk" $"Chunk binary dilation expects 2D slice chunks with depth 1, got {chunk.Size}."
    if chunkWidth <> uint64 width || chunkHeight <> uint64 height then
        invalidArg "chunk" $"Chunk binary dilation expects chunks of size {width}x{height}x1, got {chunk.Size}."

let private validateTypedSliceChunk<'T when 'T: equality> operatorName width height (chunk: Chunk<'T>) =
    let chunkWidth, chunkHeight, chunkDepth = chunk.Size
    if chunkDepth <> 1UL then
        invalidArg "chunk" $"Chunk {operatorName} expects 2D slice chunks with depth 1, got {chunk.Size}."
    if chunkWidth <> uint64 width || chunkHeight <> uint64 height then
        invalidArg "chunk" $"Chunk {operatorName} expects chunks of size {width}x{height}x1, got {chunk.Size}."

let private orSpanIntoRange (target: Span<byte>) targetStart (source: Span<byte>) sourceStart count =
    let vectorWidth = Vector<byte>.Count
    let vectorEnd = count - (count % vectorWidth)
    let mutable i = 0
    while i < vectorEnd do
        let mutable targetPart = target.Slice(targetStart + i, vectorWidth)
        let mutable sourcePart = source.Slice(sourceStart + i, vectorWidth)
        let targetSlice = MemoryMarshal.CreateReadOnlySpan(&MemoryMarshal.GetReference(targetPart), vectorWidth)
        let sourceSlice = MemoryMarshal.CreateReadOnlySpan(&MemoryMarshal.GetReference(sourcePart), vectorWidth)
        let destination: Span<byte> = target.Slice(targetStart + i, vectorWidth)
        let targetVector = MemoryMarshal.Read<Vector<byte>>(targetSlice)
        let sourceVector = MemoryMarshal.Read<Vector<byte>>(sourceSlice)
        let mutable result: Vector<byte> = Vector.BitwiseOr(targetVector, sourceVector)
        MemoryMarshal.Write(destination, &result)
        i <- i + vectorWidth
    while i < count do
        target[targetStart + i] <- target[targetStart + i] ||| source[sourceStart + i]
        i <- i + 1

let private andSpanIntoRange (target: Span<byte>) targetStart (source: Span<byte>) sourceStart count =
    let vectorWidth = Vector<byte>.Count
    let vectorEnd = count - (count % vectorWidth)
    let mutable i = 0
    while i < vectorEnd do
        let mutable targetPart = target.Slice(targetStart + i, vectorWidth)
        let mutable sourcePart = source.Slice(sourceStart + i, vectorWidth)
        let targetSlice = MemoryMarshal.CreateReadOnlySpan(&MemoryMarshal.GetReference(targetPart), vectorWidth)
        let sourceSlice = MemoryMarshal.CreateReadOnlySpan(&MemoryMarshal.GetReference(sourcePart), vectorWidth)
        let destination: Span<byte> = target.Slice(targetStart + i, vectorWidth)
        let targetVector = MemoryMarshal.Read<Vector<byte>>(targetSlice)
        let sourceVector = MemoryMarshal.Read<Vector<byte>>(sourceSlice)
        let mutable result: Vector<byte> = Vector.BitwiseAnd(targetVector, sourceVector)
        MemoryMarshal.Write(destination, &result)
        i <- i + vectorWidth
    while i < count do
        target[targetStart + i] <- target[targetStart + i] &&& source[sourceStart + i]
        i <- i + 1

let private createLinePlan width height center dx dy dz length windowLength =
    let left = length - length / 2 - 1
    let right = length / 2
    let samples = ResizeArray<LineSamplePlan>(length)
    let mutable erodeXStart = 0
    let mutable erodeXStop = width
    let mutable erodeYStart = 0
    let mutable erodeYStop = height

    let mutable t = -left
    while t <= right do
        let xShift = t * dx
        let yShift = t * dy
        let z = center + t * dz
        let xStart = max 0 (-xShift)
        let xStop = min width (width - xShift)
        let yStart = max 0 (-yShift)
        let yStop = min height (height - yShift)
        let valid = z >= 0 && z < windowLength && xStop > xStart && yStop > yStart

        if valid then
            samples.Add(
                { Z = z
                  XShift = xShift
                  YShift = yShift
                  XStart = xStart
                  XStop = xStop
                  YStart = yStart
                  YStop = yStop }
            )
            erodeXStart <- max erodeXStart xStart
            erodeXStop <- min erodeXStop xStop
            erodeYStart <- max erodeYStart yStart
            erodeYStop <- min erodeYStop yStop
        else
            erodeXStart <- 0
            erodeXStop <- 0
            erodeYStart <- 0
            erodeYStop <- 0

        t <- t + 1

    { Left = left
      Right = right
      Samples = samples.ToArray()
      ErodeXStart = erodeXStart
      ErodeXStop = erodeXStop
      ErodeYStart = erodeYStart
      ErodeYStop = erodeYStop }

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

let private dilateLineChunkSlice width height (window: ChunkSlice[]) center dx dy dz length (plan: LinePlan) =
    let output = Chunk.create<uint8> (uint64 width, uint64 height, 1UL)
    try
        clearChunk output
        let outputPixels = Chunk.span<uint8> output

        if tryDilateLineChunkSliceSimd width height window center dx dy dz plan.Left plan.Right outputPixels then
            output
        else
            for sample in plan.Samples do
                let inputPixels = Chunk.span<uint8> window[sample.Z].Chunk
                for y in sample.YStart .. sample.YStop - 1 do
                    orSpanIntoRange
                        outputPixels
                        (y * width + sample.XStart)
                        inputPixels
                        ((y + sample.YShift) * width + sample.XStart + sample.XShift)
                        (sample.XStop - sample.XStart)

            output
    with
    | _ ->
        Chunk.decRef output
        reraise()

let private erodeLineChunkSlice width height (window: ChunkSlice[]) center dx dy dz length (plan: LinePlan) =
    let output = Chunk.create<uint8> (uint64 width, uint64 height, 1UL)
    try
        clearChunk output
        let outputPixels = Chunk.span<uint8> output

        if tryErodeLineChunkSliceSimd width height window center dx dy dz plan.Left plan.Right outputPixels then
            output
        else
            if plan.ErodeXStop > plan.ErodeXStart && plan.ErodeYStop > plan.ErodeYStart then
                let count = plan.ErodeXStop - plan.ErodeXStart
                for y in plan.ErodeYStart .. plan.ErodeYStop - 1 do
                    outputPixels.Slice(y * width + plan.ErodeXStart, count).Fill(binaryForeground)

                for sample in plan.Samples do
                    let inputPixels = Chunk.span<uint8> window[sample.Z].Chunk
                    for y in plan.ErodeYStart .. plan.ErodeYStop - 1 do
                        andSpanIntoRange
                            outputPixels
                            (y * width + plan.ErodeXStart)
                            inputPixels
                            ((y + sample.YShift) * width + plan.ErodeXStart + sample.XShift)
                            count

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
            let mutable linePlan = Unchecked.defaultof<LinePlan>

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
                    linePlan <- createLinePlan width height prePad dx dy dz length lineWindowLength
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
                                    outputs[0] <- lineOperator width height windows[0] prePad dx dy dz length linePlan
                                finally
                                    releaseWindow windows[0]
                                    releasedWindows[0] <- true
                            else
                                Parallel.For(
                                    0,
                                    batchCount,
                                    fun i ->
                                        try
                                            outputs[i] <- lineOperator width height windows[i] prePad dx dy dz length linePlan
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

let private binaryDifferenceStage name : Stage<Chunk<uint8> * Chunk<uint8>, Chunk<uint8>> =
    let mapper _debug ((a, b): Chunk<uint8> * Chunk<uint8>) =
        try
            if a.Size <> b.Size then
                invalidArg "b" $"{name} expects paired chunks with identical sizes, got {a.Size} and {b.Size}."

            let output = Chunk.create<uint8> a.Size
            try
                let aPixels = Chunk.span<uint8> a
                let bPixels = Chunk.span<uint8> b
                let outputPixels = Chunk.span<uint8> output
                let mutable i = 0
                while i < aPixels.Length do
                    outputPixels[i] <- if aPixels[i] <> binaryBackground && bPixels[i] = binaryBackground then binaryForeground else binaryBackground
                    i <- i + 1
                output
            with
            | _ ->
                Chunk.decRef output
                reraise()
        finally
            Chunk.decRef a
            Chunk.decRef b

    Stage.map name mapper (fun n -> 3UL * n) id

let binaryWhiteTopHatZonohedral radius : Stage<Chunk<uint8>, Chunk<uint8>> =
    fork (Stage.map "chunkBinaryWhiteTopHatZonohedral.input" (fun _ chunk -> chunk) id id, binaryOpeningZonohedral radius)
    --> binaryDifferenceStage "chunkBinaryWhiteTopHatZonohedral"

let binaryWhiteTopHatZonohedralParallel radius windowSize : Stage<Chunk<uint8>, Chunk<uint8>> =
    fork (Stage.map $"chunkBinaryWhiteTopHatZonohedral.parallel{windowSize}.input" (fun _ chunk -> chunk) id id, binaryOpeningZonohedralParallel radius windowSize)
    --> binaryDifferenceStage $"chunkBinaryWhiteTopHatZonohedral.parallel{windowSize}"

let binaryBlackTopHatZonohedral radius : Stage<Chunk<uint8>, Chunk<uint8>> =
    fork (binaryClosingZonohedral radius, Stage.map "chunkBinaryBlackTopHatZonohedral.input" (fun _ chunk -> chunk) id id)
    --> binaryDifferenceStage "chunkBinaryBlackTopHatZonohedral"

let binaryBlackTopHatZonohedralParallel radius windowSize : Stage<Chunk<uint8>, Chunk<uint8>> =
    fork (binaryClosingZonohedralParallel radius windowSize, Stage.map $"chunkBinaryBlackTopHatZonohedral.parallel{windowSize}.input" (fun _ chunk -> chunk) id id)
    --> binaryDifferenceStage $"chunkBinaryBlackTopHatZonohedral.parallel{windowSize}"

let binaryGradientZonohedral radius : Stage<Chunk<uint8>, Chunk<uint8>> =
    fork (binaryDilateZonohedral radius, binaryErodeZonohedral radius)
    --> binaryDifferenceStage "chunkBinaryGradientZonohedral"

let binaryGradientZonohedralParallel radius windowSize : Stage<Chunk<uint8>, Chunk<uint8>> =
    fork (binaryDilateZonohedralParallel radius windowSize, binaryErodeZonohedralParallel radius windowSize)
    --> binaryDifferenceStage $"chunkBinaryGradientZonohedral.parallel{windowSize}"
