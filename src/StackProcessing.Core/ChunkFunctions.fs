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

type private TypedChunkSlice<'T when 'T: equality> =
    { Index: int
      Chunk: Chunk<'T> }

type private ConnectedComponentChunkWindow =
    { LabelChunks: Chunk<uint32> list
      ObjectCount: uint32 }

module private NativeMedian =
    [<Literal>]
    let LibraryPath = "spnth"

    [<DllImport(LibraryPath, EntryPoint = "sp_median_uint16_nth_slab")>]
    extern void medianUInt16NthSlab(
        nativeint slices,
        nativeint output,
        int width,
        int height,
        int windowLength,
        int radius,
        int outputStart,
        int outputCount)

    [<DllImport(LibraryPath, EntryPoint = "sp_median_uint8_nth_slab")>]
    extern void medianUInt8NthSlab(
        nativeint slices,
        nativeint output,
        int width,
        int height,
        int windowLength,
        int radius,
        int outputStart,
        int outputCount)

    [<DllImport(LibraryPath, EntryPoint = "sp_median_int32_nth_slab")>]
    extern void medianInt32NthSlab(
        nativeint slices,
        nativeint output,
        int width,
        int height,
        int windowLength,
        int radius,
        int outputStart,
        int outputCount)

    [<DllImport(LibraryPath, EntryPoint = "sp_median_float32_nth_slab")>]
    extern void medianFloat32NthSlab(
        nativeint slices,
        nativeint output,
        int width,
        int height,
        int windowLength,
        int radius,
        int outputStart,
        int outputCount)

    [<DllImport(LibraryPath, EntryPoint = "sp_convolve_float32_slab")>]
    extern void convolveFloat32Slab(
        nativeint slices,
        nativeint output,
        nativeint kernel,
        int width,
        int height,
        int windowLength,
        int kernelWidth,
        int kernelHeight,
        int kernelDepth,
        int outputStart,
        int outputCount)

    [<DllImport(LibraryPath, EntryPoint = "sp_convolve_float32_slices")>]
    extern void convolveFloat32Slices(
        nativeint slices,
        nativeint outputs,
        nativeint kernel,
        int width,
        int height,
        int windowLength,
        int kernelWidth,
        int kernelHeight,
        int kernelDepth,
        int outputStart,
        int outputCount)

    [<DllImport(LibraryPath, EntryPoint = "sp_convolve_uint8_slices")>]
    extern void convolveUInt8Slices(
        nativeint slices,
        nativeint outputs,
        nativeint kernel,
        int width,
        int height,
        int windowLength,
        int kernelWidth,
        int kernelHeight,
        int kernelDepth,
        int outputStart,
        int outputCount)

    let ensureAvailable () =
        let mutable handle = nativeint 0
        let searchPath = Nullable(DllImportSearchPath.AssemblyDirectory)
        if NativeLibrary.TryLoad(LibraryPath, typeof<DenseHistogram>.Assembly, searchPath, &handle) then
            NativeLibrary.Free(handle)
        else
            invalidOp "Native median helper 'spnth' was not found. Build it with native/StackProcessing.NativeMedian/build.sh so the platform library is placed in the solution lib directory and copied to the application output."

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

type private KernelTap =
    { WindowZ: int
      Dx: int
      Dy: int
      Weight: float32 }

type private KernelPlan =
    { Width: int
      Height: int
      Depth: int
      PadX: int
      PadY: int
      PadZ: int
      Taps: KernelTap[]
      UniformDivisor: int option }

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

let private validateConnectedComponentsSlice width height (chunk: Chunk<uint8>) =
    let chunkWidth, chunkHeight, chunkDepth = chunk.Size
    if chunkDepth <> 1UL then
        invalidArg "chunk" $"Chunk connected components expects 2D slice chunks with depth 1, got {chunk.Size}."
    if chunkWidth <> uint64 width || chunkHeight <> uint64 height then
        invalidArg "chunk" $"Chunk connected components expects chunks of size {width}x{height}x1, got {chunk.Size}."

let private labelConnectedComponentsSliceSauf3DArrayUf width height (unionFind: DenseUInt32UnionFind) nextLabel (previousLabels: Chunk<uint32> option) (inputChunk: Chunk<uint8>) =
    let outputChunk = Chunk.create<uint32> inputChunk.Size
    try
        let inputSpan = Chunk.span<uint8> inputChunk
        let outputSpan = Chunk.span<uint32> outputChunk
        let mutable nextLabel = nextLabel

        let inline mergeCandidate candidate selected =
            if candidate = 0u then
                selected
            elif selected = 0u then
                candidate
            else
                if selected <> candidate then
                    unionFind.UnionWithoutCompression(selected, candidate)
                selected

        match previousLabels with
        | Some previousChunk ->
            let previousSpan = Chunk.span<uint32> previousChunk
            let mutable y = 0
            while y < height do
                let row = y * width
                let mutable x = 0
                while x < width do
                    let i = row + x
                    if inputSpan[i] <> binaryBackground then
                        let mutable selected = 0u
                        if x > 0 then selected <- mergeCandidate outputSpan[i - 1] selected
                        if y > 0 then selected <- mergeCandidate outputSpan[i - width] selected
                        selected <- mergeCandidate previousSpan[i] selected
                        if selected = 0u then
                            selected <- nextLabel
                            unionFind.Add selected
                            if nextLabel = UInt32.MaxValue then
                                invalidOp "Chunk connected components exhausted UInt32 labels."
                            nextLabel <- nextLabel + 1u
                        outputSpan[i] <- selected
                    else
                        outputSpan[i] <- 0u
                    x <- x + 1
                y <- y + 1
        | None ->
            let mutable y = 0
            while y < height do
                let row = y * width
                let mutable x = 0
                while x < width do
                    let i = row + x
                    if inputSpan[i] <> binaryBackground then
                        let mutable selected = 0u
                        if x > 0 then selected <- mergeCandidate outputSpan[i - 1] selected
                        if y > 0 then selected <- mergeCandidate outputSpan[i - width] selected
                        if selected = 0u then
                            selected <- nextLabel
                            unionFind.Add selected
                            if nextLabel = UInt32.MaxValue then
                                invalidOp "Chunk connected components exhausted UInt32 labels."
                            nextLabel <- nextLabel + 1u
                        outputSpan[i] <- selected
                    else
                        outputSpan[i] <- 0u
                    x <- x + 1
                y <- y + 1

        outputChunk, nextLabel
    with
    | _ ->
        Chunk.decRef outputChunk
        reraise()

let private compactConnectedComponentLabelsArrayUf (unionFind: DenseUInt32UnionFind) nextLabel (labels: ResizeArray<Chunk<uint32>>) =
    if nextLabel > uint32 Int32.MaxValue then
        invalidOp $"Chunk connected components compact relabel requires provisional labels to fit in Int32; got {nextLabel - 1u}."

    let rootCompacts = Array.zeroCreate<uint32> (int nextLabel)
    let labelCompacts = Array.zeroCreate<uint32> (int nextLabel)
    let mutable nextCompact = 1u

    let mutable label = 1u
    while label < nextLabel do
        let root = unionFind.Find label
        let rootIndex = int root
        let mutable compact = rootCompacts[rootIndex]
        if compact = 0u then
            compact <- nextCompact
            rootCompacts[rootIndex] <- compact
            if nextCompact = UInt32.MaxValue then
                invalidOp "Chunk connected components exhausted compact UInt32 labels."
            nextCompact <- nextCompact + 1u
        labelCompacts[int label] <- compact
        label <- label + 1u

    for chunk in labels do
        let outputSpan = Chunk.span<uint32> chunk
        let mutable i = 0
        while i < outputSpan.Length do
            let label = outputSpan[i]
            if label <> 0u then
                outputSpan[i] <- labelCompacts[int label]
            i <- i + 1

    nextCompact - 1u

let private labelConnectedComponentChunkWindow (inputChunks: Chunk<uint8> list) =
    let labels = ResizeArray<Chunk<uint32>>()
    let unionFind = DenseUInt32UnionFind(1024)
    let mutable previousLabels : Chunk<uint32> option = None
    let mutable width = 0
    let mutable height = 0
    let mutable initialized = false
    let mutable nextLabel = 1u

    let releaseLabels () =
        for chunk in labels do
            if chunk.RefCount.Value > 0 then
                Chunk.decRef chunk

    try
        for inputChunk in inputChunks do
            let outputChunk =
                try
                    if not initialized then
                        let chunkWidth, chunkHeight, chunkDepth = inputChunk.Size
                        if chunkDepth <> 1UL then
                            invalidArg "chunk" $"Chunk connected components expects 2D slice chunks with depth 1, got {inputChunk.Size}."
                        if chunkWidth > uint64 Int32.MaxValue || chunkHeight > uint64 Int32.MaxValue then
                            invalidArg "chunk" $"Chunk connected components dimensions must fit in Int32, got {inputChunk.Size}."
                        width <- int chunkWidth
                        height <- int chunkHeight
                        initialized <- true
                    else
                        validateConnectedComponentsSlice width height inputChunk

                    let outputChunk, updatedNextLabel =
                        labelConnectedComponentsSliceSauf3DArrayUf width height unionFind nextLabel previousLabels inputChunk
                    nextLabel <- updatedNextLabel
                    outputChunk
                finally
                    Chunk.decRef inputChunk

            labels.Add outputChunk
            previousLabels <- Some outputChunk

        let objectCount = compactConnectedComponentLabelsArrayUf unionFind nextLabel labels
        { LabelChunks = labels |> Seq.toList
          ObjectCount = objectCount }
    with
    | _ ->
        releaseLabels ()
        reraise()

let private stitchConnectedComponentChunkWindows (windows: ResizeArray<ConnectedComponentChunkWindow>) =
    let bases = Array.zeroCreate<uint32> windows.Count
    let mutable nextBase = 1u
    for i in 0 .. windows.Count - 1 do
        bases[i] <- nextBase
        let count = windows[i].ObjectCount
        if count > UInt32.MaxValue - nextBase then
            invalidOp "Chunk connected components exhausted UInt32 labels while stitching windows."
        nextBase <- nextBase + count

    let totalLabels = nextBase
    if totalLabels > uint32 Int32.MaxValue then
        invalidOp $"Chunk connected components stitch requires global labels to fit in Int32; got {totalLabels - 1u}."

    let unionFind = DenseUInt32UnionFind(int totalLabels)
    for label in 1u .. totalLabels - 1u do
        unionFind.Add label

    let inline globalLabel windowIndex localLabel =
        if localLabel = 0u then
            0u
        else
            bases[windowIndex] + localLabel - 1u

    for windowIndex in 1 .. windows.Count - 1 do
        match windows[windowIndex - 1].LabelChunks |> List.tryLast, windows[windowIndex].LabelChunks |> List.tryHead with
        | Some previousLast, Some currentFirst ->
            let previousSpan = Chunk.span<uint32> previousLast
            let currentSpan = Chunk.span<uint32> currentFirst
            if previousSpan.Length <> currentSpan.Length then
                invalidOp "Chunk connected components cannot stitch windows with mismatched boundary slice sizes."
            let mutable i = 0
            while i < previousSpan.Length do
                let previousLabel = previousSpan[i]
                let currentLabel = currentSpan[i]
                if previousLabel <> 0u && currentLabel <> 0u then
                    unionFind.UnionWithoutCompression(
                        globalLabel (windowIndex - 1) previousLabel,
                        globalLabel windowIndex currentLabel)
                i <- i + 1
        | _ -> ()

    let rootCompacts = Array.zeroCreate<uint32> (int totalLabels)
    let globalCompacts = Array.zeroCreate<uint32> (int totalLabels)
    let mutable nextCompact = 1u
    let mutable label = 1u
    while label < totalLabels do
        let root = unionFind.Find label
        let mutable compact = rootCompacts[int root]
        if compact = 0u then
            compact <- nextCompact
            rootCompacts[int root] <- compact
            if nextCompact = UInt32.MaxValue then
                invalidOp "Chunk connected components exhausted compact UInt32 labels while stitching windows."
            nextCompact <- nextCompact + 1u
        globalCompacts[int label] <- compact
        label <- label + 1u

    for windowIndex in 0 .. windows.Count - 1 do
        for chunk in windows[windowIndex].LabelChunks do
            let labels = Chunk.span<uint32> chunk
            let mutable i = 0
            while i < labels.Length do
                let localLabel = labels[i]
                if localLabel <> 0u then
                    labels[i] <- globalCompacts[int (globalLabel windowIndex localLabel)]
                i <- i + 1

    nextCompact - 1u

let private splitChunkWindow workers (inputChunks: Chunk<uint8> list) =
    let count = inputChunks.Length
    let partitions = min workers count
    let baseSize = count / partitions
    let remainder = count % partitions
    let pieces = ResizeArray<Chunk<uint8> list>(partitions)
    let mutable remaining = inputChunks

    for partition in 0 .. partitions - 1 do
        let size = baseSize + if partition < remainder then 1 else 0
        let piece = remaining |> List.take size
        pieces.Add piece
        remaining <- remaining |> List.skip size

    pieces

let private releaseConnectedComponentChunkWindow (window: ConnectedComponentChunkWindow) =
    for chunk in window.LabelChunks do
        if chunk.RefCount.Value > 0 then
            Chunk.decRef chunk

let private labelConnectedComponentChunkWindowParallel workers (inputChunks: Chunk<uint8> list) =
    if workers <= 1 || inputChunks.Length <= 1 then
        labelConnectedComponentChunkWindow inputChunks
    else
        let pieces = splitChunkWindow workers inputChunks
        let results = Array.zeroCreate<ConnectedComponentChunkWindow> pieces.Count

        try
            Parallel.For(
                0,
                pieces.Count,
                fun i ->
                    results[i] <- labelConnectedComponentChunkWindow pieces[i])
            |> ignore

            let windows = ResizeArray<ConnectedComponentChunkWindow>(results)
            let objectCount = stitchConnectedComponentChunkWindows windows
            let labels =
                results
                |> Seq.collect (fun window -> window.LabelChunks)
                |> Seq.toList

            { LabelChunks = labels
              ObjectCount = objectCount }
        with
        | _ ->
            for window in results do
                if not (isNull (box window)) then
                    releaseConnectedComponentChunkWindow window
            reraise()

let connectedComponentsSauf3DUInt8UInt32ArrayUf () : Stage<Chunk<uint8>, Chunk<uint32>> =
    let name = "chunkConnectedComponentsSauf3DUInt8UInt32ArrayUf"

    let apply _debug (input: AsyncSeq<Chunk<uint8>>) =
        asyncSeq {
            let labels = ResizeArray<Chunk<uint32>>()
            let unionFind = DenseUInt32UnionFind(1024)
            let mutable previousLabels : Chunk<uint32> option = None
            let mutable width = 0
            let mutable height = 0
            let mutable initialized = false
            let mutable nextLabel = 1u

            let releaseRetainedOutputs () =
                for chunk in labels do
                    if chunk.RefCount.Value > 0 then
                        Chunk.decRef chunk

            try
                for inputChunk in input do
                    let outputChunk =
                        try
                            if not initialized then
                                let chunkWidth, chunkHeight, chunkDepth = inputChunk.Size
                                if chunkDepth <> 1UL then
                                    invalidArg "chunk" $"Chunk connected components expects 2D slice chunks with depth 1, got {inputChunk.Size}."
                                if chunkWidth > uint64 Int32.MaxValue || chunkHeight > uint64 Int32.MaxValue then
                                    invalidArg "chunk" $"Chunk connected components dimensions must fit in Int32, got {inputChunk.Size}."
                                width <- int chunkWidth
                                height <- int chunkHeight
                                initialized <- true
                            else
                                validateConnectedComponentsSlice width height inputChunk

                            let outputChunk, updatedNextLabel =
                                labelConnectedComponentsSliceSauf3DArrayUf width height unionFind nextLabel previousLabels inputChunk
                            nextLabel <- updatedNextLabel
                            outputChunk
                        finally
                            Chunk.decRef inputChunk

                    labels.Add outputChunk
                    previousLabels <- Some outputChunk

                compactConnectedComponentLabelsArrayUf unionFind nextLabel labels |> ignore

                for chunk in labels do
                    yield chunk
            with
            | ex ->
                releaseRetainedOutputs ()
                raise ex
        }

    let memoryNeed nPixels =
        nPixels * (uint64 sizeof<uint8> + uint64 sizeof<uint32>)

    Stage.fromAsyncSeq name apply (ProfileTransition.create Streaming Streaming) (StageMemoryModel.fromSinglePeak Reduce memoryNeed) id
    |> Stage.withSliceCardinality (SlimPipeline.Domain SlimPipeline.SliceDomain.preserve)

let connectedComponentsSauf3DUInt8UInt32 () : Stage<Chunk<uint8>, Chunk<uint32>> =
    connectedComponentsSauf3DUInt8UInt32ArrayUf ()

let connectedComponentsSauf3DUInt8UInt32ParallelCollect windowSize workers : Stage<Chunk<uint8>, Chunk<uint32>> =
    if windowSize < 1 then
        invalidArg "windowSize" $"Chunk connected components parallelCollect expects positive window size, got {windowSize}."
    if workers < 1 then
        invalidArg "workers" $"Chunk connected components parallelCollect expects at least one worker, got {workers}."

    let name = $"chunkConnectedComponentsSauf3DUInt8UInt32.slabWindow.window{windowSize}.workers{workers}"

    let windowStage =
        Stage.window
            $"{name}.windowed"
            (uint windowSize)
            0u
            (fun _ chunk -> chunk)
            (uint windowSize)

    let labelSlabStage =
        Stage.map
            $"{name}.labelSlab"
            (fun _debug (window: Window<Chunk<uint8>>) ->
                labelConnectedComponentChunkWindowParallel workers window.Items)
            (fun nPixels -> nPixels * uint64 windowSize * (uint64 sizeof<uint8> + uint64 sizeof<uint32>))
            id

    let stitchApply _debug (input: AsyncSeq<ConnectedComponentChunkWindow>) =
        asyncSeq {
            let slabs = ResizeArray<ConnectedComponentChunkWindow>()

            let releaseSlabs () =
                for slab in slabs do
                    releaseConnectedComponentChunkWindow slab

            try
                for slab in input do
                    slabs.Add slab

                stitchConnectedComponentChunkWindows slabs |> ignore

                for slab in slabs do
                    for chunk in slab.LabelChunks do
                        yield chunk
            with
            | ex ->
                releaseSlabs ()
                raise ex
        }

    let stitchStage =
        Stage.fromAsyncSeq
            $"{name}.stitchSlabs"
            stitchApply
            (ProfileTransition.create Streaming Streaming)
            (StageMemoryModel.fromSinglePeak Reduce id)
            id

    windowStage --> labelSlabStage --> stitchStage
    |> Stage.withSliceCardinality (SlimPipeline.Domain SlimPipeline.SliceDomain.preserve)

let connectedComponentsSauf3DUInt8 () : Stage<Chunk<uint8>, Chunk<uint64>> =
    connectedComponentsSauf3DUInt8UInt32 ()
    --> Stage.map
            "chunkConnectedComponentsSauf3DUInt8.widenUInt64"
            (fun _ chunk ->
                try
                    Chunk.map uint64 chunk
                finally
                    Chunk.decRef chunk)
            (fun nPixels -> nPixels * uint64 sizeof<uint64>)
            id
    |> Stage.withSliceCardinality (SlimPipeline.Domain SlimPipeline.SliceDomain.preserve)

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

let private retainTypedWindow (queue: ResizeArray<TypedChunkSlice<'T>>) start length =
    let window = Array.zeroCreate<TypedChunkSlice<'T>> length
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

let private releaseTypedWindow (window: TypedChunkSlice<'T>[]) =
    for item in window do
        Chunk.decRef item.Chunk

let private createKernelPlan (kernel: float32[,,]) =
    let width = kernel.GetLength(0)
    let height = kernel.GetLength(1)
    let depth = kernel.GetLength(2)
    if width < 1 || height < 1 || depth < 1 then
        invalidArg "kernel" "Chunk convolution expects a non-empty kernel."
    if width % 2 = 0 || height % 2 = 0 || depth % 2 = 0 then
        invalidArg "kernel" $"Chunk convolution expects odd kernel dimensions, got {width}x{height}x{depth}."

    let padX = width / 2
    let padY = height / 2
    let padZ = depth / 2
    let taps = ResizeArray<KernelTap>(width * height * depth)

    for z in 0 .. depth - 1 do
        for y in 0 .. height - 1 do
            for x in 0 .. width - 1 do
                let weight = kernel[x, y, z]
                if weight <> 0.0f then
                    taps.Add(
                        { WindowZ = z
                          Dx = x - padX
                          Dy = y - padY
                          Weight = weight }
                    )

    let uniformDivisor =
        let expectedCount = width * height * depth
        if taps.Count = expectedCount then
            let expectedWeight = 1.0f / float32 expectedCount
            let mutable equal = true
            let mutable i = 0
            while equal && i < taps.Count do
                equal <- abs (taps[i].Weight - expectedWeight) <= Single.Epsilon
                i <- i + 1
            if equal then Some expectedCount else None
        else
            None

    { Width = width
      Height = height
      Depth = depth
      PadX = padX
      PadY = padY
      PadZ = padZ
      Taps = taps.ToArray()
      UniformDivisor = uniformDivisor }

let inline private clampRoundToByte (value: float32) =
    if Single.IsNaN value || value <= 0.0f then
        0uy
    elif value >= 255.0f then
        255uy
    else
        uint8 (MathF.Round value)

let inline private clampRoundToSByte (value: float32) =
    if Single.IsNaN value then
        0y
    elif value <= float32 SByte.MinValue then
        SByte.MinValue
    elif value >= float32 SByte.MaxValue then
        SByte.MaxValue
    else
        int8 (MathF.Round value)

let inline private clampRoundToUInt16 (value: float32) =
    if Single.IsNaN value || value <= 0.0f then
        0us
    elif value >= 65535.0f then
        65535us
    else
        uint16 (MathF.Round value)

let inline private clampRoundToInt16 (value: float32) =
    if Single.IsNaN value then
        0s
    elif value <= float32 Int16.MinValue then
        Int16.MinValue
    elif value >= float32 Int16.MaxValue then
        Int16.MaxValue
    else
        int16 (MathF.Round value)

let inline private clampRoundToInt32 (value: float32) =
    if Single.IsNaN value then
        0
    elif value <= float32 Int32.MinValue then
        Int32.MinValue
    elif value >= float32 Int32.MaxValue then
        Int32.MaxValue
    else
        int32 (MathF.Round value)

let private byteVectorToSingleVectors (source: ReadOnlySpan<byte>) =
    let bytes = MemoryMarshal.Read<Vector<byte>>(source)
    let mutable lo16 = Vector<uint16>.Zero
    let mutable hi16 = Vector<uint16>.Zero
    Vector.Widen(bytes, &lo16, &hi16)
    let mutable loLo32 = Vector<uint32>.Zero
    let mutable loHi32 = Vector<uint32>.Zero
    let mutable hiLo32 = Vector<uint32>.Zero
    let mutable hiHi32 = Vector<uint32>.Zero
    Vector.Widen(lo16, &loLo32, &loHi32)
    Vector.Widen(hi16, &hiLo32, &hiHi32)
    Vector.ConvertToSingle(loLo32), Vector.ConvertToSingle(loHi32), Vector.ConvertToSingle(hiLo32), Vector.ConvertToSingle(hiHi32)

let private uint16VectorToSingleVectors (source: ReadOnlySpan<uint16>) =
    let values = Vector<uint16>(source)
    let mutable lo32 = Vector<uint32>.Zero
    let mutable hi32 = Vector<uint32>.Zero
    Vector.Widen(values, &lo32, &hi32)
    Vector.ConvertToSingle(lo32), Vector.ConvertToSingle(hi32)

let private int8VectorToSingleVectors (source: ReadOnlySpan<sbyte>) =
    let values = Vector<sbyte>(source)
    let mutable lo16 = Vector<int16>.Zero
    let mutable hi16 = Vector<int16>.Zero
    Vector.Widen(values, &lo16, &hi16)
    let mutable loLo32 = Vector<int32>.Zero
    let mutable loHi32 = Vector<int32>.Zero
    let mutable hiLo32 = Vector<int32>.Zero
    let mutable hiHi32 = Vector<int32>.Zero
    Vector.Widen(lo16, &loLo32, &loHi32)
    Vector.Widen(hi16, &hiLo32, &hiHi32)
    Vector.ConvertToSingle(loLo32), Vector.ConvertToSingle(loHi32), Vector.ConvertToSingle(hiLo32), Vector.ConvertToSingle(hiHi32)

let private int16VectorToSingleVectors (source: ReadOnlySpan<int16>) =
    let values = Vector<int16>(source)
    let mutable lo32 = Vector<int32>.Zero
    let mutable hi32 = Vector<int32>.Zero
    Vector.Widen(values, &lo32, &hi32)
    Vector.ConvertToSingle(lo32), Vector.ConvertToSingle(hi32)

let private convolvePixelFloat32 width height (plan: KernelPlan) (window: TypedChunkSlice<'T>[]) x y =
    let mutable acc = 0.0f
    for tap in plan.Taps do
        let sy = y + tap.Dy
        let sx = x + tap.Dx
        if sx >= 0 && sx < width && sy >= 0 && sy < height then
            let source = MemoryMarshal.Cast<byte, float32>(window[tap.WindowZ].Chunk.Bytes.AsSpan(0, window[tap.WindowZ].Chunk.ByteLength))
            acc <- acc + source[sy * width + sx] * tap.Weight
    acc

let private convolveFloat32Slice width height (plan: KernelPlan) (window: TypedChunkSlice<'T>[]) (output: Chunk<'T>) =
    let outputPixels = MemoryMarshal.Cast<byte, float32>(output.Bytes.AsSpan(0, output.ByteLength))
    let vectorWidth = Vector<float32>.Count
    let vectorEnd = width - plan.PadX - vectorWidth

    for y in 0 .. height - 1 do
        let mutable x = 0
        while x < width do
            if x >= plan.PadX && x <= vectorEnd then
                let mutable acc = Vector<float32>.Zero
                for tap in plan.Taps do
                    let sy = y + tap.Dy
                    if sy >= 0 && sy < height then
                        let source = MemoryMarshal.Cast<byte, float32>(window[tap.WindowZ].Chunk.Bytes.AsSpan(0, window[tap.WindowZ].Chunk.ByteLength))
                        let sourceIndex = sy * width + x + tap.Dx
                        let mutable sourcePart = source.Slice(sourceIndex, vectorWidth)
                        let sourceSlice = MemoryMarshal.CreateReadOnlySpan(&MemoryMarshal.GetReference(sourcePart), vectorWidth)
                        acc <- acc + Vector<float32>(sourceSlice) * Vector<float32>(tap.Weight)
                acc.CopyTo(outputPixels.Slice(y * width + x, vectorWidth))
                x <- x + vectorWidth
            else
                outputPixels[y * width + x] <- convolvePixelFloat32 width height plan window x y
                x <- x + 1

let private convolveUInt8Slice width height (plan: KernelPlan) (window: TypedChunkSlice<'T>[]) (output: Chunk<'T>) =
    let outputPixels = MemoryMarshal.Cast<byte, uint8>(output.Bytes.AsSpan(0, output.ByteLength))
    let scalarPixel x y =
        let mutable acc = 0.0f
        for tap in plan.Taps do
            let sy = y + tap.Dy
            let sx = x + tap.Dx
            if sx >= 0 && sx < width && sy >= 0 && sy < height then
                let source = window[tap.WindowZ].Chunk.Bytes.AsSpan(0, window[tap.WindowZ].Chunk.ByteLength)
                acc <- acc + float32 source[sy * width + sx] * tap.Weight
        clampRoundToByte acc

    match plan.UniformDivisor with
    | Some divisor when plan.Taps.Length * 255 <= int UInt16.MaxValue ->
        let byteVectorWidth = Vector<byte>.Count
        let halfVectorWidth = Vector<uint16>.Count
        let vectorEnd = width - plan.PadX - byteVectorWidth
        let loBuffer: uint16[] = Array.zeroCreate halfVectorWidth
        let hiBuffer: uint16[] = Array.zeroCreate halfVectorWidth
        let divisorF = float32 divisor

        for y in 0 .. height - 1 do
            let mutable x = 0
            while x < width do
                if x >= plan.PadX && x <= vectorEnd then
                    let mutable loAcc = Vector<uint16>.Zero
                    let mutable hiAcc = Vector<uint16>.Zero

                    for tap in plan.Taps do
                        let sy = y + tap.Dy
                        if sy >= 0 && sy < height then
                            let source = window[tap.WindowZ].Chunk.Bytes.AsSpan(0, window[tap.WindowZ].Chunk.ByteLength)
                            let sourceIndex = sy * width + x + tap.Dx
                            let sourceVector = Vector<byte>(source.Slice(sourceIndex, byteVectorWidth))
                            let mutable lo = Vector<uint16>.Zero
                            let mutable hi = Vector<uint16>.Zero
                            Vector.Widen(sourceVector, &lo, &hi)
                            loAcc <- loAcc + lo
                            hiAcc <- hiAcc + hi

                    loAcc.CopyTo(loBuffer)
                    hiAcc.CopyTo(hiBuffer)

                    let rowOffset = y * width + x
                    for i in 0 .. halfVectorWidth - 1 do
                        outputPixels[rowOffset + i] <- clampRoundToByte (float32 loBuffer[i] / divisorF)
                        outputPixels[rowOffset + halfVectorWidth + i] <- clampRoundToByte (float32 hiBuffer[i] / divisorF)

                    x <- x + byteVectorWidth
                else
                    outputPixels[y * width + x] <- scalarPixel x y
                    x <- x + 1
    | _ ->
        for y in 0 .. height - 1 do
            for x in 0 .. width - 1 do
                outputPixels[y * width + x] <- scalarPixel x y

let private convolveUInt16Slice width height (plan: KernelPlan) (window: TypedChunkSlice<'T>[]) (output: Chunk<'T>) =
    let outputPixels = MemoryMarshal.Cast<byte, uint16>(output.Bytes.AsSpan(0, output.ByteLength))
    for y in 0 .. height - 1 do
        for x in 0 .. width - 1 do
            let mutable acc = 0.0f
            for tap in plan.Taps do
                let sy = y + tap.Dy
                let sx = x + tap.Dx
                if sx >= 0 && sx < width && sy >= 0 && sy < height then
                    let source = MemoryMarshal.Cast<byte, uint16>(window[tap.WindowZ].Chunk.Bytes.AsSpan(0, window[tap.WindowZ].Chunk.ByteLength))
                    acc <- acc + float32 source[sy * width + sx] * tap.Weight
            outputPixels[y * width + x] <- clampRoundToUInt16 acc

let private convolveInt32Slice width height (plan: KernelPlan) (window: TypedChunkSlice<'T>[]) (output: Chunk<'T>) =
    let outputPixels = MemoryMarshal.Cast<byte, int32>(output.Bytes.AsSpan(0, output.ByteLength))
    for y in 0 .. height - 1 do
        for x in 0 .. width - 1 do
            let mutable acc = 0.0f
            for tap in plan.Taps do
                let sy = y + tap.Dy
                let sx = x + tap.Dx
                if sx >= 0 && sx < width && sy >= 0 && sy < height then
                    let source = MemoryMarshal.Cast<byte, int32>(window[tap.WindowZ].Chunk.Bytes.AsSpan(0, window[tap.WindowZ].Chunk.ByteLength))
                    acc <- acc + float32 source[sy * width + sx] * tap.Weight
            outputPixels[y * width + x] <- clampRoundToInt32 acc

let private convolveFixedKernelSlice<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    width
    height
    (plan: KernelPlan)
    (window: TypedChunkSlice<'T>[])
    =
    let output = Chunk.create<'T> (uint64 width, uint64 height, 1UL)
    try
        output.Bytes.AsSpan(0, output.ByteLength).Clear()
        let t = typeof<'T>
        if t = typeof<float32> then
            convolveFloat32Slice width height plan window output
        elif t = typeof<uint8> then
            convolveUInt8Slice width height plan window output
        elif t = typeof<uint16> then
            convolveUInt16Slice width height plan window output
        elif t = typeof<int32> then
            convolveInt32Slice width height plan window output
        else
            invalidArg "T" $"Chunk convolution currently supports UInt8, UInt16, Int32, and Float32 chunks, got {t.Name}."
        output
    with
    | _ ->
        Chunk.decRef output
        reraise()

let private flattenKernelForNative (kernel: float32[,,]) =
    let width = kernel.GetLength(0)
    let height = kernel.GetLength(1)
    let depth = kernel.GetLength(2)
    let values = Array.zeroCreate<float32> (width * height * depth)
    let mutable i = 0
    for z in 0 .. depth - 1 do
        for y in 0 .. height - 1 do
            for x in 0 .. width - 1 do
                values[i] <- kernel[x, y, z]
                i <- i + 1
    values

let private convolveNativeFloat32Slices width height (plan: KernelPlan) (nativeKernel: float32[]) outputStart outputCount (window: Chunk<float32>[]) =
    NativeMedian.ensureAvailable ()

    let outputs =
        Array.init outputCount (fun _ -> Chunk.create<float32> (uint64 width, uint64 height, 1UL))

    let inputHandles = Array.zeroCreate<GCHandle> window.Length
    let outputHandles = Array.zeroCreate<GCHandle> outputs.Length
    let mutable retainedInputHandles = 0
    let mutable retainedOutputHandles = 0
    let mutable inputPointerHandle = Unchecked.defaultof<GCHandle>
    let mutable inputPointersPinned = false
    let mutable outputPointerHandle = Unchecked.defaultof<GCHandle>
    let mutable outputPointersPinned = false
    let mutable kernelHandle = Unchecked.defaultof<GCHandle>
    let mutable kernelPinned = false

    try
        try
            let inputPointers = Array.zeroCreate<nativeint> window.Length
            for i in 0 .. window.Length - 1 do
                inputHandles[i] <- GCHandle.Alloc(window[i].Bytes, GCHandleType.Pinned)
                retainedInputHandles <- retainedInputHandles + 1
                inputPointers[i] <- inputHandles[i].AddrOfPinnedObject()

            let outputPointers = Array.zeroCreate<nativeint> outputs.Length
            for i in 0 .. outputs.Length - 1 do
                outputHandles[i] <- GCHandle.Alloc(outputs[i].Bytes, GCHandleType.Pinned)
                retainedOutputHandles <- retainedOutputHandles + 1
                outputPointers[i] <- outputHandles[i].AddrOfPinnedObject()

            inputPointerHandle <- GCHandle.Alloc(inputPointers, GCHandleType.Pinned)
            inputPointersPinned <- true
            outputPointerHandle <- GCHandle.Alloc(outputPointers, GCHandleType.Pinned)
            outputPointersPinned <- true
            kernelHandle <- GCHandle.Alloc(nativeKernel, GCHandleType.Pinned)
            kernelPinned <- true

            NativeMedian.convolveFloat32Slices(
                inputPointerHandle.AddrOfPinnedObject(),
                outputPointerHandle.AddrOfPinnedObject(),
                kernelHandle.AddrOfPinnedObject(),
                width,
                height,
                window.Length,
                plan.Width,
                plan.Height,
                plan.Depth,
                outputStart,
                outputCount)

            outputs |> Array.toList
        with
        | _ ->
            outputs |> Array.iter Chunk.decRef
            reraise()
    finally
        if kernelPinned then
            kernelHandle.Free()
        if outputPointersPinned then
            outputPointerHandle.Free()
        if inputPointersPinned then
            inputPointerHandle.Free()
        for i in 0 .. retainedOutputHandles - 1 do
            outputHandles[i].Free()
        for i in 0 .. retainedInputHandles - 1 do
            inputHandles[i].Free()

let private convolveNativeUInt8Slices width height (plan: KernelPlan) (nativeKernel: float32[]) outputStart outputCount (window: Chunk<uint8>[]) =
    NativeMedian.ensureAvailable ()

    let outputs =
        Array.init outputCount (fun _ -> Chunk.create<uint8> (uint64 width, uint64 height, 1UL))

    let inputHandles = Array.zeroCreate<GCHandle> window.Length
    let outputHandles = Array.zeroCreate<GCHandle> outputs.Length
    let mutable retainedInputHandles = 0
    let mutable retainedOutputHandles = 0
    let mutable inputPointerHandle = Unchecked.defaultof<GCHandle>
    let mutable inputPointersPinned = false
    let mutable outputPointerHandle = Unchecked.defaultof<GCHandle>
    let mutable outputPointersPinned = false
    let mutable kernelHandle = Unchecked.defaultof<GCHandle>
    let mutable kernelPinned = false

    try
        try
            let inputPointers = Array.zeroCreate<nativeint> window.Length
            for i in 0 .. window.Length - 1 do
                inputHandles[i] <- GCHandle.Alloc(window[i].Bytes, GCHandleType.Pinned)
                retainedInputHandles <- retainedInputHandles + 1
                inputPointers[i] <- inputHandles[i].AddrOfPinnedObject()

            let outputPointers = Array.zeroCreate<nativeint> outputs.Length
            for i in 0 .. outputs.Length - 1 do
                outputHandles[i] <- GCHandle.Alloc(outputs[i].Bytes, GCHandleType.Pinned)
                retainedOutputHandles <- retainedOutputHandles + 1
                outputPointers[i] <- outputHandles[i].AddrOfPinnedObject()

            inputPointerHandle <- GCHandle.Alloc(inputPointers, GCHandleType.Pinned)
            inputPointersPinned <- true
            outputPointerHandle <- GCHandle.Alloc(outputPointers, GCHandleType.Pinned)
            outputPointersPinned <- true
            kernelHandle <- GCHandle.Alloc(nativeKernel, GCHandleType.Pinned)
            kernelPinned <- true

            NativeMedian.convolveUInt8Slices(
                inputPointerHandle.AddrOfPinnedObject(),
                outputPointerHandle.AddrOfPinnedObject(),
                kernelHandle.AddrOfPinnedObject(),
                width,
                height,
                window.Length,
                plan.Width,
                plan.Height,
                plan.Depth,
                outputStart,
                outputCount)

            outputs |> Array.toList
        with
        | _ ->
            outputs |> Array.iter Chunk.decRef
            reraise()
    finally
        if kernelPinned then
            kernelHandle.Free()
        if outputPointersPinned then
            outputPointerHandle.Free()
        if inputPointersPinned then
            inputPointerHandle.Free()
        for i in 0 .. retainedOutputHandles - 1 do
            outputHandles[i].Free()
        for i in 0 .. retainedInputHandles - 1 do
            inputHandles[i].Free()

let private chunkConvolveFixedKernelStage<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (kernel: float32[,,])
    batchSize
    =
    if batchSize < 1 then
        invalidArg "batchSize" $"Chunk convolution expects a positive batch size, got {batchSize}."

    let plan = createKernelPlan kernel
    let windowLength = plan.Depth
    let memoryNeed nPixels =
        uint64 (windowLength + batchSize) * nPixels * uint64 (Marshal.SizeOf<'T>())
    let suffix = if batchSize = 1 then "" else $".parallel{batchSize}"
    let stageName = $"chunkConvolveFixedKernel{suffix}.{typeof<'T>.Name}.{plan.Width}x{plan.Height}x{plan.Depth}"

    let zeroMaker _index (source: Chunk<'T>) =
        let width, height, depth = source.Size
        if depth <> 1UL then
            invalidArg "source" $"Chunk convolution expects 2D slice chunks with depth 1, got {source.Size}."
        zeroChunkTyped<'T> (int width) (int height)

    let releaseConsumed (window: Window<Chunk<'T>>) =
        let _emitStart, emitCount = window.EmitRange
        if emitCount = 0u then
            window.Items |> List.iter Chunk.decRef
        else
            window.Items
            |> List.truncate (int window.ReleaseCount)
            |> List.iter Chunk.decRef

    let retainWindowRefs _debug (window: Window<Chunk<'T>>) =
        window.Items |> List.iter (Chunk.incRef >> ignore)
        releaseConsumed window
        window

    let releaseWindowRefs (window: Window<Chunk<'T>>) =
        window.Items |> List.iter Chunk.decRef

    let convolveWindow (retained: Window<Chunk<'T>>) =
        let _emitStart, emitCount = retained.EmitRange
        if emitCount = 0u then
            []
        else
            let chunks = retained.Items |> List.toArray
            if chunks.Length <> windowLength then
                invalidArg "window" $"Chunk convolution expected window length {windowLength}, got {chunks.Length}."

            let first = chunks[0]
            let chunkWidth, chunkHeight, chunkDepth = first.Size
            if chunkDepth <> 1UL then
                invalidArg "window" $"Chunk convolution expects 2D slice chunks with depth 1, got {first.Size}."

            let width = int chunkWidth
            let height = int chunkHeight
            if width <= 0 || height <= 0 then
                invalidArg "window" $"Chunk convolution expects positive slice dimensions, got {first.Size}."

            let items =
                Array.init chunks.Length (fun i ->
                    validateTypedSliceChunk "convolution" width height chunks[i]
                    { Index = i; Chunk = chunks[i] })

            [ convolveFixedKernelSlice width height plan items ]

    let convolveRetained _debug (window: Window<Window<Chunk<'T>>>) =
        match window.Items with
        | [ retainedWindow ] ->
            try
                convolveWindow retainedWindow
            finally
                releaseWindowRefs retainedWindow
        | items ->
            for retainedWindow in items do
                releaseWindowRefs retainedWindow
            invalidArg "window" $"Chunk convolution expected singleton retained windows, got {items.Length}."

    let windowStage =
        Stage.window $"{stageName}.window" (uint windowLength) (uint plan.PadZ) zeroMaker 1u

    let retainStage =
        Stage.map
            $"{stageName}.retain"
            retainWindowRefs
            memoryNeed
            id

    let computeStage =
        Stage.parallelCollect
            $"{stageName}.parallelCollect"
            1
            batchSize
            1
            0
            (fun _ retained -> retained)
            convolveRetained
            memoryNeed
            id

    Stage.compose windowStage retainStage
    |> fun stage -> Stage.compose stage computeStage

let private chunkConvolveFixedKernelNativeFloat32Stage
    (kernel: float32[,,])
    batchSize
    =
    if batchSize < 1 then
        invalidArg "batchSize" $"Native chunk convolution expects a positive batch size, got {batchSize}."

    let plan = createKernelPlan kernel
    let nativeKernel = flattenKernelForNative kernel
    let outputBatchSize = batchSize
    let windowLength = plan.Depth + outputBatchSize - 1
    let memoryNeed nPixels =
        uint64 (windowLength + batchSize) * nPixels * uint64 sizeof<float32>
    let suffix = if batchSize = 1 then "" else $".parallel{batchSize}"
    let stageName = $"chunkConvolveFixedKernelNativeFloat32{suffix}.{plan.Width}x{plan.Height}x{plan.Depth}"

    let zeroMaker _index (source: Chunk<float32>) =
        let width, height, depth = source.Size
        if depth <> 1UL then
            invalidArg "source" $"Native chunk convolution expects 2D slice chunks with depth 1, got {source.Size}."
        zeroChunkTyped<float32> (int width) (int height)

    let releaseConsumed (window: Window<Chunk<float32>>) =
        let _emitStart, emitCount = window.EmitRange
        if emitCount = 0u then
            window.Items |> List.iter Chunk.decRef
        else
            window.Items
            |> List.truncate (int window.ReleaseCount)
            |> List.iter Chunk.decRef

    let retainWindowRefs _debug (window: Window<Chunk<float32>>) =
        window.Items |> List.iter (Chunk.incRef >> ignore)
        releaseConsumed window
        window

    let releaseWindowRefs (window: Window<Chunk<float32>>) =
        window.Items |> List.iter Chunk.decRef

    let convolveWindow (retained: Window<Chunk<float32>>) =
        let emitStart, emitCount = retained.EmitRange
        if emitCount = 0u then
            []
        else
            let chunks = retained.Items |> List.toArray
            if chunks.Length < int emitStart + int emitCount then
                invalidArg "window" $"Native chunk convolution expected enough slices for emit range {retained.EmitRange}, got {chunks.Length}."

            let first = chunks[0]
            let chunkWidth, chunkHeight, chunkDepth = first.Size
            if chunkDepth <> 1UL then
                invalidArg "window" $"Native chunk convolution expects 2D slice chunks with depth 1, got {first.Size}."

            let width = int chunkWidth
            let height = int chunkHeight
            if width <= 0 || height <= 0 then
                invalidArg "window" $"Native chunk convolution expects positive slice dimensions, got {first.Size}."

            for chunk in chunks do
                validateTypedSliceChunk "native convolution" width height chunk

            convolveNativeFloat32Slices width height plan nativeKernel (int emitStart) (int emitCount) chunks

    let convolveRetained _debug (window: Window<Window<Chunk<float32>>>) =
        match window.Items with
        | [ retainedWindow ] ->
            try
                convolveWindow retainedWindow
            finally
                releaseWindowRefs retainedWindow
        | items ->
            for retainedWindow in items do
                releaseWindowRefs retainedWindow
            invalidArg "window" $"Native chunk convolution expected singleton retained windows, got {items.Length}."

    let windowStage =
        Stage.window $"{stageName}.window" (uint windowLength) (uint plan.PadZ) zeroMaker (uint outputBatchSize)

    let retainStage =
        Stage.map
            $"{stageName}.retain"
            retainWindowRefs
            memoryNeed
            id

    let computeStage =
        Stage.parallelCollect
            $"{stageName}.parallelCollect"
            1
            batchSize
            1
            0
            (fun _ retained -> retained)
            convolveRetained
            memoryNeed
            id

    Stage.compose windowStage retainStage
    |> fun stage -> Stage.compose stage computeStage

let private chunkConvolveFixedKernelNativeUInt8Stage
    (kernel: float32[,,])
    batchSize
    =
    if batchSize < 1 then
        invalidArg "batchSize" $"Native UInt8 chunk convolution expects a positive batch size, got {batchSize}."

    let plan = createKernelPlan kernel
    let nativeKernel = flattenKernelForNative kernel
    let outputBatchSize = batchSize
    let windowLength = plan.Depth + outputBatchSize - 1
    let memoryNeed nPixels =
        uint64 (windowLength + batchSize) * nPixels
    let suffix = if batchSize = 1 then "" else $".parallel{batchSize}"
    let stageName = $"chunkConvolveFixedKernelNativeUInt8{suffix}.{plan.Width}x{plan.Height}x{plan.Depth}"

    let zeroMaker _index (source: Chunk<uint8>) =
        let width, height, depth = source.Size
        if depth <> 1UL then
            invalidArg "source" $"Native UInt8 chunk convolution expects 2D slice chunks with depth 1, got {source.Size}."
        zeroChunkTyped<uint8> (int width) (int height)

    let releaseConsumed (window: Window<Chunk<uint8>>) =
        let _emitStart, emitCount = window.EmitRange
        if emitCount = 0u then
            window.Items |> List.iter Chunk.decRef
        else
            window.Items
            |> List.truncate (int window.ReleaseCount)
            |> List.iter Chunk.decRef

    let retainWindowRefs _debug (window: Window<Chunk<uint8>>) =
        window.Items |> List.iter (Chunk.incRef >> ignore)
        releaseConsumed window
        window

    let releaseWindowRefs (window: Window<Chunk<uint8>>) =
        window.Items |> List.iter Chunk.decRef

    let convolveWindow (retained: Window<Chunk<uint8>>) =
        let emitStart, emitCount = retained.EmitRange
        if emitCount = 0u then
            []
        else
            let chunks = retained.Items |> List.toArray
            if chunks.Length < int emitStart + int emitCount then
                invalidArg "window" $"Native UInt8 chunk convolution expected enough slices for emit range {retained.EmitRange}, got {chunks.Length}."

            let first = chunks[0]
            let chunkWidth, chunkHeight, chunkDepth = first.Size
            if chunkDepth <> 1UL then
                invalidArg "window" $"Native UInt8 chunk convolution expects 2D slice chunks with depth 1, got {first.Size}."

            let width = int chunkWidth
            let height = int chunkHeight
            if width <= 0 || height <= 0 then
                invalidArg "window" $"Native UInt8 chunk convolution expects positive slice dimensions, got {first.Size}."

            for chunk in chunks do
                validateTypedSliceChunk "native UInt8 convolution" width height chunk

            convolveNativeUInt8Slices width height plan nativeKernel (int emitStart) (int emitCount) chunks

    let convolveRetained _debug (window: Window<Window<Chunk<uint8>>>) =
        match window.Items with
        | [ retainedWindow ] ->
            try
                convolveWindow retainedWindow
            finally
                releaseWindowRefs retainedWindow
        | items ->
            for retainedWindow in items do
                releaseWindowRefs retainedWindow
            invalidArg "window" $"Native UInt8 chunk convolution expected singleton retained windows, got {items.Length}."

    let windowStage =
        Stage.window $"{stageName}.window" (uint windowLength) (uint plan.PadZ) zeroMaker (uint outputBatchSize)

    let retainStage =
        Stage.map
            $"{stageName}.retain"
            retainWindowRefs
            memoryNeed
            id

    let computeStage =
        Stage.parallelCollect
            $"{stageName}.parallelCollect"
            1
            batchSize
            1
            0
            (fun _ retained -> retained)
            convolveRetained
            memoryNeed
            id

    Stage.compose windowStage retainStage
    |> fun stage -> Stage.compose stage computeStage

let convolveFixedKernel<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    kernel
    : Stage<Chunk<'T>, Chunk<'T>> =
    chunkConvolveFixedKernelStage<'T> kernel 1

let convolveFixedKernelParallel<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    kernel
    windowSize
    : Stage<Chunk<'T>, Chunk<'T>> =
    if windowSize <= 1 then
        convolveFixedKernel<'T> kernel
    else
        chunkConvolveFixedKernelStage<'T> kernel windowSize

let convolveFixedKernelNativeFloat32
    kernel
    : Stage<Chunk<float32>, Chunk<float32>> =
    chunkConvolveFixedKernelNativeFloat32Stage kernel 1

let convolveFixedKernelNativeFloat32Parallel
    kernel
    windowSize
    : Stage<Chunk<float32>, Chunk<float32>> =
    if windowSize <= 1 then
        convolveFixedKernelNativeFloat32 kernel
    else
        chunkConvolveFixedKernelNativeFloat32Stage kernel windowSize

let convolveFixedKernelNativeUInt8
    kernel
    : Stage<Chunk<uint8>, Chunk<uint8>> =
    chunkConvolveFixedKernelNativeUInt8Stage kernel 1

let convolveFixedKernelNativeUInt8Parallel
    kernel
    windowSize
    : Stage<Chunk<uint8>, Chunk<uint8>> =
    if windowSize <= 1 then
        convolveFixedKernelNativeUInt8 kernel
    else
        chunkConvolveFixedKernelNativeUInt8Stage kernel windowSize

let private addUInt16HistogramInto (target: uint16[]) targetStart (source: uint16[]) sourceStart =
    let width = Vector<uint16>.Count
    let vectorEnd = 256 - (256 % width)
    let mutable i = 0
    while i < vectorEnd do
        let targetVector = Vector<uint16>(target, targetStart + i)
        let sourceVector = Vector<uint16>(source, sourceStart + i)
        (targetVector + sourceVector).CopyTo(target, targetStart + i)
        i <- i + width
    while i < 256 do
        target[targetStart + i] <- target[targetStart + i] + source[sourceStart + i]
        i <- i + 1

let private subtractUInt16HistogramFrom (target: uint16[]) targetStart (source: uint16[]) sourceStart =
    let width = Vector<uint16>.Count
    let vectorEnd = 256 - (256 % width)
    let mutable i = 0
    while i < vectorEnd do
        let targetVector = Vector<uint16>(target, targetStart + i)
        let sourceVector = Vector<uint16>(source, sourceStart + i)
        (targetVector - sourceVector).CopyTo(target, targetStart + i)
        i <- i + width
    while i < 256 do
        target[targetStart + i] <- target[targetStart + i] - source[sourceStart + i]
        i <- i + 1

let private clearUInt16Histogram (histogram: uint16[]) =
    Array.Clear(histogram, 0, histogram.Length)

let private addByteHistogramInto (target: byte[]) targetStart (source: byte[]) sourceStart =
    let width = Vector<byte>.Count
    let vectorEnd = 256 - (256 % width)
    let mutable i = 0
    while i < vectorEnd do
        let targetVector = Vector<byte>(target, targetStart + i)
        let sourceVector = Vector<byte>(source, sourceStart + i)
        (targetVector + sourceVector).CopyTo(target, targetStart + i)
        i <- i + width
    while i < 256 do
        target[targetStart + i] <- target[targetStart + i] + source[sourceStart + i]
        i <- i + 1

let private subtractByteHistogramFrom (target: byte[]) targetStart (source: byte[]) sourceStart =
    let width = Vector<byte>.Count
    let vectorEnd = 256 - (256 % width)
    let mutable i = 0
    while i < vectorEnd do
        let targetVector = Vector<byte>(target, targetStart + i)
        let sourceVector = Vector<byte>(source, sourceStart + i)
        (targetVector - sourceVector).CopyTo(target, targetStart + i)
        i <- i + width
    while i < 256 do
        target[targetStart + i] <- target[targetStart + i] - source[sourceStart + i]
        i <- i + 1

let private clearByteHistogram (histogram: byte[]) =
    Array.Clear(histogram, 0, histogram.Length)

type private PhMedianProfile =
    { mutable BuildZTicks: int64
      mutable UpdateZTicks: int64
      mutable EmitTicks: int64
      mutable YInitTicks: int64
      mutable KernelInitTicks: int64
      mutable RowScanAndUpdateTicks: int64
      mutable YUpdateTicks: int64
      mutable EmittedSlices: int
      mutable ZUpdates: int }

let private phProfileEnabled () =
    String.Equals(Environment.GetEnvironmentVariable("STACKPROCESSING_PROFILE_PH"), "1", StringComparison.Ordinal)

let private createPhMedianProfile () =
    { BuildZTicks = 0L
      UpdateZTicks = 0L
      EmitTicks = 0L
      YInitTicks = 0L
      KernelInitTicks = 0L
      RowScanAndUpdateTicks = 0L
      YUpdateTicks = 0L
      EmittedSlices = 0
      ZUpdates = 0 }

let inline private timestamp () =
    System.Diagnostics.Stopwatch.GetTimestamp()

let inline private elapsedSince start =
    timestamp () - start

let private secondsFromTicks ticks =
    float ticks / float System.Diagnostics.Stopwatch.Frequency

let private printPhMedianProfile (profile: PhMedianProfile) =
    let report name ticks =
        eprintfn $"[ph-profile] {name}={secondsFromTicks ticks:F6}s"

    eprintfn $"[ph-profile] emittedSlices={profile.EmittedSlices} zUpdates={profile.ZUpdates}"
    report "buildZ" profile.BuildZTicks
    report "updateZ" profile.UpdateZTicks
    report "emitTotal" profile.EmitTicks
    report "yInit" profile.YInitTicks
    report "kernelInit" profile.KernelInitTicks
    report "rowScanAndUpdate" profile.RowScanAndUpdateTicks
    report "yUpdate" profile.YUpdateTicks

let private medianFromUInt16Histogram totalCount (histogram: uint16[]) =
    let target = totalCount / 2 + 1
    let mutable cumulative = 0
    let mutable value = 0
    while value < 255 && cumulative < target do
        cumulative <- cumulative + int histogram[value]
        if cumulative < target then
            value <- value + 1
    uint8 value

let private medianFromByteHistogram totalCount (histogram: byte[]) =
    let target = totalCount / 2 + 1
    let mutable cumulative = 0
    let mutable value = 0
    while value < 255 && cumulative < target do
        cumulative <- cumulative + int histogram[value]
        if cumulative < target then
            value <- value + 1
    uint8 value

let private medianFromUInt16HistogramAt totalCount (histogram: uint16[]) start =
    let target = totalCount / 2 + 1
    let mutable cumulative = 0
    let mutable value = 0
    while value < 255 && cumulative < target do
        cumulative <- cumulative + int histogram[start + value]
        if cumulative < target then
            value <- value + 1
    uint8 value

let private clearUInt16PrefixTree (tree: uint16[]) =
    Array.Clear(tree, 0, tree.Length)

let private buildUInt16PrefixTreeFromHistogram (tree: uint16[]) (histogram: uint16[]) =
    Array.Clear(tree, 0, tree.Length)
    Array.Copy(histogram, 0, tree, 256, 256)
    let mutable i = 255
    while i > 0 do
        tree[i] <- tree[i <<< 1] + tree[(i <<< 1) + 1]
        i <- i - 1

let private addUInt16HistogramIntoTreeAndKernel (kernelHistogram: uint16[]) (tree: uint16[]) (source: uint16[]) sourceStart =
    addUInt16HistogramInto kernelHistogram 0 source sourceStart
    let mutable bin = 0
    while bin < 256 do
        let delta = source[sourceStart + bin]
        if delta <> 0us then
            let mutable index = 256 + bin
            while index > 0 do
                tree[index] <- tree[index] + delta
                index <- index >>> 1
        bin <- bin + 1

let private subtractUInt16HistogramFromTreeAndKernel (kernelHistogram: uint16[]) (tree: uint16[]) (source: uint16[]) sourceStart =
    subtractUInt16HistogramFrom kernelHistogram 0 source sourceStart
    let mutable bin = 0
    while bin < 256 do
        let delta = source[sourceStart + bin]
        if delta <> 0us then
            let mutable index = 256 + bin
            while index > 0 do
                tree[index] <- tree[index] - delta
                index <- index >>> 1
        bin <- bin + 1

let private addZeroCountIntoTreeAndKernel (kernelHistogram: uint16[]) (tree: uint16[]) count =
    kernelHistogram[0] <- kernelHistogram[0] + count
    let mutable index = 256
    while index > 0 do
        tree[index] <- tree[index] + count
        index <- index >>> 1

let private subtractZeroCountFromTreeAndKernel (kernelHistogram: uint16[]) (tree: uint16[]) count =
    kernelHistogram[0] <- kernelHistogram[0] - count
    let mutable index = 256
    while index > 0 do
        tree[index] <- tree[index] - count
        index <- index >>> 1

let private medianFromUInt16PrefixTree totalCount (tree: uint16[]) =
    let target = totalCount / 2 + 1
    let mutable cumulative = 0
    let mutable index = 1
    while index < 256 do
        let left = index <<< 1
        let leftCount = int tree[left]
        if cumulative + leftCount >= target then
            index <- left
        else
            cumulative <- cumulative + leftCount
            index <- left + 1
    uint8 (index - 256)

let private medianFromLaneMajorUInt16Histogram totalCount lanes (histogram: uint16[]) lane =
    let target = totalCount / 2 + 1
    let mutable cumulative = 0
    let mutable value = 0
    while value < 255 && cumulative < target do
        cumulative <- cumulative + int histogram[value * lanes + lane]
        if cumulative < target then
            value <- value + 1
    uint8 value

let private countLessEqualUInt16Span (span: ReadOnlySpan<uint16>) (pivot: uint16) =
    let vectorWidth = Vector<uint16>.Count
    let vectorEnd = span.Length - (span.Length % vectorWidth)
    let pivotVector = Vector<uint16>(pivot)
    let mutable count = 0
    let mutable i = 0
    while i < vectorEnd do
        let mask = Vector.LessThanOrEqual(Vector<uint16>(span.Slice(i, vectorWidth)), pivotVector)
        let mutable lane = 0
        while lane < vectorWidth do
            if mask[lane] <> 0us then
                count <- count + 1
            lane <- lane + 1
        i <- i + vectorWidth
    while i < span.Length do
        if span[i] <= pivot then
            count <- count + 1
        i <- i + 1
    count

let private countLessEqualInt16Span (span: ReadOnlySpan<int16>) (pivot: int16) =
    let vectorWidth = Vector<int16>.Count
    let vectorEnd = span.Length - (span.Length % vectorWidth)
    let pivotVector = Vector<int16>(pivot)
    let mutable count = 0
    let mutable i = 0
    while i < vectorEnd do
        let mask = Vector.LessThanOrEqual(Vector<int16>(span.Slice(i, vectorWidth)), pivotVector)
        let mutable lane = 0
        while lane < vectorWidth do
            if mask[lane] <> 0s then
                count <- count + 1
            lane <- lane + 1
        i <- i + vectorWidth
    while i < span.Length do
        if span[i] <= pivot then
            count <- count + 1
        i <- i + 1
    count

let private countLessEqualUInt16Window width height radius windowLength (window: Chunk<uint16>[]) x y (pivot: uint16) =
    let mutable count = 0
    let mutable validCount = 0
    let xStart = max 0 (x - radius)
    let xStop = min width (x + radius + 1)
    let xCount = xStop - xStart
    let mutable z = 0
    while z < windowLength do
        let pixels = Chunk.span<uint16> window[z]
        let mutable yy = y - radius
        while yy <= y + radius do
            if yy >= 0 && yy < height && xCount > 0 then
                let row = pixels.Slice(yy * width + xStart, xCount)
                let rowReadOnly = MemoryMarshal.CreateReadOnlySpan(&MemoryMarshal.GetReference(row), row.Length)
                count <- count + countLessEqualUInt16Span rowReadOnly pivot
                validCount <- validCount + xCount
            yy <- yy + 1
        z <- z + 1
    count + (windowLength * windowLength * windowLength - validCount)

let private countLessEqualInt16Window width height radius windowLength (window: Chunk<int16>[]) x y (pivot: int16) =
    let mutable count = 0
    let mutable validCount = 0
    let xStart = max 0 (x - radius)
    let xStop = min width (x + radius + 1)
    let xCount = xStop - xStart
    let mutable z = 0
    while z < windowLength do
        let pixels = Chunk.span<int16> window[z]
        let mutable yy = y - radius
        while yy <= y + radius do
            if yy >= 0 && yy < height && xCount > 0 then
                let row = pixels.Slice(yy * width + xStart, xCount)
                let rowReadOnly = MemoryMarshal.CreateReadOnlySpan(&MemoryMarshal.GetReference(row), row.Length)
                count <- count + countLessEqualInt16Span rowReadOnly pivot
                validCount <- validCount + xCount
            yy <- yy + 1
        z <- z + 1
    let zeroPaddingCount = windowLength * windowLength * windowLength - validCount
    if pivot >= 0s then count + zeroPaddingCount else count

let private kthUInt16ByVectorRank width height radius windowLength window x y k =
    let mutable lo = 0
    let mutable hi = int UInt16.MaxValue
    while lo < hi do
        let mid = lo + (hi - lo) / 2
        let rank = countLessEqualUInt16Window width height radius windowLength window x y (uint16 mid)
        if rank > k then hi <- mid else lo <- mid + 1
    uint16 lo

let private kthInt16ByVectorRank width height radius windowLength window x y k =
    let mutable lo = int Int16.MinValue
    let mutable hi = int Int16.MaxValue
    while lo < hi do
        let mid = lo + (hi - lo) / 2
        let rank = countLessEqualInt16Window width height radius windowLength window x y (int16 mid)
        if rank > k then hi <- mid else lo <- mid + 1
    int16 lo

let private selectKFloat32InPlace (values: float32[]) length k =
    let swap i j =
        let tmp = values[i]
        values[i] <- values[j]
        values[j] <- tmp

    let mutable left = 0
    let mutable right = length - 1
    let mutable result = 0.0f
    let mutable found = false
    while not found do
        if left = right then
            result <- values[left]
            found <- true
        else
            let pivot = values[left + (right - left) / 2]
            let mutable lt = left
            let mutable i = left
            let mutable gt = right
            while i <= gt do
                let value = values[i]
                if value < pivot then
                    swap lt i
                    lt <- lt + 1
                    i <- i + 1
                elif value > pivot then
                    swap i gt
                    gt <- gt - 1
                else
                    i <- i + 1
            if k < lt then
                right <- lt - 1
            elif k > gt then
                left <- gt + 1
            else
                result <- values[k]
                found <- true
    result

let private selectKUInt8InPlace (values: uint8[]) length k =
    let swap i j =
        let tmp = values[i]
        values[i] <- values[j]
        values[j] <- tmp

    let mutable left = 0
    let mutable right = length - 1
    let mutable result = 0uy
    let mutable found = false
    while not found do
        if left = right then
            result <- values[left]
            found <- true
        else
            let pivot = values[left + (right - left) / 2]
            let mutable lt = left
            let mutable i = left
            let mutable gt = right
            while i <= gt do
                let value = values[i]
                if value < pivot then
                    swap lt i
                    lt <- lt + 1
                    i <- i + 1
                elif value > pivot then
                    swap i gt
                    gt <- gt - 1
                else
                    i <- i + 1
            if k < lt then
                right <- lt - 1
            elif k > gt then
                left <- gt + 1
            else
                result <- values[k]
                found <- true
    result

let private fillUInt8WindowScratch width height radius windowLength (window: Chunk<uint8>[]) x y (scratch: uint8[]) =
    let mutable index = 0
    let mutable z = 0
    while z < windowLength do
        let pixels = Chunk.span<uint8> window[z]
        let mutable yy = y - radius
        while yy <= y + radius do
            let mutable xx = x - radius
            while xx <= x + radius do
                scratch[index] <-
                    if xx >= 0 && xx < width && yy >= 0 && yy < height then
                        pixels[yy * width + xx]
                    else
                        0uy
                index <- index + 1
                xx <- xx + 1
            yy <- yy + 1
        z <- z + 1

let private fillFloat32WindowScratch width height radius windowLength (window: Chunk<float32>[]) x y (scratch: float32[]) =
    let mutable index = 0
    let mutable z = 0
    while z < windowLength do
        let pixels = Chunk.span<float32> window[z]
        let mutable yy = y - radius
        while yy <= y + radius do
            let mutable xx = x - radius
            while xx <= x + radius do
                scratch[index] <-
                    if xx >= 0 && xx < width && yy >= 0 && yy < height then
                        pixels[yy * width + xx]
                    else
                        0.0f
                index <- index + 1
                xx <- xx + 1
            yy <- yy + 1
        z <- z + 1

let private selectKUInt16InPlace (values: uint16[]) length k =
    let swap i j =
        let tmp = values[i]
        values[i] <- values[j]
        values[j] <- tmp

    let medianOfThree left right =
        let mid = (left + right) >>> 1
        if values[mid] < values[left] then
            swap left mid
        if values[right] < values[left] then
            swap left right
        if values[right] < values[mid] then
            swap mid right
        values[mid]

    let mutable left = 0
    let mutable right = length - 1
    let mutable result = 0us
    let mutable found = false
    while not found do
        if left = right then
            result <- values[left]
            found <- true
        else
            let pivot = medianOfThree left right
            let mutable lt = left
            let mutable i = left
            let mutable gt = right
            while i <= gt do
                let value = values[i]
                if value < pivot then
                    swap lt i
                    lt <- lt + 1
                    i <- i + 1
                elif value > pivot then
                    swap i gt
                    gt <- gt - 1
                else
                    i <- i + 1
            if k < lt then
                right <- lt - 1
            elif k > gt then
                left <- gt + 1
            else
                result <- values[k]
                found <- true
    result

let private selectKInt16InPlace (values: int16[]) length k =
    let swap i j =
        let tmp = values[i]
        values[i] <- values[j]
        values[j] <- tmp

    let mutable left = 0
    let mutable right = length - 1
    let mutable result = 0s
    let mutable found = false
    while not found do
        if left = right then
            result <- values[left]
            found <- true
        else
            let pivot = values[left + (right - left) / 2]
            let mutable lt = left
            let mutable i = left
            let mutable gt = right
            while i <= gt do
                let value = values[i]
                if value < pivot then
                    swap lt i
                    lt <- lt + 1
                    i <- i + 1
                elif value > pivot then
                    swap i gt
                    gt <- gt - 1
                else
                    i <- i + 1
            if k < lt then
                right <- lt - 1
            elif k > gt then
                left <- gt + 1
            else
                result <- values[k]
                found <- true
    result

let private fillUInt16WindowScratch width height radius windowLength (window: Chunk<uint16>[]) x y (scratch: uint16[]) =
    let mutable index = 0
    let mutable z = 0
    while z < windowLength do
        let pixels = Chunk.span<uint16> window[z]
        let mutable yy = y - radius
        while yy <= y + radius do
            let mutable xx = x - radius
            while xx <= x + radius do
                scratch[index] <-
                    if xx >= 0 && xx < width && yy >= 0 && yy < height then
                        pixels[yy * width + xx]
                    else
                        0us
                index <- index + 1
                xx <- xx + 1
            yy <- yy + 1
        z <- z + 1

let private fillInt16WindowScratch width height radius windowLength (window: Chunk<int16>[]) x y (scratch: int16[]) =
    let mutable index = 0
    let mutable z = 0
    while z < windowLength do
        let pixels = Chunk.span<int16> window[z]
        let mutable yy = y - radius
        while yy <= y + radius do
            let mutable xx = x - radius
            while xx <= x + radius do
                scratch[index] <-
                    if xx >= 0 && xx < width && yy >= 0 && yy < height then
                        pixels[yy * width + xx]
                    else
                        0s
                index <- index + 1
                xx <- xx + 1
            yy <- yy + 1
        z <- z + 1

let private medianQuickselectUInt8Slice width height radius (window: Chunk<uint8>[]) =
    let windowLength = 2 * radius + 1
    let totalCount = windowLength * windowLength * windowLength
    let medianIndex = totalCount / 2
    let output = Chunk.create<uint8> (uint64 width, uint64 height, 1UL)
    try
        let outputPixels = Chunk.span<uint8> output
        let scratch = Array.zeroCreate<uint8> totalCount
        let mutable y = 0
        while y < height do
            let rowOffset = y * width
            let mutable x = 0
            while x < width do
                fillUInt8WindowScratch width height radius windowLength window x y scratch
                outputPixels[rowOffset + x] <- selectKUInt8InPlace scratch totalCount medianIndex
                x <- x + 1
            y <- y + 1
        output
    with
    | _ ->
        Chunk.decRef output
        reraise()

let private medianNthElementUInt16Slice width height radius (window: Chunk<uint16>[]) =
    let windowLength = 2 * radius + 1
    let totalCount = windowLength * windowLength * windowLength
    let medianIndex = totalCount / 2
    let output = Chunk.create<uint16> (uint64 width, uint64 height, 1UL)
    try
        let outputPixels = Chunk.span<uint16> output
        let mutable y = 0
        while y < height do
            let rowOffset = y * width
            let mutable x = 0
            while x < width do
                outputPixels[rowOffset + x] <- kthUInt16ByVectorRank width height radius windowLength window x y medianIndex
                x <- x + 1
            y <- y + 1
        output
    with
    | _ ->
        Chunk.decRef output
        reraise()

let private medianQuickselectUInt16Slice width height radius (window: Chunk<uint16>[]) =
    let windowLength = 2 * radius + 1
    let totalCount = windowLength * windowLength * windowLength
    let medianIndex = totalCount / 2
    let output = Chunk.create<uint16> (uint64 width, uint64 height, 1UL)
    try
        let outputPixels = Chunk.span<uint16> output
        let scratch = Array.zeroCreate<uint16> totalCount
        let mutable y = 0
        while y < height do
            let rowOffset = y * width
            let mutable x = 0
            while x < width do
                fillUInt16WindowScratch width height radius windowLength window x y scratch
                outputPixels[rowOffset + x] <- selectKUInt16InPlace scratch totalCount medianIndex
                x <- x + 1
            y <- y + 1
        output
    with
    | _ ->
        Chunk.decRef output
        reraise()

let private medianSortUInt16Slice width height radius (window: Chunk<uint16>[]) =
    let windowLength = 2 * radius + 1
    let totalCount = windowLength * windowLength * windowLength
    let medianIndex = totalCount / 2
    let output = Chunk.create<uint16> (uint64 width, uint64 height, 1UL)
    try
        let outputPixels = Chunk.span<uint16> output
        let scratch = Array.zeroCreate<uint16> totalCount
        let mutable y = 0
        while y < height do
            let rowOffset = y * width
            let mutable x = 0
            while x < width do
                fillUInt16WindowScratch width height radius windowLength window x y scratch
                Array.sortInPlace scratch
                outputPixels[rowOffset + x] <- scratch[medianIndex]
                x <- x + 1
            y <- y + 1
        output
    with
    | _ ->
        Chunk.decRef output
        reraise()

let private medianQuickselectInt16Slice width height radius (window: Chunk<int16>[]) =
    let windowLength = 2 * radius + 1
    let totalCount = windowLength * windowLength * windowLength
    let medianIndex = totalCount / 2
    let output = Chunk.create<int16> (uint64 width, uint64 height, 1UL)
    try
        let outputPixels = Chunk.span<int16> output
        let scratch = Array.zeroCreate<int16> totalCount
        let mutable y = 0
        while y < height do
            let rowOffset = y * width
            let mutable x = 0
            while x < width do
                fillInt16WindowScratch width height radius windowLength window x y scratch
                outputPixels[rowOffset + x] <- selectKInt16InPlace scratch totalCount medianIndex
                x <- x + 1
            y <- y + 1
        output
    with
    | _ ->
        Chunk.decRef output
        reraise()

let private medianNthElementInt16Slice width height radius (window: Chunk<int16>[]) =
    let windowLength = 2 * radius + 1
    let totalCount = windowLength * windowLength * windowLength
    let medianIndex = totalCount / 2
    let output = Chunk.create<int16> (uint64 width, uint64 height, 1UL)
    try
        let outputPixels = Chunk.span<int16> output
        let mutable y = 0
        while y < height do
            let rowOffset = y * width
            let mutable x = 0
            while x < width do
                outputPixels[rowOffset + x] <- kthInt16ByVectorRank width height radius windowLength window x y medianIndex
                x <- x + 1
            y <- y + 1
        output
    with
    | _ ->
        Chunk.decRef output
        reraise()

let private medianNthElementFloat32Slice width height radius (window: Chunk<float32>[]) =
    let windowLength = 2 * radius + 1
    let totalCount = windowLength * windowLength * windowLength
    let medianIndex = totalCount / 2
    let output = Chunk.create<float32> (uint64 width, uint64 height, 1UL)
    try
        let outputPixels = Chunk.span<float32> output
        let scratch = Array.zeroCreate<float32> totalCount
        let mutable y = 0
        while y < height do
            let rowOffset = y * width
            let mutable x = 0
            while x < width do
                fillFloat32WindowScratch width height radius windowLength window x y scratch
                outputPixels[rowOffset + x] <- selectKFloat32InPlace scratch totalCount medianIndex
                x <- x + 1
            y <- y + 1
        output
    with
    | _ ->
        Chunk.decRef output
        reraise()

let private buildZHistogramsUInt8 width height (window: seq<Chunk<uint8>>) =
    let pixelCount = width * height
    if pixelCount > Int32.MaxValue / 256 then
        invalidArg "window" $"UInt8 PH median dense z-histogram would exceed Int32 indexing for {width}x{height} slices."

    let zHistograms = Array.zeroCreate<uint16> (pixelCount * 256)
    for chunk in window do
        let inputPixels = Chunk.span<uint8> chunk
        let mutable p = 0
        while p < pixelCount do
            let index = p * 256 + int inputPixels[p]
            zHistograms[index] <- zHistograms[index] + 1us
            p <- p + 1
    zHistograms

let private buildZHistogramsUInt8ByteBins width height (window: seq<Chunk<uint8>>) =
    let pixelCount = width * height
    if pixelCount > Int32.MaxValue / 256 then
        invalidArg "window" $"UInt8 PH median dense byte z-histogram would exceed Int32 indexing for {width}x{height} slices."

    let zHistograms = Array.zeroCreate<byte> (pixelCount * 256)
    for chunk in window do
        let inputPixels = Chunk.span<uint8> chunk
        let mutable p = 0
        while p < pixelCount do
            let index = p * 256 + int inputPixels[p]
            zHistograms[index] <- zHistograms[index] + 1uy
            p <- p + 1
    zHistograms

let private blockedZHistogramIndex xBlockCount blockWidth y xBlock bin =
    (((y * xBlockCount + xBlock) * 256 + bin) * blockWidth)

let private buildBlockedZHistogramsUInt8 width height (window: seq<Chunk<uint8>>) =
    let blockWidth = Vector<uint16>.Count
    let xBlockCount = (width + blockWidth - 1) / blockWidth
    let zHistograms = Array.zeroCreate<uint16> (height * xBlockCount * 256 * blockWidth)

    for chunk in window do
        let inputPixels = Chunk.span<uint8> chunk
        let mutable y = 0
        while y < height do
            let rowOffset = y * width
            let mutable x = 0
            while x < width do
                let xBlock = x / blockWidth
                let lane = x - xBlock * blockWidth
                let bin = int inputPixels[rowOffset + x]
                let index = blockedZHistogramIndex xBlockCount blockWidth y xBlock bin + lane
                zHistograms[index] <- zHistograms[index] + 1us
                x <- x + 1
            y <- y + 1

    zHistograms

let private addZRowToYHistograms width (zHistograms: uint16[]) y (yHistograms: uint16[]) =
    let mutable x = 0
    while x < width do
        let sourceStart = (y * width + x) * 256
        let targetStart = x * 256
        addUInt16HistogramInto yHistograms targetStart zHistograms sourceStart
        x <- x + 1

let private subtractZRowFromYHistograms width (zHistograms: uint16[]) y (yHistograms: uint16[]) =
    let mutable x = 0
    while x < width do
        let sourceStart = (y * width + x) * 256
        let targetStart = x * 256
        subtractUInt16HistogramFrom yHistograms targetStart zHistograms sourceStart
        x <- x + 1

let private addZRowToByteYHistograms width (zHistograms: byte[]) y (yHistograms: byte[]) =
    let mutable x = 0
    while x < width do
        let sourceStart = (y * width + x) * 256
        let targetStart = x * 256
        addByteHistogramInto yHistograms targetStart zHistograms sourceStart
        x <- x + 1

let private subtractZRowFromByteYHistograms width (zHistograms: byte[]) y (yHistograms: byte[]) =
    let mutable x = 0
    while x < width do
        let sourceStart = (y * width + x) * 256
        let targetStart = x * 256
        subtractByteHistogramFrom yHistograms targetStart zHistograms sourceStart
        x <- x + 1

let private addZeroZRowToYHistograms width windowLength (yHistograms: uint16[]) =
    let mutable x = 0
    while x < width do
        let targetStart = x * 256
        yHistograms[targetStart] <- yHistograms[targetStart] + uint16 windowLength
        x <- x + 1

let private subtractZeroZRowFromYHistograms width windowLength (yHistograms: uint16[]) =
    let mutable x = 0
    while x < width do
        let targetStart = x * 256
        yHistograms[targetStart] <- yHistograms[targetStart] - uint16 windowLength
        x <- x + 1

let private addZeroZRowToByteYHistograms width windowLength (yHistograms: byte[]) =
    let mutable x = 0
    while x < width do
        let targetStart = x * 256
        yHistograms[targetStart] <- yHistograms[targetStart] + byte windowLength
        x <- x + 1

let private subtractZeroZRowFromByteYHistograms width windowLength (yHistograms: byte[]) =
    let mutable x = 0
    while x < width do
        let targetStart = x * 256
        yHistograms[targetStart] <- yHistograms[targetStart] - byte windowLength
        x <- x + 1

let private addYColumnToKernelHistogram (kernelHistogram: uint16[]) (yHistograms: uint16[]) x =
    addUInt16HistogramInto kernelHistogram 0 yHistograms (x * 256)

let private subtractYColumnFromKernelHistogram (kernelHistogram: uint16[]) (yHistograms: uint16[]) x =
    subtractUInt16HistogramFrom kernelHistogram 0 yHistograms (x * 256)

let private addByteYColumnToKernelHistogram (kernelHistogram: byte[]) (yHistograms: byte[]) x =
    addByteHistogramInto kernelHistogram 0 yHistograms (x * 256)

let private subtractByteYColumnFromKernelHistogram (kernelHistogram: byte[]) (yHistograms: byte[]) x =
    subtractByteHistogramFrom kernelHistogram 0 yHistograms (x * 256)

let private updateZHistogramsUInt8 width height (zHistograms: uint16[]) (subtractChunk: Chunk<uint8>) (addChunk: Chunk<uint8>) =
    let pixelCount = width * height
    let subtractPixels = Chunk.span<uint8> subtractChunk
    let addPixels = Chunk.span<uint8> addChunk
    let mutable p = 0
    while p < pixelCount do
        let baseIndex = p * 256
        zHistograms[baseIndex + int subtractPixels[p]] <- zHistograms[baseIndex + int subtractPixels[p]] - 1us
        zHistograms[baseIndex + int addPixels[p]] <- zHistograms[baseIndex + int addPixels[p]] + 1us
        p <- p + 1

let private updateZHistogramsUInt8ByteBins width height (zHistograms: byte[]) (subtractChunk: Chunk<uint8>) (addChunk: Chunk<uint8>) =
    let pixelCount = width * height
    let subtractPixels = Chunk.span<uint8> subtractChunk
    let addPixels = Chunk.span<uint8> addChunk
    let mutable p = 0
    while p < pixelCount do
        let baseIndex = p * 256
        zHistograms[baseIndex + int subtractPixels[p]] <- zHistograms[baseIndex + int subtractPixels[p]] - 1uy
        zHistograms[baseIndex + int addPixels[p]] <- zHistograms[baseIndex + int addPixels[p]] + 1uy
        p <- p + 1

let private updateBlockedZHistogramsUInt8 width height (zHistograms: uint16[]) (subtractChunk: Chunk<uint8>) (addChunk: Chunk<uint8>) =
    let blockWidth = Vector<uint16>.Count
    let xBlockCount = (width + blockWidth - 1) / blockWidth
    let subtractPixels = Chunk.span<uint8> subtractChunk
    let addPixels = Chunk.span<uint8> addChunk
    let mutable y = 0
    while y < height do
        let rowOffset = y * width
        let mutable x = 0
        while x < width do
            let xBlock = x / blockWidth
            let lane = x - xBlock * blockWidth
            let subtractIndex = blockedZHistogramIndex xBlockCount blockWidth y xBlock (int subtractPixels[rowOffset + x]) + lane
            let addIndex = blockedZHistogramIndex xBlockCount blockWidth y xBlock (int addPixels[rowOffset + x]) + lane
            zHistograms[subtractIndex] <- zHistograms[subtractIndex] - 1us
            zHistograms[addIndex] <- zHistograms[addIndex] + 1us
            x <- x + 1
        y <- y + 1

let private copyUInt16HistogramInto (target: uint16[]) targetStart (source: uint16[]) =
    Array.Copy(source, 0, target, targetStart, 256)

let private addUInt16ArrayInto (target: uint16[]) (source: uint16[]) =
    if target.Length <> source.Length then
        invalidArg "source" $"UInt16 histogram arrays must have identical length; got {target.Length} and {source.Length}."
    let width = Vector<uint16>.Count
    let vectorEnd = target.Length - (target.Length % width)
    let mutable i = 0
    while i < vectorEnd do
        let targetVector = Vector<uint16>(target, i)
        let sourceVector = Vector<uint16>(source, i)
        (targetVector + sourceVector).CopyTo(target, i)
        i <- i + width
    while i < target.Length do
        target[i] <- target[i] + source[i]
        i <- i + 1

let private subtractUInt16ArrayFrom (target: uint16[]) (source: uint16[]) =
    if target.Length <> source.Length then
        invalidArg "source" $"UInt16 histogram arrays must have identical length; got {target.Length} and {source.Length}."
    let width = Vector<uint16>.Count
    let vectorEnd = target.Length - (target.Length % width)
    let mutable i = 0
    while i < vectorEnd do
        let targetVector = Vector<uint16>(target, i)
        let sourceVector = Vector<uint16>(source, i)
        (targetVector - sourceVector).CopyTo(target, i)
        i <- i + width
    while i < target.Length do
        target[i] <- target[i] - source[i]
        i <- i + 1

let private buildXHistogramsUInt8 width height radius (chunk: Chunk<uint8>) =
    let windowLength = 2 * radius + 1
    let pixelCount = width * height
    let inputPixels = Chunk.span<uint8> chunk
    let xHistograms = Array.zeroCreate<uint16> (pixelCount * 256)
    let histogram = Array.zeroCreate<uint16> 256

    for y in 0 .. height - 1 do
        clearUInt16Histogram histogram
        let rowOffset = y * width

        for dx in -radius .. radius do
            if dx >= 0 && dx < width then
                histogram[int inputPixels[rowOffset + dx]] <- histogram[int inputPixels[rowOffset + dx]] + 1us
            else
                histogram[0] <- histogram[0] + 1us

        for x in 0 .. width - 1 do
            copyUInt16HistogramInto xHistograms ((rowOffset + x) * 256) histogram

            if x < width - 1 then
                let leaving = x - radius
                let entering = x + radius + 1
                if leaving >= 0 && leaving < width then
                    histogram[int inputPixels[rowOffset + leaving]] <- histogram[int inputPixels[rowOffset + leaving]] - 1us
                else
                    histogram[0] <- histogram[0] - 1us

                if entering >= 0 && entering < width then
                    histogram[int inputPixels[rowOffset + entering]] <- histogram[int inputPixels[rowOffset + entering]] + 1us
                else
                    histogram[0] <- histogram[0] + 1us

    xHistograms

let private buildZeroXyHistogramsUInt8 width height windowLength =
    let pixelCount = width * height
    let xyHistograms = Array.zeroCreate<uint16> (pixelCount * 256)
    let count = uint16 (windowLength * windowLength)
    let mutable p = 0
    while p < pixelCount do
        xyHistograms[p * 256] <- count
        p <- p + 1
    xyHistograms

let private addXHistogramToYHistogram (target: uint16[]) (xHistograms: uint16[]) sourcePixel =
    addUInt16HistogramInto target 0 xHistograms (sourcePixel * 256)

let private subtractXHistogramFromYHistogram (target: uint16[]) (xHistograms: uint16[]) sourcePixel =
    subtractUInt16HistogramFrom target 0 xHistograms (sourcePixel * 256)

let private buildXyHistogramsUInt8 width height radius (chunk: Chunk<uint8>) =
    let windowLength = 2 * radius + 1
    let xHistograms = buildXHistogramsUInt8 width height radius chunk
    let xyHistograms = Array.zeroCreate<uint16> (width * height * 256)
    let histogram = Array.zeroCreate<uint16> 256
    let zeroXCount = uint16 windowLength

    for x in 0 .. width - 1 do
        clearUInt16Histogram histogram

        for dy in -radius .. radius do
            if dy >= 0 && dy < height then
                addXHistogramToYHistogram histogram xHistograms (dy * width + x)
            else
                histogram[0] <- histogram[0] + zeroXCount

        for y in 0 .. height - 1 do
            copyUInt16HistogramInto xyHistograms ((y * width + x) * 256) histogram

            if y < height - 1 then
                let leaving = y - radius
                let entering = y + radius + 1
                if leaving >= 0 && leaving < height then
                    subtractXHistogramFromYHistogram histogram xHistograms (leaving * width + x)
                else
                    histogram[0] <- histogram[0] - zeroXCount

                if entering >= 0 && entering < height then
                    addXHistogramToYHistogram histogram xHistograms (entering * width + x)
                else
                    histogram[0] <- histogram[0] + zeroXCount

    xyHistograms

let private emitMedianSliceFromXyKernelHistograms width height radius windowLength (kernelHistograms: uint16[]) =
    let totalCount = windowLength * windowLength * windowLength
    if totalCount > int UInt16.MaxValue then
        invalidArg "radius" $"UInt8 x-first dense PH median stores counts as UInt16 and supports at most {UInt16.MaxValue} samples; got {totalCount}."

    let output = Chunk.create<uint8> (uint64 width, uint64 height, 1UL)
    try
        let outputPixels = Chunk.span<uint8> output
        let pixelCount = width * height
        let mutable p = 0
        while p < pixelCount do
            outputPixels[p] <- medianFromUInt16HistogramAt totalCount kernelHistograms (p * 256)
            p <- p + 1
        output
    with
    | _ ->
        Chunk.decRef output
        reraise()

let private addStoredXHistogramToYBlock (target: uint16[]) targetStart (xHistograms: uint16[]) sourceStart =
    addUInt16HistogramInto target targetStart xHistograms sourceStart

let private subtractStoredXHistogramFromYBlock (target: uint16[]) targetStart (xHistograms: uint16[]) sourceStart =
    subtractUInt16HistogramFrom target targetStart xHistograms sourceStart

let private medianPerreaultHebertUInt8DenseSliceXBlock width height radius (window: ChunkSlice[]) =
    let windowLength = 2 * radius + 1
    let totalCount = windowLength * windowLength * windowLength
    if totalCount > int UInt16.MaxValue then
        invalidArg "radius" $"UInt8 x-block dense PH median stores counts as UInt16 and supports at most {UInt16.MaxValue} samples; got {totalCount}."

    let blockWidth = Vector<uint16>.Count
    let output = Chunk.create<uint8> (uint64 width, uint64 height, 1UL)
    try
        let outputPixels = Chunk.span<uint8> output
        let xHistograms = Array.zeroCreate<uint16> (windowLength * height * blockWidth * 256)
        let yHistograms = Array.zeroCreate<uint16> (windowLength * blockWidth * 256)
        let kernelHistograms = Array.zeroCreate<uint16> (blockWidth * 256)
        let zeroXCount = uint16 windowLength

        let xHistogramStart zi y lane =
            (((zi * height + y) * blockWidth + lane) * 256)

        let yHistogramStart zi lane =
            ((zi * blockWidth + lane) * 256)

        let kernelHistogramStart lane =
            lane * 256

        let addXRowToY row =
            let mutable zi = 0
            while zi < windowLength do
                let mutable lane = 0
                while lane < blockWidth do
                    let targetStart = yHistogramStart zi lane
                    if row >= 0 && row < height then
                        addStoredXHistogramToYBlock yHistograms targetStart xHistograms (xHistogramStart zi row lane)
                    else
                        yHistograms[targetStart] <- yHistograms[targetStart] + zeroXCount
                    lane <- lane + 1
                zi <- zi + 1

        let subtractXRowFromY row =
            let mutable zi = 0
            while zi < windowLength do
                let mutable lane = 0
                while lane < blockWidth do
                    let targetStart = yHistogramStart zi lane
                    if row >= 0 && row < height then
                        subtractStoredXHistogramFromYBlock yHistograms targetStart xHistograms (xHistogramStart zi row lane)
                    else
                        yHistograms[targetStart] <- yHistograms[targetStart] - zeroXCount
                    lane <- lane + 1
                zi <- zi + 1

        let addYHistogramsToKernel () =
            let mutable zi = 0
            while zi < windowLength do
                let mutable lane = 0
                while lane < blockWidth do
                    addUInt16HistogramInto kernelHistograms (kernelHistogramStart lane) yHistograms (yHistogramStart zi lane)
                    lane <- lane + 1
                zi <- zi + 1

        let mutable blockStart = 0
        while blockStart < width do
            let lanes = min blockWidth (width - blockStart)
            Array.Clear(xHistograms, 0, xHistograms.Length)

            let mutable zi = 0
            while zi < windowLength do
                let pixels = Chunk.span<uint8> window[zi].Chunk
                let mutable y = 0
                while y < height do
                    let rowOffset = y * width
                    let mutable lane = 0
                    while lane < lanes do
                        let x = blockStart + lane
                        let start = xHistogramStart zi y lane
                        let mutable dx = -radius
                        while dx <= radius do
                            let sx = x + dx
                            if sx >= 0 && sx < width then
                                xHistograms[start + int pixels[rowOffset + sx]] <- xHistograms[start + int pixels[rowOffset + sx]] + 1us
                            else
                                xHistograms[start] <- xHistograms[start] + 1us
                            dx <- dx + 1
                        lane <- lane + 1
                    y <- y + 1
                zi <- zi + 1

            Array.Clear(yHistograms, 0, yHistograms.Length)
            for yy in -radius .. radius do
                addXRowToY yy

            let mutable y = 0
            while y < height do
                Array.Clear(kernelHistograms, 0, kernelHistograms.Length)
                addYHistogramsToKernel ()

                let rowOffset = y * width
                let mutable lane = 0
                while lane < lanes do
                    outputPixels[rowOffset + blockStart + lane] <-
                        medianFromUInt16HistogramAt totalCount kernelHistograms (kernelHistogramStart lane)
                    lane <- lane + 1

                if y < height - 1 then
                    subtractXRowFromY (y - radius)
                    addXRowToY (y + radius + 1)

                y <- y + 1

            blockStart <- blockStart + blockWidth

        output
    with
    | _ ->
        Chunk.decRef output
        reraise()

let private fillZRowBlockLaneMajor width height windowLength (zHistograms: uint16[]) row xStart xSlots (target: uint16[]) =
    Array.Clear(target, 0, target.Length)
    if row >= 0 && row < height then
        let mutable slot = 0
        while slot < xSlots do
            let x = xStart + slot
            if x >= 0 && x < width then
                let sourceStart = (row * width + x) * 256
                let mutable bin = 0
                while bin < 256 do
                    target[bin * xSlots + slot] <- zHistograms[sourceStart + bin]
                    bin <- bin + 1
            else
                target[slot] <- uint16 windowLength
            slot <- slot + 1
    else
        let count = uint16 windowLength
        let mutable slot = 0
        while slot < xSlots do
            target[slot] <- count
            slot <- slot + 1

let private addLaneMajorUInt16ArrayInto (target: uint16[]) (source: uint16[]) =
    addUInt16ArrayInto target source

let private subtractLaneMajorUInt16ArrayFrom (target: uint16[]) (source: uint16[]) =
    subtractUInt16ArrayFrom target source

let private medianPerreaultHebertUInt8DenseSliceFromZHistogramsTransposedXBlock width height radius windowLength (zHistograms: uint16[]) =
    let totalCount = windowLength * windowLength * windowLength
    if totalCount > int UInt16.MaxValue then
        invalidArg "radius" $"UInt8 transposed x-block dense PH median stores counts as UInt16 and supports at most {UInt16.MaxValue} samples; got {totalCount}."

    let blockWidth = Vector<uint16>.Count
    let output = Chunk.create<uint8> (uint64 width, uint64 height, 1UL)
    try
        let outputPixels = Chunk.span<uint8> output

        let mutable blockStart = 0
        while blockStart < width do
            let lanes = min blockWidth (width - blockStart)
            let xStart = blockStart - radius
            let xSlots = lanes + 2 * radius
            let yBlock = Array.zeroCreate<uint16> (256 * xSlots)
            let rowBlock = Array.zeroCreate<uint16> (256 * xSlots)
            let kernelBlock = Array.zeroCreate<uint16> (256 * lanes)

            Array.Clear(yBlock, 0, yBlock.Length)
            for yy in -radius .. radius do
                fillZRowBlockLaneMajor width height windowLength zHistograms yy xStart xSlots rowBlock
                addLaneMajorUInt16ArrayInto yBlock rowBlock

            let mutable y = 0
            while y < height do
                Array.Clear(kernelBlock, 0, kernelBlock.Length)
                let mutable bin = 0
                while bin < 256 do
                    let yBinStart = bin * xSlots
                    let kernelBinStart = bin * lanes
                    let mutable lane = 0
                    while lane < lanes do
                        let mutable sum = 0
                        let mutable dx = 0
                        while dx < windowLength do
                            sum <- sum + int yBlock[yBinStart + lane + dx]
                            dx <- dx + 1
                        kernelBlock[kernelBinStart + lane] <- uint16 sum
                        lane <- lane + 1
                    bin <- bin + 1

                let rowOffset = y * width
                let mutable lane = 0
                while lane < lanes do
                    outputPixels[rowOffset + blockStart + lane] <-
                        medianFromLaneMajorUInt16Histogram totalCount lanes kernelBlock lane
                    lane <- lane + 1

                if y < height - 1 then
                    fillZRowBlockLaneMajor width height windowLength zHistograms (y - radius) xStart xSlots rowBlock
                    subtractLaneMajorUInt16ArrayFrom yBlock rowBlock
                    fillZRowBlockLaneMajor width height windowLength zHistograms (y + radius + 1) xStart xSlots rowBlock
                    addLaneMajorUInt16ArrayInto yBlock rowBlock

                y <- y + 1

            blockStart <- blockStart + blockWidth

        output
    with
    | _ ->
        Chunk.decRef output
        reraise()

let private medianPerreaultHebertUInt8DenseSliceFromZHistogramsWithProfile width height radius windowLength (zHistograms: uint16[]) (profile: PhMedianProfile option) =
    let totalCount = windowLength * windowLength * windowLength
    if totalCount > int UInt16.MaxValue then
        invalidArg "radius" $"UInt8 dense PH median first version stores counts as UInt16 and supports at most {UInt16.MaxValue} samples; got {totalCount}."

    let emitStart =
        match profile with
        | Some _ -> timestamp ()
        | None -> 0L
    let output = Chunk.create<uint8> (uint64 width, uint64 height, 1UL)
    try
        let outputPixels = Chunk.span<uint8> output
        let rowProfileMode =
            match profile with
            | Some _ -> Environment.GetEnvironmentVariable("STACKPROCESSING_PROFILE_PH_ROW_MODE")
            | None -> null
        let yHistograms = Array.zeroCreate<uint16> (width * 256)
        let kernelHistogram = Array.zeroCreate<uint16> 256
        let zeroYColumnCount = uint16 (windowLength * windowLength)

        let addYRow y =
            if y >= 0 && y < height then
                addZRowToYHistograms width zHistograms y yHistograms
            else
                addZeroZRowToYHistograms width windowLength yHistograms

        let subtractYRow y =
            if y >= 0 && y < height then
                subtractZRowFromYHistograms width zHistograms y yHistograms
            else
                subtractZeroZRowFromYHistograms width windowLength yHistograms

        let mutable sectionStart =
            match profile with
            | Some _ -> timestamp ()
            | None -> 0L
        for yy in -radius .. radius do
            addYRow yy
        match profile with
        | Some p -> p.YInitTicks <- p.YInitTicks + elapsedSince sectionStart
        | None -> ()

        for y in 0 .. height - 1 do
            match profile with
            | Some _ -> sectionStart <- timestamp ()
            | None -> ()
            clearUInt16Histogram kernelHistogram

            for xx in -radius .. radius do
                if xx >= 0 && xx < width then
                    addYColumnToKernelHistogram kernelHistogram yHistograms xx
                else
                    kernelHistogram[0] <- kernelHistogram[0] + zeroYColumnCount
            match profile with
            | Some p -> p.KernelInitTicks <- p.KernelInitTicks + elapsedSince sectionStart
            | None -> ()

            let rowOffset = y * width
            match profile with
            | Some _ -> sectionStart <- timestamp ()
            | None -> ()
            for x in 0 .. width - 1 do
                if not (String.Equals(rowProfileMode, "update-only", StringComparison.Ordinal)) then
                    outputPixels[rowOffset + x] <- medianFromUInt16Histogram totalCount kernelHistogram

                if x < width - 1 && not (String.Equals(rowProfileMode, "median-only", StringComparison.Ordinal)) then
                    let leaving = x - radius
                    let entering = x + radius + 1
                    if leaving >= 0 && leaving < width then
                        subtractYColumnFromKernelHistogram kernelHistogram yHistograms leaving
                    else
                        kernelHistogram[0] <- kernelHistogram[0] - zeroYColumnCount

                    if entering >= 0 && entering < width then
                        addYColumnToKernelHistogram kernelHistogram yHistograms entering
                    else
                        kernelHistogram[0] <- kernelHistogram[0] + zeroYColumnCount
            match profile with
            | Some p -> p.RowScanAndUpdateTicks <- p.RowScanAndUpdateTicks + elapsedSince sectionStart
            | None -> ()

            if y < height - 1 then
                match profile with
                | Some _ -> sectionStart <- timestamp ()
                | None -> ()
                subtractYRow (y - radius)
                addYRow (y + radius + 1)
                match profile with
                | Some p -> p.YUpdateTicks <- p.YUpdateTicks + elapsedSince sectionStart
                | None -> ()

        match profile with
        | Some p ->
            p.EmitTicks <- p.EmitTicks + elapsedSince emitStart
            p.EmittedSlices <- p.EmittedSlices + 1
        | None -> ()
        output
    with
    | _ ->
        Chunk.decRef output
        reraise()

let private fillBlockedZRowLaneMajor width height windowLength (zHistograms: uint16[]) row xStart xSlots (target: uint16[]) =
    Array.Clear(target, 0, target.Length)
    let blockWidth = Vector<uint16>.Count
    let xBlockCount = (width + blockWidth - 1) / blockWidth
    if row >= 0 && row < height then
        let mutable slot = 0
        while slot < xSlots do
            let x = xStart + slot
            if x >= 0 && x < width then
                let xBlock = x / blockWidth
                let lane = x - xBlock * blockWidth
                let mutable bin = 0
                while bin < 256 do
                    target[bin * xSlots + slot] <-
                        zHistograms[blockedZHistogramIndex xBlockCount blockWidth row xBlock bin + lane]
                    bin <- bin + 1
            else
                target[slot] <- uint16 windowLength
            slot <- slot + 1
    else
        let count = uint16 windowLength
        let mutable slot = 0
        while slot < xSlots do
            target[slot] <- count
            slot <- slot + 1

let private medianPerreaultHebertUInt8DenseSliceFromBlockedZHistograms width height radius windowLength (zHistograms: uint16[]) =
    let totalCount = windowLength * windowLength * windowLength
    if totalCount > int UInt16.MaxValue then
        invalidArg "radius" $"UInt8 blocked-z dense PH median stores counts as UInt16 and supports at most {UInt16.MaxValue} samples; got {totalCount}."

    let blockWidth = Vector<uint16>.Count
    let output = Chunk.create<uint8> (uint64 width, uint64 height, 1UL)
    try
        let outputPixels = Chunk.span<uint8> output

        let mutable blockStart = 0
        while blockStart < width do
            let lanes = min blockWidth (width - blockStart)
            let xStart = blockStart - radius
            let xSlots = lanes + 2 * radius
            let yBlock = Array.zeroCreate<uint16> (256 * xSlots)
            let rowBlock = Array.zeroCreate<uint16> (256 * xSlots)
            let kernelBlock = Array.zeroCreate<uint16> (256 * lanes)

            Array.Clear(yBlock, 0, yBlock.Length)
            for yy in -radius .. radius do
                fillBlockedZRowLaneMajor width height windowLength zHistograms yy xStart xSlots rowBlock
                addLaneMajorUInt16ArrayInto yBlock rowBlock

            let mutable y = 0
            while y < height do
                Array.Clear(kernelBlock, 0, kernelBlock.Length)
                let mutable bin = 0
                while bin < 256 do
                    let yBinStart = bin * xSlots
                    let kernelBinStart = bin * lanes
                    let mutable lane = 0
                    while lane < lanes do
                        let mutable sum = 0
                        let mutable dx = 0
                        while dx < windowLength do
                            sum <- sum + int yBlock[yBinStart + lane + dx]
                            dx <- dx + 1
                        kernelBlock[kernelBinStart + lane] <- uint16 sum
                        lane <- lane + 1
                    bin <- bin + 1

                let rowOffset = y * width
                let mutable lane = 0
                while lane < lanes do
                    outputPixels[rowOffset + blockStart + lane] <-
                        medianFromLaneMajorUInt16Histogram totalCount lanes kernelBlock lane
                    lane <- lane + 1

                if y < height - 1 then
                    fillBlockedZRowLaneMajor width height windowLength zHistograms (y - radius) xStart xSlots rowBlock
                    subtractLaneMajorUInt16ArrayFrom yBlock rowBlock
                    fillBlockedZRowLaneMajor width height windowLength zHistograms (y + radius + 1) xStart xSlots rowBlock
                    addLaneMajorUInt16ArrayInto yBlock rowBlock

                y <- y + 1

            blockStart <- blockStart + blockWidth

        output
    with
    | _ ->
        Chunk.decRef output
        reraise()

let private medianPerreaultHebertUInt8DenseSliceFromZHistograms width height radius windowLength (zHistograms: uint16[]) =
    medianPerreaultHebertUInt8DenseSliceFromZHistogramsWithProfile width height radius windowLength zHistograms None

let private writeMedianPerreaultHebertUInt8DenseYBandFromZHistograms width height radius windowLength totalCount (zHistograms: uint16[]) (outputBytes: byte[]) yStart yStop =
    let zeroYColumnCount = uint16 (windowLength * windowLength)
    let yHistograms = Array.zeroCreate<uint16> (width * 256)
    let kernelHistogram = Array.zeroCreate<uint16> 256

    let addYRow y =
        if y >= 0 && y < height then
            addZRowToYHistograms width zHistograms y yHistograms
        else
            addZeroZRowToYHistograms width windowLength yHistograms

    let subtractYRow y =
        if y >= 0 && y < height then
            subtractZRowFromYHistograms width zHistograms y yHistograms
        else
            subtractZeroZRowFromYHistograms width windowLength yHistograms

    for yy in yStart - radius .. yStart + radius do
        addYRow yy

    let mutable y = yStart
    while y < yStop do
        clearUInt16Histogram kernelHistogram

        for xx in -radius .. radius do
            if xx >= 0 && xx < width then
                addYColumnToKernelHistogram kernelHistogram yHistograms xx
            else
                kernelHistogram[0] <- kernelHistogram[0] + zeroYColumnCount

        let rowOffset = y * width
        let mutable x = 0
        while x < width do
            outputBytes[rowOffset + x] <- medianFromUInt16Histogram totalCount kernelHistogram

            if x < width - 1 then
                let leaving = x - radius
                let entering = x + radius + 1
                if leaving >= 0 && leaving < width then
                    subtractYColumnFromKernelHistogram kernelHistogram yHistograms leaving
                else
                    kernelHistogram[0] <- kernelHistogram[0] - zeroYColumnCount

                if entering >= 0 && entering < width then
                    addYColumnToKernelHistogram kernelHistogram yHistograms entering
                else
                    kernelHistogram[0] <- kernelHistogram[0] + zeroYColumnCount

            x <- x + 1

        if y < yStop - 1 then
            subtractYRow (y - radius)
            addYRow (y + radius + 1)

        y <- y + 1

let private writeMedianPerreaultHebertUInt8DenseYBandFromByteZHistograms width height radius windowLength totalCount (zHistograms: byte[]) (outputBytes: byte[]) yStart yStop =
    let zeroYColumnCount = byte (windowLength * windowLength)
    let yHistograms = Array.zeroCreate<byte> (width * 256)
    let kernelHistogram = Array.zeroCreate<byte> 256

    let addYRow y =
        if y >= 0 && y < height then
            addZRowToByteYHistograms width zHistograms y yHistograms
        else
            addZeroZRowToByteYHistograms width windowLength yHistograms

    let subtractYRow y =
        if y >= 0 && y < height then
            subtractZRowFromByteYHistograms width zHistograms y yHistograms
        else
            subtractZeroZRowFromByteYHistograms width windowLength yHistograms

    for yy in yStart - radius .. yStart + radius do
        addYRow yy

    let mutable y = yStart
    while y < yStop do
        clearByteHistogram kernelHistogram

        for xx in -radius .. radius do
            if xx >= 0 && xx < width then
                addByteYColumnToKernelHistogram kernelHistogram yHistograms xx
            else
                kernelHistogram[0] <- kernelHistogram[0] + zeroYColumnCount

        let rowOffset = y * width
        let mutable x = 0
        while x < width do
            outputBytes[rowOffset + x] <- medianFromByteHistogram totalCount kernelHistogram

            if x < width - 1 then
                let leaving = x - radius
                let entering = x + radius + 1
                if leaving >= 0 && leaving < width then
                    subtractByteYColumnFromKernelHistogram kernelHistogram yHistograms leaving
                else
                    kernelHistogram[0] <- kernelHistogram[0] - zeroYColumnCount

                if entering >= 0 && entering < width then
                    addByteYColumnToKernelHistogram kernelHistogram yHistograms entering
                else
                    kernelHistogram[0] <- kernelHistogram[0] + zeroYColumnCount

            x <- x + 1

        if y < yStop - 1 then
            subtractYRow (y - radius)
            addYRow (y + radius + 1)

        y <- y + 1

let private medianPerreaultHebertUInt8DenseSliceFromZHistogramsYBands width height radius windowLength workers (zHistograms: uint16[]) =
    if workers <= 1 || height <= 1 then
        medianPerreaultHebertUInt8DenseSliceFromZHistograms width height radius windowLength zHistograms
    else
        let totalCount = windowLength * windowLength * windowLength
        if totalCount > int UInt16.MaxValue then
            invalidArg "radius" $"UInt8 dense PH median y-band version stores counts as UInt16 and supports at most {UInt16.MaxValue} samples; got {totalCount}."

        let bandCount = min workers height
        let output = Chunk.create<uint8> (uint64 width, uint64 height, 1UL)
        try
            let outputBytes = output.Bytes
            let options = ParallelOptions(MaxDegreeOfParallelism = workers)

            let runBand band =
                let yStart = band * height / bandCount
                let yStop = (band + 1) * height / bandCount
                writeMedianPerreaultHebertUInt8DenseYBandFromZHistograms width height radius windowLength totalCount zHistograms outputBytes yStart yStop

            Parallel.For(0, bandCount, options, Action<int> runBand) |> ignore
            output
        with
        | _ ->
            Chunk.decRef output
            reraise()

let private medianPerreaultHebertUInt8DenseSliceFromByteZHistogramsYBands width height radius windowLength workers (zHistograms: byte[]) =
    let totalCount = windowLength * windowLength * windowLength
    if totalCount > int Byte.MaxValue then
        invalidArg "radius" $"UInt8 dense PH median byte-bin y-band version supports at most {Byte.MaxValue} samples; got {totalCount}."

    let output = Chunk.create<uint8> (uint64 width, uint64 height, 1UL)
    try
        let outputBytes = output.Bytes
        if workers <= 1 || height <= 1 then
            writeMedianPerreaultHebertUInt8DenseYBandFromByteZHistograms width height radius windowLength totalCount zHistograms outputBytes 0 height
        else
            let bandCount = min workers height
            let options = ParallelOptions(MaxDegreeOfParallelism = workers)

            let runBand band =
                let yStart = band * height / bandCount
                let yStop = (band + 1) * height / bandCount
                writeMedianPerreaultHebertUInt8DenseYBandFromByteZHistograms width height radius windowLength totalCount zHistograms outputBytes yStart yStop

            Parallel.For(0, bandCount, options, Action<int> runBand) |> ignore
        output
    with
    | _ ->
        Chunk.decRef output
        reraise()

let private medianPerreaultHebertUInt8DenseSliceFromZHistogramsWithTree width height radius windowLength (zHistograms: uint16[]) =
    let totalCount = windowLength * windowLength * windowLength
    if totalCount > int UInt16.MaxValue then
        invalidArg "radius" $"UInt8 dense PH median prefix tree stores counts as UInt16 and supports at most {UInt16.MaxValue} samples; got {totalCount}."

    let output = Chunk.create<uint8> (uint64 width, uint64 height, 1UL)
    try
        let outputPixels = Chunk.span<uint8> output
        let yHistograms = Array.zeroCreate<uint16> (width * 256)
        let kernelHistogram = Array.zeroCreate<uint16> 256
        let kernelTree = Array.zeroCreate<uint16> 512
        let zeroYColumnCount = uint16 (windowLength * windowLength)

        let addYRow y =
            if y >= 0 && y < height then
                addZRowToYHistograms width zHistograms y yHistograms
            else
                addZeroZRowToYHistograms width windowLength yHistograms

        let subtractYRow y =
            if y >= 0 && y < height then
                subtractZRowFromYHistograms width zHistograms y yHistograms
            else
                subtractZeroZRowFromYHistograms width windowLength yHistograms

        for yy in -radius .. radius do
            addYRow yy

        for y in 0 .. height - 1 do
            clearUInt16Histogram kernelHistogram
            clearUInt16PrefixTree kernelTree

            for xx in -radius .. radius do
                if xx >= 0 && xx < width then
                    addYColumnToKernelHistogram kernelHistogram yHistograms xx
                else
                    kernelHistogram[0] <- kernelHistogram[0] + zeroYColumnCount

            buildUInt16PrefixTreeFromHistogram kernelTree kernelHistogram

            let rowOffset = y * width
            for x in 0 .. width - 1 do
                outputPixels[rowOffset + x] <- medianFromUInt16PrefixTree totalCount kernelTree

                if x < width - 1 then
                    let leaving = x - radius
                    let entering = x + radius + 1
                    if leaving >= 0 && leaving < width then
                        subtractUInt16HistogramFromTreeAndKernel kernelHistogram kernelTree yHistograms (leaving * 256)
                    else
                        subtractZeroCountFromTreeAndKernel kernelHistogram kernelTree zeroYColumnCount

                    if entering >= 0 && entering < width then
                        addUInt16HistogramIntoTreeAndKernel kernelHistogram kernelTree yHistograms (entering * 256)
                    else
                        addZeroCountIntoTreeAndKernel kernelHistogram kernelTree zeroYColumnCount

            if y < height - 1 then
                subtractYRow (y - radius)
                addYRow (y + radius + 1)

        output
    with
    | _ ->
        Chunk.decRef output
        reraise()

let private medianPerreaultHebertUInt8DenseSlice width height radius (window: ChunkSlice[]) =
    let windowLength = 2 * radius + 1
    let totalCount = windowLength * windowLength * windowLength
    if totalCount > int UInt16.MaxValue then
        invalidArg "radius" $"UInt8 dense PH median first version stores counts as UInt16 and supports at most {UInt16.MaxValue} samples; got {totalCount}."

    let zHistograms = window |> Seq.map _.Chunk |> buildZHistogramsUInt8 width height
    medianPerreaultHebertUInt8DenseSliceFromZHistograms width height radius windowLength zHistograms

let medianPerreaultHebertUInt8Dense radius : Stage<Chunk<uint8>, Chunk<uint8>> =
    if radius < 0 then
        invalidArg "radius" $"UInt8 PH median expects a non-negative radius, got {radius}."

    let windowLength = 2 * radius + 1
    let memoryNeed nPixels =
        // Full-slice z histograms: nPixels * 256 UInt16 counts, plus one output slice and row/kernel histograms.
        nPixels * 512UL + nPixels + uint64 windowLength * nPixels

    let zeroMaker _index (source: Chunk<uint8>) =
        let width, height, depth = source.Size
        if depth <> 1UL then
            invalidArg "source" $"Chunk PH median expects 2D slice chunks with depth 1, got {source.Size}."
        zeroChunk (int width) (int height)

    let mapper _debug (window: Window<Chunk<uint8>>) =
        let releaseConsumed () =
            window.Items
            |> List.truncate (int window.ReleaseCount)
            |> List.iter Chunk.decRef

        try
            let emitStart, emitCount = window.EmitRange
            if emitCount = 0u then
                releaseConsumed ()
                []
            else
                let chunks = window.Items |> List.toArray
                if chunks.Length <> windowLength then
                    invalidArg "window" $"Chunk PH median expected window length {windowLength}, got {chunks.Length}."

                let first = chunks[0]
                let chunkWidth, chunkHeight, chunkDepth = first.Size
                if chunkDepth <> 1UL then
                    invalidArg "window" $"Chunk PH median expects 2D slice chunks with depth 1, got {first.Size}."

                let width = int chunkWidth
                let height = int chunkHeight
                if width <= 0 || height <= 0 then
                    invalidArg "window" $"Chunk PH median expects positive slice dimensions, got {first.Size}."

                let windowItems: ChunkSlice[] =
                    chunks
                    |> Array.mapi (fun index chunk ->
                        validateSliceChunk width height chunk
                        { Index = index
                          Chunk = chunk })

                let output = medianPerreaultHebertUInt8DenseSlice width height radius windowItems
                releaseConsumed ()
                [ output ]
        with
        | _ ->
            releaseConsumed ()
            reraise()

    Stage.parallelCollect
        $"chunkMedianPerreaultHebertUInt8Dense.radius{radius}"
        windowLength
        1
        1
        radius
        zeroMaker
        mapper
        memoryNeed
        id

let medianPerreaultHebertUInt8DenseRolling radius : Stage<Chunk<uint8>, Chunk<uint8>> =
    if radius < 0 then
        invalidArg "radius" $"UInt8 rolling PH median expects a non-negative radius, got {radius}."

    let windowLength = 2 * radius + 1
    let memoryNeed nPixels =
        nPixels * 512UL + nPixels + uint64 windowLength * nPixels

    let transition = ProfileTransition.create Streaming Streaming
    let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed

    let apply (_debug: bool) (input: AsyncSeq<Chunk<uint8>>) =
        asyncSeq {
            let profile =
                if phProfileEnabled () then Some(createPhMedianProfile ()) else None
            let queue = ResizeArray<Chunk<uint8>>()
            let mutable width = 0
            let mutable height = 0
            let mutable initialized = false
            let mutable zInitialized = false
            let mutable zHistograms = Array.empty<uint16>
            let mutable realCount = 0
            let mutable emittedCount = 0

            let addPadding () =
                queue.Add(zeroChunk width height)

            let ensureInitialized (chunk: Chunk<uint8>) =
                let chunkWidth, chunkHeight, chunkDepth = chunk.Size
                if chunkDepth <> 1UL then
                    invalidArg "chunk" $"Chunk rolling PH median expects 2D slice chunks with depth 1, got {chunk.Size}."

                if not initialized then
                    width <- int chunkWidth
                    height <- int chunkHeight
                    if width <= 0 || height <= 0 then
                        invalidArg "chunk" $"Chunk rolling PH median expects positive slice dimensions, got {chunk.Size}."
                    for _ in 1 .. radius do
                        addPadding ()
                    initialized <- true
                else
                    validateSliceChunk width height chunk

            let buildInitialZHistograms () =
                let window =
                    queue
                    |> Seq.take windowLength
                let start = timestamp ()
                zHistograms <- buildZHistogramsUInt8 width height window
                match profile with
                | Some p -> p.BuildZTicks <- p.BuildZTicks + elapsedSince start
                | None -> ()
                zInitialized <- true

            let emitCurrent () =
                let output = medianPerreaultHebertUInt8DenseSliceFromZHistogramsWithProfile width height radius windowLength zHistograms profile
                emittedCount <- emittedCount + 1
                output

            let advanceToNewestWindow () =
                let leaving = queue[0]
                let entering = queue[windowLength]
                let start = timestamp ()
                updateZHistogramsUInt8 width height zHistograms leaving entering
                match profile with
                | Some p ->
                    p.UpdateZTicks <- p.UpdateZTicks + elapsedSince start
                    p.ZUpdates <- p.ZUpdates + 1
                | None -> ()
                queue.RemoveAt(0)
                Chunk.decRef leaving

            let tryEmitAfterAppend () =
                seq {
                    if initialized && emittedCount < realCount then
                        if not zInitialized && queue.Count = windowLength then
                            buildInitialZHistograms ()
                            yield emitCurrent ()
                        elif zInitialized && queue.Count = windowLength + 1 then
                            advanceToNewestWindow ()
                            yield emitCurrent ()
                }

            try
                for chunk in input do
                    ensureInitialized chunk
                    queue.Add chunk
                    realCount <- realCount + 1

                    for output in tryEmitAfterAppend () do
                        yield output

                if initialized then
                    while emittedCount < realCount do
                        addPadding ()
                        for output in tryEmitAfterAppend () do
                            yield output
            finally
                match profile with
                | Some p -> printPhMedianProfile p
                | None -> ()
                for chunk in queue do
                    Chunk.decRef chunk
                queue.Clear()
        }

    Stage.fromAsyncSeq
        $"chunkMedianPerreaultHebertUInt8DenseRolling.radius{radius}"
        apply
        transition
        memoryModel
        id

let private medianNativeNthSlice<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (invokeNative: nativeint -> nativeint -> int -> int -> int -> int -> int -> int -> unit)
    width
    height
    radius
    (window: Chunk<'T>[])
    =
    NativeMedian.ensureAvailable ()

    let output = Chunk.create<'T> (uint64 width, uint64 height, 1UL)
    let handles = Array.zeroCreate<GCHandle> window.Length
    let mutable retainedHandles = 0
    let mutable outputHandle = Unchecked.defaultof<GCHandle>
    let mutable outputPinned = false
    let mutable pointerHandle = Unchecked.defaultof<GCHandle>
    let mutable pointersPinned = false
    try
        try
            let pointers = Array.zeroCreate<nativeint> window.Length
            for i in 0 .. window.Length - 1 do
                handles[i] <- GCHandle.Alloc(window[i].Bytes, GCHandleType.Pinned)
                retainedHandles <- retainedHandles + 1
                pointers[i] <- handles[i].AddrOfPinnedObject()

            outputHandle <- GCHandle.Alloc(output.Bytes, GCHandleType.Pinned)
            outputPinned <- true
            pointerHandle <- GCHandle.Alloc(pointers, GCHandleType.Pinned)
            pointersPinned <- true

            invokeNative
                (pointerHandle.AddrOfPinnedObject())
                (outputHandle.AddrOfPinnedObject())
                width
                height
                window.Length
                radius
                radius
                1

            output
        with
        | _ ->
            Chunk.decRef output
            reraise()
    finally
        if pointersPinned then
            pointerHandle.Free()
        if outputPinned then
            outputHandle.Free()
        for i in 0 .. retainedHandles - 1 do
            handles[i].Free()

let private medianNativeUInt8NthSlice width height radius (window: Chunk<uint8>[]) =
    medianNativeNthSlice<uint8>
        (fun slices output width height windowLength radius outputStart outputCount ->
            NativeMedian.medianUInt8NthSlab(slices, output, width, height, windowLength, radius, outputStart, outputCount))
        width
        height
        radius
        window

let private medianNativeUInt16NthSlice width height radius (window: Chunk<uint16>[]) =
    medianNativeNthSlice<uint16>
        (fun slices output width height windowLength radius outputStart outputCount ->
            NativeMedian.medianUInt16NthSlab(slices, output, width, height, windowLength, radius, outputStart, outputCount))
        width
        height
        radius
        window

let private medianNativeInt32NthSlice width height radius (window: Chunk<int32>[]) =
    medianNativeNthSlice<int32>
        (fun slices output width height windowLength radius outputStart outputCount ->
            NativeMedian.medianInt32NthSlab(slices, output, width, height, windowLength, radius, outputStart, outputCount))
        width
        height
        radius
        window

let private medianNativeFloat32NthSlice width height radius (window: Chunk<float32>[]) =
    medianNativeNthSlice<float32>
        (fun slices output width height windowLength radius outputStart outputCount ->
            NativeMedian.medianFloat32NthSlab(slices, output, width, height, windowLength, radius, outputStart, outputCount))
        width
        height
        radius
        window

let private medianNthElementStage<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    name
    radius
    (sliceMedian: int -> int -> int -> Chunk<'T>[] -> Chunk<'T>)
    : Stage<Chunk<'T>, Chunk<'T>> =
    if radius < 0 then
        invalidArg "radius" $"{name} expects a non-negative radius, got {radius}."

    let windowLength = 2 * radius + 1
    let memoryNeed nPixels =
        uint64 (windowLength + 1) * nPixels * uint64 (Marshal.SizeOf<'T>())

    let transition = ProfileTransition.create Streaming Streaming
    let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed

    let apply (_debug: bool) (input: AsyncSeq<Chunk<'T>>) =
        asyncSeq {
            let queue = ResizeArray<Chunk<'T>>()
            let mutable width = 0
            let mutable height = 0
            let mutable initialized = false
            let mutable realCount = 0
            let mutable emittedCount = 0

            let addPadding () =
                queue.Add(zeroChunkTyped<'T> width height)

            let ensureInitialized (chunk: Chunk<'T>) =
                let chunkWidth, chunkHeight, chunkDepth = chunk.Size
                if chunkDepth <> 1UL then
                    invalidArg "chunk" $"{name} expects 2D slice chunks with depth 1, got {chunk.Size}."

                if not initialized then
                    width <- int chunkWidth
                    height <- int chunkHeight
                    if width <= 0 || height <= 0 then
                        invalidArg "chunk" $"{name} expects positive slice dimensions, got {chunk.Size}."
                    for _ in 1 .. radius do
                        addPadding ()
                    initialized <- true
                else
                    validateTypedSliceChunk<'T> name width height chunk

            let emitCurrent () =
                let window = Array.zeroCreate<Chunk<'T>> windowLength
                let mutable i = 0
                while i < windowLength do
                    window[i] <- queue[i]
                    i <- i + 1
                let output = sliceMedian width height radius window
                emittedCount <- emittedCount + 1
                output

            let advanceToNewestWindow () =
                let leaving = queue[0]
                queue.RemoveAt(0)
                Chunk.decRef leaving

            let tryEmitAfterAppend () =
                seq {
                    if initialized && emittedCount < realCount then
                        if queue.Count = windowLength then
                            yield emitCurrent ()
                        elif queue.Count = windowLength + 1 then
                            advanceToNewestWindow ()
                            yield emitCurrent ()
                }

            try
                for chunk in input do
                    ensureInitialized chunk
                    queue.Add chunk
                    realCount <- realCount + 1

                    for output in tryEmitAfterAppend () do
                        yield output

                if initialized then
                    while emittedCount < realCount do
                        addPadding ()
                        for output in tryEmitAfterAppend () do
                            yield output
            finally
                for chunk in queue do
                    Chunk.decRef chunk
                queue.Clear()
        }

    Stage.fromAsyncSeq
        $"{name}.radius{radius}"
        apply
        transition
        memoryModel
        id

let private medianNthElementParallelCollectStage<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    name
    radius
    workers
    (sliceMedian: int -> int -> int -> Chunk<'T>[] -> Chunk<'T>)
    : Stage<Chunk<'T>, Chunk<'T>> =
    if workers < 1 then
        invalidArg "workers" $"{name} expects at least one worker, got {workers}."
    if workers = 1 then
        medianNthElementStage<'T> name radius sliceMedian
    else
        if radius < 0 then
            invalidArg "radius" $"{name} expects a non-negative radius, got {radius}."

        let windowLength = 2 * radius + 1
        let memoryNeed nPixels =
            uint64 (windowLength + workers + 1) * nPixels * uint64 (Marshal.SizeOf<'T>())

        let transition = ProfileTransition.create Streaming Streaming
        let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed

        let apply (_debug: bool) (input: AsyncSeq<Chunk<'T>>) =
            asyncSeq {
                let queue = ResizeArray<TypedChunkSlice<'T>>()
                let mutable width = 0
                let mutable height = 0
                let mutable initialized = false
                let mutable realCount = 0
                let mutable emittedCount = 0
                let mutable lastIndex = -1

                let addPadding index =
                    let chunk = zeroChunkTyped<'T> width height
                    queue.Add({ Index = index; Chunk = chunk })

                let ensureInitialized (chunk: Chunk<'T>) =
                    let chunkWidth, chunkHeight, chunkDepth = chunk.Size
                    if chunkDepth <> 1UL then
                        invalidArg "chunk" $"{name} expects 2D slice chunks with depth 1, got {chunk.Size}."

                    if not initialized then
                        width <- int chunkWidth
                        height <- int chunkHeight
                        if width <= 0 || height <= 0 then
                            invalidArg "chunk" $"{name} expects positive slice dimensions, got {chunk.Size}."
                        for i in radius .. -1 .. 1 do
                            addPadding -i
                        initialized <- true
                    else
                        validateTypedSliceChunk<'T> name width height chunk

                let releasePrefix count =
                    for _ in 1 .. count do
                        let removed = queue[0]
                        queue.RemoveAt(0)
                        Chunk.decRef removed.Chunk

                let tryProcessBatch draining =
                    if initialized then
                        let remainingRealOutputs = realCount - emittedCount
                        let availableWindows = queue.Count - windowLength + 1
                        let availableOutputs = min remainingRealOutputs availableWindows
                        let batchCount =
                            if draining then
                                min workers availableOutputs
                            elif availableOutputs >= workers then
                                workers
                            else
                                0

                        if batchCount > 0 then
                            let windows = Array.zeroCreate<TypedChunkSlice<'T>[]> batchCount
                            let releasedWindows = Array.zeroCreate<bool> batchCount
                            let outputs = Array.zeroCreate<Chunk<'T>> batchCount
                            try
                                for i in 0 .. batchCount - 1 do
                                    windows[i] <- retainTypedWindow queue i windowLength

                                Parallel.For(
                                    0,
                                    batchCount,
                                    fun i ->
                                        try
                                            let chunks = windows[i] |> Array.map _.Chunk
                                            outputs[i] <- sliceMedian width height radius chunks
                                        finally
                                            releaseTypedWindow windows[i]
                                            releasedWindows[i] <- true)
                                |> ignore

                                releasePrefix batchCount
                                emittedCount <- emittedCount + batchCount
                                Some outputs
                            with
                            | _ ->
                                for i in 0 .. windows.Length - 1 do
                                    if not releasedWindows[i] && not (isNull (box windows[i])) then
                                        releaseTypedWindow windows[i]
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
                        for i in 1 .. radius do
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
            $"{name}.parallelCollect.radius{radius}.workers{workers}"
            apply
            transition
            memoryModel
            id

let medianNthElementUInt16 radius : Stage<Chunk<uint16>, Chunk<uint16>> =
    medianNthElementStage<uint16> "chunkMedianNthElementUInt16" radius medianNthElementUInt16Slice

let medianNthElementInt16 radius : Stage<Chunk<int16>, Chunk<int16>> =
    medianNthElementStage<int16> "chunkMedianNthElementInt16" radius medianNthElementInt16Slice

let medianNthElementFloat32 radius : Stage<Chunk<float32>, Chunk<float32>> =
    medianNthElementStage<float32> "chunkMedianNthElementFloat32" radius medianNthElementFloat32Slice

let medianQuickselectUInt8 radius : Stage<Chunk<uint8>, Chunk<uint8>> =
    medianNthElementStage<uint8> "chunkMedianQuickselectUInt8" radius medianQuickselectUInt8Slice

let medianQuickselectUInt16 radius : Stage<Chunk<uint16>, Chunk<uint16>> =
    medianNthElementStage<uint16> "chunkMedianQuickselectUInt16" radius medianQuickselectUInt16Slice

let medianQuickselectUInt16ParallelCollect radius workers : Stage<Chunk<uint16>, Chunk<uint16>> =
    medianNthElementParallelCollectStage<uint16> "chunkMedianQuickselectUInt16" radius workers medianQuickselectUInt16Slice

let medianSortUInt16 radius : Stage<Chunk<uint16>, Chunk<uint16>> =
    medianNthElementStage<uint16> "chunkMedianSortUInt16" radius medianSortUInt16Slice

let medianSortUInt16ParallelCollect radius workers : Stage<Chunk<uint16>, Chunk<uint16>> =
    medianNthElementParallelCollectStage<uint16> "chunkMedianSortUInt16" radius workers medianSortUInt16Slice

let medianNativeNthElementUInt8 radius : Stage<Chunk<uint8>, Chunk<uint8>> =
    medianNthElementStage<uint8> "chunkMedianNativeNthElementUInt8" radius medianNativeUInt8NthSlice

let medianNativeNthElementUInt8ParallelCollect radius workers : Stage<Chunk<uint8>, Chunk<uint8>> =
    medianNthElementParallelCollectStage<uint8> "chunkMedianNativeNthElementUInt8" radius workers medianNativeUInt8NthSlice

let medianNativeNthElementUInt16 radius : Stage<Chunk<uint16>, Chunk<uint16>> =
    medianNthElementStage<uint16> "chunkMedianNativeNthElementUInt16" radius medianNativeUInt16NthSlice

let medianNativeNthElementUInt16ParallelCollect radius workers : Stage<Chunk<uint16>, Chunk<uint16>> =
    medianNthElementParallelCollectStage<uint16> "chunkMedianNativeNthElementUInt16" radius workers medianNativeUInt16NthSlice

let medianNativeNthElementInt32 radius : Stage<Chunk<int32>, Chunk<int32>> =
    medianNthElementStage<int32> "chunkMedianNativeNthElementInt32" radius medianNativeInt32NthSlice

let medianNativeNthElementInt32ParallelCollect radius workers : Stage<Chunk<int32>, Chunk<int32>> =
    medianNthElementParallelCollectStage<int32> "chunkMedianNativeNthElementInt32" radius workers medianNativeInt32NthSlice

let medianNativeNthElementFloat32 radius : Stage<Chunk<float32>, Chunk<float32>> =
    medianNthElementStage<float32> "chunkMedianNativeNthElementFloat32" radius medianNativeFloat32NthSlice

let medianNativeNthElementFloat32ParallelCollect radius workers : Stage<Chunk<float32>, Chunk<float32>> =
    medianNthElementParallelCollectStage<float32> "chunkMedianNativeNthElementFloat32" radius workers medianNativeFloat32NthSlice

let medianQuickselectInt16 radius : Stage<Chunk<int16>, Chunk<int16>> =
    medianNthElementStage<int16> "chunkMedianQuickselectInt16" radius medianQuickselectInt16Slice

let private medianItkWrappedSlice<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    width
    height
    radius
    (window: Chunk<'T>[])
    =
    let chunkWindow =
        { Items = window |> Array.toList
          EmitRange = uint radius, 1u
          ReleaseCount = 0u }

    let slab = Chunk.toSlabWith $"chunkMedianItkWrapped.radius{radius}" chunkWindow
    try
        let medianImage = ImageFunctions.median (uint radius) slab.Image
        try
            match Chunk.ofSlab { Image = medianImage; EmitRange = slab.EmitRange } with
            | [ output ] -> output
            | outputs ->
                outputs |> List.iter Chunk.decRef
                invalidOp $"Chunk ITK-wrapped median expected exactly one emitted slice, got {outputs.Length}."
        finally
            medianImage.decRefCount()
    finally
        slab.Image.decRefCount()

let medianItkWrappedParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    radius
    workers
    : Stage<Chunk<'T>, Chunk<'T>> =
    medianNthElementParallelCollectStage<'T>
        $"chunkMedianItkWrapped.{typeof<'T>.Name}"
        radius
        workers
        medianItkWrappedSlice<'T>

let medianItkWrapped<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> radius : Stage<Chunk<'T>, Chunk<'T>> =
    medianItkWrappedParallelCollect<'T> radius 1

let private medianPerreaultHebertUInt8DenseRollingByteBinsYBands radius workers : Stage<Chunk<uint8>, Chunk<uint8>> =
    if radius < 0 then
        invalidArg "radius" $"UInt8 rolling PH median byte-bin y-band version expects a non-negative radius, got {radius}."
    if workers < 1 then
        invalidArg "workers" $"UInt8 rolling PH median byte-bin y-band version expects at least one worker, got {workers}."

    let windowLength = 2 * radius + 1
    let totalCount = windowLength * windowLength * windowLength
    if totalCount > int Byte.MaxValue then
        invalidArg "radius" $"UInt8 rolling PH median byte-bin y-band version supports at most {Byte.MaxValue} samples; got {totalCount}."

    let memoryNeed nPixels =
        nPixels * 256UL + nPixels + uint64 windowLength * nPixels

    let transition = ProfileTransition.create Streaming Streaming
    let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed

    let apply (_debug: bool) (input: AsyncSeq<Chunk<uint8>>) =
        asyncSeq {
            let queue = ResizeArray<Chunk<uint8>>()
            let mutable width = 0
            let mutable height = 0
            let mutable initialized = false
            let mutable zInitialized = false
            let mutable zHistograms = Array.empty<byte>
            let mutable realCount = 0
            let mutable emittedCount = 0

            let addPadding () =
                queue.Add(zeroChunk width height)

            let ensureInitialized (chunk: Chunk<uint8>) =
                let chunkWidth, chunkHeight, chunkDepth = chunk.Size
                if chunkDepth <> 1UL then
                    invalidArg "chunk" $"Chunk rolling PH median byte-bin y-band version expects 2D slice chunks with depth 1, got {chunk.Size}."

                if not initialized then
                    width <- int chunkWidth
                    height <- int chunkHeight
                    if width <= 0 || height <= 0 then
                        invalidArg "chunk" $"Chunk rolling PH median byte-bin y-band version expects positive slice dimensions, got {chunk.Size}."
                    for _ in 1 .. radius do
                        addPadding ()
                    initialized <- true
                else
                    validateSliceChunk width height chunk

            let buildInitialZHistograms () =
                let window =
                    queue
                    |> Seq.take windowLength
                zHistograms <- buildZHistogramsUInt8ByteBins width height window
                zInitialized <- true

            let emitCurrent () =
                let output = medianPerreaultHebertUInt8DenseSliceFromByteZHistogramsYBands width height radius windowLength workers zHistograms
                emittedCount <- emittedCount + 1
                output

            let advanceToNewestWindow () =
                let leaving = queue[0]
                let entering = queue[windowLength]
                updateZHistogramsUInt8ByteBins width height zHistograms leaving entering
                queue.RemoveAt(0)
                Chunk.decRef leaving

            let tryEmitAfterAppend () =
                seq {
                    if initialized && emittedCount < realCount then
                        if not zInitialized && queue.Count = windowLength then
                            buildInitialZHistograms ()
                            yield emitCurrent ()
                        elif zInitialized && queue.Count = windowLength + 1 then
                            advanceToNewestWindow ()
                            yield emitCurrent ()
                }

            try
                for chunk in input do
                    ensureInitialized chunk
                    queue.Add chunk
                    realCount <- realCount + 1

                    for output in tryEmitAfterAppend () do
                        yield output

                if initialized then
                    while emittedCount < realCount do
                        addPadding ()
                        for output in tryEmitAfterAppend () do
                            yield output
            finally
                for chunk in queue do
                    Chunk.decRef chunk
                queue.Clear()
        }

    Stage.fromAsyncSeq
        $"chunkMedianPerreaultHebertUInt8DenseRollingByteBinsYBands.radius{radius}.workers{workers}"
        apply
        transition
        memoryModel
        id

let medianPerreaultHebertUInt8DenseRollingYBands radius workers : Stage<Chunk<uint8>, Chunk<uint8>> =
    if radius < 0 then
        invalidArg "radius" $"UInt8 rolling PH median y-band version expects a non-negative radius, got {radius}."
    if workers < 1 then
        invalidArg "workers" $"UInt8 rolling PH median y-band version expects at least one worker, got {workers}."
    let windowLength = 2 * radius + 1
    if windowLength * windowLength * windowLength <= int Byte.MaxValue then
        medianPerreaultHebertUInt8DenseRollingByteBinsYBands radius workers
    elif workers = 1 then
        medianPerreaultHebertUInt8DenseRolling radius
    else
        let memoryNeed nPixels =
            nPixels * 512UL + nPixels + uint64 windowLength * nPixels + uint64 workers * 512UL * uint64 (int (sqrt (float nPixels)) + 1)

        let transition = ProfileTransition.create Streaming Streaming
        let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed

        let apply (_debug: bool) (input: AsyncSeq<Chunk<uint8>>) =
            asyncSeq {
                let queue = ResizeArray<Chunk<uint8>>()
                let mutable width = 0
                let mutable height = 0
                let mutable initialized = false
                let mutable zInitialized = false
                let mutable zHistograms = Array.empty<uint16>
                let mutable realCount = 0
                let mutable emittedCount = 0

                let addPadding () =
                    queue.Add(zeroChunk width height)

                let ensureInitialized (chunk: Chunk<uint8>) =
                    let chunkWidth, chunkHeight, chunkDepth = chunk.Size
                    if chunkDepth <> 1UL then
                        invalidArg "chunk" $"Chunk rolling PH median y-band version expects 2D slice chunks with depth 1, got {chunk.Size}."

                    if not initialized then
                        width <- int chunkWidth
                        height <- int chunkHeight
                        if width <= 0 || height <= 0 then
                            invalidArg "chunk" $"Chunk rolling PH median y-band version expects positive slice dimensions, got {chunk.Size}."
                        for _ in 1 .. radius do
                            addPadding ()
                        initialized <- true
                    else
                        validateSliceChunk width height chunk

                let buildInitialZHistograms () =
                    let window =
                        queue
                        |> Seq.take windowLength
                    zHistograms <- buildZHistogramsUInt8 width height window
                    zInitialized <- true

                let emitCurrent () =
                    let output = medianPerreaultHebertUInt8DenseSliceFromZHistogramsYBands width height radius windowLength workers zHistograms
                    emittedCount <- emittedCount + 1
                    output

                let advanceToNewestWindow () =
                    let leaving = queue[0]
                    let entering = queue[windowLength]
                    updateZHistogramsUInt8 width height zHistograms leaving entering
                    queue.RemoveAt(0)
                    Chunk.decRef leaving

                let tryEmitAfterAppend () =
                    seq {
                        if initialized && emittedCount < realCount then
                            if not zInitialized && queue.Count = windowLength then
                                buildInitialZHistograms ()
                                yield emitCurrent ()
                            elif zInitialized && queue.Count = windowLength + 1 then
                                advanceToNewestWindow ()
                                yield emitCurrent ()
                    }

                try
                    for chunk in input do
                        ensureInitialized chunk
                        queue.Add chunk
                        realCount <- realCount + 1

                        for output in tryEmitAfterAppend () do
                            yield output

                    if initialized then
                        while emittedCount < realCount do
                            addPadding ()
                            for output in tryEmitAfterAppend () do
                                yield output
                finally
                    for chunk in queue do
                        Chunk.decRef chunk
                    queue.Clear()
            }

        Stage.fromAsyncSeq
            $"chunkMedianPerreaultHebertUInt8DenseRollingYBands.radius{radius}.workers{workers}"
            apply
            transition
            memoryModel
            id

let medianPerreaultHebertUInt8DenseRollingTree radius : Stage<Chunk<uint8>, Chunk<uint8>> =
    if radius < 0 then
        invalidArg "radius" $"UInt8 rolling PH median prefix tree expects a non-negative radius, got {radius}."

    let windowLength = 2 * radius + 1
    let memoryNeed nPixels =
        nPixels * 512UL + nPixels + uint64 windowLength * nPixels

    let transition = ProfileTransition.create Streaming Streaming
    let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed

    let apply (_debug: bool) (input: AsyncSeq<Chunk<uint8>>) =
        asyncSeq {
            let queue = ResizeArray<Chunk<uint8>>()
            let mutable width = 0
            let mutable height = 0
            let mutable initialized = false
            let mutable zInitialized = false
            let mutable zHistograms = Array.empty<uint16>
            let mutable realCount = 0
            let mutable emittedCount = 0

            let addPadding () =
                queue.Add(zeroChunk width height)

            let ensureInitialized (chunk: Chunk<uint8>) =
                let chunkWidth, chunkHeight, chunkDepth = chunk.Size
                if chunkDepth <> 1UL then
                    invalidArg "chunk" $"Chunk rolling PH median prefix tree expects 2D slice chunks with depth 1, got {chunk.Size}."

                if not initialized then
                    width <- int chunkWidth
                    height <- int chunkHeight
                    if width <= 0 || height <= 0 then
                        invalidArg "chunk" $"Chunk rolling PH median prefix tree expects positive slice dimensions, got {chunk.Size}."
                    for _ in 1 .. radius do
                        addPadding ()
                    initialized <- true
                else
                    validateSliceChunk width height chunk

            let buildInitialZHistograms () =
                let window =
                    queue
                    |> Seq.take windowLength
                zHistograms <- buildZHistogramsUInt8 width height window
                zInitialized <- true

            let emitCurrent () =
                let output = medianPerreaultHebertUInt8DenseSliceFromZHistogramsWithTree width height radius windowLength zHistograms
                emittedCount <- emittedCount + 1
                output

            let advanceToNewestWindow () =
                let leaving = queue[0]
                let entering = queue[windowLength]
                updateZHistogramsUInt8 width height zHistograms leaving entering
                queue.RemoveAt(0)
                Chunk.decRef leaving

            let tryEmitAfterAppend () =
                seq {
                    if initialized && emittedCount < realCount then
                        if not zInitialized && queue.Count = windowLength then
                            buildInitialZHistograms ()
                            yield emitCurrent ()
                        elif zInitialized && queue.Count = windowLength + 1 then
                            advanceToNewestWindow ()
                            yield emitCurrent ()
                }

            try
                for chunk in input do
                    ensureInitialized chunk
                    queue.Add chunk
                    realCount <- realCount + 1

                    for output in tryEmitAfterAppend () do
                        yield output

                if initialized then
                    while emittedCount < realCount do
                        addPadding ()
                        for output in tryEmitAfterAppend () do
                            yield output
            finally
                for chunk in queue do
                    Chunk.decRef chunk
                queue.Clear()
        }

    Stage.fromAsyncSeq
        $"chunkMedianPerreaultHebertUInt8DenseRollingTree.radius{radius}"
        apply
        transition
        memoryModel
        id

let medianPerreaultHebertUInt8DenseRollingTransposedXBlock radius : Stage<Chunk<uint8>, Chunk<uint8>> =
    if radius < 0 then
        invalidArg "radius" $"UInt8 rolling transposed x-block PH median expects a non-negative radius, got {radius}."

    let windowLength = 2 * radius + 1
    let memoryNeed nPixels =
        nPixels * 512UL + nPixels + uint64 windowLength * nPixels

    let transition = ProfileTransition.create Streaming Streaming
    let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed

    let apply (_debug: bool) (input: AsyncSeq<Chunk<uint8>>) =
        asyncSeq {
            let queue = ResizeArray<Chunk<uint8>>()
            let mutable width = 0
            let mutable height = 0
            let mutable initialized = false
            let mutable zInitialized = false
            let mutable zHistograms = Array.empty<uint16>
            let mutable realCount = 0
            let mutable emittedCount = 0

            let addPadding () =
                queue.Add(zeroChunk width height)

            let ensureInitialized (chunk: Chunk<uint8>) =
                let chunkWidth, chunkHeight, chunkDepth = chunk.Size
                if chunkDepth <> 1UL then
                    invalidArg "chunk" $"Chunk rolling transposed x-block PH median expects 2D slice chunks with depth 1, got {chunk.Size}."

                if not initialized then
                    width <- int chunkWidth
                    height <- int chunkHeight
                    if width <= 0 || height <= 0 then
                        invalidArg "chunk" $"Chunk rolling transposed x-block PH median expects positive slice dimensions, got {chunk.Size}."
                    for _ in 1 .. radius do
                        addPadding ()
                    initialized <- true
                else
                    validateSliceChunk width height chunk

            let buildInitialZHistograms () =
                let window =
                    queue
                    |> Seq.take windowLength
                zHistograms <- buildZHistogramsUInt8 width height window
                zInitialized <- true

            let emitCurrent () =
                let output = medianPerreaultHebertUInt8DenseSliceFromZHistogramsTransposedXBlock width height radius windowLength zHistograms
                emittedCount <- emittedCount + 1
                output

            let advanceToNewestWindow () =
                let leaving = queue[0]
                let entering = queue[windowLength]
                updateZHistogramsUInt8 width height zHistograms leaving entering
                queue.RemoveAt(0)
                Chunk.decRef leaving

            let tryEmitAfterAppend () =
                seq {
                    if initialized && emittedCount < realCount then
                        if not zInitialized && queue.Count = windowLength then
                            buildInitialZHistograms ()
                            yield emitCurrent ()
                        elif zInitialized && queue.Count = windowLength + 1 then
                            advanceToNewestWindow ()
                            yield emitCurrent ()
                }

            try
                for chunk in input do
                    ensureInitialized chunk
                    queue.Add chunk
                    realCount <- realCount + 1

                    for output in tryEmitAfterAppend () do
                        yield output

                if initialized then
                    while emittedCount < realCount do
                        addPadding ()
                        for output in tryEmitAfterAppend () do
                            yield output
            finally
                for chunk in queue do
                    Chunk.decRef chunk
                queue.Clear()
        }

    Stage.fromAsyncSeq
        $"chunkMedianPerreaultHebertUInt8DenseRollingTransposedXBlock.radius{radius}"
        apply
        transition
        memoryModel
        id

let medianPerreaultHebertUInt8DenseRollingBlockedZ radius : Stage<Chunk<uint8>, Chunk<uint8>> =
    if radius < 0 then
        invalidArg "radius" $"UInt8 rolling blocked-z PH median expects a non-negative radius, got {radius}."

    let windowLength = 2 * radius + 1
    let memoryNeed nPixels =
        nPixels * 512UL + nPixels + uint64 windowLength * nPixels

    let transition = ProfileTransition.create Streaming Streaming
    let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed

    let apply (_debug: bool) (input: AsyncSeq<Chunk<uint8>>) =
        asyncSeq {
            let queue = ResizeArray<Chunk<uint8>>()
            let mutable width = 0
            let mutable height = 0
            let mutable initialized = false
            let mutable zInitialized = false
            let mutable zHistograms = Array.empty<uint16>
            let mutable realCount = 0
            let mutable emittedCount = 0

            let addPadding () =
                queue.Add(zeroChunk width height)

            let ensureInitialized (chunk: Chunk<uint8>) =
                let chunkWidth, chunkHeight, chunkDepth = chunk.Size
                if chunkDepth <> 1UL then
                    invalidArg "chunk" $"Chunk rolling blocked-z PH median expects 2D slice chunks with depth 1, got {chunk.Size}."

                if not initialized then
                    width <- int chunkWidth
                    height <- int chunkHeight
                    if width <= 0 || height <= 0 then
                        invalidArg "chunk" $"Chunk rolling blocked-z PH median expects positive slice dimensions, got {chunk.Size}."
                    for _ in 1 .. radius do
                        addPadding ()
                    initialized <- true
                else
                    validateSliceChunk width height chunk

            let buildInitialZHistograms () =
                let window =
                    queue
                    |> Seq.take windowLength
                zHistograms <- buildBlockedZHistogramsUInt8 width height window
                zInitialized <- true

            let emitCurrent () =
                let output = medianPerreaultHebertUInt8DenseSliceFromBlockedZHistograms width height radius windowLength zHistograms
                emittedCount <- emittedCount + 1
                output

            let advanceToNewestWindow () =
                let leaving = queue[0]
                let entering = queue[windowLength]
                updateBlockedZHistogramsUInt8 width height zHistograms leaving entering
                queue.RemoveAt(0)
                Chunk.decRef leaving

            let tryEmitAfterAppend () =
                seq {
                    if initialized && emittedCount < realCount then
                        if not zInitialized && queue.Count = windowLength then
                            buildInitialZHistograms ()
                            yield emitCurrent ()
                        elif zInitialized && queue.Count = windowLength + 1 then
                            advanceToNewestWindow ()
                            yield emitCurrent ()
                }

            try
                for chunk in input do
                    ensureInitialized chunk
                    queue.Add chunk
                    realCount <- realCount + 1

                    for output in tryEmitAfterAppend () do
                        yield output

                if initialized then
                    while emittedCount < realCount do
                        addPadding ()
                        for output in tryEmitAfterAppend () do
                            yield output
            finally
                for chunk in queue do
                    Chunk.decRef chunk
                queue.Clear()
        }

    Stage.fromAsyncSeq
        $"chunkMedianPerreaultHebertUInt8DenseRollingBlockedZ.radius{radius}"
        apply
        transition
        memoryModel
        id

let medianPerreaultHebertUInt8DenseXFirstMaterialized radius : Stage<Chunk<uint8>, Chunk<uint8>> =
    if radius < 0 then
        invalidArg "radius" $"UInt8 x-first PH median expects a non-negative radius, got {radius}."

    let windowLength = 2 * radius + 1
    let memoryNeed nPixels =
        // This deliberately materialized experiment carries xy histograms for the active z window.
        uint64 (windowLength + 2) * nPixels * 512UL + nPixels

    let transition = ProfileTransition.create Streaming Streaming
    let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed

    let apply (_debug: bool) (input: AsyncSeq<Chunk<uint8>>) =
        asyncSeq {
            let queue = ResizeArray<uint16[]>()
            let mutable width = 0
            let mutable height = 0
            let mutable initialized = false
            let mutable zeroXyHistograms = Array.empty<uint16>
            let mutable kernelHistograms = Array.empty<uint16>
            let mutable kernelInitialized = false
            let mutable realCount = 0
            let mutable emittedCount = 0

            let ensureInitialized (chunk: Chunk<uint8>) =
                let chunkWidth, chunkHeight, chunkDepth = chunk.Size
                if chunkDepth <> 1UL then
                    invalidArg "chunk" $"Chunk x-first PH median expects 2D slice chunks with depth 1, got {chunk.Size}."

                if not initialized then
                    width <- int chunkWidth
                    height <- int chunkHeight
                    if width <= 0 || height <= 0 then
                        invalidArg "chunk" $"Chunk x-first PH median expects positive slice dimensions, got {chunk.Size}."
                    zeroXyHistograms <- buildZeroXyHistogramsUInt8 width height windowLength
                    for _ in 1 .. radius do
                        queue.Add(zeroXyHistograms)
                    initialized <- true
                else
                    validateSliceChunk width height chunk

            let buildInitialKernelHistograms () =
                kernelHistograms <- Array.zeroCreate<uint16> (width * height * 256)
                for i in 0 .. windowLength - 1 do
                    addUInt16ArrayInto kernelHistograms queue[i]
                kernelInitialized <- true

            let emitCurrent () =
                let output = emitMedianSliceFromXyKernelHistograms width height radius windowLength kernelHistograms
                emittedCount <- emittedCount + 1
                output

            let advanceToNewestWindow () =
                subtractUInt16ArrayFrom kernelHistograms queue[0]
                addUInt16ArrayInto kernelHistograms queue[windowLength]
                queue.RemoveAt(0)

            let tryEmitAfterAppend () =
                seq {
                    if initialized && emittedCount < realCount then
                        if not kernelInitialized && queue.Count = windowLength then
                            buildInitialKernelHistograms ()
                            yield emitCurrent ()
                        elif kernelInitialized && queue.Count = windowLength + 1 then
                            advanceToNewestWindow ()
                            yield emitCurrent ()
                }

            try
                for chunk in input do
                    ensureInitialized chunk
                    let xyHistograms = buildXyHistogramsUInt8 width height radius chunk
                    Chunk.decRef chunk
                    queue.Add(xyHistograms)
                    realCount <- realCount + 1

                    for output in tryEmitAfterAppend () do
                        yield output

                if initialized then
                    while emittedCount < realCount do
                        queue.Add(zeroXyHistograms)
                        for output in tryEmitAfterAppend () do
                            yield output
            finally
                queue.Clear()
        }

    Stage.fromAsyncSeq
        $"chunkMedianPerreaultHebertUInt8DenseXFirstMaterialized.radius{radius}"
        apply
        transition
        memoryModel
        id

let medianPerreaultHebertUInt8DenseXBlock radius : Stage<Chunk<uint8>, Chunk<uint8>> =
    if radius < 0 then
        invalidArg "radius" $"UInt8 x-block PH median expects a non-negative radius, got {radius}."

    let windowLength = 2 * radius + 1
    let memoryNeed nPixels =
        let blockWidth = uint64 Vector<uint16>.Count
        uint64 windowLength * nPixels + uint64 windowLength * blockWidth * 256UL * 2UL + nPixels

    let zeroMaker _index (source: Chunk<uint8>) =
        let width, height, depth = source.Size
        if depth <> 1UL then
            invalidArg "source" $"Chunk x-block PH median expects 2D slice chunks with depth 1, got {source.Size}."
        zeroChunk (int width) (int height)

    let mapper _debug (window: Window<Chunk<uint8>>) =
        let releaseConsumed () =
            window.Items
            |> List.truncate (int window.ReleaseCount)
            |> List.iter Chunk.decRef

        try
            let _emitStart, emitCount = window.EmitRange
            if emitCount = 0u then
                releaseConsumed ()
                []
            else
                let chunks = window.Items |> List.toArray
                if chunks.Length <> windowLength then
                    invalidArg "window" $"Chunk x-block PH median expected window length {windowLength}, got {chunks.Length}."

                let first = chunks[0]
                let chunkWidth, chunkHeight, chunkDepth = first.Size
                if chunkDepth <> 1UL then
                    invalidArg "window" $"Chunk x-block PH median expects 2D slice chunks with depth 1, got {first.Size}."

                let width = int chunkWidth
                let height = int chunkHeight
                if width <= 0 || height <= 0 then
                    invalidArg "window" $"Chunk x-block PH median expects positive slice dimensions, got {first.Size}."

                let windowItems: ChunkSlice[] =
                    chunks
                    |> Array.mapi (fun index chunk ->
                        validateSliceChunk width height chunk
                        { Index = index
                          Chunk = chunk })

                let output = medianPerreaultHebertUInt8DenseSliceXBlock width height radius windowItems
                releaseConsumed ()
                [ output ]
        with
        | _ ->
            releaseConsumed ()
            reraise()

    Stage.parallelCollect
        $"chunkMedianPerreaultHebertUInt8DenseXBlock.radius{radius}"
        windowLength
        1
        1
        radius
        zeroMaker
        mapper
        memoryNeed
        id

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
    if a.Size <> b.Size then
        invalidArg "b" $"ChunkFunctions.{name} expects chunks with identical sizes, got {a.Size} and {b.Size}."

let map2Chunk<'T, 'U, 'V when 'T: equality and 'U: equality and 'V: equality
                                and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType
                                and 'U: (new: unit -> 'U) and 'U: struct and 'U :> ValueType
                                and 'V: (new: unit -> 'V) and 'V: struct and 'V :> ValueType>
    name
    f
    (a: Chunk<'T>)
    (b: Chunk<'U>) =
    validateSameSize name a b
    let output = Chunk.create<'V> a.Size
    try
        let aSpan = Chunk.span<'T> a
        let bSpan = Chunk.span<'U> b
        let outputSpan = Chunk.span<'V> output
        let mutable i = 0
        while i < aSpan.Length do
            outputSpan[i] <- f aSpan[i] bSpan[i]
            i <- i + 1
        output
    with
    | _ ->
        Chunk.decRef output
        reraise()

let releaseBinaryChunk name f memoryNeed : Stage<Chunk<'T> * Chunk<'U>, Chunk<'V>> =
    let mapper _debug (a, b) =
        try
            f a b
        finally
            Chunk.decRef a
            Chunk.decRef b

    Stage.map name mapper memoryNeed id

let copy<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> : Stage<Chunk<'T>, Chunk<'T>> =
    let mapper chunk =
        let output = Chunk.create<'T> chunk.Size
        try
            chunk.Bytes.AsSpan(0, chunk.ByteLength).CopyTo(output.Bytes.AsSpan(0, output.ByteLength))
            output
        with
        | _ ->
            Chunk.decRef output
            reraise()

    releaseUnaryChunk $"chunkCopy.{typeof<'T>.Name}" mapper (fun n -> 2UL * chunkMemoryNeed<'T> n)

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
    releaseBinaryChunk name (map2Chunk name f) (fun n -> n * uint64 (chunkElementBytes<'T> + chunkElementBytes<'U> + chunkElementBytes<'V>))

let inline sum<'T when 'T: equality
                    and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType
                    and 'T: (static member ( + ) : 'T * 'T -> 'T)
                    and 'T: (static member Zero : 'T)> (chunk: Chunk<'T>) : 'T =
    Chunk.fold (fun acc value -> acc + value) LanguagePrimitives.GenericZero chunk

let inline prod<'T when 'T: equality
                     and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType
                     and 'T: (static member ( * ) : 'T * 'T -> 'T)
                     and 'T: (static member One : 'T)> (chunk: Chunk<'T>) : 'T =
    Chunk.fold (fun acc value -> acc * value) LanguagePrimitives.GenericOne chunk

let inline minMax<'T when 'T: equality and 'T: comparison
                       and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (chunk: Chunk<'T>) =
    let values = Chunk.span<'T> chunk
    if values.Length = 0 then
        invalidArg "chunk" "ChunkFunctions.minMax cannot reduce an empty chunk."
    let mutable mn = values[0]
    let mutable mx = values[0]
    let mutable i = 1
    while i < values.Length do
        let value = values[i]
        if value < mn then mn <- value
        if value > mx then mx <- value
        i <- i + 1
    mn, mx

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

let private mapFloat32Vector name (scalarOp: float32 -> float32) (vectorOp: Vector<float32> -> Vector<float32>) chunk =
    let output = Chunk.create<float32> chunk.Size
    try
        let input = Chunk.span<float32> chunk
        let outputSpan = Chunk.span<float32> output
        let width = Vector<float32>.Count
        let vectorEnd = input.Length - (input.Length % width)
        let mutable i = 0
        while i < vectorEnd do
            let result = vectorOp (Vector<float32>(input.Slice(i, width)))
            result.CopyTo(outputSpan.Slice(i, width))
            i <- i + width
        while i < input.Length do
            outputSpan[i] <- scalarOp input[i]
            i <- i + 1
        output
    with
    | _ ->
        Chunk.decRef output
        reraise()

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
    let output = Chunk.create<'T> chunk.Size
    try
        let t = typeof<'T>
        if t = typeof<uint8> then
            let threshold = byte (Math.Clamp(Math.Ceiling(threshold), 0.0, 255.0))
            let inputPixels = chunk.Bytes.AsSpan(0, chunk.ByteLength)
            let outputPixels = output.Bytes.AsSpan(0, output.ByteLength)
            let mutable i = 0
            while i < inputPixels.Length do
                outputPixels[i] <- if inputPixels[i] >= threshold then 1uy else 0uy
                i <- i + 1
        elif t = typeof<int8> then
            let threshold = sbyte (Math.Clamp(Math.Ceiling(threshold), float SByte.MinValue, float SByte.MaxValue))
            let inputPixels = MemoryMarshal.Cast<byte, sbyte>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
            let outputPixels = MemoryMarshal.Cast<byte, sbyte>(output.Bytes.AsSpan(0, output.ByteLength))
            let mutable i = 0
            while i < inputPixels.Length do
                outputPixels[i] <- if inputPixels[i] >= threshold then 1y else 0y
                i <- i + 1
        elif t = typeof<uint16> then
            let threshold = uint16 (Math.Clamp(Math.Ceiling(threshold), 0.0, float UInt16.MaxValue))
            let inputPixels = MemoryMarshal.Cast<byte, uint16>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
            let outputPixels = MemoryMarshal.Cast<byte, uint16>(output.Bytes.AsSpan(0, output.ByteLength))
            let mutable i = 0
            while i < inputPixels.Length do
                outputPixels[i] <- if inputPixels[i] >= threshold then 1us else 0us
                i <- i + 1
        elif t = typeof<int16> then
            let threshold = int16 (Math.Clamp(Math.Ceiling(threshold), float Int16.MinValue, float Int16.MaxValue))
            let inputPixels = MemoryMarshal.Cast<byte, int16>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
            let outputPixels = MemoryMarshal.Cast<byte, int16>(output.Bytes.AsSpan(0, output.ByteLength))
            let mutable i = 0
            while i < inputPixels.Length do
                outputPixels[i] <- if inputPixels[i] >= threshold then 1s else 0s
                i <- i + 1
        elif t = typeof<int32> then
            let threshold = int32 (Math.Clamp(Math.Ceiling(threshold), float Int32.MinValue, float Int32.MaxValue))
            let inputPixels = MemoryMarshal.Cast<byte, int32>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
            let outputPixels = MemoryMarshal.Cast<byte, int32>(output.Bytes.AsSpan(0, output.ByteLength))
            let mutable i = 0
            while i < inputPixels.Length do
                outputPixels[i] <- if inputPixels[i] >= threshold then 1 else 0
                i <- i + 1
        elif t = typeof<float32> then
            let threshold = float32 threshold
            let inputPixels = MemoryMarshal.Cast<byte, float32>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
            let outputPixels = MemoryMarshal.Cast<byte, float32>(output.Bytes.AsSpan(0, output.ByteLength))
            let mutable i = 0
            while i < inputPixels.Length do
                outputPixels[i] <- if inputPixels[i] >= threshold then 1.0f else 0.0f
                i <- i + 1
        else
            invalidArg "T" $"ChunkFunctions.thresholdNative supports Int8, UInt8, Int16, UInt16, Int32, and Float32 chunks, got {t.Name}."
        output
    with
    | _ ->
        Chunk.decRef output
        reraise()

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
    let output = Chunk.create<uint8> chunk.Size
    try
        let t = typeof<'T>
        let outputPixels = output.Bytes.AsSpan(0, output.ByteLength)
        if t = typeof<uint8> then
            chunk.Bytes.AsSpan(0, chunk.ByteLength).CopyTo(outputPixels)
        elif t = typeof<int8> then
            let inputPixels = MemoryMarshal.Cast<byte, sbyte>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
            let mutable i = 0
            while i < inputPixels.Length do
                let value = inputPixels[i]
                outputPixels[i] <- if value <= 0y then 0uy else uint8 value
                i <- i + 1
        elif t = typeof<uint16> then
            let inputPixels = MemoryMarshal.Cast<byte, uint16>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
            let mutable i = 0
            while i < inputPixels.Length do
                let value = inputPixels[i]
                outputPixels[i] <- if value >= 255us then 255uy else uint8 value
                i <- i + 1
        elif t = typeof<int16> then
            let inputPixels = MemoryMarshal.Cast<byte, int16>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
            let mutable i = 0
            while i < inputPixels.Length do
                let value = inputPixels[i]
                outputPixels[i] <- if value <= 0s then 0uy elif value >= 255s then 255uy else uint8 value
                i <- i + 1
        elif t = typeof<int32> then
            let inputPixels = MemoryMarshal.Cast<byte, int32>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
            let mutable i = 0
            while i < inputPixels.Length do
                let value = inputPixels[i]
                outputPixels[i] <- if value <= 0 then 0uy elif value >= 255 then 255uy else uint8 value
                i <- i + 1
        elif t = typeof<float32> then
            let inputPixels = MemoryMarshal.Cast<byte, float32>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
            let mutable i = 0
            while i < inputPixels.Length do
                outputPixels[i] <- clampRoundToByte inputPixels[i]
                i <- i + 1
        else
            invalidArg "T" $"ChunkFunctions.castToUInt8 supports Int8, UInt8, Int16, UInt16, Int32, and Float32 chunks, got {t.Name}."
        output
    with
    | _ ->
        Chunk.decRef output
        reraise()

let castToUInt8<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> : Stage<Chunk<'T>, Chunk<uint8>> =
    let mapper _debug chunk =
        try
            castChunkToUInt8 chunk
        finally
            Chunk.decRef chunk

    Stage.map $"chunkCastToUInt8.{typeof<'T>.Name}" mapper id id

let private castChunkToFloat32<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (chunk: Chunk<'T>) =
    let output = Chunk.create<float32> chunk.Size
    try
        let outputPixels = Chunk.span<float32> output
        let t = typeof<'T>
        if t = typeof<float32> then
            let inputPixels = MemoryMarshal.Cast<byte, float32>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
            inputPixels.CopyTo(outputPixels)
        elif t = typeof<uint8> then
            let inputPixels = chunk.Bytes.AsSpan(0, chunk.ByteLength)
            let byteVectorWidth = Vector<byte>.Count
            let floatVectorWidth = Vector<float32>.Count
            let vectorEnd = inputPixels.Length - (inputPixels.Length % byteVectorWidth)
            let mutable i = 0
            while i < vectorEnd do
                let mutable inputPart = inputPixels.Slice(i, byteVectorWidth)
                let inputSlice = MemoryMarshal.CreateReadOnlySpan(&MemoryMarshal.GetReference(inputPart), byteVectorWidth)
                let a, b, c, d = byteVectorToSingleVectors inputSlice
                a.CopyTo(outputPixels.Slice(i, floatVectorWidth))
                b.CopyTo(outputPixels.Slice(i + floatVectorWidth, floatVectorWidth))
                c.CopyTo(outputPixels.Slice(i + 2 * floatVectorWidth, floatVectorWidth))
                d.CopyTo(outputPixels.Slice(i + 3 * floatVectorWidth, floatVectorWidth))
                i <- i + byteVectorWidth
            while i < inputPixels.Length do
                outputPixels[i] <- float32 inputPixels[i]
                i <- i + 1
        elif t = typeof<int8> then
            let inputPixels = MemoryMarshal.Cast<byte, sbyte>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
            let int8VectorWidth = Vector<sbyte>.Count
            let floatVectorWidth = Vector<float32>.Count
            let vectorEnd = inputPixels.Length - (inputPixels.Length % int8VectorWidth)
            let mutable i = 0
            while i < vectorEnd do
                let mutable inputPart = inputPixels.Slice(i, int8VectorWidth)
                let inputSlice = MemoryMarshal.CreateReadOnlySpan(&MemoryMarshal.GetReference(inputPart), int8VectorWidth)
                let a, b, c, d = int8VectorToSingleVectors inputSlice
                a.CopyTo(outputPixels.Slice(i, floatVectorWidth))
                b.CopyTo(outputPixels.Slice(i + floatVectorWidth, floatVectorWidth))
                c.CopyTo(outputPixels.Slice(i + 2 * floatVectorWidth, floatVectorWidth))
                d.CopyTo(outputPixels.Slice(i + 3 * floatVectorWidth, floatVectorWidth))
                i <- i + int8VectorWidth
            while i < inputPixels.Length do
                outputPixels[i] <- float32 inputPixels[i]
                i <- i + 1
        elif t = typeof<uint16> then
            let inputPixels = MemoryMarshal.Cast<byte, uint16>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
            let uint16VectorWidth = Vector<uint16>.Count
            let floatVectorWidth = Vector<float32>.Count
            let vectorEnd = inputPixels.Length - (inputPixels.Length % uint16VectorWidth)
            let mutable i = 0
            while i < vectorEnd do
                let mutable inputPart = inputPixels.Slice(i, uint16VectorWidth)
                let inputSlice = MemoryMarshal.CreateReadOnlySpan(&MemoryMarshal.GetReference(inputPart), uint16VectorWidth)
                let a, b = uint16VectorToSingleVectors inputSlice
                a.CopyTo(outputPixels.Slice(i, floatVectorWidth))
                b.CopyTo(outputPixels.Slice(i + floatVectorWidth, floatVectorWidth))
                i <- i + uint16VectorWidth
            while i < inputPixels.Length do
                outputPixels[i] <- float32 inputPixels[i]
                i <- i + 1
        elif t = typeof<int16> then
            let inputPixels = MemoryMarshal.Cast<byte, int16>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
            let int16VectorWidth = Vector<int16>.Count
            let floatVectorWidth = Vector<float32>.Count
            let vectorEnd = inputPixels.Length - (inputPixels.Length % int16VectorWidth)
            let mutable i = 0
            while i < vectorEnd do
                let mutable inputPart = inputPixels.Slice(i, int16VectorWidth)
                let inputSlice = MemoryMarshal.CreateReadOnlySpan(&MemoryMarshal.GetReference(inputPart), int16VectorWidth)
                let a, b = int16VectorToSingleVectors inputSlice
                a.CopyTo(outputPixels.Slice(i, floatVectorWidth))
                b.CopyTo(outputPixels.Slice(i + floatVectorWidth, floatVectorWidth))
                i <- i + int16VectorWidth
            while i < inputPixels.Length do
                outputPixels[i] <- float32 inputPixels[i]
                i <- i + 1
        elif t = typeof<int32> then
            let inputPixels = MemoryMarshal.Cast<byte, int32>(chunk.Bytes.AsSpan(0, chunk.ByteLength))
            let vectorWidth = Vector<int32>.Count
            let vectorEnd = inputPixels.Length - (inputPixels.Length % vectorWidth)
            let mutable i = 0
            while i < vectorEnd do
                Vector.ConvertToSingle(Vector<int32>(inputPixels.Slice(i, vectorWidth))).CopyTo(outputPixels.Slice(i, vectorWidth))
                i <- i + vectorWidth
            while i < inputPixels.Length do
                outputPixels[i] <- float32 inputPixels[i]
                i <- i + 1
        else
            invalidArg "T" $"ChunkFunctions.castToFloat32 supports Int8, UInt8, Int16, UInt16, Int32, and Float32 chunks, got {t.Name}."
        output
    with
    | _ ->
        Chunk.decRef output
        reraise()

let castToFloat32<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> : Stage<Chunk<'T>, Chunk<float32>> =
    let mapper _debug chunk =
        try
            castChunkToFloat32 chunk
        finally
            Chunk.decRef chunk

    Stage.map $"chunkCastToFloat32.{typeof<'T>.Name}" mapper id id

let private castFloat32ChunkTo<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (chunk: Chunk<float32>) =
    let output = Chunk.create<'T> chunk.Size
    try
        let inputPixels = Chunk.span<float32> chunk
        let t = typeof<'T>
        if t = typeof<float32> then
            let outputPixels = MemoryMarshal.Cast<byte, float32>(output.Bytes.AsSpan(0, output.ByteLength))
            inputPixels.CopyTo(outputPixels)
        elif t = typeof<uint8> then
            let outputPixels = output.Bytes.AsSpan(0, output.ByteLength)
            let mutable i = 0
            while i < inputPixels.Length do
                outputPixels[i] <- clampRoundToByte inputPixels[i]
                i <- i + 1
        elif t = typeof<int8> then
            let outputPixels = MemoryMarshal.Cast<byte, sbyte>(output.Bytes.AsSpan(0, output.ByteLength))
            let mutable i = 0
            while i < inputPixels.Length do
                outputPixels[i] <- clampRoundToSByte inputPixels[i]
                i <- i + 1
        elif t = typeof<uint16> then
            let outputPixels = MemoryMarshal.Cast<byte, uint16>(output.Bytes.AsSpan(0, output.ByteLength))
            let mutable i = 0
            while i < inputPixels.Length do
                outputPixels[i] <- clampRoundToUInt16 inputPixels[i]
                i <- i + 1
        elif t = typeof<int16> then
            let outputPixels = MemoryMarshal.Cast<byte, int16>(output.Bytes.AsSpan(0, output.ByteLength))
            let mutable i = 0
            while i < inputPixels.Length do
                outputPixels[i] <- clampRoundToInt16 inputPixels[i]
                i <- i + 1
        elif t = typeof<int32> then
            let outputPixels = MemoryMarshal.Cast<byte, int32>(output.Bytes.AsSpan(0, output.ByteLength))
            let mutable i = 0
            while i < inputPixels.Length do
                outputPixels[i] <- clampRoundToInt32 inputPixels[i]
                i <- i + 1
        else
            invalidArg "T" $"ChunkFunctions.castFromFloat32 supports Int8, UInt8, Int16, UInt16, Int32, and Float32 chunks, got {t.Name}."
        output
    with
    | _ ->
        Chunk.decRef output
        reraise()

let castFromFloat32<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> : Stage<Chunk<float32>, Chunk<'T>> =
    let mapper _debug chunk =
        try
            castFloat32ChunkTo<'T> chunk
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
