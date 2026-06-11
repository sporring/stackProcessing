module StackConnectedComponents

open System
open System.Collections.Generic
open System.Threading.Tasks
open FSharp.Control
open SlimPipeline
open StackCore

type private ConnectedComponentChunkWindow =
    { LabelChunks: Chunk<uint32> list
      ObjectCount: uint32 }

let private binaryBackground = 0uy
let private binaryForeground = 1uy

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

