module StackMedian

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

type private ChunkSlice =
    { Index: int
      Chunk: Chunk<uint8> }

type private TypedChunkSlice<'T when 'T: equality> =
    { Index: int
      Chunk: Chunk<'T> }

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
    ChunkKernel.NativeMedian.ensureAvailable ()

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
            ChunkKernel.NativeMedian.medianUInt8NthSlab(slices, output, width, height, windowLength, radius, outputStart, outputCount))
        width
        height
        radius
        window

let private medianNativeUInt16NthSlice width height radius (window: Chunk<uint16>[]) =
    medianNativeNthSlice<uint16>
        (fun slices output width height windowLength radius outputStart outputCount ->
            ChunkKernel.NativeMedian.medianUInt16NthSlab(slices, output, width, height, windowLength, radius, outputStart, outputCount))
        width
        height
        radius
        window

let private medianNativeInt32NthSlice width height radius (window: Chunk<int32>[]) =
    medianNativeNthSlice<int32>
        (fun slices output width height windowLength radius outputStart outputCount ->
            ChunkKernel.NativeMedian.medianInt32NthSlab(slices, output, width, height, windowLength, radius, outputStart, outputCount))
        width
        height
        radius
        window

let private medianNativeFloat32NthSlice width height radius (window: Chunk<float32>[]) =
    medianNativeNthSlice<float32>
        (fun slices output width height windowLength radius outputStart outputCount ->
            ChunkKernel.NativeMedian.medianFloat32NthSlab(slices, output, width, height, windowLength, radius, outputStart, outputCount))
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

