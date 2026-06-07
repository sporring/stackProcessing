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

type private TypedChunkSlice<'T when 'T: equality> =
    { Index: int
      Chunk: Chunk<'T> }

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

    let transition = ProfileTransition.create Streaming Streaming
    let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed
    let suffix = if batchSize = 1 then "" else $".parallel{batchSize}"

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
                    invalidArg "chunk" $"Chunk convolution expects 2D slice chunks with depth 1, got {chunk.Size}."

                if not initialized then
                    width <- int chunkWidth
                    height <- int chunkHeight
                    if width <= 0 || height <= 0 then
                        invalidArg "chunk" $"Chunk convolution expects positive slice dimensions, got {chunk.Size}."
                    for i in plan.PadZ .. -1 .. 1 do
                        addPadding -i
                    initialized <- true
                else
                    validateTypedSliceChunk "convolution" width height chunk

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
                            min batchSize availableOutputs
                        elif availableOutputs >= batchSize then
                            batchSize
                        else
                            0

                    if batchCount > 0 then
                        let windows = Array.zeroCreate<TypedChunkSlice<'T>[]> batchCount
                        let releasedWindows = Array.zeroCreate<bool> batchCount
                        let outputs = Array.zeroCreate<Chunk<'T>> batchCount
                        try
                            for i in 0 .. batchCount - 1 do
                                windows[i] <- retainTypedWindow queue i windowLength

                            if batchCount = 1 then
                                try
                                    outputs[0] <- convolveFixedKernelSlice width height plan windows[0]
                                finally
                                    releaseTypedWindow windows[0]
                                    releasedWindows[0] <- true
                            else
                                Parallel.For(
                                    0,
                                    batchCount,
                                    fun i ->
                                        try
                                            outputs[i] <- convolveFixedKernelSlice width height plan windows[i]
                                        finally
                                            releaseTypedWindow windows[i]
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
                    for i in 1 .. plan.PadZ do
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
        $"chunkConvolveFixedKernel{suffix}.{typeof<'T>.Name}.{plan.Width}x{plan.Height}x{plan.Depth}"
        apply
        transition
        memoryModel
        id

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

let private medianFromUInt16Histogram totalCount (histogram: uint16[]) =
    let target = totalCount / 2 + 1
    let mutable cumulative = 0
    let mutable value = 0
    while value < 255 && cumulative < target do
        cumulative <- cumulative + int histogram[value]
        if cumulative < target then
            value <- value + 1
    uint8 value

let private buildZHistogramsUInt8 width height (window: ChunkSlice[]) =
    let pixelCount = width * height
    if pixelCount > Int32.MaxValue / 256 then
        invalidArg "window" $"UInt8 PH median dense z-histogram would exceed Int32 indexing for {width}x{height} slices."

    let zHistograms = Array.zeroCreate<uint16> (pixelCount * 256)
    for item in window do
        let inputPixels = Chunk.span<uint8> item.Chunk
        let mutable p = 0
        while p < pixelCount do
            let index = p * 256 + int inputPixels[p]
            zHistograms[index] <- zHistograms[index] + 1us
            p <- p + 1
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

let private addYColumnToKernelHistogram (kernelHistogram: uint16[]) (yHistograms: uint16[]) x =
    addUInt16HistogramInto kernelHistogram 0 yHistograms (x * 256)

let private subtractYColumnFromKernelHistogram (kernelHistogram: uint16[]) (yHistograms: uint16[]) x =
    subtractUInt16HistogramFrom kernelHistogram 0 yHistograms (x * 256)

let private medianPerreaultHebertUInt8DenseSlice width height radius (window: ChunkSlice[]) =
    let windowLength = 2 * radius + 1
    let totalCount = windowLength * windowLength * windowLength
    if totalCount > int UInt16.MaxValue then
        invalidArg "radius" $"UInt8 dense PH median first version stores counts as UInt16 and supports at most {UInt16.MaxValue} samples; got {totalCount}."

    let output = Chunk.create<uint8> (uint64 width, uint64 height, 1UL)
    try
        let outputPixels = Chunk.span<uint8> output
        let zHistograms = buildZHistogramsUInt8 width height window
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

        for yy in -radius .. radius do
            addYRow yy

        for y in 0 .. height - 1 do
            clearUInt16Histogram kernelHistogram

            for xx in -radius .. radius do
                if xx >= 0 && xx < width then
                    addYColumnToKernelHistogram kernelHistogram yHistograms xx
                else
                    kernelHistogram[0] <- kernelHistogram[0] + zeroYColumnCount

            let rowOffset = y * width
            for x in 0 .. width - 1 do
                outputPixels[rowOffset + x] <- medianFromUInt16Histogram totalCount kernelHistogram

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

            if y < height - 1 then
                subtractYRow (y - radius)
                addYRow (y + radius + 1)

        output
    with
    | _ ->
        Chunk.decRef output
        reraise()

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
