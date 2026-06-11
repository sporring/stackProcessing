module StackConvolve

open System
open System.Collections.Generic
open System.Numerics
open System.Runtime.InteropServices
open SlimPipeline
open StackCore

module ChunkKernel = ChunkCore.ChunkFunctions

type private TypedChunkSlice<'T when 'T: equality> =
    { Index: int
      Chunk: Chunk<'T> }

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
    ChunkKernel.NativeMedian.ensureAvailable ()

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

            ChunkKernel.NativeMedian.convolveFloat32Slices(
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
    ChunkKernel.NativeMedian.ensureAvailable ()

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

            ChunkKernel.NativeMedian.convolveUInt8Slices(
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

type ConvolveAxis =
    | X
    | Y
    | Z

let private validateAxisKernel (kernel: float32[]) =
    if isNull kernel then
        nullArg "kernel"
    if kernel.Length = 0 || kernel.Length % 2 = 0 then
        invalidArg "kernel" $"Chunk axis convolution expects a non-empty odd-length kernel, got {kernel.Length}."
    kernel.Length / 2

let private convolveNativeAxisChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    axis
    kernel
    (items: Chunk<'T>[])
    =
    match axis with
    | X ->
        if items.Length <> 1 then
            invalidArg "items" $"Chunk native convolveX expects one slice, got {items.Length}."
        ChunkKernel.convolveNativeX<'T> kernel items[0]
    | Y ->
        if items.Length <> 1 then
            invalidArg "items" $"Chunk native convolveY expects one slice, got {items.Length}."
        ChunkKernel.convolveNativeY<'T> kernel items[0]
    | Z ->
        if items.Length <> kernel.Length then
            invalidArg "items" $"Chunk native convolveZ expects {kernel.Length} slices, got {items.Length}."
        ChunkKernel.convolveNativeZ<'T> kernel items

let convolveNativeAxisParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    axis
    (kernel: float32[])
    workers
    : Stage<Chunk<'T>, Chunk<'T>> =
    let radius = validateAxisKernel kernel
    if workers < 1 then
        invalidArg "workers" $"Chunk native axis convolution expects at least one worker, got {workers}."

    let windowLength, padding =
        match axis with
        | X
        | Y -> 1, 0
        | Z -> kernel.Length, radius

    let axisName =
        match axis with
        | X -> "X"
        | Y -> "Y"
        | Z -> "Z"

    let memoryNeed nPixels =
        uint64 (windowLength + workers) * nPixels * uint64 (Marshal.SizeOf<'T>())

    let zeroMaker _index (source: Chunk<'T>) =
        let width, height, depth = source.Size
        if depth <> 1UL then
            invalidArg "source" $"Chunk native convolve{axisName} expects 2D slice chunks with depth 1, got {source.Size}."
        zeroChunkTyped<'T> (int width) (int height)

    let mapper _debug (window: Window<Chunk<'T>>) =
        let items = window.Items |> List.toArray
        try
            [ convolveNativeAxisChunk<'T> axis kernel items ]
        finally
            items |> Array.iter Chunk.decRef

    Stage.parallelCollect
        $"chunkConvolveNative{axisName}.parallelCollect.{typeof<'T>.Name}.k{kernel.Length}.workers{workers}"
        windowLength
        workers
        1
        padding
        zeroMaker
        mapper
        memoryNeed
        id

let convolveNativeXParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> kernel workers =
    convolveNativeAxisParallelCollect<'T> X kernel workers

let convolveNativeYParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> kernel workers =
    convolveNativeAxisParallelCollect<'T> Y kernel workers

let convolveNativeZParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> kernel workers =
    convolveNativeAxisParallelCollect<'T> Z kernel workers

let finiteDiffKernel1D order =
    ChunkKernel.finiteDiffKernel1D order

let finiteDiffNativeXParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> order workers =
    convolveNativeXParallelCollect<'T> (finiteDiffKernel1D order) workers

let finiteDiffNativeYParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> order workers =
    convolveNativeYParallelCollect<'T> (finiteDiffKernel1D order) workers

let finiteDiffNativeZParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> order workers =
    convolveNativeZParallelCollect<'T> (finiteDiffKernel1D order) workers

let separableConvolveNativeParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    xKernel
    yKernel
    zKernel
    workers
    : Stage<Chunk<'T>, Chunk<'T>> =
    convolveNativeXParallelCollect<'T> xKernel workers
    |> fun stage -> Stage.compose stage (convolveNativeYParallelCollect<'T> yKernel workers)
    |> fun stage -> Stage.compose stage (convolveNativeZParallelCollect<'T> zKernel workers)

let boxKernel radius =
    if radius < 0 then
        invalidArg "radius" $"Chunk box filter expects a non-negative radius, got {radius}."
    let length = 2 * radius + 1
    let weight = 1.0f / float32 length
    Array.create length weight

let gaussianKernel sigma radius =
    if sigma <= 0.0 then
        invalidArg "sigma" $"Chunk Gaussian filter expects positive sigma, got {sigma}."
    if radius < 0 then
        invalidArg "radius" $"Chunk Gaussian filter expects a non-negative radius, got {radius}."
    let kernel =
        Array.init (2 * radius + 1) (fun i ->
            let x = float (i - radius)
            float32 (Math.Exp(-(x * x) / (2.0 * sigma * sigma))))
    let sum = kernel |> Array.sum
    kernel |> Array.map (fun value -> value / sum)

let defaultGaussianRadius sigma =
    if sigma <= 0.0 then
        invalidArg "sigma" $"Chunk Gaussian filter expects positive sigma, got {sigma}."
    max 1 (int (Math.Ceiling(3.0 * sigma)))

let boxFilterNativeParallelCollectXYZ<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    radiusX
    radiusY
    radiusZ
    workers
    : Stage<Chunk<'T>, Chunk<'T>> =
    let xKernel = boxKernel radiusX
    let yKernel = boxKernel radiusY
    let zKernel = boxKernel radiusZ
    separableConvolveNativeParallelCollect<'T> xKernel yKernel zKernel workers

let boxFilterNativeParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    radius
    workers
    : Stage<Chunk<'T>, Chunk<'T>> =
    boxFilterNativeParallelCollectXYZ<'T> radius radius radius workers

let gaussianFilterNativeParallelCollectXYZ<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    sigmaX
    radiusX
    sigmaY
    radiusY
    sigmaZ
    radiusZ
    workers
    : Stage<Chunk<'T>, Chunk<'T>> =
    let xKernel = gaussianKernel sigmaX radiusX
    let yKernel = gaussianKernel sigmaY radiusY
    let zKernel = gaussianKernel sigmaZ radiusZ
    separableConvolveNativeParallelCollect<'T> xKernel yKernel zKernel workers

let gaussianFilterNativeParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    sigma
    radius
    workers
    : Stage<Chunk<'T>, Chunk<'T>> =
    gaussianFilterNativeParallelCollectXYZ<'T> sigma radius sigma radius sigma radius workers

let sobelXNativeParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    workers
    : Stage<Chunk<'T>, Chunk<'T>> =
    let derivative = [| -1.0f; 0.0f; 1.0f |]
    let smooth = [| 0.25f; 0.5f; 0.25f |]
    separableConvolveNativeParallelCollect<'T> derivative smooth smooth workers

let sobelYNativeParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    workers
    : Stage<Chunk<'T>, Chunk<'T>> =
    let derivative = [| -1.0f; 0.0f; 1.0f |]
    let smooth = [| 0.25f; 0.5f; 0.25f |]
    separableConvolveNativeParallelCollect<'T> smooth derivative smooth workers

let sobelZNativeParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    workers
    : Stage<Chunk<'T>, Chunk<'T>> =
    let derivative = [| -1.0f; 0.0f; 1.0f |]
    let smooth = [| 0.25f; 0.5f; 0.25f |]
    separableConvolveNativeParallelCollect<'T> smooth smooth derivative workers

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
