open System
open System.Buffers
open System.Diagnostics
open System.Globalization
open System.IO
open System.Numerics
open System.Runtime.CompilerServices
open System.Runtime.InteropServices
open System.Threading
open Image
open Microsoft.FSharp.NativeInterop

#nowarn "9"

type PixelType =
    | UInt8
    | UInt16
    | Float32

type Shape =
    { Width: int
      Height: int
      Depth: int }

type Row =
    { Backend: string
      Variant: string
      PixelType: string
      Shape: string
      Threshold: string
      Repeat: int
      ExitCode: int
      Seconds: string
      ManagedAllocatedBytes: string
      Error: string }

let private invariant = CultureInfo.InvariantCulture

let private parseArgs (args: string array) =
    let rec loop i (acc: Map<string,string>) =
        if i >= args.Length then
            acc
        elif args[i].StartsWith("--", StringComparison.Ordinal) then
            if i + 1 >= args.Length || args[i + 1].StartsWith("--", StringComparison.Ordinal) then
                loop (i + 1) (acc.Add(args[i].Substring(2), "true"))
            else
                loop (i + 2) (acc.Add(args[i].Substring(2), args[i + 1]))
        else
            loop (i + 1) acc
    loop 0 Map.empty

let private optional name fallback (opts: Map<string,string>) =
    opts.TryFind name |> Option.defaultValue fallback

let private optionalBool name fallback (opts: Map<string,string>) =
    opts.TryFind name
    |> Option.map (fun value ->
        match value.Trim().ToLowerInvariant() with
        | "1" | "true" | "yes" | "y" -> true
        | "0" | "false" | "no" | "n" -> false
        | _ -> invalidArg name $"Expected a boolean value for --{name}, got '{value}'.")
    |> Option.defaultValue fallback

let private parseShape (text: string) =
    let parts = text.Split('x', StringSplitOptions.RemoveEmptyEntries)
    if parts.Length <> 3 then invalidArg "shape" $"Expected WxHxD shape, got '{text}'."
    { Width = Int32.Parse(parts[0], invariant)
      Height = Int32.Parse(parts[1], invariant)
      Depth = Int32.Parse(parts[2], invariant) }

let private parsePixelType (text: string) =
    match text.Trim().ToLowerInvariant() with
    | "uint8" -> UInt8
    | "uint16" -> UInt16
    | "float32" -> Float32
    | _ -> invalidArg "pixel-type" $"Unsupported pixel type '{text}'."

let private shapeText shape =
    $"{shape.Width}x{shape.Height}x{shape.Depth}"

let private elementCount shape =
    int64 shape.Width * int64 shape.Height * int64 shape.Depth

let private requireIntLength shape =
    let count = elementCount shape
    if count > int64 Int32.MaxValue then
        invalidArg "shape" $"The experiment uses CLR arrays and requires <= Int32.MaxValue elements; got {count}."
    int count

let private identityDirection dim =
    [ for row in 0 .. dim - 1 do
        for col in 0 .. dim - 1 do
            if row = col then 1.0 else 0.0 ]

let private vectorUInt32 values = new itk.simple.VectorUInt32(values |> Seq.map uint32 |> Seq.toList)
let private vectorDouble values = new itk.simple.VectorDouble(values |> Seq.toList)

let private setImportBuffer<'T> (importer: itk.simple.ImportImageFilter) (buffer: nativeint) =
    let t = typeof<'T>
    if t = typeof<uint8> then importer.SetBufferAsUInt8(buffer)
    elif t = typeof<uint16> then importer.SetBufferAsUInt16(buffer)
    elif t = typeof<float32> then importer.SetBufferAsFloat(buffer)
    else invalidArg "T" $"Unsupported import type {t.Name}."

let private importedImage<'T> shape (source: 'T[]) =
    use importer = new itk.simple.ImportImageFilter()
    importer.SetSize(vectorUInt32 [ shape.Width; shape.Height; shape.Depth ])
    importer.SetSpacing(vectorDouble [ 1.0; 1.0; 1.0 ])
    importer.SetOrigin(vectorDouble [ 0.0; 0.0; 0.0 ])
    importer.SetDirection(vectorDouble (identityDirection 3))
    let handle = GCHandle.Alloc(source, GCHandleType.Pinned)
    try
        setImportBuffer<'T> importer (handle.AddrOfPinnedObject())
        use imported = importer.Execute()
        let copy = new itk.simple.Image(imported)
        copy.MakeUnique()
        copy
    finally
        handle.Free()

let private copyUInt8ImageToPooledArray length (image: itk.simple.Image) =
    if image.GetPixelID() <> itk.simple.PixelIDValueEnum.sitkUInt8 then
        invalidArg "image" $"Expected UInt8 SimpleITK output, got {image.GetPixelID()}."
    if image.GetNumberOfComponentsPerPixel() <> 1u then
        invalidArg "image" $"Expected scalar SimpleITK output, got {image.GetNumberOfComponentsPerPixel()} components per pixel."
    let output = ArrayPool<uint8>.Shared.Rent(length)
    try
        Marshal.Copy(image.GetConstBufferAsUInt8(), output, 0, length)
        output[0] |> ignore
    finally
        ArrayPool<uint8>.Shared.Return(output)

let private fillUInt8 (buffer: uint8[]) length =
    for i in 0 .. length - 1 do
        buffer[i] <- byte (i &&& 0xff)

let private fillUInt16 (buffer: uint16[]) length =
    for i in 0 .. length - 1 do
        buffer[i] <- uint16 (i &&& 0xffff)

let private fillFloat32 (buffer: float32[]) length =
    for i in 0 .. length - 1 do
        buffer[i] <- float32 (i &&& 0xff)

type private ExperimentImageStorage<'T when 'T: equality> =
    | ItkImage of itk.simple.Image
    | ArrayPool1D of buffer: 'T[] * length: int * release: (unit -> unit)

type private ExperimentImage<'T when 'T: equality>(shape: Shape, storage: ExperimentImageStorage<'T>) =
    let mutable storageOpt = Some storage
    let mutable refCount = 1

    member _.Shape = shape

    member _.Length = requireIntLength shape

    member _.Storage =
        match storageOpt with
        | Some storage -> storage
        | None -> ObjectDisposedException(nameof ExperimentImage<'T>) |> raise

    member image.IncRef() =
        if Interlocked.Increment(&refCount) <= 1 then
            ObjectDisposedException(nameof ExperimentImage<'T>) |> raise
        image

    member image.TryArrayPool1D() =
        match image.Storage with
        | ArrayPool1D(buffer, length, _) -> ValueSome(buffer, length)
        | ItkImage _ -> ValueNone

    member image.ToSimpleITK() =
        match image.Storage with
        | ItkImage image -> image
        | ArrayPool1D(buffer, length, _) ->
            if length <> image.Length then
                invalidOp $"ExperimentImage length mismatch: storage length {length}, shape length {image.Length}."
            importedImage<'T> shape buffer

    member _.DecRef() =
        let remaining = Interlocked.Decrement(&refCount)
        if remaining = 0 then
            match storageOpt with
            | None -> ()
            | Some storage ->
                storageOpt <- None
                match storage with
                | ItkImage image -> image.Dispose()
                | ArrayPool1D(_, _, release) -> release()
        elif remaining < 0 then
            invalidOp "ExperimentImage.DecRef called after the image had already been released."

    interface IDisposable with
        member image.Dispose() = image.DecRef()

module private ExperimentImage =
    let rentArrayPool<'T when 'T: equality> shape =
        let length = requireIntLength shape
        let buffer = ArrayPool<'T>.Shared.Rent(length)
        let release () =
            ArrayPool<'T>.Shared.Return(buffer, RuntimeHelpers.IsReferenceOrContainsReferences<'T>())
        new ExperimentImage<'T>(shape, ArrayPool1D(buffer, length, release))

    let fromItkTransfer<'T when 'T: equality> shape (image: itk.simple.Image) =
        new ExperimentImage<'T>(shape, ItkImage image)

let private thresholdUInt8Array threshold (input: uint8[]) (output: uint8[]) length =
    for i in 0 .. length - 1 do
        output[i] <- if input[i] >= threshold then 255uy else 0uy

let private thresholdUInt8While threshold (input: uint8[]) (output: uint8[]) length =
    let mutable i = 0
    while i < length do
        output[i] <- if input[i] >= threshold then 255uy else 0uy
        i <- i + 1

let private thresholdUInt8BoolWhile threshold (input: uint8[]) (output: bool[]) length =
    let mutable i = 0
    while i < length do
        output[i] <- input[i] >= threshold
        i <- i + 1

[<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
let private thresholdUInt8WhileOptimized threshold (input: uint8[]) (output: uint8[]) length =
    let mutable i = 0
    while i < length do
        output[i] <- if input[i] >= threshold then 255uy else 0uy
        i <- i + 1

let private thresholdUInt8NativePtr threshold (input: uint8[]) (output: uint8[]) length =
    let inputHandle = GCHandle.Alloc(input, GCHandleType.Pinned)
    let outputHandle = GCHandle.Alloc(output, GCHandleType.Pinned)
    try
        let inputPtr = NativePtr.ofNativeInt<uint8> (inputHandle.AddrOfPinnedObject())
        let outputPtr = NativePtr.ofNativeInt<uint8> (outputHandle.AddrOfPinnedObject())
        let mutable i = 0
        while i < length do
            NativePtr.set outputPtr i (if NativePtr.get inputPtr i >= threshold then 255uy else 0uy)
            i <- i + 1
    finally
        inputHandle.Free()
        outputHandle.Free()

let private thresholdUInt8Vector (threshold: byte) (input: uint8[]) (output: uint8[]) length =
    let width = Vector<byte>.Count
    let thresholdVector = Vector<byte>(threshold)
    let mutable i = 0
    let vectorEnd = length - (length % width)
    while i < vectorEnd do
        let values = Vector<byte>(input, i)
        let mask = Vector.GreaterThanOrEqual(values, thresholdVector)
        mask.CopyTo(output, i)
        i <- i + width
    while i < length do
        output[i] <- if input[i] >= threshold then 255uy else 0uy
        i <- i + 1

let private thresholdUInt8OneVector (threshold: byte) (input: uint8[]) (output: uint8[]) length =
    let width = Vector<byte>.Count
    let thresholdVector = Vector<byte>(threshold)
    let oneVector = Vector<byte>(1uy)
    let mutable i = 0
    let vectorEnd = length - (length % width)
    while i < vectorEnd do
        let values = Vector<byte>(input, i)
        let mask = Vector.GreaterThanOrEqual(values, thresholdVector)
        Vector.BitwiseAnd(mask, oneVector).CopyTo(output, i)
        i <- i + width
    while i < length do
        output[i] <- if input[i] >= threshold then 1uy else 0uy
        i <- i + 1

let private thresholdUInt16Array threshold (input: uint16[]) (output: uint8[]) length =
    for i in 0 .. length - 1 do
        output[i] <- if input[i] >= threshold then 255uy else 0uy

let private thresholdUInt16While threshold (input: uint16[]) (output: uint8[]) length =
    let mutable i = 0
    while i < length do
        output[i] <- if input[i] >= threshold then 255uy else 0uy
        i <- i + 1

let private thresholdUInt16BoolWhile threshold (input: uint16[]) (output: bool[]) length =
    let mutable i = 0
    while i < length do
        output[i] <- input[i] >= threshold
        i <- i + 1

let private thresholdUInt16ToUInt16MaxWhile threshold (input: uint16[]) (output: uint16[]) length =
    let mutable i = 0
    while i < length do
        output[i] <- if input[i] >= threshold then UInt16.MaxValue else 0us
        i <- i + 1

let private thresholdUInt16ToUInt16OneWhile threshold (input: uint16[]) (output: uint16[]) length =
    let mutable i = 0
    while i < length do
        output[i] <- if input[i] >= threshold then 1us else 0us
        i <- i + 1

let private thresholdUInt16ToUInt16OneArrayMap threshold (input: uint16[]) =
    input |> Array.map (fun value -> if value >= threshold then 1us else 0us)

[<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
let private thresholdUInt16WhileOptimized threshold (input: uint16[]) (output: uint8[]) length =
    let mutable i = 0
    while i < length do
        output[i] <- if input[i] >= threshold then 255uy else 0uy
        i <- i + 1

let private thresholdUInt16NativePtr threshold (input: uint16[]) (output: uint8[]) length =
    let inputHandle = GCHandle.Alloc(input, GCHandleType.Pinned)
    let outputHandle = GCHandle.Alloc(output, GCHandleType.Pinned)
    try
        let inputPtr = NativePtr.ofNativeInt<uint16> (inputHandle.AddrOfPinnedObject())
        let outputPtr = NativePtr.ofNativeInt<uint8> (outputHandle.AddrOfPinnedObject())
        let mutable i = 0
        while i < length do
            NativePtr.set outputPtr i (if NativePtr.get inputPtr i >= threshold then 255uy else 0uy)
            i <- i + 1
    finally
        inputHandle.Free()
        outputHandle.Free()

let private thresholdUInt16ToUInt16MaxVector (threshold: uint16) (input: uint16[]) (output: uint16[]) length =
    let width = Vector<uint16>.Count
    let thresholdVector = Vector<uint16>(threshold)
    let mutable i = 0
    let vectorEnd = length - (length % width)
    while i < vectorEnd do
        let values = Vector<uint16>(input, i)
        let mask = Vector.GreaterThanOrEqual(values, thresholdVector)
        mask.CopyTo(output, i)
        i <- i + width
    while i < length do
        output[i] <- if input[i] >= threshold then UInt16.MaxValue else 0us
        i <- i + 1

let private thresholdUInt16ToUInt16OneVector (threshold: uint16) (input: uint16[]) (output: uint16[]) length =
    let width = Vector<uint16>.Count
    let thresholdVector = Vector<uint16>(threshold)
    let oneVector = Vector<uint16>(1us)
    let mutable i = 0
    let vectorEnd = length - (length % width)
    while i < vectorEnd do
        let values = Vector<uint16>(input, i)
        let mask = Vector.GreaterThanOrEqual(values, thresholdVector)
        let normalized = Vector.BitwiseAnd(mask, oneVector)
        normalized.CopyTo(output, i)
        i <- i + width
    while i < length do
        output[i] <- if input[i] >= threshold then 1us else 0us
        i <- i + 1

let private thresholdFloat32Array threshold (input: float32[]) (output: uint8[]) length =
    for i in 0 .. length - 1 do
        output[i] <- if input[i] >= threshold then 255uy else 0uy

let private thresholdFloat32While threshold (input: float32[]) (output: uint8[]) length =
    let mutable i = 0
    while i < length do
        output[i] <- if input[i] >= threshold then 255uy else 0uy
        i <- i + 1

let private thresholdFloat32BoolWhile threshold (input: float32[]) (output: bool[]) length =
    let mutable i = 0
    while i < length do
        output[i] <- input[i] >= threshold
        i <- i + 1

let private thresholdFloat32ToFloat32AllBitsWhile threshold (input: float32[]) (output: float32[]) length =
    let mutable i = 0
    while i < length do
        output[i] <- if input[i] >= threshold then -1.0f else 0.0f
        i <- i + 1

let private thresholdFloat32ToFloat32OneWhile threshold (input: float32[]) (output: float32[]) length =
    let mutable i = 0
    while i < length do
        output[i] <- if input[i] >= threshold then 1.0f else 0.0f
        i <- i + 1

let private thresholdFloat32ToFloat32OneArrayMap threshold (input: float32[]) =
    input |> Array.map (fun value -> if value >= threshold then 1.0f else 0.0f)

[<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
let private thresholdFloat32WhileOptimized threshold (input: float32[]) (output: uint8[]) length =
    let mutable i = 0
    while i < length do
        output[i] <- if input[i] >= threshold then 255uy else 0uy
        i <- i + 1

let private thresholdFloat32NativePtr threshold (input: float32[]) (output: uint8[]) length =
    let inputHandle = GCHandle.Alloc(input, GCHandleType.Pinned)
    let outputHandle = GCHandle.Alloc(output, GCHandleType.Pinned)
    try
        let inputPtr = NativePtr.ofNativeInt<float32> (inputHandle.AddrOfPinnedObject())
        let outputPtr = NativePtr.ofNativeInt<uint8> (outputHandle.AddrOfPinnedObject())
        let mutable i = 0
        while i < length do
            NativePtr.set outputPtr i (if NativePtr.get inputPtr i >= threshold then 255uy else 0uy)
            i <- i + 1
    finally
        inputHandle.Free()
        outputHandle.Free()

let private thresholdFloat32Vector (threshold: float32) (input: float32[]) (output: uint8[]) length =
    let width = Vector<float32>.Count
    let thresholdVector = Vector<float32>(threshold)
    let mutable i = 0
    let vectorEnd = length - (length % width)
    while i < vectorEnd do
        let values = Vector<float32>(input, i)
        let mask = Vector.GreaterThanOrEqual(values, thresholdVector)
        let mutable lane = 0
        while lane < width do
            output[i + lane] <- if mask[lane] <> 0 then 255uy else 0uy
            lane <- lane + 1
        i <- i + width
    while i < length do
        output[i] <- if input[i] >= threshold then 255uy else 0uy
        i <- i + 1

let private thresholdFloat32ToFloat32AllBitsVector (threshold: float32) (input: float32[]) (output: float32[]) length =
    let width = Vector<float32>.Count
    let thresholdVector = Vector<float32>(threshold)
    let trueVector = Vector<float32>(-1.0f)
    let falseVector = Vector<float32>(0.0f)
    let mutable i = 0
    let vectorEnd = length - (length % width)
    while i < vectorEnd do
        let values = Vector<float32>(input, i)
        let mask = Vector.GreaterThanOrEqual(values, thresholdVector)
        let selected = Vector.ConditionalSelect(mask, trueVector, falseVector)
        selected.CopyTo(output, i)
        i <- i + width
    while i < length do
        output[i] <- if input[i] >= threshold then -1.0f else 0.0f
        i <- i + 1

let private thresholdFloat32ToFloat32OneVector (threshold: float32) (input: float32[]) (output: float32[]) length =
    let width = Vector<float32>.Count
    let thresholdVector = Vector<float32>(threshold)
    let trueVector = Vector<float32>(1.0f)
    let falseVector = Vector<float32>(0.0f)
    let mutable i = 0
    let vectorEnd = length - (length % width)
    while i < vectorEnd do
        let values = Vector<float32>(input, i)
        let mask = Vector.GreaterThanOrEqual(values, thresholdVector)
        let selected = Vector.ConditionalSelect(mask, trueVector, falseVector)
        selected.CopyTo(output, i)
        i <- i + width
    while i < length do
        output[i] <- if input[i] >= threshold then 1.0f else 0.0f
        i <- i + 1

let private thresholdUInt8SpanOneVector (threshold: byte) (input: Span<uint8>) (output: Span<uint8>) =
    let width = Vector<byte>.Count
    let thresholdVector = Vector<byte>(threshold)
    let oneVector = Vector<byte>(1uy)
    let mutable i = 0
    let vectorEnd = input.Length - (input.Length % width)
    while i < vectorEnd do
        let values = Vector<byte>(input.Slice(i, width))
        let mask = Vector.GreaterThanOrEqual(values, thresholdVector)
        let selected = Vector.BitwiseAnd(mask, oneVector)
        let mutable lane = 0
        while lane < width do
            output[i + lane] <- selected[lane]
            lane <- lane + 1
        i <- i + width
    while i < input.Length do
        output[i] <- if input[i] >= threshold then 1uy else 0uy
        i <- i + 1

let private thresholdUInt16SpanOneVector (threshold: uint16) (input: Span<uint16>) (output: Span<uint16>) =
    let width = Vector<uint16>.Count
    let thresholdVector = Vector<uint16>(threshold)
    let oneVector = Vector<uint16>(1us)
    let mutable i = 0
    let vectorEnd = input.Length - (input.Length % width)
    while i < vectorEnd do
        let values = Vector<uint16>(input.Slice(i, width))
        let mask = Vector.GreaterThanOrEqual(values, thresholdVector)
        Vector.BitwiseAnd(mask, oneVector).CopyTo(output.Slice(i, width))
        i <- i + width
    while i < input.Length do
        output[i] <- if input[i] >= threshold then 1us else 0us
        i <- i + 1

let private thresholdFloat32SpanOneVector (threshold: float32) (input: Span<float32>) (output: Span<float32>) =
    let width = Vector<float32>.Count
    let thresholdVector = Vector<float32>(threshold)
    let trueVector = Vector<float32>(1.0f)
    let falseVector = Vector<float32>.Zero
    let mutable i = 0
    let vectorEnd = input.Length - (input.Length % width)
    while i < vectorEnd do
        let values = Vector<float32>(input.Slice(i, width))
        let mask = Vector.GreaterThanOrEqual(values, thresholdVector)
        Vector.ConditionalSelect(mask, trueVector, falseVector).CopyTo(output.Slice(i, width))
        i <- i + width
    while i < input.Length do
        output[i] <- if input[i] >= threshold then 1.0f else 0.0f
        i <- i + 1

let private makeInTypeThresholdFilter threshold =
    let filter = new itk.simple.BinaryThresholdImageFilter()
    filter.SetLowerThreshold(threshold)
    filter.SetUpperThreshold(Double.PositiveInfinity)
    filter.SetInsideValue(1uy)
    filter.SetOutsideValue(0uy)
    filter

let private thresholdExperimentImageInType<'T when 'T: equality> threshold (input: ExperimentImage<'T>) =
    match input.TryArrayPool1D() with
    | ValueSome(inputBuffer, length) ->
        let output = ExperimentImage.rentArrayPool<'T> input.Shape
        match output.TryArrayPool1D() with
        | ValueSome(outputBuffer, outputLength) when outputLength = length ->
            let t = typeof<'T>
            if t = typeof<uint8> then
                thresholdUInt8OneVector (byte threshold) (unbox<uint8[]> inputBuffer) (unbox<uint8[]> outputBuffer) length
            elif t = typeof<uint16> then
                thresholdUInt16ToUInt16OneVector (uint16 threshold) (unbox<uint16[]> inputBuffer) (unbox<uint16[]> outputBuffer) length
            elif t = typeof<float32> then
                thresholdFloat32ToFloat32OneVector (float32 threshold) (unbox<float32[]> inputBuffer) (unbox<float32[]> outputBuffer) length
            else
                output.DecRef()
                invalidArg "T" $"Unsupported ExperimentImage threshold type {t.Name}."
            output
        | _ ->
            output.DecRef()
            invalidOp "ArrayPool-backed ExperimentImage output did not expose ArrayPool storage."
    | ValueNone ->
        use filter = makeInTypeThresholdFilter threshold
        let result = filter.Execute(input.ToSimpleITK())
        ExperimentImage.fromItkTransfer<'T> input.Shape result

let private thresholdUInt8Span threshold (input: Span<uint8>) (output: Span<uint8>) =
    for i in 0 .. input.Length - 1 do
        output[i] <- if input[i] >= threshold then 255uy else 0uy

let private thresholdUInt16Span threshold (input: Span<uint16>) (output: Span<uint8>) =
    for i in 0 .. input.Length - 1 do
        output[i] <- if input[i] >= threshold then 255uy else 0uy

let private thresholdFloat32Span threshold (input: Span<float32>) (output: Span<uint8>) =
    for i in 0 .. input.Length - 1 do
        output[i] <- if input[i] >= threshold then 255uy else 0uy

let private thresholdUInt8OneWhile threshold (input: uint8[]) (output: uint8[]) length =
    let mutable i = 0
    while i < length do
        output[i] <- if input[i] >= threshold then 1uy else 0uy
        i <- i + 1

let private thresholdUInt8OneArrayMap threshold (input: uint8[]) =
    input |> Array.map (fun value -> if value >= threshold then 1uy else 0uy)

let private thresholdUInt16OneWhile threshold (input: uint16[]) (output: uint8[]) length =
    let mutable i = 0
    while i < length do
        output[i] <- if input[i] >= threshold then 1uy else 0uy
        i <- i + 1

let private thresholdUInt16OneSpan threshold (input: Span<uint16>) (output: Span<uint8>) =
    let mutable i = 0
    while i < input.Length do
        output[i] <- if input[i] >= threshold then 1uy else 0uy
        i <- i + 1

let private thresholdFloat32OneWhile threshold (input: float32[]) (output: uint8[]) length =
    let mutable i = 0
    while i < length do
        output[i] <- if input[i] >= threshold then 1uy else 0uy
        i <- i + 1

let private thresholdFloat32OneSpan threshold (input: Span<float32>) (output: Span<uint8>) =
    let mutable i = 0
    while i < input.Length do
        output[i] <- if input[i] >= threshold then 1uy else 0uy
        i <- i + 1

let private measure action =
    GC.Collect()
    GC.WaitForPendingFinalizers()
    GC.Collect()
    let allocatedBefore = GC.GetTotalAllocatedBytes(true)
    let sw = Stopwatch.StartNew()
    action()
    sw.Stop()
    let allocatedAfter = GC.GetTotalAllocatedBytes(true)
    sw.Elapsed.TotalSeconds, allocatedAfter - allocatedBefore

let private makeFilter threshold =
    let filter = new itk.simple.BinaryThresholdImageFilter()
    filter.SetLowerThreshold(threshold)
    filter.SetUpperThreshold(Double.PositiveInfinity)
    filter.SetInsideValue(255uy)
    filter.SetOutsideValue(0uy)
    filter

type private ThresholdRuntime(threshold: float) =
    let filter = makeFilter threshold

    member _.Execute(image: itk.simple.Image) =
        filter.Execute(image)

    interface IDisposable with
        member _.Dispose() =
            filter.Dispose()

let private measureItk (image: itk.simple.Image) threshold =
    use filter = makeFilter threshold
    use warmup = filter.Execute(image)
    measure (fun () ->
        use result = filter.Execute(image)
        if result.GetSize().Count = 0 then
            failwith "unreachable")

let private measureFilterLifecycleCreateEachTime iterations (image: itk.simple.Image) threshold =
    use warmupFilter = makeFilter threshold
    use warmup = warmupFilter.Execute(image)
    measure (fun () ->
        let mutable checksum = 0UL
        for _ in 1 .. iterations do
            use filter = makeFilter threshold
            use result = filter.Execute(image)
            checksum <- checksum + uint64 (result.GetSize().Count)
        if checksum = 0UL then
            failwith "unreachable")

let private measureFilterLifecycleRuntime iterations (image: itk.simple.Image) threshold =
    use runtime = new ThresholdRuntime(threshold)
    use warmup = runtime.Execute(image)
    measure (fun () ->
        let mutable checksum = 0UL
        for _ in 1 .. iterations do
            use result = runtime.Execute(image)
            checksum <- checksum + uint64 (result.GetSize().Count)
        if checksum = 0UL then
            failwith "unreachable")

let private measureArrayPoolItkRoundtrip<'T> shape fill threshold =
    let length = requireIntLength shape
    let input = ArrayPool<'T>.Shared.Rent(length)
    let mutable inputReturned = false
    try
        fill input length
        GC.Collect()
        GC.WaitForPendingFinalizers()
        GC.Collect()
        let allocatedBefore = GC.GetTotalAllocatedBytes(true)
        let sw = Stopwatch.StartNew()
        try
            use image = importedImage shape input
            ArrayPool<'T>.Shared.Return(input, RuntimeHelpers.IsReferenceOrContainsReferences<'T>())
            inputReturned <- true
            use filter = makeFilter threshold
            use result = filter.Execute(image)
            copyUInt8ImageToPooledArray length result
            sw.Stop()
            let allocatedAfter = GC.GetTotalAllocatedBytes(true)
            sw.Elapsed.TotalSeconds, allocatedAfter - allocatedBefore
        with _ ->
            sw.Stop()
            reraise()
    finally
        if not inputReturned then
            ArrayPool<'T>.Shared.Return(input, RuntimeHelpers.IsReferenceOrContainsReferences<'T>())

let private row backend variant pixel shape (threshold: float) repeat exitCode seconds allocated (error: string) =
    { Backend = backend
      Variant = variant
      PixelType = pixel
      Shape = shape
      Threshold = threshold.ToString("R", invariant)
      Repeat = repeat
      ExitCode = exitCode
      Seconds = seconds
      ManagedAllocatedBytes = allocated
      Error = error.Replace("\n", " ") }

let private successful backend variant pixel shape (threshold: float) repeat (seconds: float) allocated =
    row backend variant pixel shape threshold repeat 0 (seconds.ToString("F9", invariant)) (string allocated) ""

let private failed backend variant pixel shape (threshold: float) repeat (ex: exn) =
    row backend variant pixel shape threshold repeat 1 "" "" (ex.GetType().Name + ": " + ex.Message)

let private measureArrayActions pixelName shapeName reportedThreshold repeat actions =
    actions
    |> List.collect (fun (variant, action) ->
        try
            action()
            let seconds, allocated = measure action
            [ successful "arraypool" variant pixelName shapeName reportedThreshold repeat seconds allocated ]
        with ex ->
            [ failed "arraypool" variant pixelName shapeName reportedThreshold repeat ex ])

let private measureActions backend pixelName shapeName reportedThreshold repeat actions =
    actions
    |> List.collect (fun (variant, action) ->
        try
            action()
            let seconds, allocated = measure action
            [ successful backend variant pixelName shapeName reportedThreshold repeat seconds allocated ]
        with ex ->
            [ failed backend variant pixelName shapeName reportedThreshold repeat ex ])

let private withPooledBuffers<'T> shape name fill body =
    let length = requireIntLength shape
    let input = ArrayPool<'T>.Shared.Rent(length)
    let output = ArrayPool<uint8>.Shared.Rent(length)
    try
        fill input length
        body length input output
    finally
        ArrayPool<'T>.Shared.Return(input, RuntimeHelpers.IsReferenceOrContainsReferences<'T>())
        ArrayPool<uint8>.Shared.Return(output)

let private withPooledBuffersAndBoolMask<'T> shape fill body =
    let length = requireIntLength shape
    let input = ArrayPool<'T>.Shared.Rent(length)
    let boolOutput = ArrayPool<bool>.Shared.Rent(length)
    try
        fill input length
        body length input boolOutput
    finally
        ArrayPool<'T>.Shared.Return(input, RuntimeHelpers.IsReferenceOrContainsReferences<'T>())
        ArrayPool<bool>.Shared.Return(boolOutput)

let private withPooledUInt16Mask shape body =
    let length = requireIntLength shape
    let input = ArrayPool<uint16>.Shared.Rent(length)
    let output = ArrayPool<uint16>.Shared.Rent(length)
    try
        fillUInt16 input length
        body length input output
    finally
        ArrayPool<uint16>.Shared.Return(input)
        ArrayPool<uint16>.Shared.Return(output)

let private withPooledFloat32Mask shape body =
    let length = requireIntLength shape
    let input = ArrayPool<float32>.Shared.Rent(length)
    let output = ArrayPool<float32>.Shared.Rent(length)
    try
        fillFloat32 input length
        body length input output
    finally
        ArrayPool<float32>.Shared.Return(input)
        ArrayPool<float32>.Shared.Return(output)

let private runCase pixelType shape threshold repeat =
    let shapeName = shapeText shape
    let addItkRow pixelName input rows =
        let itkRow =
            try
                use image = importedImage shape input
                let seconds, allocated = measureItk image threshold
                successful "simpleitk" "BinaryThresholdImageFilter" pixelName shapeName threshold repeat seconds allocated
            with ex ->
                failed "simpleitk" "BinaryThresholdImageFilter" pixelName shapeName threshold repeat ex
        itkRow :: rows

    match pixelType with
    | UInt8 ->
        withPooledBuffers<uint8> shape "UInt8" fillUInt8 (fun length input output ->
            let typedThreshold = byte threshold
            let rows =
                [ "arraypool-loop", fun () -> thresholdUInt8Array typedThreshold input output length
                  "arraypool-while", fun () -> thresholdUInt8While typedThreshold input output length
                  "arraypool-while-aggressive", fun () -> thresholdUInt8WhileOptimized typedThreshold input output length
                  "arraypool-nativeptr", fun () -> thresholdUInt8NativePtr typedThreshold input output length
                  "arraypool-vector", fun () -> thresholdUInt8Vector typedThreshold input output length
                  "arraypool-span", fun () -> thresholdUInt8Span typedThreshold (input.AsSpan(0, length)) (output.AsSpan(0, length)) ]
                |> measureArrayActions "UInt8" shapeName threshold repeat
                |> addItkRow "UInt8" input
            let roundtripRow =
                try
                    let seconds, allocated = measureArrayPoolItkRoundtrip<uint8> shape fillUInt8 threshold
                    successful "arraypool-itk" "import-return-threshold-export" "UInt8" shapeName threshold repeat seconds allocated
                with ex ->
                    failed "arraypool-itk" "import-return-threshold-export" "UInt8" shapeName threshold repeat ex
            let boolRows =
                withPooledBuffersAndBoolMask<uint8> shape fillUInt8 (fun boolLength boolInput boolOutput ->
                    [ "arraypool-bool-while", fun () -> thresholdUInt8BoolWhile typedThreshold boolInput boolOutput boolLength ]
                    |> measureArrayActions "UInt8" shapeName threshold repeat)
            roundtripRow :: (rows @ boolRows))
    | UInt16 ->
        withPooledBuffers<uint16> shape "UInt16" fillUInt16 (fun length input output ->
            let typedThreshold = uint16 threshold
            let rows =
                [ "arraypool-loop", fun () -> thresholdUInt16Array typedThreshold input output length
                  "arraypool-while", fun () -> thresholdUInt16While typedThreshold input output length
                  "arraypool-while-aggressive", fun () -> thresholdUInt16WhileOptimized typedThreshold input output length
                  "arraypool-nativeptr", fun () -> thresholdUInt16NativePtr typedThreshold input output length
                  "arraypool-span", fun () -> thresholdUInt16Span typedThreshold (input.AsSpan(0, length)) (output.AsSpan(0, length)) ]
                |> measureArrayActions "UInt16" shapeName threshold repeat
                |> addItkRow "UInt16" input
            let roundtripRow =
                try
                    let seconds, allocated = measureArrayPoolItkRoundtrip<uint16> shape fillUInt16 threshold
                    successful "arraypool-itk" "import-return-threshold-export" "UInt16" shapeName threshold repeat seconds allocated
                with ex ->
                    failed "arraypool-itk" "import-return-threshold-export" "UInt16" shapeName threshold repeat ex
            let boolRows =
                withPooledBuffersAndBoolMask<uint16> shape fillUInt16 (fun boolLength boolInput boolOutput ->
                    [ "arraypool-bool-while", fun () -> thresholdUInt16BoolWhile typedThreshold boolInput boolOutput boolLength ]
                    |> measureArrayActions "UInt16" shapeName threshold repeat)
            let uint16MaskRows =
                withPooledUInt16Mask shape (fun maskLength maskInput maskOutput ->
                    [ "arraypool-uint16mask-max-while", fun () -> thresholdUInt16ToUInt16MaxWhile typedThreshold maskInput maskOutput maskLength
                      "arraypool-uint16mask-max-vector", fun () -> thresholdUInt16ToUInt16MaxVector typedThreshold maskInput maskOutput maskLength
                      "arraypool-uint16mask-one-while", fun () -> thresholdUInt16ToUInt16OneWhile typedThreshold maskInput maskOutput maskLength
                      "arraypool-uint16mask-one-vector", fun () -> thresholdUInt16ToUInt16OneVector typedThreshold maskInput maskOutput maskLength ]
                    |> measureArrayActions "UInt16" shapeName threshold repeat)
            roundtripRow :: (rows @ boolRows @ uint16MaskRows))
    | Float32 ->
        withPooledBuffers<float32> shape "Float32" fillFloat32 (fun length input output ->
            let typedThreshold = float32 threshold
            let rows =
                [ "arraypool-loop", fun () -> thresholdFloat32Array typedThreshold input output length
                  "arraypool-while", fun () -> thresholdFloat32While typedThreshold input output length
                  "arraypool-while-aggressive", fun () -> thresholdFloat32WhileOptimized typedThreshold input output length
                  "arraypool-nativeptr", fun () -> thresholdFloat32NativePtr typedThreshold input output length
                  "arraypool-vector", fun () -> thresholdFloat32Vector typedThreshold input output length
                  "arraypool-span", fun () -> thresholdFloat32Span typedThreshold (input.AsSpan(0, length)) (output.AsSpan(0, length)) ]
                |> measureArrayActions "Float32" shapeName threshold repeat
                |> addItkRow "Float32" input
            let roundtripRow =
                try
                    let seconds, allocated = measureArrayPoolItkRoundtrip<float32> shape fillFloat32 threshold
                    successful "arraypool-itk" "import-return-threshold-export" "Float32" shapeName threshold repeat seconds allocated
                with ex ->
                    failed "arraypool-itk" "import-return-threshold-export" "Float32" shapeName threshold repeat ex
            let boolRows =
                withPooledBuffersAndBoolMask<float32> shape fillFloat32 (fun boolLength boolInput boolOutput ->
                    [ "arraypool-bool-while", fun () -> thresholdFloat32BoolWhile typedThreshold boolInput boolOutput boolLength ]
                    |> measureArrayActions "Float32" shapeName threshold repeat)
            let float32MaskRows =
                withPooledFloat32Mask shape (fun maskLength maskInput maskOutput ->
                    [ "arraypool-float32mask-allbits-while", fun () -> thresholdFloat32ToFloat32AllBitsWhile typedThreshold maskInput maskOutput maskLength
                      "arraypool-float32mask-allbits-vector", fun () -> thresholdFloat32ToFloat32AllBitsVector typedThreshold maskInput maskOutput maskLength
                      "arraypool-float32mask-one-while", fun () -> thresholdFloat32ToFloat32OneWhile typedThreshold maskInput maskOutput maskLength
                      "arraypool-float32mask-one-vector", fun () -> thresholdFloat32ToFloat32OneVector typedThreshold maskInput maskOutput maskLength ]
                    |> measureArrayActions "Float32" shapeName threshold repeat)
            roundtripRow :: (rows @ boolRows @ float32MaskRows))

let private runFilterLifecycleFor<'T> pixelName fill shape threshold repeat iterations =
    let shapeName = shapeText shape
    let length = requireIntLength shape
    let input = ArrayPool<'T>.Shared.Rent(length)
    try
        fill input length
        use image = importedImage<'T> shape input
        [ try
              let seconds, allocated = measureFilterLifecycleCreateEachTime iterations image threshold
              successful "simpleitk" "create-dispose-filter-per-execute" pixelName shapeName threshold repeat seconds allocated
          with ex ->
              failed "simpleitk" "create-dispose-filter-per-execute" pixelName shapeName threshold repeat ex
          try
              let seconds, allocated = measureFilterLifecycleRuntime iterations image threshold
              successful "simpleitk" "reuse-threshold-runtime-filter" pixelName shapeName threshold repeat seconds allocated
          with ex ->
              failed "simpleitk" "reuse-threshold-runtime-filter" pixelName shapeName threshold repeat ex ]
    finally
        ArrayPool<'T>.Shared.Return(input, RuntimeHelpers.IsReferenceOrContainsReferences<'T>())

let private runFilterLifecycleCase pixelType shape threshold repeat iterations =
    match pixelType with
    | UInt8 -> runFilterLifecycleFor<uint8> "UInt8" fillUInt8 shape threshold repeat iterations
    | UInt16 -> runFilterLifecycleFor<uint16> "UInt16" fillUInt16 shape threshold repeat iterations
    | Float32 -> runFilterLifecycleFor<float32> "Float32" fillFloat32 shape threshold repeat iterations

let private typedArrayToBytes<'T> (length: int) (input: 'T[]) =
    let byteCount = length * Marshal.SizeOf<'T>()
    let bytes = ArrayPool<byte>.Shared.Rent(byteCount)
    Buffer.BlockCopy(input, 0, bytes, 0, byteCount)
    bytes, byteCount

let private runChunkStorageUInt8 shape threshold repeat =
    let shapeName = shapeText shape
    let length = requireIntLength shape
    let input = ArrayPool<uint8>.Shared.Rent(length)
    let output = ArrayPool<uint8>.Shared.Rent(length)
    try
        fillUInt8 input length
        let typedThreshold = byte threshold
        [ "typed-array-direct-while-0-1", fun () -> thresholdUInt8OneWhile typedThreshold input output length
          "typed-array-direct-vector-0-1", fun () -> thresholdUInt8OneVector typedThreshold input output length
          "byte-backed-span-vector-0-1", fun () -> thresholdUInt8OneVector typedThreshold input output length ]
        |> measureArrayActions "UInt8" shapeName threshold repeat
    finally
        ArrayPool<uint8>.Shared.Return(input)
        ArrayPool<uint8>.Shared.Return(output)

let private runChunkStorageUInt16 shape threshold repeat =
    let shapeName = shapeText shape
    let length = requireIntLength shape
    let input = ArrayPool<uint16>.Shared.Rent(length)
    let output = ArrayPool<uint8>.Shared.Rent(length)
    let mutable inputBytesOpt: byte[] option = None
    try
        fillUInt16 input length
        let inputBytes, byteCount = typedArrayToBytes length input
        inputBytesOpt <- Some inputBytes
        let typedThreshold = uint16 threshold
        [ "typed-array-direct-while-0-1", fun () -> thresholdUInt16OneWhile typedThreshold input output length
          "byte-backed-span-while-0-1", fun () ->
              let inputSpan = MemoryMarshal.Cast<byte, uint16>(inputBytes.AsSpan(0, byteCount))
              thresholdUInt16OneSpan typedThreshold inputSpan (output.AsSpan(0, length)) ]
        |> measureArrayActions "UInt16" shapeName threshold repeat
    finally
        ArrayPool<uint16>.Shared.Return(input)
        ArrayPool<uint8>.Shared.Return(output)
        inputBytesOpt |> Option.iter ArrayPool<byte>.Shared.Return

let private runChunkStorageFloat32 shape threshold repeat =
    let shapeName = shapeText shape
    let length = requireIntLength shape
    let input = ArrayPool<float32>.Shared.Rent(length)
    let output = ArrayPool<uint8>.Shared.Rent(length)
    let mutable inputBytesOpt: byte[] option = None
    try
        fillFloat32 input length
        let inputBytes, byteCount = typedArrayToBytes length input
        inputBytesOpt <- Some inputBytes
        let typedThreshold = float32 threshold
        [ "typed-array-direct-while-0-1", fun () -> thresholdFloat32OneWhile typedThreshold input output length
          "byte-backed-span-while-0-1", fun () ->
              let inputSpan = MemoryMarshal.Cast<byte, float32>(inputBytes.AsSpan(0, byteCount))
              thresholdFloat32OneSpan typedThreshold inputSpan (output.AsSpan(0, length)) ]
        |> measureArrayActions "Float32" shapeName threshold repeat
    finally
        ArrayPool<float32>.Shared.Return(input)
        ArrayPool<uint8>.Shared.Return(output)
        inputBytesOpt |> Option.iter ArrayPool<byte>.Shared.Return

let private runChunkStorageCase pixelType shape threshold repeat =
    match pixelType with
    | UInt8 -> runChunkStorageUInt8 shape threshold repeat
    | UInt16 -> runChunkStorageUInt16 shape threshold repeat
    | Float32 -> runChunkStorageFloat32 shape threshold repeat

let private runImageClassArrayPoolCase<'T when 'T: equality> pixelName shape threshold repeat fill directAction =
    let shapeName = shapeText shape
    let image = ExperimentImage.rentArrayPool<'T> shape
    try
        match image.TryArrayPool1D() with
        | ValueSome(buffer, length) -> fill buffer length
        | ValueNone -> invalidOp "Newly rented ExperimentImage did not expose ArrayPool storage."

        let imageAction () =
            use output = thresholdExperimentImageInType threshold image
            output.Length |> ignore

        let rows =
            [ "arraypool-image-intype-vector-dispatch", imageAction ]
            |> measureActions "experiment-image" pixelName shapeName threshold repeat

        directAction rows
    finally
        image.DecRef()

let private runImageClassItkCase<'T when 'T: equality> pixelName shape threshold repeat fill =
    let shapeName = shapeText shape
    let length = requireIntLength shape
    let input = ArrayPool<'T>.Shared.Rent(length)
    try
        fill input length
        let image = ExperimentImage.fromItkTransfer<'T> shape (importedImage<'T> shape input)
        try
            [ "simpleitk-image-intype-filter-dispatch", fun () ->
                  use output = thresholdExperimentImageInType threshold image
                  output.Length |> ignore ]
            |> measureActions "experiment-image" pixelName shapeName threshold repeat
        finally
            image.DecRef()
    finally
        ArrayPool<'T>.Shared.Return(input, RuntimeHelpers.IsReferenceOrContainsReferences<'T>())

let private runImageClassUInt8 shape threshold repeat =
    let shapeName = shapeText shape
    let typedThreshold = byte threshold
    runImageClassArrayPoolCase<uint8>
        "UInt8"
        shape
        threshold
        repeat
        fillUInt8
        (fun imageRows ->
            withPooledBuffers<uint8> shape "UInt8" fillUInt8 (fun length input output ->
                let directRows =
                    [ "direct-intype-vector", fun () -> thresholdUInt8OneVector typedThreshold input output length ]
                    |> measureActions "arraypool" "UInt8" shapeName threshold repeat
                directRows @ imageRows))
    @ runImageClassItkCase<uint8> "UInt8" shape threshold repeat fillUInt8

let private runImageClassUInt16 shape threshold repeat =
    let shapeName = shapeText shape
    let typedThreshold = uint16 threshold
    runImageClassArrayPoolCase<uint16>
        "UInt16"
        shape
        threshold
        repeat
        fillUInt16
        (fun imageRows ->
            withPooledUInt16Mask shape (fun length input output ->
                let directRows =
                    [ "direct-intype-vector", fun () -> thresholdUInt16ToUInt16OneVector typedThreshold input output length ]
                    |> measureActions "arraypool" "UInt16" shapeName threshold repeat
                directRows @ imageRows))
    @ runImageClassItkCase<uint16> "UInt16" shape threshold repeat fillUInt16

let private runImageClassFloat32 shape threshold repeat =
    let shapeName = shapeText shape
    let typedThreshold = float32 threshold
    runImageClassArrayPoolCase<float32>
        "Float32"
        shape
        threshold
        repeat
        fillFloat32
        (fun imageRows ->
            withPooledFloat32Mask shape (fun length input output ->
                let directRows =
                    [ "direct-intype-vector", fun () -> thresholdFloat32ToFloat32OneVector typedThreshold input output length ]
                    |> measureActions "arraypool" "Float32" shapeName threshold repeat
                directRows @ imageRows))
    @ runImageClassItkCase<float32> "Float32" shape threshold repeat fillFloat32

let private runImageClassCase pixelType shape threshold repeat =
    match pixelType with
    | UInt8 -> runImageClassUInt8 shape threshold repeat
    | UInt16 -> runImageClassUInt16 shape threshold repeat
    | Float32 -> runImageClassFloat32 shape threshold repeat

let private sliceShape shape =
    { Width = shape.Width
      Height = shape.Height
      Depth = 1 }

let private measureImageClassSliceWise<'T when 'T: equality> pixelName shape threshold repeat fill =
    let fullShapeName = shapeText shape
    let oneSlice = sliceShape shape
    let sliceLength = requireIntLength oneSlice
    let input = ExperimentImage.rentArrayPool<'T> oneSlice
    try
        match input.TryArrayPool1D() with
        | ValueSome(buffer, length) when length = sliceLength -> fill buffer length
        | ValueSome(_, length) -> invalidOp $"Slice image length mismatch: got {length}, expected {sliceLength}."
        | ValueNone -> invalidOp "Newly rented slice ExperimentImage did not expose ArrayPool storage."

        [ "arraypool-image-slicewise-intype-vector-dispatch", fun () ->
              let mutable z = 0
              while z < shape.Depth do
                  use output = thresholdExperimentImageInType threshold input
                  output.Length |> ignore
                  z <- z + 1 ]
        |> measureActions "experiment-image" pixelName fullShapeName threshold repeat
    finally
        input.DecRef()

let private measureDirectSliceWiseUInt8 shape threshold repeat =
    let fullShapeName = shapeText shape
    let oneSlice = sliceShape shape
    let sliceLength = requireIntLength oneSlice
    let input = ArrayPool<uint8>.Shared.Rent(sliceLength)
    let output = ArrayPool<uint8>.Shared.Rent(sliceLength)
    try
        fillUInt8 input sliceLength
        let typedThreshold = byte threshold
        [ "direct-slicewise-intype-vector", fun () ->
              let mutable z = 0
              while z < shape.Depth do
                  thresholdUInt8OneVector typedThreshold input output sliceLength
                  z <- z + 1 ]
        |> measureActions "arraypool" "UInt8" fullShapeName threshold repeat
    finally
        ArrayPool<uint8>.Shared.Return(input)
        ArrayPool<uint8>.Shared.Return(output)

let private measureDirectSliceWiseUInt16 shape threshold repeat =
    let fullShapeName = shapeText shape
    let oneSlice = sliceShape shape
    let sliceLength = requireIntLength oneSlice
    let input = ArrayPool<uint16>.Shared.Rent(sliceLength)
    let output = ArrayPool<uint16>.Shared.Rent(sliceLength)
    try
        fillUInt16 input sliceLength
        let typedThreshold = uint16 threshold
        [ "direct-slicewise-intype-vector", fun () ->
              let mutable z = 0
              while z < shape.Depth do
                  thresholdUInt16ToUInt16OneVector typedThreshold input output sliceLength
                  z <- z + 1 ]
        |> measureActions "arraypool" "UInt16" fullShapeName threshold repeat
    finally
        ArrayPool<uint16>.Shared.Return(input)
        ArrayPool<uint16>.Shared.Return(output)

let private measureDirectSliceWiseFloat32 shape threshold repeat =
    let fullShapeName = shapeText shape
    let oneSlice = sliceShape shape
    let sliceLength = requireIntLength oneSlice
    let input = ArrayPool<float32>.Shared.Rent(sliceLength)
    let output = ArrayPool<float32>.Shared.Rent(sliceLength)
    try
        fillFloat32 input sliceLength
        let typedThreshold = float32 threshold
        [ "direct-slicewise-intype-vector", fun () ->
              let mutable z = 0
              while z < shape.Depth do
                  thresholdFloat32ToFloat32OneVector typedThreshold input output sliceLength
                  z <- z + 1 ]
        |> measureActions "arraypool" "Float32" fullShapeName threshold repeat
    finally
        ArrayPool<float32>.Shared.Return(input)
        ArrayPool<float32>.Shared.Return(output)

let private runImageClassSliceWiseCase pixelType shape threshold repeat =
    match pixelType with
    | UInt8 ->
        measureDirectSliceWiseUInt8 shape threshold repeat
        @ measureImageClassSliceWise<uint8> "UInt8" shape threshold repeat fillUInt8
    | UInt16 ->
        measureDirectSliceWiseUInt16 shape threshold repeat
        @ measureImageClassSliceWise<uint16> "UInt16" shape threshold repeat fillUInt16
    | Float32 ->
        measureDirectSliceWiseFloat32 shape threshold repeat
        @ measureImageClassSliceWise<float32> "Float32" shape threshold repeat fillFloat32

let private chunkGeometryForSlice shape =
    uint64 shape.Width, uint64 shape.Height, 1UL

let private measureChunkSliceWiseUInt8 shape threshold repeat =
    let fullShapeName = shapeText shape
    let size = chunkGeometryForSlice shape
    let input = StackCore.Chunk.create<uint8> size
    try
        let inputSpan = StackCore.Chunk.span<uint8> input
        for i in 0 .. inputSpan.Length - 1 do
            inputSpan[i] <- byte (i &&& 0xff)
        let typedThreshold = byte threshold
        [ "chunk-slicewise-intype-vector", fun () ->
              let inputSpan = StackCore.Chunk.span<uint8> input
              let mutable z = 0
              while z < shape.Depth do
                  let output = StackCore.Chunk.create<uint8> size
                  try
                      thresholdUInt8OneVector typedThreshold input.Bytes output.Bytes inputSpan.Length
                  finally
                      StackCore.Chunk.decRef output
                  z <- z + 1 ]
        |> measureActions "chunk" "UInt8" fullShapeName threshold repeat
    finally
        StackCore.Chunk.decRef input

let private measureChunkSliceWiseUInt16 shape threshold repeat =
    let fullShapeName = shapeText shape
    let size = chunkGeometryForSlice shape
    let input = StackCore.Chunk.create<uint16> size
    try
        let inputSpan = StackCore.Chunk.span<uint16> input
        for i in 0 .. inputSpan.Length - 1 do
            inputSpan[i] <- uint16 (i &&& 0xffff)
        let typedThreshold = uint16 threshold
        [ "chunk-slicewise-intype-vector", fun () ->
              let inputSpan = StackCore.Chunk.span<uint16> input
              let mutable z = 0
              while z < shape.Depth do
                  let output = StackCore.Chunk.create<uint16> size
                  try
                      thresholdUInt16SpanOneVector typedThreshold inputSpan (StackCore.Chunk.span<uint16> output)
                  finally
                      StackCore.Chunk.decRef output
                  z <- z + 1 ]
        |> measureActions "chunk" "UInt16" fullShapeName threshold repeat
    finally
        StackCore.Chunk.decRef input

let private measureChunkSliceWiseFloat32 shape threshold repeat =
    let fullShapeName = shapeText shape
    let size = chunkGeometryForSlice shape
    let input = StackCore.Chunk.create<float32> size
    try
        let inputSpan = StackCore.Chunk.span<float32> input
        for i in 0 .. inputSpan.Length - 1 do
            inputSpan[i] <- float32 (i &&& 0xff)
        let typedThreshold = float32 threshold
        [ "chunk-slicewise-intype-vector", fun () ->
              let inputSpan = StackCore.Chunk.span<float32> input
              let mutable z = 0
              while z < shape.Depth do
                  let output = StackCore.Chunk.create<float32> size
                  try
                      thresholdFloat32SpanOneVector typedThreshold inputSpan (StackCore.Chunk.span<float32> output)
                  finally
                      StackCore.Chunk.decRef output
                  z <- z + 1 ]
        |> measureActions "chunk" "Float32" fullShapeName threshold repeat
    finally
        StackCore.Chunk.decRef input

let private runChunkSliceWiseCase pixelType shape threshold repeat =
    match pixelType with
    | UInt8 ->
        measureDirectSliceWiseUInt8 shape threshold repeat
        @ measureChunkSliceWiseUInt8 shape threshold repeat
    | UInt16 ->
        measureDirectSliceWiseUInt16 shape threshold repeat
        @ measureChunkSliceWiseUInt16 shape threshold repeat
    | Float32 ->
        measureDirectSliceWiseFloat32 shape threshold repeat
        @ measureChunkSliceWiseFloat32 shape threshold repeat

let private measureMapStyleUInt8 shape threshold repeat =
    let fullShapeName = shapeText shape
    let oneSlice = sliceShape shape
    let sliceLength = requireIntLength oneSlice
    let input = Array.zeroCreate<uint8> sliceLength
    let output = Array.zeroCreate<uint8> sliceLength
    fillUInt8 input sliceLength
    let typedThreshold = byte threshold
    [ "while-exact-array-slicewise", fun () ->
          let mutable z = 0
          while z < shape.Depth do
              thresholdUInt8OneWhile typedThreshold input output sliceLength
              z <- z + 1
      "array-map-exact-array-slicewise", fun () ->
          let mutable z = 0
          let mutable checksum = 0
          while z < shape.Depth do
              let mapped = thresholdUInt8OneArrayMap typedThreshold input
              checksum <- checksum + int mapped[0]
              z <- z + 1
          checksum |> ignore ]
    |> measureActions "map-style" "UInt8" fullShapeName threshold repeat

let private measureMapStyleUInt16 shape threshold repeat =
    let fullShapeName = shapeText shape
    let oneSlice = sliceShape shape
    let sliceLength = requireIntLength oneSlice
    let input = Array.zeroCreate<uint16> sliceLength
    let output = Array.zeroCreate<uint16> sliceLength
    fillUInt16 input sliceLength
    let typedThreshold = uint16 threshold
    [ "while-exact-array-slicewise", fun () ->
          let mutable z = 0
          while z < shape.Depth do
              thresholdUInt16ToUInt16OneWhile typedThreshold input output sliceLength
              z <- z + 1
      "array-map-exact-array-slicewise", fun () ->
          let mutable z = 0
          let mutable checksum = 0
          while z < shape.Depth do
              let mapped = thresholdUInt16ToUInt16OneArrayMap typedThreshold input
              checksum <- checksum + int mapped[0]
              z <- z + 1
          checksum |> ignore ]
    |> measureActions "map-style" "UInt16" fullShapeName threshold repeat

let private measureMapStyleFloat32 shape threshold repeat =
    let fullShapeName = shapeText shape
    let oneSlice = sliceShape shape
    let sliceLength = requireIntLength oneSlice
    let input = Array.zeroCreate<float32> sliceLength
    let output = Array.zeroCreate<float32> sliceLength
    fillFloat32 input sliceLength
    let typedThreshold = float32 threshold
    [ "while-exact-array-slicewise", fun () ->
          let mutable z = 0
          while z < shape.Depth do
              thresholdFloat32ToFloat32OneWhile typedThreshold input output sliceLength
              z <- z + 1
      "array-map-exact-array-slicewise", fun () ->
          let mutable z = 0
          let mutable checksum = 0.0f
          while z < shape.Depth do
              let mapped = thresholdFloat32ToFloat32OneArrayMap typedThreshold input
              checksum <- checksum + mapped[0]
              z <- z + 1
          checksum |> ignore ]
    |> measureActions "map-style" "Float32" fullShapeName threshold repeat

let private runMapStyleSliceWiseCase pixelType shape threshold repeat =
    match pixelType with
    | UInt8 -> measureMapStyleUInt8 shape threshold repeat
    | UInt16 -> measureMapStyleUInt16 shape threshold repeat
    | Float32 -> measureMapStyleFloat32 shape threshold repeat

let private measureChunkMapUInt8 shape threshold repeat =
    let fullShapeName = shapeText shape
    let size = chunkGeometryForSlice shape
    let input = StackCore.Chunk.create<uint8> size
    let reusableOutput = StackCore.Chunk.create<uint8> size
    try
        let inputSpan = StackCore.Chunk.span<uint8> input
        for i in 0 .. inputSpan.Length - 1 do
            inputSpan[i] <- byte (i &&& 0xff)
        let typedThreshold = byte threshold
        [ "chunk-mapInto-slicewise", fun () ->
              let mutable z = 0
              while z < shape.Depth do
                  StackCore.Chunk.mapInto (fun value -> if value >= typedThreshold then 1uy else 0uy) input reusableOutput
                  z <- z + 1
          "chunk-map-slicewise", fun () ->
              let mutable z = 0
              while z < shape.Depth do
                  let output = StackCore.Chunk.map (fun value -> if value >= typedThreshold then 1uy else 0uy) input
                  try
                      output.ByteLength |> ignore
                  finally
                      StackCore.Chunk.decRef output
                  z <- z + 1 ]
        |> measureActions "chunk-map" "UInt8" fullShapeName threshold repeat
    finally
        StackCore.Chunk.decRef reusableOutput
        StackCore.Chunk.decRef input

let private measureChunkMapUInt16 shape threshold repeat =
    let fullShapeName = shapeText shape
    let size = chunkGeometryForSlice shape
    let input = StackCore.Chunk.create<uint16> size
    let reusableOutput = StackCore.Chunk.create<uint16> size
    try
        let inputSpan = StackCore.Chunk.span<uint16> input
        for i in 0 .. inputSpan.Length - 1 do
            inputSpan[i] <- uint16 (i &&& 0xffff)
        let typedThreshold = uint16 threshold
        [ "chunk-mapInto-slicewise", fun () ->
              let mutable z = 0
              while z < shape.Depth do
                  StackCore.Chunk.mapInto (fun value -> if value >= typedThreshold then 1us else 0us) input reusableOutput
                  z <- z + 1
          "chunk-map-slicewise", fun () ->
              let mutable z = 0
              while z < shape.Depth do
                  let output = StackCore.Chunk.map (fun value -> if value >= typedThreshold then 1us else 0us) input
                  try
                      output.ByteLength |> ignore
                  finally
                      StackCore.Chunk.decRef output
                  z <- z + 1 ]
        |> measureActions "chunk-map" "UInt16" fullShapeName threshold repeat
    finally
        StackCore.Chunk.decRef reusableOutput
        StackCore.Chunk.decRef input

let private measureChunkMapFloat32 shape threshold repeat =
    let fullShapeName = shapeText shape
    let size = chunkGeometryForSlice shape
    let input = StackCore.Chunk.create<float32> size
    let reusableOutput = StackCore.Chunk.create<float32> size
    try
        let inputSpan = StackCore.Chunk.span<float32> input
        for i in 0 .. inputSpan.Length - 1 do
            inputSpan[i] <- float32 (i &&& 0xff)
        let typedThreshold = float32 threshold
        [ "chunk-mapInto-slicewise", fun () ->
              let mutable z = 0
              while z < shape.Depth do
                  StackCore.Chunk.mapInto (fun value -> if value >= typedThreshold then 1.0f else 0.0f) input reusableOutput
                  z <- z + 1
          "chunk-map-slicewise", fun () ->
              let mutable z = 0
              while z < shape.Depth do
                  let output = StackCore.Chunk.map (fun value -> if value >= typedThreshold then 1.0f else 0.0f) input
                  try
                      output.ByteLength |> ignore
                  finally
                      StackCore.Chunk.decRef output
                  z <- z + 1 ]
        |> measureActions "chunk-map" "Float32" fullShapeName threshold repeat
    finally
        StackCore.Chunk.decRef reusableOutput
        StackCore.Chunk.decRef input

let private runChunkMapCase pixelType shape threshold repeat =
    match pixelType with
    | UInt8 -> measureChunkMapUInt8 shape threshold repeat
    | UInt16 -> measureChunkMapUInt16 shape threshold repeat
    | Float32 -> measureChunkMapFloat32 shape threshold repeat

let private imageSizeList shape =
    [ uint shape.Width; uint shape.Height; uint shape.Depth ]

let private chunkSize shape =
    uint64 shape.Width, uint64 shape.Height, uint64 shape.Depth

let private measureHistogramUInt8 includeImageFunctions shape threshold repeat =
    let shapeName = shapeText shape
    let length = requireIntLength shape
    let pixels = Array.zeroCreate<uint8> length
    fillUInt8 pixels length
    let image = Image.Image<uint8>.ofFlatArray(imageSizeList shape, pixels, "histogram-benchmark")
    let chunk = StackCore.Chunk.create<uint8> (chunkSize shape)
    let leftEdges16 = [ for edge in 0 .. 16 .. 240 -> float edge ]
    let leftEdges256 = [ for edge in 0 .. 255 -> float edge ]
    try
        let values = StackCore.Chunk.span<uint8> chunk
        for i in 0 .. values.Length - 1 do
            values[i] <- pixels[i]
        let baselineActions =
            if includeImageFunctions then
                [ "imagefunctions-histogram", fun () ->
                      let histogram = ImageFunctions.histogram image
                      if Map.isEmpty histogram then invalidOp "ImageFunctions.histogram returned an empty histogram." ]
            else
                []

        let chunkActions =
            [ "chunk-histogram-dense", fun () ->
                  let histogram = ChunkFunctions.histogramDense chunk
                  if Map.isEmpty histogram then invalidOp "ChunkFunctions.histogramDense returned an empty histogram."
              "chunk-histogram-leftedges-16", fun () ->
                  let histogram = ChunkFunctions.histogramLeftEdges leftEdges16 chunk
                  if Map.isEmpty histogram then invalidOp "ChunkFunctions.histogramLeftEdges returned an empty histogram."
              "chunk-histogram-leftedges-256", fun () ->
                  let histogram = ChunkFunctions.histogramLeftEdges leftEdges256 chunk
                  if Map.isEmpty histogram then invalidOp "ChunkFunctions.histogramLeftEdges returned an empty histogram." ]

        baselineActions @ chunkActions
        |> measureActions "histogram" "UInt8" shapeName threshold repeat
    finally
        StackCore.Chunk.decRef chunk
        image.decRefCount()

let private measureHistogramUInt16 includeImageFunctions shape threshold repeat =
    let shapeName = shapeText shape
    let length = requireIntLength shape
    let pixels = Array.zeroCreate<uint16> length
    fillUInt16 pixels length
    let image = Image.Image<uint16>.ofFlatArray(imageSizeList shape, pixels, "histogram-benchmark")
    let chunk = StackCore.Chunk.create<uint16> (chunkSize shape)
    let leftEdges256 = [ for edge in 0 .. 256 .. 65280 -> float edge ]
    let leftEdges4096 = [ for edge in 0 .. 4096 .. 61440 -> float edge ]
    try
        let values = StackCore.Chunk.span<uint16> chunk
        for i in 0 .. values.Length - 1 do
            values[i] <- pixels[i]
        let baselineActions =
            if includeImageFunctions then
                [ "imagefunctions-histogram", fun () ->
                      let histogram = ImageFunctions.histogram image
                      if Map.isEmpty histogram then invalidOp "ImageFunctions.histogram returned an empty histogram." ]
            else
                []

        let chunkActions =
            [ "chunk-histogram-dense", fun () ->
                  let histogram = ChunkFunctions.histogramDense chunk
                  if Map.isEmpty histogram then invalidOp "ChunkFunctions.histogramDense returned an empty histogram."
              "chunk-histogram-leftedges-256", fun () ->
                  let histogram = ChunkFunctions.histogramLeftEdges leftEdges256 chunk
                  if Map.isEmpty histogram then invalidOp "ChunkFunctions.histogramLeftEdges returned an empty histogram."
              "chunk-histogram-leftedges-4096", fun () ->
                  let histogram = ChunkFunctions.histogramLeftEdges leftEdges4096 chunk
                  if Map.isEmpty histogram then invalidOp "ChunkFunctions.histogramLeftEdges returned an empty histogram." ]

        baselineActions @ chunkActions
        |> measureActions "histogram" "UInt16" shapeName threshold repeat
    finally
        StackCore.Chunk.decRef chunk
        image.decRefCount()

let private measureHistogramFloat32 includeImageFunctions shape threshold repeat =
    let shapeName = shapeText shape
    let length = requireIntLength shape
    let pixels = Array.zeroCreate<float32> length
    fillFloat32 pixels length
    let image = Image.Image<float32>.ofFlatArray(imageSizeList shape, pixels, "histogram-benchmark")
    let chunk = StackCore.Chunk.create<float32> (chunkSize shape)
    let leftEdges16 = [ for edge in 0 .. 16 .. 240 -> float edge ]
    let leftEdges256 = [ for edge in 0 .. 255 -> float edge ]
    try
        let values = StackCore.Chunk.span<float32> chunk
        for i in 0 .. values.Length - 1 do
            values[i] <- pixels[i]
        let baselineActions =
            if includeImageFunctions then
                [ "imagefunctions-histogram", fun () ->
                      let histogram = ImageFunctions.histogram image
                      if Map.isEmpty histogram then invalidOp "ImageFunctions.histogram returned an empty histogram." ]
            else
                []

        let chunkActions =
            [ "chunk-histogram-sparse", fun () ->
                  let histogram = ChunkFunctions.histogram chunk
                  if Map.isEmpty histogram then invalidOp "ChunkFunctions.histogram returned an empty histogram."
              "chunk-histogram-leftedges-16", fun () ->
                  let histogram = ChunkFunctions.histogramLeftEdges leftEdges16 chunk
                  if Map.isEmpty histogram then invalidOp "ChunkFunctions.histogramLeftEdges returned an empty histogram."
              "chunk-histogram-leftedges-256", fun () ->
                  let histogram = ChunkFunctions.histogramLeftEdges leftEdges256 chunk
                  if Map.isEmpty histogram then invalidOp "ChunkFunctions.histogramLeftEdges returned an empty histogram." ]

        baselineActions @ chunkActions
        |> measureActions "histogram" "Float32" shapeName threshold repeat
    finally
        StackCore.Chunk.decRef chunk
        image.decRefCount()

let private runHistogramCase includeImageFunctions pixelType shape threshold repeat =
    match pixelType with
    | UInt8 -> measureHistogramUInt8 includeImageFunctions shape threshold repeat
    | UInt16 -> measureHistogramUInt16 includeImageFunctions shape threshold repeat
    | Float32 -> measureHistogramFloat32 includeImageFunctions shape threshold repeat

let private writeRows output rows =
    let exists = File.Exists output
    use writer = new StreamWriter(output, append = true)
    if not exists then
        writer.WriteLine("backend,variant,pixelType,shape,threshold,repeat,exitCode,seconds,managedAllocatedBytes,error")
    for row in rows do
        let fields =
            [ row.Backend
              row.Variant
              row.PixelType
              row.Shape
              row.Threshold
              string row.Repeat
              string row.ExitCode
              row.Seconds
              row.ManagedAllocatedBytes
              row.Error ]
            |> List.map (fun value -> "\"" + value.Replace("\"", "\"\"") + "\"")
        writer.WriteLine(String.Join(",", fields))

let private usage () =
    printfn "In-memory threshold benchmark"
    printfn "  dotnet run --project benchmarks/InMemoryThreshold.Benchmarks -- --output tmp/in-memory-threshold.csv --shapes 128x128x128,256x256x256,1024x1024x1024 --pixel-types UInt8,UInt16,Float32 --repeat 3 --threshold 128"
    printfn "  dotnet run --project benchmarks/InMemoryThreshold.Benchmarks -- --mode filter-lifecycle --output tmp/filter-lifecycle.csv --shapes 1024x1024x1 --pixel-types UInt8 --repeat 3 --iterations 1024 --threshold 128"
    printfn "  dotnet benchmarks/InMemoryThreshold.Benchmarks/bin/Release/net10.0/InMemoryThreshold.Benchmarks.dll --mode chunk-storage --output tmp/chunk-storage-threshold.csv --shapes 256x256x256,512x512x512 --pixel-types UInt8,UInt16,Float32 --repeat 3 --threshold 128"
    printfn "  dotnet benchmarks/InMemoryThreshold.Benchmarks/bin/Release/net10.0/InMemoryThreshold.Benchmarks.dll --mode chunk-slicewise --output tmp/chunk-slicewise-threshold.csv --shapes 256x256x256,512x512x512 --pixel-types UInt8,UInt16,Float32 --repeat 3 --threshold 128"
    printfn "  dotnet benchmarks/InMemoryThreshold.Benchmarks/bin/Release/net10.0/InMemoryThreshold.Benchmarks.dll --mode chunk-map --output tmp/chunk-map-threshold.csv --shapes 256x256x256,512x512x512 --pixel-types UInt8,UInt16,Float32 --repeat 3 --threshold 128"
    printfn "  dotnet benchmarks/InMemoryThreshold.Benchmarks/bin/Release/net10.0/InMemoryThreshold.Benchmarks.dll --mode map-style-slicewise --output tmp/map-style-slicewise-threshold.csv --shapes 256x256x256,512x512x512 --pixel-types UInt8,UInt16,Float32 --repeat 3 --threshold 128"
    printfn "  dotnet benchmarks/InMemoryThreshold.Benchmarks/bin/Release/net10.0/InMemoryThreshold.Benchmarks.dll --mode image-class --output tmp/image-class-threshold.csv --shapes 256x256x256,512x512x512 --pixel-types UInt8,UInt16,Float32 --repeat 3 --threshold 128"
    printfn "  dotnet benchmarks/InMemoryThreshold.Benchmarks/bin/Release/net10.0/InMemoryThreshold.Benchmarks.dll --mode image-class-slicewise --output tmp/image-class-slicewise-threshold.csv --shapes 256x256x256,512x512x512 --pixel-types UInt8,UInt16,Float32 --repeat 3 --threshold 128"
    printfn "  dotnet benchmarks/InMemoryThreshold.Benchmarks/bin/Release/net10.0/InMemoryThreshold.Benchmarks.dll --mode histogram --include-imagefunctions false --output tmp/chunk-histogram.csv --shapes 256x256x256,512x512x512 --pixel-types UInt8,UInt16,Float32 --repeat 3"

[<EntryPoint>]
let main args =
    try
        let opts = parseArgs args
        if opts.ContainsKey("help") then
            usage()
            0
        else
            let output = optional "output" "benchmarks/results/in-memory-threshold.csv" opts
            let mode = optional "mode" "threshold" opts
            let shapes =
                optional "shapes" (if mode = "filter-lifecycle" then "1024x1024x1" else "128x128x128,256x256x256,1024x1024x1024") opts
                |> fun text -> text.Split(',', StringSplitOptions.RemoveEmptyEntries ||| StringSplitOptions.TrimEntries)
                |> Array.map parseShape
            let pixelTypes =
                optional "pixel-types" "UInt8,UInt16,Float32" opts
                |> fun text -> text.Split(',', StringSplitOptions.RemoveEmptyEntries ||| StringSplitOptions.TrimEntries)
                |> Array.map parsePixelType
            let repeat = optional "repeat" "3" opts |> int
            let iterations = optional "iterations" "1024" opts |> int
            let threshold = optional "threshold" "128" opts |> fun text -> Double.Parse(text, invariant)
            let includeImageFunctions = optionalBool "include-imagefunctions" true opts

            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))) |> ignore
            for shape in shapes do
                for pixelType in pixelTypes do
                    for r in 1 .. repeat do
                        let rows =
                            try
                                match mode with
                                | "threshold" -> runCase pixelType shape threshold r
                                | "filter-lifecycle" -> runFilterLifecycleCase pixelType shape threshold r iterations
                                | "chunk-storage" -> runChunkStorageCase pixelType shape threshold r
                                | "chunk-slicewise" -> runChunkSliceWiseCase pixelType shape threshold r
                                | "chunk-map" -> runChunkMapCase pixelType shape threshold r
                                | "map-style-slicewise" -> runMapStyleSliceWiseCase pixelType shape threshold r
                                | "image-class" -> runImageClassCase pixelType shape threshold r
                                | "image-class-slicewise" -> runImageClassSliceWiseCase pixelType shape threshold r
                                | "histogram" -> runHistogramCase includeImageFunctions pixelType shape threshold r
                                | _ -> invalidArg "mode" $"Unsupported mode '{mode}'."
                            with ex ->
                                let pixelName = string pixelType
                                [ failed "setup" "allocate-import" pixelName (shapeText shape) threshold r ex ]
                        writeRows output rows
                        printfn "wrote %s %A repeat %d" (shapeText shape) pixelType r
            0
    with ex ->
        eprintfn "%s: %s" (ex.GetType().Name) ex.Message
        2
