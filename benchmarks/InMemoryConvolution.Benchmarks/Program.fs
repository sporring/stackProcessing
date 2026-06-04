open System
open System.Buffers
open System.Diagnostics
open System.Globalization
open System.IO
open System.Numerics
open System.Runtime.InteropServices

type PixelType =
    | UInt8
    | UInt16
    | Float32

type Shape =
    { Width: int
      Height: int
      Depth: int }

let private invariant = CultureInfo.InvariantCulture

let private parseArgs (args: string array) =
    let rec loop i (acc: Map<string,string>) =
        if i >= args.Length then acc
        elif args[i].StartsWith("--", StringComparison.Ordinal) then
            if i + 1 >= args.Length || args[i + 1].StartsWith("--", StringComparison.Ordinal) then
                loop (i + 1) (acc.Add(args[i].Substring(2), "true"))
            else
                loop (i + 2) (acc.Add(args[i].Substring(2), args[i + 1]))
        else
            loop (i + 1) acc
    loop 0 Map.empty

let private optional name fallback opts =
    opts |> Map.tryFind name |> Option.defaultValue fallback

let private parseShape (text: string) =
    let parts = text.Split('x', StringSplitOptions.RemoveEmptyEntries ||| StringSplitOptions.TrimEntries)
    if parts.Length <> 3 then invalidArg "shape" $"Expected WxHxD, got {text}."
    { Width = Int32.Parse(parts[0], invariant)
      Height = Int32.Parse(parts[1], invariant)
      Depth = Int32.Parse(parts[2], invariant) }

let private parsePixelType (text: string) =
    match text.Trim().ToLowerInvariant() with
    | "uint8" -> UInt8
    | "uint16" -> UInt16
    | "float32" -> Float32
    | _ -> invalidArg "pixel-type" $"Unsupported pixel type {text}."

let private shapeText s = $"{s.Width}x{s.Height}x{s.Depth}"

let private length s =
    let n = int64 s.Width * int64 s.Height * int64 s.Depth
    if n > int64 Int32.MaxValue then invalidArg "shape" $"Too many elements for CLR array: {n}."
    int n

let private vectorUInt32 values = new itk.simple.VectorUInt32(values |> Seq.map uint32 |> Seq.toList)
let private vectorDouble values = new itk.simple.VectorDouble(values |> Seq.toList)

let private identityDirection dim =
    [ for row in 0 .. dim - 1 do
        for col in 0 .. dim - 1 do
            if row = col then 1.0 else 0.0 ]

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

let private fillUInt8 (buffer: uint8[]) n =
    for i in 0 .. n - 1 do
        buffer[i] <- byte (i &&& 0xff)

let private fillUInt16 (buffer: uint16[]) n =
    for i in 0 .. n - 1 do
        buffer[i] <- uint16 (i &&& 0xff)

let private fillFloat32 (buffer: float32[]) n =
    for i in 0 .. n - 1 do
        buffer[i] <- float32 (i &&& 0xff)

let private measure action =
    GC.Collect()
    GC.WaitForPendingFinalizers()
    GC.Collect()
    let before = GC.GetTotalAllocatedBytes(true)
    let sw = Stopwatch.StartNew()
    action()
    sw.Stop()
    let after = GC.GetTotalAllocatedBytes(true)
    sw.Elapsed.TotalSeconds, after - before

let private index width height x y z =
    (z * height + y) * width + x

let private stencil3x3x3Offsets width height =
    let sliceStride = width * height
    [| for dz in -1 .. 1 do
        for dy in -1 .. 1 do
            for dx in -1 .. 1 do
                yield dz * sliceStride + dy * width + dx |]

let private convolve3x3x3UInt8 shape (input: uint8[]) (output: uint8[]) =
    let w, h, d = shape.Width, shape.Height, shape.Depth
    Array.Clear(output, 0, w * h * d)
    for z in 1 .. d - 2 do
        for y in 1 .. h - 2 do
            for x in 1 .. w - 2 do
                let mutable sum = 0
                for dz in -1 .. 1 do
                    for dy in -1 .. 1 do
                        for dx in -1 .. 1 do
                            sum <- sum + int input[index w h (x + dx) (y + dy) (z + dz)]
                output[index w h x y z] <- byte (min 255 sum)

let private convolve3x3x3UInt8Unrolled shape (input: uint8[]) (output: uint8[]) =
    let w, h, d = shape.Width, shape.Height, shape.Depth
    let offsets = stencil3x3x3Offsets w h
    Array.Clear(output, 0, w * h * d)
    for z in 1 .. d - 2 do
        for y in 1 .. h - 2 do
            for x in 1 .. w - 2 do
                let baseIndex = index w h x y z
                let mutable sum = 0
                let mutable k = 0
                while k < offsets.Length do
                    sum <- sum + int input[baseIndex + offsets[k]]
                    k <- k + 1
                output[baseIndex] <- byte (min 255 sum)

let private convolve3x3x3UInt16 shape (input: uint16[]) (output: uint16[]) =
    let w, h, d = shape.Width, shape.Height, shape.Depth
    Array.Clear(output, 0, w * h * d)
    for z in 1 .. d - 2 do
        for y in 1 .. h - 2 do
            for x in 1 .. w - 2 do
                let mutable sum = 0
                for dz in -1 .. 1 do
                    for dy in -1 .. 1 do
                        for dx in -1 .. 1 do
                            sum <- sum + int input[index w h (x + dx) (y + dy) (z + dz)]
                output[index w h x y z] <- uint16 (min (int UInt16.MaxValue) sum)

let private convolve3x3x3UInt16Unrolled shape (input: uint16[]) (output: uint16[]) =
    let w, h, d = shape.Width, shape.Height, shape.Depth
    let offsets = stencil3x3x3Offsets w h
    Array.Clear(output, 0, w * h * d)
    for z in 1 .. d - 2 do
        for y in 1 .. h - 2 do
            for x in 1 .. w - 2 do
                let baseIndex = index w h x y z
                let mutable sum = 0
                let mutable k = 0
                while k < offsets.Length do
                    sum <- sum + int input[baseIndex + offsets[k]]
                    k <- k + 1
                output[baseIndex] <- uint16 (min (int UInt16.MaxValue) sum)

let private convolve3x3x3UInt16VectorX shape (input: uint16[]) (output: uint16[]) =
    let w, h, d = shape.Width, shape.Height, shape.Depth
    let offsets = stencil3x3x3Offsets w h
    let lanes = Vector<uint16>.Count
    Array.Clear(output, 0, w * h * d)
    for z in 1 .. d - 2 do
        for y in 1 .. h - 2 do
            let mutable x = 1
            let vectorLastStart = w - lanes - 1
            while x <= vectorLastStart do
                let baseIndex = index w h x y z
                let mutable acc = Vector<uint16>.Zero
                let mutable k = 0
                while k < offsets.Length do
                    acc <- acc + Vector<uint16>(input, baseIndex + offsets[k])
                    k <- k + 1
                acc.CopyTo(output, baseIndex)
                x <- x + lanes
            while x <= w - 2 do
                let baseIndex = index w h x y z
                let mutable sum = 0
                let mutable k = 0
                while k < offsets.Length do
                    sum <- sum + int input[baseIndex + offsets[k]]
                    k <- k + 1
                output[baseIndex] <- uint16 (min (int UInt16.MaxValue) sum)
                x <- x + 1

let private convolve3x3x3Float32 shape (input: float32[]) (output: float32[]) =
    let w, h, d = shape.Width, shape.Height, shape.Depth
    Array.Clear(output, 0, w * h * d)
    for z in 1 .. d - 2 do
        for y in 1 .. h - 2 do
            for x in 1 .. w - 2 do
                let mutable sum = 0.0f
                for dz in -1 .. 1 do
                    for dy in -1 .. 1 do
                        for dx in -1 .. 1 do
                            sum <- sum + input[index w h (x + dx) (y + dy) (z + dz)]
                output[index w h x y z] <- sum

let private convolve3x3x3Float32Unrolled shape (input: float32[]) (output: float32[]) =
    let w, h, d = shape.Width, shape.Height, shape.Depth
    let offsets = stencil3x3x3Offsets w h
    Array.Clear(output, 0, w * h * d)
    for z in 1 .. d - 2 do
        for y in 1 .. h - 2 do
            for x in 1 .. w - 2 do
                let baseIndex = index w h x y z
                let mutable sum = 0.0f
                let mutable k = 0
                while k < offsets.Length do
                    sum <- sum + input[baseIndex + offsets[k]]
                    k <- k + 1
                output[baseIndex] <- sum

let private convolve3x3x3Float32VectorX shape (input: float32[]) (output: float32[]) =
    let w, h, d = shape.Width, shape.Height, shape.Depth
    let offsets = stencil3x3x3Offsets w h
    let lanes = Vector<float32>.Count
    Array.Clear(output, 0, w * h * d)
    for z in 1 .. d - 2 do
        for y in 1 .. h - 2 do
            let mutable x = 1
            let vectorLastStart = w - lanes - 1
            while x <= vectorLastStart do
                let baseIndex = index w h x y z
                let mutable acc = Vector<float32>.Zero
                let mutable k = 0
                while k < offsets.Length do
                    acc <- acc + Vector<float32>(input, baseIndex + offsets[k])
                    k <- k + 1
                acc.CopyTo(output, baseIndex)
                x <- x + lanes
            while x <= w - 2 do
                let baseIndex = index w h x y z
                let mutable sum = 0.0f
                let mutable k = 0
                while k < offsets.Length do
                    sum <- sum + input[baseIndex + offsets[k]]
                    k <- k + 1
                output[baseIndex] <- sum
                x <- x + 1

let private makeKernel<'T> (value: 'T) =
    let kernel = Array.create 27 value
    importedImage { Width = 3; Height = 3; Depth = 3 } kernel

let private measureItkConvolution<'T> shape (input: 'T[]) kernelValue =
    use image = importedImage shape input
    use kernel = makeKernel<'T> kernelValue
    use filter = new itk.simple.ConvolutionImageFilter()
    filter.SetOutputRegionMode(itk.simple.ConvolutionImageFilter.OutputRegionModeType.SAME)
    filter.SetBoundaryCondition(itk.simple.ConvolutionImageFilter.BoundaryConditionType.ZERO_PAD)
    use warmup = filter.Execute(image, kernel)
    measure (fun () ->
        use result = filter.Execute(image, kernel)
        if result.GetSize().Count = 0 then failwith "unreachable")

let private writeHeaderIfNeeded output =
    if not (File.Exists output) then
        File.WriteAllText(output, "backend,variant,pixelType,shape,repeat,exitCode,seconds,managedAllocatedBytes,error\n")

let private appendRow output backend variant pixel shape repeat exitCode (seconds: string) (allocated: string) (error: string) =
    writeHeaderIfNeeded output
    let q (s: string) = "\"" + s.Replace("\"", "\"\"") + "\""
    let fields =
        [ backend; variant; pixel; shapeText shape; string repeat; string exitCode; seconds; allocated; error.Replace("\n", " ") ]
        |> List.map q
    File.AppendAllText(output, String.Join(",", fields) + "\n")

let private runCase output shape pixel repeat =
    let n = length shape
    match pixel with
    | UInt8 ->
        let input = ArrayPool<uint8>.Shared.Rent(n)
        let outputBuffer = ArrayPool<uint8>.Shared.Rent(n)
        try
            fillUInt8 input n
            let seconds, allocated = measure (fun () -> convolve3x3x3UInt8 shape input outputBuffer)
            appendRow output "arraypool" "flat-interior-3x3x3" "UInt8" shape repeat 0 (seconds.ToString("F9", invariant)) (string allocated) ""
            let seconds, allocated = measure (fun () -> convolve3x3x3UInt8Unrolled shape input outputBuffer)
            appendRow output "arraypool" "flat-unrolled-3x3x3" "UInt8" shape repeat 0 (seconds.ToString("F9", invariant)) (string allocated) ""
            try
                let seconds, allocated = measureItkConvolution<uint8> shape input 1uy
                appendRow output "simpleitk" "ConvolutionImageFilter-3x3x3" "UInt8" shape repeat 0 (seconds.ToString("F9", invariant)) (string allocated) ""
            with ex ->
                appendRow output "simpleitk" "ConvolutionImageFilter-3x3x3" "UInt8" shape repeat 1 "" "" (ex.GetType().Name + ": " + ex.Message)
        finally
            ArrayPool<uint8>.Shared.Return(input)
            ArrayPool<uint8>.Shared.Return(outputBuffer)
    | UInt16 ->
        let input = ArrayPool<uint16>.Shared.Rent(n)
        let outputBuffer = ArrayPool<uint16>.Shared.Rent(n)
        try
            fillUInt16 input n
            let seconds, allocated = measure (fun () -> convolve3x3x3UInt16 shape input outputBuffer)
            appendRow output "arraypool" "flat-interior-3x3x3" "UInt16" shape repeat 0 (seconds.ToString("F9", invariant)) (string allocated) ""
            let seconds, allocated = measure (fun () -> convolve3x3x3UInt16Unrolled shape input outputBuffer)
            appendRow output "arraypool" "flat-unrolled-3x3x3" "UInt16" shape repeat 0 (seconds.ToString("F9", invariant)) (string allocated) ""
            let seconds, allocated = measure (fun () -> convolve3x3x3UInt16VectorX shape input outputBuffer)
            appendRow output "arraypool" "flat-vectorx-3x3x3" "UInt16" shape repeat 0 (seconds.ToString("F9", invariant)) (string allocated) ""
            try
                let seconds, allocated = measureItkConvolution<uint16> shape input 1us
                appendRow output "simpleitk" "ConvolutionImageFilter-3x3x3" "UInt16" shape repeat 0 (seconds.ToString("F9", invariant)) (string allocated) ""
            with ex ->
                appendRow output "simpleitk" "ConvolutionImageFilter-3x3x3" "UInt16" shape repeat 1 "" "" (ex.GetType().Name + ": " + ex.Message)
        finally
            ArrayPool<uint16>.Shared.Return(input)
            ArrayPool<uint16>.Shared.Return(outputBuffer)
    | Float32 ->
        let input = ArrayPool<float32>.Shared.Rent(n)
        let outputBuffer = ArrayPool<float32>.Shared.Rent(n)
        try
            fillFloat32 input n
            let seconds, allocated = measure (fun () -> convolve3x3x3Float32 shape input outputBuffer)
            appendRow output "arraypool" "flat-interior-3x3x3" "Float32" shape repeat 0 (seconds.ToString("F9", invariant)) (string allocated) ""
            let seconds, allocated = measure (fun () -> convolve3x3x3Float32Unrolled shape input outputBuffer)
            appendRow output "arraypool" "flat-unrolled-3x3x3" "Float32" shape repeat 0 (seconds.ToString("F9", invariant)) (string allocated) ""
            let seconds, allocated = measure (fun () -> convolve3x3x3Float32VectorX shape input outputBuffer)
            appendRow output "arraypool" "flat-vectorx-3x3x3" "Float32" shape repeat 0 (seconds.ToString("F9", invariant)) (string allocated) ""
            try
                let seconds, allocated = measureItkConvolution<float32> shape input 1.0f
                appendRow output "simpleitk" "ConvolutionImageFilter-3x3x3" "Float32" shape repeat 0 (seconds.ToString("F9", invariant)) (string allocated) ""
            with ex ->
                appendRow output "simpleitk" "ConvolutionImageFilter-3x3x3" "Float32" shape repeat 1 "" "" (ex.GetType().Name + ": " + ex.Message)
        finally
            ArrayPool<float32>.Shared.Return(input)
            ArrayPool<float32>.Shared.Return(outputBuffer)

[<EntryPoint>]
let main args =
    try
        let opts = parseArgs args
        let output = optional "output" "benchmarks/results/in-memory-convolution.csv" opts
        let shapes =
            optional "shapes" "128x128x128,256x256x256" opts
            |> fun text -> text.Split(',', StringSplitOptions.RemoveEmptyEntries ||| StringSplitOptions.TrimEntries)
            |> Array.map parseShape
        let pixelTypes =
            optional "pixel-types" "UInt8,UInt16,Float32" opts
            |> fun text -> text.Split(',', StringSplitOptions.RemoveEmptyEntries ||| StringSplitOptions.TrimEntries)
            |> Array.map parsePixelType
        let repeat = optional "repeat" "3" opts |> int
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))) |> ignore
        if File.Exists output then File.Delete output
        for shape in shapes do
            for pixel in pixelTypes do
                for r in 1 .. repeat do
                    runCase output shape pixel r
                    printfn "wrote %s %A repeat %d" (shapeText shape) pixel r
        0
    with ex ->
        eprintfn "%s: %s" (ex.GetType().Name) ex.Message
        2
