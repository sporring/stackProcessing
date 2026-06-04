open System
open System.Buffers
open System.Diagnostics
open System.Globalization
open System.IO
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
        else loop (i + 1) acc
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

let private index width height x y z =
    (z * height + y) * width + x

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
    for i in 0 .. n - 1 do buffer[i] <- byte ((i * 17 + i / 7) &&& 0xff)

let private fillUInt16 (buffer: uint16[]) n =
    for i in 0 .. n - 1 do buffer[i] <- uint16 ((i * 17 + i / 7) &&& 0xffff)

let private fillFloat32 (buffer: float32[]) n =
    for i in 0 .. n - 1 do buffer[i] <- float32 ((i * 17 + i / 7) &&& 0xff)

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

let private clampByte (x: float32) = byte (max 0.0f (min 255.0f x))
let private clampUInt16 (x: float32) = uint16 (max 0.0f (min (float32 UInt16.MaxValue) x))

let private boxMeanRollingX<'T> radius shape (input: 'T[]) (output: float32[]) toFloat =
    let w, h, d = shape.Width, shape.Height, shape.Depth
    let side = radius * 2 + 1
    let count = float32 (side * side * side)
    Array.Clear(output, 0, w * h * d)
    for z in radius .. d - radius - 1 do
        for y in radius .. h - radius - 1 do
            let mutable sum = 0.0f
            for dz in -radius .. radius do
                for dy in -radius .. radius do
                    for xx in 0 .. side - 1 do
                        sum <- sum + toFloat input[index w h xx (y + dy) (z + dz)]
            output[index w h radius y z] <- sum / count
            for x in radius + 1 .. w - radius - 1 do
                let removeX = x - radius - 1
                let addX = x + radius
                for dz in -radius .. radius do
                    for dy in -radius .. radius do
                        sum <- sum - toFloat input[index w h removeX (y + dy) (z + dz)]
                        sum <- sum + toFloat input[index w h addX (y + dy) (z + dz)]
                output[index w h x y z] <- sum / count

let private boxMeanSeparable<'T> radius shape (input: 'T[]) (tmpX: float32[]) (tmpY: float32[]) (output: float32[]) toFloat =
    let w, h, d = shape.Width, shape.Height, shape.Depth
    let side = radius * 2 + 1
    let sideF = float32 side
    Array.Clear(tmpX, 0, w * h * d)
    Array.Clear(tmpY, 0, w * h * d)
    Array.Clear(output, 0, w * h * d)
    for z in 0 .. d - 1 do
        for y in 0 .. h - 1 do
            let mutable sum = 0.0f
            for x in 0 .. side - 1 do
                sum <- sum + toFloat input[index w h x y z]
            tmpX[index w h radius y z] <- sum / sideF
            for x in radius + 1 .. w - radius - 1 do
                sum <- sum - toFloat input[index w h (x - radius - 1) y z]
                sum <- sum + toFloat input[index w h (x + radius) y z]
                tmpX[index w h x y z] <- sum / sideF
    for z in 0 .. d - 1 do
        for x in radius .. w - radius - 1 do
            let mutable sum = 0.0f
            for y in 0 .. side - 1 do
                sum <- sum + tmpX[index w h x y z]
            tmpY[index w h x radius z] <- sum / sideF
            for y in radius + 1 .. h - radius - 1 do
                sum <- sum - tmpX[index w h x (y - radius - 1) z]
                sum <- sum + tmpX[index w h x (y + radius) z]
                tmpY[index w h x y z] <- sum / sideF
    for y in radius .. h - radius - 1 do
        for x in radius .. w - radius - 1 do
            let mutable sum = 0.0f
            for z in 0 .. side - 1 do
                sum <- sum + tmpY[index w h x y z]
            output[index w h x y radius] <- sum / sideF
            for z in radius + 1 .. d - radius - 1 do
                sum <- sum - tmpY[index w h x y (z - radius - 1)]
                sum <- sum + tmpY[index w h x y (z + radius)]
                output[index w h x y z] <- sum / sideF

let private measureItkBoxMean<'T> radius shape (input: 'T[]) =
    use image = importedImage shape input
    use filter = new itk.simple.BoxMeanImageFilter()
    filter.SetRadius(vectorUInt32 [ radius; radius; radius ])
    use warmup = filter.Execute(image)
    measure (fun () ->
        use result = filter.Execute(image)
        if result.GetSize().Count = 0 then failwith "unreachable")

let private writeHeaderIfNeeded output =
    if not (File.Exists output) then
        File.WriteAllText(output, "backend,variant,pixelType,shape,radius,repeat,exitCode,seconds,managedAllocatedBytes,error\n")

let private appendRow output backend variant pixel shape radius repeat exitCode (seconds: string) (allocated: string) (error: string) =
    writeHeaderIfNeeded output
    let q (s: string) = "\"" + s.Replace("\"", "\"\"") + "\""
    [ backend; variant; pixel; shapeText shape; string radius; string repeat; string exitCode; seconds; allocated; error.Replace("\n", " ") ]
    |> List.map q
    |> fun fields -> File.AppendAllText(output, String.Join(",", fields) + "\n")

let private runTyped<'T> output pixelName radius shape repeat fill toFloat =
    let n = length shape
    let input = ArrayPool<'T>.Shared.Rent(n)
    let rolling = ArrayPool<float32>.Shared.Rent(n)
    let tmpX = ArrayPool<float32>.Shared.Rent(n)
    let tmpY = ArrayPool<float32>.Shared.Rent(n)
    let sep = ArrayPool<float32>.Shared.Rent(n)
    try
        fill input n
        let seconds, allocated = measure (fun () -> boxMeanRollingX radius shape input rolling toFloat)
        appendRow output "arraypool" "rollingx-boxmean" pixelName shape radius repeat 0 (seconds.ToString("F9", invariant)) (string allocated) ""
        let seconds, allocated = measure (fun () -> boxMeanSeparable radius shape input tmpX tmpY sep toFloat)
        appendRow output "arraypool" "separable-rolling-boxmean" pixelName shape radius repeat 0 (seconds.ToString("F9", invariant)) (string allocated) ""
        try
            let seconds, allocated = measureItkBoxMean<'T> radius shape input
            appendRow output "simpleitk" "BoxMeanImageFilter" pixelName shape radius repeat 0 (seconds.ToString("F9", invariant)) (string allocated) ""
        with ex ->
            appendRow output "simpleitk" "BoxMeanImageFilter" pixelName shape radius repeat 1 "" "" (ex.GetType().Name + ": " + ex.Message)
    finally
        ArrayPool<'T>.Shared.Return(input)
        ArrayPool<float32>.Shared.Return(rolling)
        ArrayPool<float32>.Shared.Return(tmpX)
        ArrayPool<float32>.Shared.Return(tmpY)
        ArrayPool<float32>.Shared.Return(sep)

[<EntryPoint>]
let main args =
    try
        let opts = parseArgs args
        let output = optional "output" "benchmarks/results/in-memory-boxmean.csv" opts
        let shapes =
            optional "shapes" "128x128x128" opts
            |> fun text -> text.Split(',', StringSplitOptions.RemoveEmptyEntries ||| StringSplitOptions.TrimEntries)
            |> Array.map parseShape
        let pixelTypes =
            optional "pixel-types" "UInt8,UInt16,Float32" opts
            |> fun text -> text.Split(',', StringSplitOptions.RemoveEmptyEntries ||| StringSplitOptions.TrimEntries)
            |> Array.map parsePixelType
        let radius = optional "radius" "5" opts |> int
        let repeat = optional "repeat" "3" opts |> int
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))) |> ignore
        if File.Exists output then File.Delete output
        for shape in shapes do
            for pixel in pixelTypes do
                for r in 1 .. repeat do
                    match pixel with
                    | UInt8 -> runTyped<uint8> output "UInt8" radius shape r fillUInt8 float32
                    | UInt16 -> runTyped<uint16> output "UInt16" radius shape r fillUInt16 float32
                    | Float32 -> runTyped<float32> output "Float32" radius shape r fillFloat32 id
                    printfn "wrote %s %A radius %d repeat %d" (shapeText shape) pixel radius r
        0
    with ex ->
        eprintfn "%s: %s" (ex.GetType().Name) ex.Message
        2
