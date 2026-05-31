open System
open System.Diagnostics
open System.Globalization
open System.IO
open StackProcessing

type PixelType =
    | UInt8
    | UInt16
    | Float32

type Shape =
    { Width: uint
      Height: uint
      Depth: uint }

let private invariant = CultureInfo.InvariantCulture

let private fail message =
    eprintfn "%s" message
    2

let private usage () =
    """
StackProcessing benchmark runner

Generate:
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- generate --output DIR --shape 512x512x64 --pixel-type UInt8 [--pattern ramp|binary]

Run:
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run --operation copy|threshold|uniformConvolve|median|dilate|connectedComponents --pixel-type UInt8|UInt16|Float32 --input DIR --output DIR [--radius N] [--kernel-size N] [--threshold X] [--window N] [--available-memory BYTES]
"""
    |> printfn "%s"
    0

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

let private require name (opts: Map<string,string>) =
    match opts.TryFind name with
    | Some value -> value
    | None -> failwith $"missing required --{name}"

let private optional name fallback (opts: Map<string,string>) =
    opts.TryFind name |> Option.defaultValue fallback

let private writeInternalSeconds (elapsed: TimeSpan) =
    let path = Environment.GetEnvironmentVariable("BENCHMARK_INTERNAL_SECONDS_PATH")
    if not (String.IsNullOrWhiteSpace path) then
        File.WriteAllText(path, elapsed.TotalSeconds.ToString("F9", invariant))

let private benchmarkSource availableMemory =
    sourceWithOptimizer false availableMemory

let private parsePixelType value =
    match value with
    | "UInt8" | "uint8" -> UInt8
    | "UInt16" | "uint16" -> UInt16
    | "Float32" | "float32" -> Float32
    | _ -> failwith $"unsupported pixel type '{value}'"

let private parseShape (value: string) =
    let parts = value.Split('x', StringSplitOptions.RemoveEmptyEntries)
    if parts.Length <> 3 then
        failwith $"shape must be WxHxD, got '{value}'"
    { Width = UInt32.Parse(parts[0], invariant)
      Height = UInt32.Parse(parts[1], invariant)
      Depth = UInt32.Parse(parts[2], invariant) }

let private ensureCleanDirectory path =
    if Directory.Exists path then
        Directory.Delete(path, true)
    Directory.CreateDirectory path |> ignore

let private outputFile outputDir z =
    Path.Combine(outputDir, sprintf "slice_%05d.tiff" z)

let private generateUInt8 pattern shape outputDir =
    ensureCleanDirectory outputDir
    for z in 0u .. shape.Depth - 1u do
        let arr =
            Array2D.init (int shape.Width) (int shape.Height) (fun x y ->
                match pattern with
                | "binary" -> if (x + y + int z) % 7 = 0 then 255uy else 0uy
                | _ -> byte ((x * 3 + y * 5 + int z * 11) % 256))
        let img = Image<uint8>.ofArray2D(arr, name = "input", index = int z)
        img.toFile(outputFile outputDir (int z))
        img.decRefCount()

let private generateUInt16 pattern shape outputDir =
    ensureCleanDirectory outputDir
    for z in 0u .. shape.Depth - 1u do
        let arr =
            Array2D.init (int shape.Width) (int shape.Height) (fun x y ->
                match pattern with
                | "binary" -> if (x + y + int z) % 7 = 0 then 255us else 0us
                | _ -> uint16 ((x * 3 + y * 5 + int z * 11) % 256))
        let img = Image<uint16>.ofArray2D(arr, name = "input", index = int z)
        img.toFile(outputFile outputDir (int z))
        img.decRefCount()

let private generateFloat32 pattern shape outputDir =
    ensureCleanDirectory outputDir
    for z in 0u .. shape.Depth - 1u do
        let arr =
            Array2D.init (int shape.Width) (int shape.Height) (fun x y ->
                match pattern with
                | "binary" -> if (x + y + int z) % 7 = 0 then 255.0f else 0.0f
                | _ -> float32 ((x * 3 + y * 5 + int z * 11) % 256))
        let img = Image<float32>.ofArray2D(arr, name = "input", index = int z)
        img.toFile(outputFile outputDir (int z))
        img.decRefCount()

let private generate opts =
    let shape = require "shape" opts |> parseShape
    let pixelType = require "pixel-type" opts |> parsePixelType
    let output = require "output" opts
    let pattern = optional "pattern" "ramp" opts
    match pixelType with
    | UInt8 -> generateUInt8 pattern shape output
    | UInt16 -> generateUInt16 pattern shape output
    | Float32 -> generateFloat32 pattern shape output
    0

let private runTyped<'T when 'T: equality> operation input output radius thresholdValue availableMemory =
    ensureCleanDirectory output
    let src = benchmarkSource availableMemory
    match operation with
    | "copy" ->
        src
        |> read<'T> input ".tiff"
        >=> write output ".tiff"
        |> sink
    | "threshold" ->
        src
        |> read<'T> input ".tiff"
        >=> threshold thresholdValue infinity
        >=> write output ".tiff"
        |> sink
    | "median" ->
        src
        |> read<'T> input ".tiff"
        >=> smoothWMedian<'T> radius None
        >=> write output ".tiff"
        |> sink
    | _ -> failwith $"unsupported operation '{operation}'"
    0

let private runBinaryDilateTyped<'T when 'T: equality> input output radius availableMemory =
    ensureCleanDirectory output
    let src = benchmarkSource availableMemory
    src
    |> read<'T> input ".tiff"
    >=> threshold 128.0 infinity
    >=> dilate radius
    >=> write output ".tiff"
    |> sink
    0

let private uniformKernel3D (kernelSize: uint) =
    let size = max 1u kernelSize
    let value = 1.0 / float (size * size * size)
    Array3D.create (int size) (int size) (int size) value
    |> fun values -> Image<float>.ofArray3D(values, name = $"uniformKernel{size}")

let private runUniformConvolveTyped<'T when 'T: equality> input output kernelSize availableMemory =
    ensureCleanDirectory output
    let kernel = uniformKernel3D kernelSize
    let src = benchmarkSource availableMemory
    try
        src
        |> read<'T> input ".tiff"
        >=> cast<'T, float>
        >=> convolve kernel None None None
        >=> cast<float, 'T>
        >=> write output ".tiff"
        |> sink
    finally
        kernel.decRefCount()
    0

let private runConnectedComponents input output windowSize availableMemory =
    ensureCleanDirectory output
    let window = max 1u windowSize
    let width, height, depth = getStackSize input ".tiff"
    let src = benchmarkSource availableMemory

    if connectedComponentsFullVolumeFits availableMemory width height depth then
        src
        |> read<uint8> input ".tiff"
        >=> threshold 128.0 infinity
        >=> connectedComponentsLabels (Some depth)
        >=> cast<uint64, uint8>
        >=> writeSlabSlices output ".tiff" depth
        |> sink
        0
    else
        let tmp = output + "-labels"
        ensureCleanDirectory tmp
        let tmpSuffix = ".mha"
        try
            let table =
                src
                |> read<uint8> input ".tiff"
                >=> threshold 128.0 infinity
                >=> connectedComponents (Some window)
                >=> teeFst (writeSlabSlices tmp tmpSuffix window)
                >=> makeConnectedComponentTranslationTable (Some window)
                |> drain

            src
            |> read<uint64> tmp tmpSuffix
            >=> updateConnectedComponents (Some window) table
            >=> cast<uint64, uint8>
            >=> write output ".tiff"
            |> sink
            0
        finally
            if Directory.Exists tmp then
                Directory.Delete(tmp, true)

let private run opts =
    let operation = require "operation" opts
    let pixelType = require "pixel-type" opts |> parsePixelType
    let input = require "input" opts
    let output = require "output" opts
    let radius = optional "radius" "1" opts |> UInt32.Parse
    let windowSize = optional "window" "16" opts |> UInt32.Parse
    let kernelSize = optional "kernel-size" "3" opts |> UInt32.Parse
    let thresholdValue = optional "threshold" "128" opts |> fun s -> Double.Parse(s, invariant)
    let availableMemory = optional "available-memory" (string (1024UL * 1024UL * 1024UL * 1024UL)) opts |> UInt64.Parse
    let stopwatch = Stopwatch.StartNew()
    let exitCode =
        match operation, pixelType with
        | "uniformConvolve", UInt8 -> runUniformConvolveTyped<uint8> input output kernelSize availableMemory
        | "uniformConvolve", UInt16 -> runUniformConvolveTyped<uint16> input output kernelSize availableMemory
        | "uniformConvolve", Float32 -> runUniformConvolveTyped<float32> input output kernelSize availableMemory
        | "dilate", UInt8 -> runBinaryDilateTyped<uint8> input output radius availableMemory
        | "dilate", UInt16 -> runBinaryDilateTyped<uint16> input output radius availableMemory
        | "dilate", Float32 -> runBinaryDilateTyped<float32> input output radius availableMemory
        | "connectedComponents", UInt8 -> runConnectedComponents input output windowSize availableMemory
        | "connectedComponents", _ -> failwith "connectedComponents benchmark is currently defined for UInt8 masks only"
        | _, UInt8 -> runTyped<uint8> operation input output radius thresholdValue availableMemory
        | _, UInt16 -> runTyped<uint16> operation input output radius thresholdValue availableMemory
        | _, Float32 -> runTyped<float32> operation input output radius thresholdValue availableMemory
    stopwatch.Stop()
    writeInternalSeconds stopwatch.Elapsed
    exitCode

[<EntryPoint>]
let main args =
    try
        match args with
        | [| |] -> usage ()
        | _ when args[0] = "generate" -> args[1..] |> parseArgs |> generate
        | _ when args[0] = "run" -> args[1..] |> parseArgs |> run
        | _ -> usage ()
    with ex ->
        fail ex.Message
