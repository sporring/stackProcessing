open System
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
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run --operation copy|threshold|smoothWGauss|median|dilate|connectedComponents --pixel-type UInt8|UInt16|Float32 --input DIR --output DIR [--radius N] [--sigma X] [--threshold X] [--window N] [--available-memory BYTES]
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
                | "binary" -> if (x + y + int z) % 7 = 0 then 65535us else 0us
                | _ -> uint16 ((x * 97 + y * 193 + int z * 389) % 65536))
        let img = Image<uint16>.ofArray2D(arr, name = "input", index = int z)
        img.toFile(outputFile outputDir (int z))
        img.decRefCount()

let private generateFloat32 pattern shape outputDir =
    ensureCleanDirectory outputDir
    for z in 0u .. shape.Depth - 1u do
        let arr =
            Array2D.init (int shape.Width) (int shape.Height) (fun x y ->
                match pattern with
                | "binary" -> if (x + y + int z) % 7 = 0 then 1.0f else 0.0f
                | _ -> float32 ((x * 3 + y * 5 + int z * 11) % 4096) / 4095.0f)
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
    let src, _ = commandLineSource availableMemory [||]
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
    | "dilate" ->
        src
        |> read<'T> input ".tiff"
        >=> grayscaleDilate<'T> radius None
        >=> write output ".tiff"
        |> sink
    | _ -> failwith $"unsupported operation '{operation}'"
    0

let private runGaussian input output sigma availableMemory =
    ensureCleanDirectory output
    let src, _ = commandLineSource availableMemory [||]
    src
    |> read<float> input ".tiff"
    >=> smoothWGauss sigma None None None
    >=> cast<float, float32>
    >=> write output ".tiff"
    |> sink
    0

let private runConnectedComponents input output windowSize availableMemory =
    ensureCleanDirectory output
    let tmp = output + "-labels"
    ensureCleanDirectory tmp
    let tmpSuffix = ".mha"
    let window = max 1u windowSize
    let src, _ = commandLineSource availableMemory [||]
    let table =
        src
        |> read<uint8> input ".tiff"
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

let private run opts =
    let operation = require "operation" opts
    let pixelType = require "pixel-type" opts |> parsePixelType
    let input = require "input" opts
    let output = require "output" opts
    let radius = optional "radius" "1" opts |> UInt32.Parse
    let windowSize = optional "window" "16" opts |> UInt32.Parse
    let sigma = optional "sigma" "1.5" opts |> fun s -> Double.Parse(s, invariant)
    let thresholdValue = optional "threshold" "128" opts |> fun s -> Double.Parse(s, invariant)
    let availableMemory = optional "available-memory" (string (8UL * 1024UL * 1024UL * 1024UL)) opts |> UInt64.Parse
    match operation, pixelType with
    | "smoothWGauss", _ -> runGaussian input output sigma availableMemory
    | "connectedComponents", UInt8 -> runConnectedComponents input output windowSize availableMemory
    | "connectedComponents", _ -> failwith "connectedComponents benchmark is currently defined for UInt8 masks only"
    | _, UInt8 -> runTyped<uint8> operation input output radius thresholdValue availableMemory
    | _, UInt16 -> runTyped<uint16> operation input output radius thresholdValue availableMemory
    | _, Float32 -> runTyped<float32> operation input output radius thresholdValue availableMemory

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
