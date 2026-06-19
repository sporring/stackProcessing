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

let private usage () =
    """
TiffPathBench

Commands:
  generate --pixel-type UInt8|UInt16|Float32 --shape WxHxD --output DIR [--write-byte-order native|opposite] [--write-compression none|lzw|deflate|packbits]
  copy --pixel-type UInt8|UInt16|Float32 --input DIR --output DIR [--write-byte-order native|opposite] [--write-compression none|lzw|deflate|packbits] [--repeat N]
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

let private parsePixelType (value: string) =
    match value with
    | "UInt8" | "uint8" -> UInt8
    | "UInt16" | "uint16" -> UInt16
    | "Float32" | "float32" -> Float32
    | _ -> failwith $"unsupported pixel type '{value}'"

let private parseShape (value: string) =
    let parts = value.Split('x', 'X')
    if parts.Length <> 3 then
        invalidArg "shape" $"Expected shape WxHxD, got '{value}'."
    { Width = UInt32.Parse(string parts[0], invariant)
      Height = UInt32.Parse(string parts[1], invariant)
      Depth = UInt32.Parse(string parts[2], invariant) }

let private parseTiffByteOrder (value: string) =
    match value with
    | "native" | "Native" | "NATIVE" -> StackIO.TiffByteOrder.Native
    | "opposite" | "Opposite" | "OPPOSITE" | "swapped" | "Swapped" -> StackIO.TiffByteOrder.Opposite
    | _ -> failwith $"unsupported TIFF byte order '{value}'"

let private parseTiffCompression (value: string) =
    match value with
    | "none" | "None" | "NONE" | "uncompressed" | "Uncompressed" -> StackIO.TiffCompression.None
    | "lzw" | "Lzw" | "LZW" -> StackIO.TiffCompression.Lzw
    | "deflate" | "Deflate" | "DEFLATE" | "zip" | "Zip" | "ZIP" -> StackIO.TiffCompression.Deflate
    | "packbits" | "PackBits" | "PACKBITS" -> StackIO.TiffCompression.PackBits
    | _ -> failwith $"unsupported TIFF compression '{value}'"

let private writeInternalSeconds (elapsed: TimeSpan) =
    let path = Environment.GetEnvironmentVariable("BENCHMARK_INTERNAL_SECONDS_PATH")
    if not (String.IsNullOrWhiteSpace path) then
        File.WriteAllText(path, elapsed.TotalSeconds.ToString("F9", invariant))

let private cleanDirectory path =
    if Directory.Exists path then
        Directory.Delete(path, true)
    Directory.CreateDirectory(path) |> ignore

let private sourcePlan () =
    sourceWithOptimizer false (1024UL * 1024UL * 1024UL * 1024UL)

let private optionsFromArgs opts : StackIO.TiffWriteOptions =
    { Compression = optional "write-compression" "none" opts |> parseTiffCompression
      ByteOrder = optional "write-byte-order" "native" opts |> parseTiffByteOrder }

let private generateTyped<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> shape output options =
    cleanDirectory output
    sourcePlan ()
    |> coordinateX<'T> shape.Width shape.Height shape.Depth
    >=> writeTiffWithOptions<'T> options output ".tiff"
    |> sink

let private copyTyped<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> input output options =
    cleanDirectory output
    sourcePlan ()
    |> read<'T> input ".tiff"
    >=> writeTiffWithOptions<'T> options output ".tiff"
    |> sink

let private generate opts =
    let pixelType = require "pixel-type" opts |> parsePixelType
    let shape = require "shape" opts |> parseShape
    let output = require "output" opts
    let options = optionsFromArgs opts

    match pixelType with
    | UInt8 -> generateTyped<uint8> shape output options
    | UInt16 -> generateTyped<uint16> shape output options
    | Float32 -> generateTyped<float32> shape output options
    0

let private copy opts =
    let pixelType = require "pixel-type" opts |> parsePixelType
    let input = require "input" opts
    let output = require "output" opts
    let options = optionsFromArgs opts
    let repeat = optional "repeat" "1" opts |> Int32.Parse

    let mutable lastElapsed = TimeSpan.Zero
    for i in 1 .. repeat do
        let iterationOutput =
            if repeat = 1 then
                output
            else
                Path.Combine(output, sprintf "repeat_%03d" i)
        let stopwatch = Stopwatch.StartNew()
        match pixelType with
        | UInt8 -> copyTyped<uint8> input iterationOutput options
        | UInt16 -> copyTyped<uint16> input iterationOutput options
        | Float32 -> copyTyped<float32> input iterationOutput options
        stopwatch.Stop()
        lastElapsed <- stopwatch.Elapsed
        let seconds = stopwatch.Elapsed.TotalSeconds.ToString("F6", invariant)
        printfn $"repeat={i},seconds={seconds}"

    writeInternalSeconds lastElapsed
    0

[<EntryPoint>]
let main args =
    try
        match args with
        | [| |] -> usage ()
        | _ when args[0] = "generate" -> args[1..] |> parseArgs |> generate
        | _ when args[0] = "copy" -> args[1..] |> parseArgs |> copy
        | _ -> usage ()
    with ex ->
        eprintfn "%s" ex.Message
        2
