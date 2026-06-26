open System
open System.Buffers
open System.Diagnostics
open System.Globalization
open System.IO
open System.Numerics
open System.Runtime.CompilerServices
open System.Runtime.InteropServices
open System.Text.Json
open System.Text.Json.Nodes
open System.Threading.Tasks
open BitMiracle.LibTiff.Classic
open FSharp.Control
open Image
open StackProcessing
open ZarrNET.Core
open ZarrNET.Core.Nodes
open ZarrNET.Core.OmeZarr.Coordinates
open ZarrNET.Core.Zarr
open ZarrNET.Core.Zarr.Store

type PixelType =
    | UInt8
    | UInt16
    | UInt32
    | Int32
    | Float32

let private unsupportedPixelType context supported pixelType =
    invalidArg "pixelType" $"{context} supports {supported}; got {pixelType}."

type ChunkConvolvePixelType =
    | ChunkUInt8
    | ChunkInt8
    | ChunkUInt16
    | ChunkInt16
    | ChunkFloat32

type ChunkAxisConvolveAxis =
    | AxisX
    | AxisY
    | AxisZ

type ChunkAxisConvolveVariant =
    | AxisNativeGeneric
    | AxisNative1D

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
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- generate --output DIR --shape 512x512x64 --pixel-type UInt8|UInt16|UInt32|Int32|Float32 [--pattern ramp|binary]

Run:
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run --operation copy|threshold|median|median-native-nth|dilate|connectedComponents --pixel-type UInt8|UInt16|Int32|Float32 --input DIR --output DIR [--radius N] [--threshold X] [--window N] [--available-memory BYTES]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-zarr --operation copy|copyThickThin|zarrToTiff|tiffToZarr|threshold|convolve|dilate --pixel-type UInt8|UInt16|Float32 --shape WxHxD --input ZARR_OR_TIFF_DIR --output ZARR_OR_TIFF_DIR [--radius N] [--kernel-size N] [--threshold X] [--workers N] [--chunk-size N] [--available-memory BYTES]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-zarr-convolve-breakdown --variant read|readWrite|readConvolve|convolve --pixel-type UInt8|UInt16|Float32 --shape WxHxD --input ZARR --output ZARR [--kernel-size N] [--workers N] [--available-memory BYTES]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-zarr-direct-copy --pixel-type UInt8|UInt16|Float32 --shape WxHxD --input ZARR --output ZARR
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-zarr-direct-threshold --pixel-type UInt8|UInt16|Float32 --shape WxHxD --input ZARR --output ZARR [--threshold X]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-zarr-direct-threshold-raw --pixel-type UInt8|UInt16|Float32 --shape WxHxD --input ZARR --output ZARR [--threshold X]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-zarr-direct-threshold-intype --pixel-type UInt8|UInt16|Float32 --shape WxHxD --input ZARR --output ZARR [--threshold X]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-zarr-direct-threshold-hotloop --pixel-type UInt8|UInt16|Float32 --input ZARR --variant byte-mask-one|byte-intype-max|byte-intype-one|typed-intype-max|typed-intype-one|typed-copy-intype-max|typed-copy-intype-one [--iterations N] [--threshold X]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-zarr-chunk-copy --pixel-type UInt8|UInt16|Float32 --shape WxHxD --input ZARR --output ZARR [--available-memory BYTES]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-zarr-readonly --pixel-type UInt8|UInt16|Float32 --input ZARR [--available-memory BYTES]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-zarr-writeonly --pixel-type UInt8|UInt16|Float32 --shape WxHxD --output ZARR [--available-memory BYTES]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-tiff-thick-readonly --pixel-type UInt8|UInt16|Float32 --input DIR [--chunk-size N] [--available-memory BYTES]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-tiff-thick-split-drain --pixel-type UInt8|UInt16|Float32 --input DIR [--chunk-size N] [--available-memory BYTES]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-zarr-thick-writeonly --pixel-type UInt8|UInt16|Float32 --shape WxHxD --output ZARR [--chunk-size N] [--available-memory BYTES]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-zarr-thick-writeonly-pattern --pixel-type UInt8 --shape WxHxD --output ZARR [--chunk-size N] [--available-memory BYTES]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-zarr-thick-writeonly-directlocal --pixel-type UInt8|UInt16|Float32 --shape WxHxD --output ZARR [--chunk-size N] [--workers N] [--available-memory BYTES]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-chunk-fft-float32-zarr --shape WxHxD --input DIR --output ZARR [--chunk-size N] [--compression none|blosc] [--available-memory BYTES]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-chunk-fft-xy-float32-zarr --shape WxHxD --input DIR --output ZARR [--chunk-size N] [--workers N] [--available-memory BYTES]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-chunk-fft-z-complex64-zarr --shape WxHxD --input ZARR --output ZARR [--chunk-size N]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-chunk-fft-native-float32-zarr --shape WxHxD --input DIR --output ZARR [--chunk-size N] [--workers N] [--available-memory BYTES]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-fft3d-kernel --shape WxHxD [--variant simpleitk|lowlevel|lowlevel-xy-plan-z|lowlevel-xy-z-plan|lowlevel-3d|lowlevel-r2c-3d|all] [--iterations N]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-chunk-fft3d-stage --shape WxHxD [--variant complex-xy|real-xy|real-xy-roundtrip|all] [--iterations N]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-chunk-fft3d-stage-io --shape WxHxD --input DIR --output DIR [--available-memory BYTES]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-chunk-fft3d-stage-overhead --shape WxHxD [--iterations N]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-chunk-fft3d-spectral-zarr --shape WxHxD --output ZARR [--variant write|read|roundtrip] [--chunk-size N] [--iterations N] [--available-memory BYTES]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-chunk-fft3d-zarr-roundtrip-io --shape WxHxD --input DIR --output DIR [--temp-zarr ZARR] [--chunk-size N] [--available-memory BYTES]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-chunk-fft3d-zarr-subchunked-roundtrip-io --shape WxHxD --input DIR --output DIR [--temp-zarr ZARR] [--chunk-size N] [--available-memory BYTES]

ArrayPool experiment:
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-arraypool --operation copy|threshold|connectedComponents --pixel-type UInt8|UInt16|Float32 --input DIR --output DIR [--threshold X]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-arraypool-slice --operation copy|threshold --pixel-type UInt8|UInt16|Float32 --input DIR --output DIR [--threshold X]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-arraypool-slice-reuse --operation copy|threshold --pixel-type UInt8|UInt16|Float32 --input DIR --output DIR [--threshold X]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-byte-slice-reuse --operation copy|threshold --pixel-type UInt8 --input DIR --output DIR [--threshold X]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-byte-float32-slice-reuse --operation copy|threshold --pixel-type Float32 --input DIR --output DIR [--threshold X]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-libtiff-direct-copy --operation copy --pixel-type UInt8|UInt16|Float32 --input DIR --output DIR
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-libtiff-direct-threshold --operation threshold --pixel-type UInt8|UInt16|Float32 --input DIR --output DIR [--threshold X]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-libtiff-direct-threshold-intype --operation threshold --pixel-type UInt8|UInt16|Float32 --input DIR --output DIR [--threshold X]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-libtiff-direct-threshold-hotloop --pixel-type UInt8|UInt16|Float32 --input DIR --variant byte-mask-one|byte-intype-max|byte-intype-one|typed-intype-max|typed-intype-one|typed-copy-intype-max|typed-copy-intype-one [--iterations N] [--threshold X]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-chunk-threshold-parallel --pixel-type UInt8|UInt16|Float32 --input DIR --output DIR [--threshold X] [--workers N] [--available-memory BYTES]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-chunk-histogram --pixel-type UInt8|UInt16|Float32 --input DIR --variant dense|sparse|leftedges [--window-size N] [--bins N] [--available-memory BYTES]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-chunk-simd-reductions --pixel-type UInt8|UInt16|Float32|Float64 --shape WxHxD [--variant computeStats-current|sum-scalar|moments-scalar|sum-vector|moments-vector|sum-vector-accurate|moments-vector-accurate] [--iterations N]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-chunk-cast --shape WxHxD [--variant uint8-to-float32|uint16-to-float32|float32-to-uint8|float32-to-uint16] [--iterations N]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-chunk-pixelwise-float32 --shape WxHxD [--variant scalar-add|vector-add|scalar-mul|vector-mul|scalar-threshold|vector-threshold|scalar-pair-add|vector-pair-add|scalar-pair-mul|vector-pair-mul|scalar-absdiff|vector-absdiff|scalar-blend|vector-blend] [--iterations N]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-chunk-structure-tensor-layout --shape WxHxD [--variant aos-outer|soa-outer|soa-outer-vector|components-outer-vector|aos-smooth|soa-smooth-vector|components-smooth-vector|aos-eigenvalues|soa-eigenvalues|components-eigenvalues|components-eigenvalues-jacobi6|components-eigenvalues-jacobi8|components-eigensystem|components-eigensystem-chunk-algebra|components-eigensystem-jacobi-alloc|aos-pipeline|soa-pipeline-vector|components-pipeline-vector|components-pipeline-jacobi6|components-pipeline-jacobi8|components-pipeline-eigensystem|components-pipeline-eigensystem-chunk-algebra|components-pipeline-eigensystem-jacobi-alloc|aos-to-soa|soa-to-aos|all] [--iterations N]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-chunk-connected-components --input DIR [--output DIR] [--threshold X] [--available-memory BYTES]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-chunk-dilate --input DIR --output DIR [--radius N] [--threshold X] [--workers N] [--available-memory BYTES]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-chunk-convolve --pixel-type UInt8|Int8|UInt16|Int16|Float32 --input DIR --output DIR [--kernel-size N] [--workers N] [--available-memory BYTES]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-chunk-axis-convolve-kernel --shape WxHxD [--kernel-size N] [--iterations N] [--axis x|y|z|all] [--variant native-generic|native-1d|all]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-threshold-kernel --pixel-type UInt8|UInt16|Float32 --shape WxHxD --output-type mask|intype [--threshold X]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-libtiff-strip-copy --operation copy --pixel-type UInt8|UInt16|Float32 --input DIR --output DIR
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-libtiff-raw-strip-copy --operation copy --pixel-type UInt8|UInt16|Float32 --input DIR --output DIR
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-bitmiracle-raw-strip-copy --operation copy --pixel-type UInt8|UInt16|Float32 --input DIR --output DIR
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-native-libtiff-raw-strip-copy --operation copy --pixel-type UInt8|UInt16|Float32 --input DIR --output DIR
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-native-libtiff-raw-strip-chunk-copy --operation copy --pixel-type UInt8 --input DIR --output DIR
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-native-libtiff-raw-strip-volume-chunk-copy --operation copy --pixel-type UInt8 --input DIR --output DIR
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-stack-read-write --pixel-type UInt8|UInt16|Float32 --input DIR --output DIR [--available-memory BYTES] [--debug-level N]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-stack-tiff-path-copy --pixel-type UInt8|UInt16|Float32 --input DIR --output DIR [--write-byte-order native|opposite] [--write-compression none|lzw|deflate|packbits] [--available-memory BYTES] [--debug-level N]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-native-libtiff-raw-strip-readonly --pixel-type UInt8|UInt16|Float32 --input DIR
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-native-libtiff-raw-strip-chunk-readonly --pixel-type UInt8 --input DIR
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-native-libtiff-raw-strip-writeonly --pixel-type UInt8|UInt16|Float32 --input DIR --output DIR
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-native-libtiff-raw-strip-chunk-writeonly --pixel-type UInt8 --input DIR --output DIR
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-native-libtiff-scanline-copy --operation copy --pixel-type UInt8|UInt16|Float32 --input DIR --output DIR
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-tifflibrary-raw-strip-copy --operation copy --pixel-type UInt8|UInt16|Float32 --input DIR --output DIR
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-imagesharp-copy --operation copy --pixel-type UInt8|UInt16 --input DIR --output DIR
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-libtiff-direct-readonly --pixel-type UInt8|UInt16|Float32 --input DIR
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-libtiff-direct-writeonly --pixel-type UInt8|UInt16|Float32 --input DIR --output DIR
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

let private optionalValue name (opts: Map<string,string>) =
    opts.TryFind name

let private writeInternalSeconds (elapsed: TimeSpan) =
    let path = Environment.GetEnvironmentVariable("BENCHMARK_INTERNAL_SECONDS_PATH")
    if not (String.IsNullOrWhiteSpace path) then
        File.WriteAllText(path, elapsed.TotalSeconds.ToString("F9", invariant))

let private benchmarkSource availableMemory =
    sourceWithOptimizer false availableMemory

let private benchmarkSourceWithDebug debugLevel availableMemory =
    if debugLevel >= 2u then
        debug debugLevel false availableMemory
    else
        benchmarkSource availableMemory

let private runTask (task: Threading.Tasks.Task<'T>) : 'T =
    task.GetAwaiter().GetResult()

let private runUnitTask (task: Threading.Tasks.Task) : unit =
    task.GetAwaiter().GetResult()

let private parsePixelType value =
    match value with
    | "UInt8" | "uint8" -> UInt8
    | "UInt16" | "uint16" -> UInt16
    | "UInt32" | "uint32" -> UInt32
    | "Int32" | "int32" | "Int" | "int" -> Int32
    | "Float32" | "float32" -> Float32
    | _ -> failwith $"unsupported pixel type '{value}'"

let private parseTiffByteOrder value =
    match value with
    | "native" | "Native" | "NATIVE" -> StackIO.TiffByteOrder.Native
    | "opposite" | "Opposite" | "OPPOSITE" | "swapped" | "Swapped" -> StackIO.TiffByteOrder.Opposite
    | _ -> failwith $"unsupported TIFF byte order '{value}'"

let private parseTiffCompression value =
    match value with
    | "none" | "None" | "NONE" | "uncompressed" | "Uncompressed" -> StackIO.TiffCompression.None
    | "lzw" | "Lzw" | "LZW" -> StackIO.TiffCompression.Lzw
    | "deflate" | "Deflate" | "DEFLATE" | "zip" | "Zip" | "ZIP" -> StackIO.TiffCompression.Deflate
    | "packbits" | "PackBits" | "PACKBITS" -> StackIO.TiffCompression.PackBits
    | _ -> failwith $"unsupported TIFF compression '{value}'"

let private parseChunkConvolvePixelType value =
    match value with
    | "UInt8" | "uint8" -> ChunkUInt8
    | "Int8" | "int8" -> ChunkInt8
    | "UInt16" | "uint16" -> ChunkUInt16
    | "Int16" | "int16" -> ChunkInt16
    | "Float32" | "float32" -> ChunkFloat32
    | _ -> failwith $"unsupported chunk convolve pixel type '{value}'"

let private parseChunkAxisConvolveAxis value =
    match value with
    | "x" | "X" -> [ AxisX ]
    | "y" | "Y" -> [ AxisY ]
    | "z" | "Z" -> [ AxisZ ]
    | "all" | "All" | "ALL" -> [ AxisX; AxisY; AxisZ ]
    | _ -> failwith $"unsupported chunk axis convolution axis '{value}'"

let private parseChunkAxisConvolveVariant value =
    match value with
    | "native-generic" | "NativeGeneric" | "generic" | "native" | "Native" -> [ AxisNativeGeneric ]
    | "native-1d" | "Native1D" | "specialized" | "specialised" -> [ AxisNative1D ]
    | "all" | "All" | "ALL" -> [ AxisNativeGeneric; AxisNative1D ]
    | _ -> failwith $"unsupported chunk axis convolution variant '{value}'"

let private parseZarrCompression value =
    match value with
    | "none" | "None" | "NONE" | "uncompressed" | "Uncompressed" -> ZarrCompression.None
    | "blosc" | "Blosc" | "blosc-lz4" | "BloscLz4" | "lz4" -> ZarrCompression.BloscLz4
    | _ -> failwith $"unsupported Zarr compression '{value}'"

let private zarrDataType pixelType =
    match pixelType with
    | UInt8 -> "uint8"
    | UInt16 -> "uint16"
    | UInt32 -> "uint32"
    | Int32 -> "int32"
    | Float32 -> "float32"

let private parseShape (value: string) =
    let parts = value.Split('x', StringSplitOptions.RemoveEmptyEntries)
    if parts.Length <> 3 then
        failwith $"shape must be WxHxD, got '{value}'"
    { Width = UInt32.Parse(parts[0], invariant)
      Height = UInt32.Parse(parts[1], invariant)
      Depth = UInt32.Parse(parts[2], invariant) }

let private precleanedDirectories = System.Collections.Generic.HashSet<string>(StringComparer.Ordinal)
let private precleanedDirectoriesLock = obj()

let private fullPath path =
    Path.GetFullPath path

let private registerExternallyPrecleanedDirectories () =
    let value = Environment.GetEnvironmentVariable("BENCHMARK_PRECLEANED_OUTPUTS")
    if not (String.IsNullOrWhiteSpace value) then
        let parts = value.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries)
        lock precleanedDirectoriesLock (fun () ->
            for path in parts do
                precleanedDirectories.Add(fullPath path) |> ignore)

registerExternallyPrecleanedDirectories ()

let private cleanDirectoryNow path =
    if Directory.Exists path then
        Directory.Delete(path, true)
    Directory.CreateDirectory path |> ignore

let private consumePrecleanedDirectory normalized =
    lock precleanedDirectoriesLock (fun () ->
        if precleanedDirectories.Contains normalized then
            precleanedDirectories.Remove normalized |> ignore
            true
        else
            false)

let private ensureCleanDirectory path =
    let normalized = fullPath path
    if consumePrecleanedDirectory normalized then
        Directory.CreateDirectory normalized |> ignore
    else
        cleanDirectoryNow normalized

let private precleanDirectoryForTimedRun path =
    let normalized = fullPath path
    if consumePrecleanedDirectory normalized then
        Directory.CreateDirectory normalized |> ignore
    else
        cleanDirectoryNow normalized
        lock precleanedDirectoriesLock (fun () ->
            precleanedDirectories.Add normalized |> ignore)

let private outputFile outputDir z =
    Path.Combine(outputDir, sprintf "slice_%05d.tiff" z)

type private PooledVolume<'T>(width: uint, height: uint, depth: uint, buffer: 'T[], length: int, name: string) =
    let mutable refCount = 1
    let mutable returned = false

    member _.Width = width
    member _.Height = height
    member _.Depth = depth
    member _.Buffer = buffer
    member _.Length = length
    member _.Name = name
    member _.Span = buffer.AsSpan(0, length)

    member _.incRefCount() =
        if returned then
            invalidOp $"Cannot increment reference count for returned pooled volume '{name}'."
        refCount <- refCount + 1

    member _.decRefCount() =
        if returned then
            ()
        else
            refCount <- refCount - 1
            if refCount < 0 then
                invalidOp $"Reference count became negative for pooled volume '{name}'."
            elif refCount = 0 then
                returned <- true
                ArrayPool<'T>.Shared.Return(buffer, RuntimeHelpers.IsReferenceOrContainsReferences<'T>())

let private rentVolume<'T> width height depth name =
    let length64 = uint64 width * uint64 height * uint64 depth
    if length64 > uint64 Int32.MaxValue then
        invalidArg "shape" $"ArrayPool experiment currently expects fewer than {Int32.MaxValue} elements per volume; got {length64}."
    let length = int length64
    let buffer = ArrayPool<'T>.Shared.Rent(length)
    PooledVolume<'T>(width, height, depth, buffer, length, name)

let private scalarTiffLayout<'T> () =
    let t = typeof<'T>
    if t = typeof<uint8> then 8, SampleFormat.UINT, 1
    elif t = typeof<int8> then 8, SampleFormat.INT, 1
    elif t = typeof<uint16> then 16, SampleFormat.UINT, 2
    elif t = typeof<int16> then 16, SampleFormat.INT, 2
    elif t = typeof<int32> then 32, SampleFormat.INT, 4
    elif t = typeof<float32> then 32, SampleFormat.IEEEFP, 4
    else
        invalidArg "T" $"ArrayPool benchmark supports UInt8, Int8, UInt16, Int16, Int32, and Float32 TIFF stacks; got {t.Name}."

let private scalarTiffLayoutForPixelType pixelType =
    match pixelType with
    | UInt8 -> 8, SampleFormat.UINT, 1
    | UInt16 -> 16, SampleFormat.UINT, 2
    | UInt32 -> 32, SampleFormat.UINT, 4
    | Int32 -> 32, SampleFormat.INT, 4
    | Float32 -> 32, SampleFormat.IEEEFP, 4

let private tiffFieldInt (tiff: Tiff) tag fallback =
    let field = tiff.GetField(tag)
    if isNull field || field.Length = 0 then fallback else field[0].ToInt()

let private tiffFieldIntDefaulted (tiff: Tiff) tag fallback =
    let field = tiff.GetFieldDefaulted(tag)
    if isNull field || field.Length = 0 then fallback else field[0].ToInt()

let private stackTiffFiles inputDir =
    if not (Directory.Exists inputDir) then
        invalidOp $"Input stack directory does not exist: {inputDir}"

    Directory.EnumerateFiles(inputDir)
    |> Seq.filter (fun path ->
        path.EndsWith(".tif", StringComparison.OrdinalIgnoreCase)
        || path.EndsWith(".tiff", StringComparison.OrdinalIgnoreCase))
    |> Seq.sort
    |> Seq.toArray

let private readTiffPageBytes (fileName: string) =
    use tiff = Tiff.Open(fileName, "r")
    if isNull tiff then
        invalidOp $"Could not open '{fileName}' for TIFF reading."

    let width = uint (tiffFieldInt tiff TiffTag.IMAGEWIDTH 0)
    let height = uint (tiffFieldInt tiff TiffTag.IMAGELENGTH 0)
    if width = 0u || height = 0u then
        invalidOp $"TIFF slice '{fileName}' has invalid page dimensions {width}x{height}."

    let bitsPerSample = tiffFieldIntDefaulted tiff TiffTag.BITSPERSAMPLE 1
    let samplesPerPixel = tiffFieldIntDefaulted tiff TiffTag.SAMPLESPERPIXEL 1
    if samplesPerPixel <> 1 then
        invalidOp $"TIFF slice '{fileName}' has {samplesPerPixel} samples per pixel; the ArrayPool benchmark expects scalar images."

    let sampleFormat =
        tiffFieldIntDefaulted tiff TiffTag.SAMPLEFORMAT (int SampleFormat.UINT)
        |> enum<SampleFormat>

    let actualBytesPerSample =
        match sampleFormat, bitsPerSample with
        | SampleFormat.UINT, 8
        | SampleFormat.INT, 8 -> 1
        | SampleFormat.UINT, 16
        | SampleFormat.INT, 16 -> 2
        | SampleFormat.INT, 32 -> 4
        | SampleFormat.IEEEFP, 32 -> 4
        | SampleFormat.IEEEFP, 64 -> 8
        | _ -> invalidOp $"Unsupported TIFF scalar layout in '{fileName}': {bitsPerSample}-bit {sampleFormat}."

    let rowBytes = int width * actualBytesPerSample
    let scanlineSize = max rowBytes (tiff.ScanlineSize())
    let scanline = Array.zeroCreate<byte> scanlineSize
    let pageBytes = Array.zeroCreate<byte> (rowBytes * int height)

    for row in 0 .. int height - 1 do
        if not (tiff.ReadScanline(scanline, row)) then
            invalidOp $"Failed to read TIFF scanline {row} from '{fileName}'."
        Buffer.BlockCopy(scanline, 0, pageBytes, row * rowBytes, rowBytes)

    width, height, bitsPerSample, sampleFormat, actualBytesPerSample, pageBytes

let private readArrayPoolTiffSlice<'T> fileName name =
    let expectedBits, expectedFormat, expectedBytesPerSample = scalarTiffLayout<'T> ()
    use tiff = Tiff.Open(fileName, "r")
    if isNull tiff then
        invalidOp $"Could not open '{fileName}' for TIFF reading."

    let width = uint (tiffFieldInt tiff TiffTag.IMAGEWIDTH 0)
    let height = uint (tiffFieldInt tiff TiffTag.IMAGELENGTH 0)
    if width = 0u || height = 0u then
        invalidOp $"TIFF slice '{fileName}' has invalid page dimensions {width}x{height}."

    let bitsPerSample = tiffFieldIntDefaulted tiff TiffTag.BITSPERSAMPLE 1
    let samplesPerPixel = tiffFieldIntDefaulted tiff TiffTag.SAMPLESPERPIXEL 1
    if samplesPerPixel <> 1 then
        invalidOp $"TIFF slice '{fileName}' has {samplesPerPixel} samples per pixel; the ArrayPool benchmark expects scalar images."

    let sampleFormat =
        tiffFieldIntDefaulted tiff TiffTag.SAMPLEFORMAT (int SampleFormat.UINT)
        |> enum<SampleFormat>

    if bitsPerSample <> expectedBits || sampleFormat <> expectedFormat then
        invalidOp $"Input slice '{fileName}' does not match {typeof<'T>.Name}: got {bitsPerSample}-bit {sampleFormat}."

    let rowBytes = int width * expectedBytesPerSample
    let scanlineSize = max rowBytes (tiff.ScanlineSize())
    let scanline = ArrayPool<byte>.Shared.Rent(scanlineSize)
    let slice = rentVolume<'T> width height 1u name
    try
        try
            for row in 0 .. int height - 1 do
                if not (tiff.ReadScanline(scanline, row)) then
                    invalidOp $"Failed to read TIFF scanline {row} from '{fileName}'."
                Buffer.BlockCopy(scanline, 0, slice.Buffer, row * rowBytes, rowBytes)
            slice
        with
        | ex ->
            slice.decRefCount()
            raise ex
    finally
        ArrayPool<byte>.Shared.Return(scanline)

let private inspectTiffSlice<'T> fileName =
    let expectedBits, expectedFormat, expectedBytesPerSample = scalarTiffLayout<'T> ()
    use tiff = Tiff.Open(fileName, "r")
    if isNull tiff then
        invalidOp $"Could not open '{fileName}' for TIFF reading."

    let width = uint (tiffFieldInt tiff TiffTag.IMAGEWIDTH 0)
    let height = uint (tiffFieldInt tiff TiffTag.IMAGELENGTH 0)
    if width = 0u || height = 0u then
        invalidOp $"TIFF slice '{fileName}' has invalid page dimensions {width}x{height}."

    let bitsPerSample = tiffFieldIntDefaulted tiff TiffTag.BITSPERSAMPLE 1
    let samplesPerPixel = tiffFieldIntDefaulted tiff TiffTag.SAMPLESPERPIXEL 1
    if samplesPerPixel <> 1 then
        invalidOp $"TIFF slice '{fileName}' has {samplesPerPixel} samples per pixel; the ArrayPool benchmark expects scalar images."

    let sampleFormat =
        tiffFieldIntDefaulted tiff TiffTag.SAMPLEFORMAT (int SampleFormat.UINT)
        |> enum<SampleFormat>

    if bitsPerSample <> expectedBits || sampleFormat <> expectedFormat then
        invalidOp $"Input slice '{fileName}' does not match {typeof<'T>.Name}: got {bitsPerSample}-bit {sampleFormat}."

    let rowBytes = int width * expectedBytesPerSample
    width, height, rowBytes, max rowBytes (tiff.ScanlineSize())

let private compressionName (compression: Compression) =
    compression.ToString()

[<Struct; StructLayout(LayoutKind.Sequential)>]
type private NativeTiffInfo =
    val mutable Width: uint32
    val mutable Height: uint32
    val mutable RowsPerStrip: uint32
    val mutable Strips: uint32
    val mutable BitsPerSample: uint16
    val mutable SampleFormat: uint16
    val mutable SamplesPerPixel: uint16
    val mutable PlanarConfig: uint16
    val mutable Compression: uint16
    val mutable IsTiled: int32
    val mutable IsByteSwapped: int32
    val mutable PageBytes: uint64
    val mutable RawPageBytes: uint64

module private NativeLibTiff =
    [<Literal>]
    let Ok = 0

    [<Literal>]
    let CompressionNone = 1us

    [<Literal>]
    let PlanarConfigContig = 1us

    [<Literal>]
    let SampleFormatUInt = 1us

    [<Literal>]
    let SampleFormatInt = 2us

    [<Literal>]
    let SampleFormatIeeeFp = 3us

    [<DllImport("sp_libtiff_shim", EntryPoint = "sp_tiff_read_info", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)>]
    extern int readInfo(string path, NativeTiffInfo& info)

    [<DllImport("sp_libtiff_shim", EntryPoint = "sp_tiff_read_raw_page", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)>]
    extern int readRawPage(string path, byte[] buffer, UIntPtr capacity, uint64& bytesRead)

    [<DllImport("sp_libtiff_shim", EntryPoint = "sp_tiff_read_raw_page_into", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)>]
    extern int readRawPageInto(string path, byte[] buffer, UIntPtr bufferOffset, UIntPtr capacity, uint64& bytesRead)

    [<DllImport("sp_libtiff_shim", EntryPoint = "sp_tiff_read_scanline_page", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)>]
    extern int readScanlinePage(string path, byte[] buffer, UIntPtr capacity, uint64& bytesRead)

    [<DllImport("sp_libtiff_shim", EntryPoint = "sp_tiff_write_raw_page", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)>]
    extern int writeRawPage(string path, byte[] buffer, UIntPtr count, uint32 width, uint32 height, uint16 bitsPerSample, uint16 sampleFormat)

    [<DllImport("sp_libtiff_shim", EntryPoint = "sp_tiff_write_raw_page_from", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)>]
    extern int writeRawPageFrom(string path, byte[] buffer, UIntPtr bufferOffset, UIntPtr count, UIntPtr capacity, uint32 width, uint32 height, uint16 bitsPerSample, uint16 sampleFormat)

    [<DllImport("sp_libtiff_shim", EntryPoint = "sp_tiff_write_scanline_page", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)>]
    extern int writeScanlinePage(string path, byte[] buffer, UIntPtr count, uint32 width, uint32 height, uint16 bitsPerSample, uint16 sampleFormat)

    let describeStatus status =
        match status with
        | 0 -> "ok"
        | -1 -> "open failed"
        | -2 -> "missing required field"
        | -3 -> "unsupported layout"
        | -4 -> "buffer too small"
        | -5 -> "I/O failed"
        | -6 -> "size overflow"
        | _ -> $"native libtiff shim error {status}"

    let failStatus operation fileName status =
        invalidOp $"Native libtiff {operation} failed for '{fileName}': {describeStatus status}."

let private scalarNativeTiffLayoutForPixelType pixelType =
    match pixelType with
    | UInt8 -> 8us, NativeLibTiff.SampleFormatUInt, 1
    | UInt16 -> 16us, NativeLibTiff.SampleFormatUInt, 2
    | Int32 -> 32us, NativeLibTiff.SampleFormatInt, 4
    | Float32 -> 32us, NativeLibTiff.SampleFormatIeeeFp, 4
    | _ -> unsupportedPixelType "Native libtiff scalar layout" "UInt8, UInt16, Int32, and Float32" pixelType

let private scalarTiffLibraryLayoutForPixelType pixelType =
    match pixelType with
    | UInt8 -> 8us, 1us, 1
    | UInt16 -> 16us, 1us, 2
    | Int32 -> 32us, 2us, 4
    | Float32 -> 32us, 3us, 4
    | _ -> unsupportedPixelType "TiffLibrary scalar layout" "UInt8, UInt16, Int32, and Float32" pixelType

let private awaitTask (task: System.Threading.Tasks.Task<'T>) =
    task.GetAwaiter().GetResult()

let private awaitUnitTask (task: System.Threading.Tasks.Task) =
    task.GetAwaiter().GetResult()

let private awaitValueTask (task: ValueTask<'T>) =
    task.GetAwaiter().GetResult()

let private awaitUnitValueTask (task: ValueTask) =
    task.GetAwaiter().GetResult()

let private inspectOpenDirectTiffSlice pixelType fileName (tiff: Tiff) =
    let expectedBits, expectedFormat, expectedBytesPerSample = scalarTiffLayoutForPixelType pixelType

    let width = uint (tiffFieldInt tiff TiffTag.IMAGEWIDTH 0)
    let height = uint (tiffFieldInt tiff TiffTag.IMAGELENGTH 0)
    if width = 0u || height = 0u then
        invalidOp $"TIFF slice '{fileName}' has invalid page dimensions {width}x{height}."

    let bitsPerSample = tiffFieldIntDefaulted tiff TiffTag.BITSPERSAMPLE 1
    let samplesPerPixel = tiffFieldIntDefaulted tiff TiffTag.SAMPLESPERPIXEL 1
    if samplesPerPixel <> 1 then
        invalidOp $"TIFF slice '{fileName}' has {samplesPerPixel} samples per pixel; the direct LibTiff copy benchmark expects scalar images."

    let sampleFormat =
        tiffFieldIntDefaulted tiff TiffTag.SAMPLEFORMAT (int SampleFormat.UINT)
        |> enum<SampleFormat>

    if bitsPerSample <> expectedBits || sampleFormat <> expectedFormat then
        invalidOp $"Input slice '{fileName}' does not match {pixelType}: got {bitsPerSample}-bit {sampleFormat}."

    let compression =
        tiffFieldIntDefaulted tiff TiffTag.COMPRESSION (int Compression.NONE)
        |> enum<Compression>

    if compression <> Compression.NONE then
        invalidOp $"Input slice '{fileName}' is compressed with {compressionName compression}; the direct LibTiff copy benchmark is intentionally uncompressed."

    let rowBytes = int width * expectedBytesPerSample
    width, height, rowBytes, max rowBytes (tiff.ScanlineSize())

let private inspectOpenStripTiffSlice pixelType fileName (tiff: Tiff) =
    let width, height, rowBytes, _scanlineSize = inspectOpenDirectTiffSlice pixelType fileName tiff
    if tiff.IsTiled() then
        invalidOp $"Input slice '{fileName}' is tiled; the strip LibTiff copy benchmark expects stripped TIFF slices."

    let planarConfig =
        tiffFieldIntDefaulted tiff TiffTag.PLANARCONFIG (int PlanarConfig.CONTIG)
        |> enum<PlanarConfig>

    if planarConfig <> PlanarConfig.CONTIG then
        invalidOp $"Input slice '{fileName}' has planar configuration {planarConfig}; the strip LibTiff copy benchmark expects contiguous scalar images."

    let strips = tiff.NumberOfStrips()
    if strips < 1 then
        invalidOp $"Input slice '{fileName}' has no readable strips."

    let rowsPerStrip = uint (tiffFieldIntDefaulted tiff TiffTag.ROWSPERSTRIP (int height))
    if rowsPerStrip = 0u then
        invalidOp $"Input slice '{fileName}' has invalid ROWSPERSTRIP=0."

    let stripBytes = tiff.StripSize()
    let pageBytes = rowBytes * int height
    if stripBytes <= 0 then
        invalidOp $"Input slice '{fileName}' has invalid decoded strip size {stripBytes}."

    width, height, rowBytes, pageBytes, strips, stripBytes, rowsPerStrip

let private inspectOpenRawStripTiffSlice pixelType fileName (tiff: Tiff) =
    let width, height, rowBytes, pageBytes, strips, _stripBytes, rowsPerStrip = inspectOpenStripTiffSlice pixelType fileName tiff
    if tiff.IsByteSwapped() then
        invalidOp $"Input slice '{fileName}' has non-native byte order; the raw-strip LibTiff copy benchmark bypasses byte swapping."

    let rawStripSizes =
        Array.init strips (fun strip ->
            let size = tiff.RawStripSize(strip)
            if size <= 0L then
                invalidOp $"Input slice '{fileName}' has invalid raw strip size {size} for strip {strip}."
            if size > int64 Int32.MaxValue then
                invalidOp $"Input slice '{fileName}' has raw strip {strip} larger than Int32.MaxValue bytes."
            int size)

    let rawPageBytes = rawStripSizes |> Array.sum
    if rawPageBytes <> pageBytes then
        invalidOp $"Input slice '{fileName}' has {rawPageBytes} raw strip bytes, expected {pageBytes} decoded bytes for an uncompressed scalar page."

    width, height, rowBytes, pageBytes, strips, rawStripSizes, rowsPerStrip

let private inspectDirectTiffSlice pixelType fileName =
    use tiff = Tiff.Open(fileName, "r")
    if isNull tiff then
        invalidOp $"Could not open '{fileName}' for TIFF reading."

    inspectOpenDirectTiffSlice pixelType fileName tiff

let private readDirectByteTiffSliceInto pixelType fileName expectedWidth expectedHeight expectedRowBytes (pageBuffer: byte[]) (scratch: byte[]) =
    use tiff = Tiff.Open(fileName, "r")
    if isNull tiff then
        invalidOp $"Could not open '{fileName}' for TIFF reading."

    let width, height, rowBytes, scanlineSize = inspectOpenDirectTiffSlice pixelType fileName tiff
    if width <> expectedWidth || height <> expectedHeight then
        invalidOp $"Input slice '{fileName}' has shape {width}x{height}, expected {expectedWidth}x{expectedHeight}."
    if rowBytes <> expectedRowBytes then
        invalidOp $"Input slice '{fileName}' has {rowBytes} logical bytes per row, expected {expectedRowBytes}."
    if pageBuffer.Length < rowBytes * int height then
        invalidArg "pageBuffer" $"Direct TIFF page buffer too small: need {rowBytes * int height}, got {pageBuffer.Length}."
    if scratch.Length < scanlineSize then
        invalidArg "scratch" $"Direct TIFF scratch buffer too small: need {scanlineSize}, got {scratch.Length}."

    if scanlineSize <= rowBytes then
        for row in 0 .. int expectedHeight - 1 do
            if not (tiff.ReadScanline(pageBuffer, row * rowBytes, row, int16 0)) then
                invalidOp $"Failed to read TIFF scanline {row} from '{fileName}'."
    else
        for row in 0 .. int expectedHeight - 1 do
            if not (tiff.ReadScanline(scratch, row)) then
                invalidOp $"Failed to read TIFF scanline {row} from '{fileName}'."
            Buffer.BlockCopy(scratch, 0, pageBuffer, row * rowBytes, rowBytes)

let private writeDirectByteTiffPage pixelType fileName width height rowBytes (pageBuffer: byte[]) =
    let bitsPerSample, sampleFormat, bytesPerSample = scalarTiffLayoutForPixelType pixelType
    if rowBytes <> int width * bytesPerSample then
        invalidArg "rowBytes" $"Expected {int width * bytesPerSample} row bytes for {pixelType}, got {rowBytes}."

    use tiff = Tiff.Open(fileName, ImageIO.tiffWriteMode fileName)
    if isNull tiff then
        invalidOp $"Could not open '{fileName}' for TIFF writing."

    tiff.SetField(TiffTag.IMAGEWIDTH, int width) |> ignore
    tiff.SetField(TiffTag.IMAGELENGTH, int height) |> ignore
    tiff.SetField(TiffTag.SAMPLESPERPIXEL, 1) |> ignore
    tiff.SetField(TiffTag.BITSPERSAMPLE, bitsPerSample) |> ignore
    tiff.SetField(TiffTag.SAMPLEFORMAT, sampleFormat) |> ignore
    tiff.SetField(TiffTag.PHOTOMETRIC, Photometric.MINISBLACK) |> ignore
    tiff.SetField(TiffTag.PLANARCONFIG, PlanarConfig.CONTIG) |> ignore
    tiff.SetField(TiffTag.ROWSPERSTRIP, int height) |> ignore
    tiff.SetField(TiffTag.COMPRESSION, Compression.NONE) |> ignore

    for row in 0 .. int height - 1 do
        if not (tiff.WriteScanline(pageBuffer, row * rowBytes, row, int16 0)) then
            invalidOp $"Failed to write TIFF scanline {row} to '{fileName}'."

    if not (tiff.WriteDirectory()) then
        invalidOp $"Failed to write TIFF directory to '{fileName}'."

let private readEncodedStripTiffSliceInto pixelType fileName expectedWidth expectedHeight expectedRowBytes expectedPageBytes expectedStrips (pageBuffer: byte[]) =
    use tiff = Tiff.Open(fileName, "r")
    if isNull tiff then
        invalidOp $"Could not open '{fileName}' for TIFF reading."

    let width, height, rowBytes, pageBytes, strips, stripBytes, _rowsPerStrip = inspectOpenStripTiffSlice pixelType fileName tiff
    if width <> expectedWidth || height <> expectedHeight then
        invalidOp $"Input slice '{fileName}' has shape {width}x{height}, expected {expectedWidth}x{expectedHeight}."
    if rowBytes <> expectedRowBytes || pageBytes <> expectedPageBytes then
        invalidOp $"Input slice '{fileName}' layout changed: rowBytes={rowBytes}, pageBytes={pageBytes}; expected {expectedRowBytes}, {expectedPageBytes}."
    if strips <> expectedStrips then
        invalidOp $"Input slice '{fileName}' has {strips} strips, expected {expectedStrips}."
    if pageBuffer.Length < pageBytes then
        invalidArg "pageBuffer" $"Direct TIFF page buffer too small: need {pageBytes}, got {pageBuffer.Length}."

    let mutable offset = 0
    for strip in 0 .. strips - 1 do
        let remaining = pageBytes - offset
        let requested = min stripBytes remaining
        let bytesRead = tiff.ReadEncodedStrip(strip, pageBuffer, offset, requested)
        if bytesRead < 0 then
            invalidOp $"ReadEncodedStrip failed for strip {strip} from '{fileName}'."
        if bytesRead > remaining then
            invalidOp $"ReadEncodedStrip read {bytesRead} decoded bytes for strip {strip} from '{fileName}', but only {remaining} bytes remain."
        offset <- offset + bytesRead

    if offset <> pageBytes then
        invalidOp $"Decoded strip reads produced {offset} bytes from '{fileName}', expected {pageBytes}."

let private writeEncodedStripTiffPage pixelType fileName width height rowBytes pageBytes strips stripBytes rowsPerStrip (pageBuffer: byte[]) =
    let bitsPerSample, sampleFormat, bytesPerSample = scalarTiffLayoutForPixelType pixelType
    if rowBytes <> int width * bytesPerSample then
        invalidArg "rowBytes" $"Expected {int width * bytesPerSample} row bytes for {pixelType}, got {rowBytes}."
    if pageBytes <> rowBytes * int height then
        invalidArg "pageBytes" $"Expected {rowBytes * int height} page bytes for {pixelType}, got {pageBytes}."
    if strips < 1 then
        invalidArg "strips" "Expected at least one output strip."
    if stripBytes < 1 then
        invalidArg "stripBytes" "Expected positive output strip byte count."

    use tiff = Tiff.Open(fileName, ImageIO.tiffWriteMode fileName)
    if isNull tiff then
        invalidOp $"Could not open '{fileName}' for TIFF writing."

    tiff.SetField(TiffTag.IMAGEWIDTH, int width) |> ignore
    tiff.SetField(TiffTag.IMAGELENGTH, int height) |> ignore
    tiff.SetField(TiffTag.SAMPLESPERPIXEL, 1) |> ignore
    tiff.SetField(TiffTag.BITSPERSAMPLE, bitsPerSample) |> ignore
    tiff.SetField(TiffTag.SAMPLEFORMAT, sampleFormat) |> ignore
    tiff.SetField(TiffTag.PHOTOMETRIC, Photometric.MINISBLACK) |> ignore
    tiff.SetField(TiffTag.PLANARCONFIG, PlanarConfig.CONTIG) |> ignore
    tiff.SetField(TiffTag.ROWSPERSTRIP, int rowsPerStrip) |> ignore
    tiff.SetField(TiffTag.COMPRESSION, Compression.NONE) |> ignore

    let mutable offset = 0
    for strip in 0 .. strips - 1 do
        let remaining = pageBytes - offset
        let count = min stripBytes remaining
        let written = tiff.WriteEncodedStrip(strip, pageBuffer, offset, count)
        if written < 0 then
            invalidOp $"Failed to write encoded TIFF strip {strip} to '{fileName}'."
        offset <- offset + count

    if offset <> pageBytes then
        invalidOp $"Wrote {offset} decoded strip bytes to '{fileName}', expected {pageBytes}."

    if not (tiff.WriteDirectory()) then
        invalidOp $"Failed to write TIFF directory to '{fileName}'."

let private readRawStripTiffSliceInto pixelType fileName expectedWidth expectedHeight expectedRowBytes expectedPageBytes expectedStrips expectedRawStripSizes (pageBuffer: byte[]) =
    use tiff = Tiff.Open(fileName, "r")
    if isNull tiff then
        invalidOp $"Could not open '{fileName}' for TIFF reading."

    let width, height, rowBytes, pageBytes, strips, rawStripSizes, _rowsPerStrip = inspectOpenRawStripTiffSlice pixelType fileName tiff
    if width <> expectedWidth || height <> expectedHeight then
        invalidOp $"Input slice '{fileName}' has shape {width}x{height}, expected {expectedWidth}x{expectedHeight}."
    if rowBytes <> expectedRowBytes || pageBytes <> expectedPageBytes then
        invalidOp $"Input slice '{fileName}' layout changed: rowBytes={rowBytes}, pageBytes={pageBytes}; expected {expectedRowBytes}, {expectedPageBytes}."
    if strips <> expectedStrips then
        invalidOp $"Input slice '{fileName}' has {strips} strips, expected {expectedStrips}."
    if rawStripSizes <> expectedRawStripSizes then
        invalidOp $"Input slice '{fileName}' raw strip layout changed."
    if pageBuffer.Length < pageBytes then
        invalidArg "pageBuffer" $"Raw-strip TIFF page buffer too small: need {pageBytes}, got {pageBuffer.Length}."

    let mutable offset = 0
    for strip in 0 .. strips - 1 do
        let count = rawStripSizes[strip]
        let bytesRead = tiff.ReadRawStrip(strip, pageBuffer, offset, count)
        if bytesRead < 0 then
            invalidOp $"ReadRawStrip failed for strip {strip} from '{fileName}'."
        if bytesRead <> count then
            invalidOp $"ReadRawStrip read {bytesRead} bytes for strip {strip} from '{fileName}', expected {count}."
        offset <- offset + count

    if offset <> pageBytes then
        invalidOp $"Raw strip reads produced {offset} bytes from '{fileName}', expected {pageBytes}."

let private writeRawStripTiffPage pixelType fileName width height rowBytes pageBytes (pageBuffer: byte[]) =
    let bitsPerSample, sampleFormat, bytesPerSample = scalarTiffLayoutForPixelType pixelType
    if rowBytes <> int width * bytesPerSample then
        invalidArg "rowBytes" $"Expected {int width * bytesPerSample} row bytes for {pixelType}, got {rowBytes}."
    if pageBytes <> rowBytes * int height then
        invalidArg "pageBytes" $"Expected {rowBytes * int height} page bytes for {pixelType}, got {pageBytes}."

    use tiff = Tiff.Open(fileName, ImageIO.tiffWriteMode fileName)
    if isNull tiff then
        invalidOp $"Could not open '{fileName}' for TIFF writing."

    tiff.SetField(TiffTag.IMAGEWIDTH, int width) |> ignore
    tiff.SetField(TiffTag.IMAGELENGTH, int height) |> ignore
    tiff.SetField(TiffTag.SAMPLESPERPIXEL, 1) |> ignore
    tiff.SetField(TiffTag.BITSPERSAMPLE, bitsPerSample) |> ignore
    tiff.SetField(TiffTag.SAMPLEFORMAT, sampleFormat) |> ignore
    tiff.SetField(TiffTag.PHOTOMETRIC, Photometric.MINISBLACK) |> ignore
    tiff.SetField(TiffTag.PLANARCONFIG, PlanarConfig.CONTIG) |> ignore
    tiff.SetField(TiffTag.ROWSPERSTRIP, int height) |> ignore
    tiff.SetField(TiffTag.COMPRESSION, Compression.NONE) |> ignore

    let written = tiff.WriteRawStrip(0, pageBuffer, 0, pageBytes)
    if written < 0 then
        invalidOp $"Failed to write raw TIFF strip to '{fileName}'."
    if written <> pageBytes then
        invalidOp $"WriteRawStrip wrote {written} bytes to '{fileName}', expected {pageBytes}."

    if not (tiff.WriteDirectory()) then
        invalidOp $"Failed to write TIFF directory to '{fileName}'."

let private readNativeTiffInfo pixelType fileName =
    let mutable info = NativeTiffInfo()
    let status = NativeLibTiff.readInfo(fileName, &info)
    if status <> NativeLibTiff.Ok then
        NativeLibTiff.failStatus "read-info" fileName status

    let expectedBits, expectedSampleFormat, expectedBytesPerSample = scalarNativeTiffLayoutForPixelType pixelType
    if info.Width = 0u || info.Height = 0u then
        invalidOp $"Native libtiff saw invalid page dimensions {info.Width}x{info.Height} for '{fileName}'."
    if info.BitsPerSample <> expectedBits || info.SampleFormat <> expectedSampleFormat then
        invalidOp $"Native libtiff input slice '{fileName}' does not match {pixelType}: got {info.BitsPerSample}-bit sample-format {info.SampleFormat}."
    if info.SamplesPerPixel <> 1us then
        invalidOp $"Native libtiff input slice '{fileName}' has {info.SamplesPerPixel} samples per pixel; expected scalar images."
    if info.PlanarConfig <> NativeLibTiff.PlanarConfigContig then
        invalidOp $"Native libtiff input slice '{fileName}' has planar configuration {info.PlanarConfig}; expected contiguous scalar images."
    if info.Compression <> NativeLibTiff.CompressionNone then
        invalidOp $"Native libtiff input slice '{fileName}' is compressed with code {info.Compression}; expected uncompressed TIFF."
    if info.IsTiled <> 0 then
        invalidOp $"Native libtiff input slice '{fileName}' is tiled; expected stripped TIFF."
    if info.IsByteSwapped <> 0 then
        invalidOp $"Native libtiff input slice '{fileName}' has non-native byte order; raw-strip copy bypasses byte swapping."
    if info.PageBytes <> info.RawPageBytes then
        invalidOp $"Native libtiff input slice '{fileName}' has {info.RawPageBytes} raw strip bytes but {info.PageBytes} logical page bytes."
    if info.PageBytes > uint64 Int32.MaxValue then
        invalidOp $"Native libtiff input slice '{fileName}' has a page larger than Int32.MaxValue bytes."

    let rowBytes = int info.Width * expectedBytesPerSample
    let expectedPageBytes = rowBytes * int info.Height
    if uint64 expectedPageBytes <> info.PageBytes then
        invalidOp $"Native libtiff input slice '{fileName}' has inconsistent page size {info.PageBytes}; expected {expectedPageBytes}."

    info, rowBytes, expectedBytesPerSample

let private readNativeRawTiffSliceInto pixelType fileName expectedWidth expectedHeight expectedRowBytes expectedPageBytes (pageBuffer: byte[]) =
    let info, rowBytes, _bytesPerSample = readNativeTiffInfo pixelType fileName
    if info.Width <> expectedWidth || info.Height <> expectedHeight then
        invalidOp $"Native libtiff input slice '{fileName}' has shape {info.Width}x{info.Height}, expected {expectedWidth}x{expectedHeight}."
    if rowBytes <> expectedRowBytes || int info.PageBytes <> expectedPageBytes then
        invalidOp $"Native libtiff input slice '{fileName}' layout changed: rowBytes={rowBytes}, pageBytes={info.PageBytes}; expected {expectedRowBytes}, {expectedPageBytes}."
    if pageBuffer.Length < expectedPageBytes then
        invalidArg "pageBuffer" $"Native libtiff page buffer too small: need {expectedPageBytes}, got {pageBuffer.Length}."

    let mutable bytesRead = 0UL
    let status = NativeLibTiff.readRawPage(fileName, pageBuffer, UIntPtr(uint64 expectedPageBytes), &bytesRead)
    if status <> NativeLibTiff.Ok then
        NativeLibTiff.failStatus "read-raw-page" fileName status
    if bytesRead <> uint64 expectedPageBytes then
        invalidOp $"Native libtiff read {bytesRead} bytes from '{fileName}', expected {expectedPageBytes}."

let private readNativeRawTiffSliceIntoOffset pixelType fileName expectedWidth expectedHeight expectedRowBytes expectedPageBytes offset (pageBuffer: byte[]) =
    let info, rowBytes, _bytesPerSample = readNativeTiffInfo pixelType fileName
    if info.Width <> expectedWidth || info.Height <> expectedHeight then
        invalidOp $"Native libtiff input slice '{fileName}' has shape {info.Width}x{info.Height}, expected {expectedWidth}x{expectedHeight}."
    if rowBytes <> expectedRowBytes || int info.PageBytes <> expectedPageBytes then
        invalidOp $"Native libtiff input slice '{fileName}' layout changed: rowBytes={rowBytes}, pageBytes={info.PageBytes}; expected {expectedRowBytes}, {expectedPageBytes}."
    if offset < 0 || pageBuffer.Length < offset + expectedPageBytes then
        invalidArg "pageBuffer" $"Native libtiff page buffer too small: need offset {offset} + {expectedPageBytes}, got {pageBuffer.Length}."

    let mutable bytesRead = 0UL
    let status = NativeLibTiff.readRawPageInto(fileName, pageBuffer, UIntPtr(uint64 offset), UIntPtr(uint64 pageBuffer.Length), &bytesRead)
    if status <> NativeLibTiff.Ok then
        NativeLibTiff.failStatus "read-raw-page-into" fileName status
    if bytesRead <> uint64 expectedPageBytes then
        invalidOp $"Native libtiff read {bytesRead} bytes from '{fileName}', expected {expectedPageBytes}."

let private writeNativeRawTiffPage pixelType fileName width height pageBytes (pageBuffer: byte[]) =
    let bitsPerSample, sampleFormat, _bytesPerSample = scalarNativeTiffLayoutForPixelType pixelType
    if pageBuffer.Length < pageBytes then
        invalidArg "pageBuffer" $"Native libtiff page buffer too small: need {pageBytes}, got {pageBuffer.Length}."

    let status = NativeLibTiff.writeRawPage(fileName, pageBuffer, UIntPtr(uint64 pageBytes), width, height, bitsPerSample, sampleFormat)
    if status <> NativeLibTiff.Ok then
        NativeLibTiff.failStatus "write-raw-page" fileName status

let private writeNativeRawTiffPageFromOffset pixelType fileName width height pageBytes offset (pageBuffer: byte[]) =
    let bitsPerSample, sampleFormat, _bytesPerSample = scalarNativeTiffLayoutForPixelType pixelType
    if offset < 0 || pageBuffer.Length < offset + pageBytes then
        invalidArg "pageBuffer" $"Native libtiff page buffer too small: need offset {offset} + {pageBytes}, got {pageBuffer.Length}."

    let status = NativeLibTiff.writeRawPageFrom(fileName, pageBuffer, UIntPtr(uint64 offset), UIntPtr(uint64 pageBytes), UIntPtr(uint64 pageBuffer.Length), width, height, bitsPerSample, sampleFormat)
    if status <> NativeLibTiff.Ok then
        NativeLibTiff.failStatus "write-raw-page-from" fileName status

let private readNativeScanlineTiffSliceInto pixelType fileName expectedWidth expectedHeight expectedRowBytes expectedPageBytes (pageBuffer: byte[]) =
    let info, rowBytes, _bytesPerSample = readNativeTiffInfo pixelType fileName
    if info.Width <> expectedWidth || info.Height <> expectedHeight then
        invalidOp $"Native libtiff input slice '{fileName}' has shape {info.Width}x{info.Height}, expected {expectedWidth}x{expectedHeight}."
    if rowBytes <> expectedRowBytes || int info.PageBytes <> expectedPageBytes then
        invalidOp $"Native libtiff input slice '{fileName}' layout changed: rowBytes={rowBytes}, pageBytes={info.PageBytes}; expected {expectedRowBytes}, {expectedPageBytes}."
    if pageBuffer.Length < expectedPageBytes then
        invalidArg "pageBuffer" $"Native libtiff page buffer too small: need {expectedPageBytes}, got {pageBuffer.Length}."

    let mutable bytesRead = 0UL
    let status = NativeLibTiff.readScanlinePage(fileName, pageBuffer, UIntPtr(uint64 expectedPageBytes), &bytesRead)
    if status <> NativeLibTiff.Ok then
        NativeLibTiff.failStatus "read-scanline-page" fileName status
    if bytesRead <> uint64 expectedPageBytes then
        invalidOp $"Native libtiff scanline read {bytesRead} bytes from '{fileName}', expected {expectedPageBytes}."

let private writeNativeScanlineTiffPage pixelType fileName width height pageBytes (pageBuffer: byte[]) =
    let bitsPerSample, sampleFormat, _bytesPerSample = scalarNativeTiffLayoutForPixelType pixelType
    if pageBuffer.Length < pageBytes then
        invalidArg "pageBuffer" $"Native libtiff page buffer too small: need {pageBytes}, got {pageBuffer.Length}."

    let status = NativeLibTiff.writeScanlinePage(fileName, pageBuffer, UIntPtr(uint64 pageBytes), width, height, bitsPerSample, sampleFormat)
    if status <> NativeLibTiff.Ok then
        NativeLibTiff.failStatus "write-scanline-page" fileName status

let private inspectTiffLibraryTags pixelType (fileName: string) (reader: TiffLibrary.TiffFileReader) (tagReader: TiffLibrary.TiffTagReader) =
    let expectedBits, expectedSampleFormat, expectedBytesPerSample = scalarTiffLibraryLayoutForPixelType pixelType
    let width64 = TiffLibrary.TiffTagReaderExtensions.ReadImageWidth(tagReader)
    let height64 = TiffLibrary.TiffTagReaderExtensions.ReadImageLength(tagReader)
    if width64 = 0UL || height64 = 0UL || width64 > uint64 Int32.MaxValue || height64 > uint64 Int32.MaxValue then
        invalidOp $"TiffLibrary saw invalid page dimensions {width64}x{height64} for '{fileName}'."

    let bitsPerSample = TiffLibrary.TiffTagReaderExtensions.ReadBitsPerSample(tagReader).GetFirstOrDefault()
    let samplesPerPixel = TiffLibrary.TiffTagReaderExtensions.ReadSamplesPerPixel(tagReader)
    let sampleFormatValues = tagReader.ReadShortField(TiffLibrary.TiffTag.SampleFormat)
    let sampleFormat = if sampleFormatValues.IsEmpty then 1us else sampleFormatValues.GetFirstOrDefault()
    let compression = TiffLibrary.TiffTagReaderExtensions.ReadCompression(tagReader)
    let planarConfig = TiffLibrary.TiffTagReaderExtensions.ReadPlanarConfiguration(tagReader)

    if bitsPerSample <> expectedBits || sampleFormat <> expectedSampleFormat then
        invalidOp $"TiffLibrary input slice '{fileName}' does not match {pixelType}: got {bitsPerSample}-bit sample-format {sampleFormat}."
    if samplesPerPixel <> 1us then
        invalidOp $"TiffLibrary input slice '{fileName}' has {samplesPerPixel} samples per pixel; expected scalar images."
    if compression <> TiffLibrary.TiffCompression.NoCompression then
        invalidOp $"TiffLibrary input slice '{fileName}' is compressed with {compression}; expected uncompressed TIFF."
    if planarConfig <> TiffLibrary.TiffPlanarConfiguration.Chunky then
        invalidOp $"TiffLibrary input slice '{fileName}' has planar configuration {planarConfig}; expected chunky scalar images."
    if expectedBytesPerSample > 1 && reader.IsLittleEndian <> BitConverter.IsLittleEndian then
        invalidOp $"TiffLibrary input slice '{fileName}' has non-native byte order; raw-strip copy bypasses byte swapping."

    let stripOffsets = TiffLibrary.TiffTagReaderExtensions.ReadStripOffsets(tagReader)
    let stripByteCounts = TiffLibrary.TiffTagReaderExtensions.ReadStripByteCounts(tagReader)
    if stripOffsets.Count <> stripByteCounts.Count || stripOffsets.Count < 1 then
        invalidOp $"TiffLibrary input slice '{fileName}' has invalid strip offset/count layout."

    let mutable rawPageBytes64 = 0UL
    for i in 0 .. stripByteCounts.Count - 1 do
        rawPageBytes64 <- rawPageBytes64 + stripByteCounts[i]

    let width = uint32 width64
    let height = uint32 height64
    let rowBytes = int width * expectedBytesPerSample
    let pageBytes = rowBytes * int height
    if rawPageBytes64 <> uint64 pageBytes then
        invalidOp $"TiffLibrary input slice '{fileName}' has {rawPageBytes64} raw strip bytes but {pageBytes} logical page bytes."

    width, height, rowBytes, pageBytes, stripOffsets, stripByteCounts

let private readTiffLibraryInfo pixelType (fileName: string) =
    use reader: TiffLibrary.TiffFileReader = TiffLibrary.TiffFileReader.Open(fileName)
    let ifd = reader.ReadImageFileDirectory()
    use fieldReader = reader.CreateFieldReader()
    let tagReader = TiffLibrary.TiffTagReader(fieldReader, ifd)
    inspectTiffLibraryTags pixelType fileName reader tagReader

let private readTiffLibraryRawSliceInto pixelType (fileName: string) expectedWidth expectedHeight expectedRowBytes expectedPageBytes expectedStrips (pageBuffer: byte[]) =
    use reader: TiffLibrary.TiffFileReader = TiffLibrary.TiffFileReader.Open(fileName)
    let ifd = reader.ReadImageFileDirectory()
    use fieldReader = reader.CreateFieldReader()
    use contentReader = reader.CreateContentReader()
    let tagReader = TiffLibrary.TiffTagReader(fieldReader, ifd)
    let width, height, rowBytes, pageBytes, stripOffsets, stripByteCounts = inspectTiffLibraryTags pixelType fileName reader tagReader
    if width <> expectedWidth || height <> expectedHeight then
        invalidOp $"TiffLibrary input slice '{fileName}' has shape {width}x{height}, expected {expectedWidth}x{expectedHeight}."
    if rowBytes <> expectedRowBytes || pageBytes <> expectedPageBytes then
        invalidOp $"TiffLibrary input slice '{fileName}' layout changed: rowBytes={rowBytes}, pageBytes={pageBytes}; expected {expectedRowBytes}, {expectedPageBytes}."
    if stripOffsets.Count <> expectedStrips then
        invalidOp $"TiffLibrary input slice '{fileName}' has {stripOffsets.Count} strips, expected {expectedStrips}."
    if pageBuffer.Length < expectedPageBytes then
        invalidArg "pageBuffer" $"TiffLibrary page buffer too small: need {expectedPageBytes}, got {pageBuffer.Length}."

    let mutable offset = 0
    for strip in 0 .. stripOffsets.Count - 1 do
        let byteCount64 = stripByteCounts[strip]
        if byteCount64 > uint64 Int32.MaxValue then
            invalidOp $"TiffLibrary input slice '{fileName}' strip {strip} is larger than Int32.MaxValue bytes."
        let byteCount = int byteCount64
        let bytesRead = contentReader.Read(TiffLibrary.TiffStreamOffset(int64 stripOffsets[strip]), pageBuffer.AsMemory(offset, byteCount))
        if bytesRead <> byteCount then
            invalidOp $"TiffLibrary read {bytesRead} bytes from strip {strip} in '{fileName}', expected {byteCount}."
        offset <- offset + byteCount

    if offset <> expectedPageBytes then
        invalidOp $"TiffLibrary raw strip reads produced {offset} bytes from '{fileName}', expected {expectedPageBytes}."

let private writeTiffLibraryRawPage pixelType (fileName: string) width height pageBytes (pageBuffer: byte[]) =
    let bitsPerSample, sampleFormat, _bytesPerSample = scalarTiffLibraryLayoutForPixelType pixelType
    if pageBuffer.Length < pageBytes then
        invalidArg "pageBuffer" $"TiffLibrary page buffer too small: need {pageBytes}, got {pageBuffer.Length}."
    if pageBytes > Int32.MaxValue then
        invalidOp $"TiffLibrary output page too large for one standard strip: {pageBytes} bytes."

    use writer: TiffLibrary.TiffFileWriter = TiffLibrary.TiffFileWriter.OpenAsync(fileName, useBigTiff = false) |> awaitTask
    let stripOffset = writer.WriteAlignedBytesAsync(pageBuffer.AsMemory(0, pageBytes)) |> awaitTask
    use ifdWriter = writer.CreateImageFileDirectory()
    ifdWriter.WriteTagAsync(TiffLibrary.TiffTag.ImageWidth, TiffLibrary.TiffValueCollection.Single(uint32 width)) |> awaitUnitValueTask
    ifdWriter.WriteTagAsync(TiffLibrary.TiffTag.ImageLength, TiffLibrary.TiffValueCollection.Single(uint32 height)) |> awaitUnitValueTask
    ifdWriter.WriteTagAsync(TiffLibrary.TiffTag.BitsPerSample, TiffLibrary.TiffValueCollection.Single(bitsPerSample)) |> awaitUnitValueTask
    ifdWriter.WriteTagAsync(TiffLibrary.TiffTag.SampleFormat, TiffLibrary.TiffValueCollection.Single(sampleFormat)) |> awaitUnitValueTask
    ifdWriter.WriteTagAsync(TiffLibrary.TiffTag.Compression, TiffLibrary.TiffValueCollection.Single(uint16 TiffLibrary.TiffCompression.NoCompression)) |> awaitUnitValueTask
    ifdWriter.WriteTagAsync(TiffLibrary.TiffTag.PhotometricInterpretation, TiffLibrary.TiffValueCollection.Single(uint16 TiffLibrary.TiffPhotometricInterpretation.BlackIsZero)) |> awaitUnitValueTask
    ifdWriter.WriteTagAsync(TiffLibrary.TiffTag.SamplesPerPixel, TiffLibrary.TiffValueCollection.Single(1us)) |> awaitUnitValueTask
    ifdWriter.WriteTagAsync(TiffLibrary.TiffTag.PlanarConfiguration, TiffLibrary.TiffValueCollection.Single(uint16 TiffLibrary.TiffPlanarConfiguration.Chunky)) |> awaitUnitValueTask
    ifdWriter.WriteTagAsync(TiffLibrary.TiffTag.RowsPerStrip, TiffLibrary.TiffValueCollection.Single(uint32 height)) |> awaitUnitValueTask
    ifdWriter.WriteTagAsync(TiffLibrary.TiffTag.StripOffsets, TiffLibrary.TiffValueCollection.Single(uint32 stripOffset.Offset)) |> awaitUnitValueTask
    ifdWriter.WriteTagAsync(TiffLibrary.TiffTag.StripByteCounts, TiffLibrary.TiffValueCollection.Single(uint32 pageBytes)) |> awaitUnitValueTask
    let ifdOffset = ifdWriter.FlushAsync() |> awaitTask
    writer.SetFirstImageFileDirectoryOffset(ifdOffset)
    writer.FlushAsync() |> awaitUnitTask

let private readArrayPoolTiffSliceInto<'T> fileName expectedWidth expectedHeight rowBytes (scanline: byte[]) (slice: PooledVolume<'T>) =
    use tiff = Tiff.Open(fileName, "r")
    if isNull tiff then
        invalidOp $"Could not open '{fileName}' for TIFF reading."

    let width = uint (tiffFieldInt tiff TiffTag.IMAGEWIDTH 0)
    let height = uint (tiffFieldInt tiff TiffTag.IMAGELENGTH 0)
    if width <> expectedWidth || height <> expectedHeight then
        invalidOp $"Input slice '{fileName}' has shape {width}x{height}, expected {expectedWidth}x{expectedHeight}."

    for row in 0 .. int expectedHeight - 1 do
        if not (tiff.ReadScanline(scanline, row)) then
            invalidOp $"Failed to read TIFF scanline {row} from '{fileName}'."
        Buffer.BlockCopy(scanline, 0, slice.Buffer, row * rowBytes, rowBytes)

let private writeArrayPoolTiffPageWithRowBuffer<'T> fileName width height rowBytes (rowBuffer: byte[]) (buffer: 'T[]) elementOffset =
    let bitsPerSample, sampleFormat, bytesPerSample = scalarTiffLayout<'T> ()
    use tiff = Tiff.Open(fileName, ImageIO.tiffWriteMode fileName)
    if isNull tiff then
        invalidOp $"Could not open '{fileName}' for TIFF writing."

    tiff.SetField(TiffTag.IMAGEWIDTH, int width) |> ignore
    tiff.SetField(TiffTag.IMAGELENGTH, int height) |> ignore
    tiff.SetField(TiffTag.SAMPLESPERPIXEL, 1) |> ignore
    tiff.SetField(TiffTag.BITSPERSAMPLE, bitsPerSample) |> ignore
    tiff.SetField(TiffTag.SAMPLEFORMAT, sampleFormat) |> ignore
    tiff.SetField(TiffTag.PHOTOMETRIC, Photometric.MINISBLACK) |> ignore
    tiff.SetField(TiffTag.PLANARCONFIG, PlanarConfig.CONTIG) |> ignore
    tiff.SetField(TiffTag.ROWSPERSTRIP, int height) |> ignore
    tiff.SetField(TiffTag.COMPRESSION, Compression.NONE) |> ignore

    for row in 0 .. int height - 1 do
        Buffer.BlockCopy(buffer, (elementOffset * bytesPerSample) + (row * rowBytes), rowBuffer, 0, rowBytes)
        if not (tiff.WriteScanline(rowBuffer, row)) then
            invalidOp $"Failed to write TIFF scanline {row} to '{fileName}'."

    if not (tiff.WriteDirectory()) then
        invalidOp $"Failed to write TIFF directory to '{fileName}'."

let private writeByteTiffPageFor<'T> fileName width height rowBytes (pageBuffer: byte[]) =
    let bitsPerSample, sampleFormat, _bytesPerSample = scalarTiffLayout<'T> ()
    use tiff = Tiff.Open(fileName, ImageIO.tiffWriteMode fileName)
    if isNull tiff then
        invalidOp $"Could not open '{fileName}' for TIFF writing."

    tiff.SetField(TiffTag.IMAGEWIDTH, int width) |> ignore
    tiff.SetField(TiffTag.IMAGELENGTH, int height) |> ignore
    tiff.SetField(TiffTag.SAMPLESPERPIXEL, 1) |> ignore
    tiff.SetField(TiffTag.BITSPERSAMPLE, bitsPerSample) |> ignore
    tiff.SetField(TiffTag.SAMPLEFORMAT, sampleFormat) |> ignore
    tiff.SetField(TiffTag.PHOTOMETRIC, Photometric.MINISBLACK) |> ignore
    tiff.SetField(TiffTag.PLANARCONFIG, PlanarConfig.CONTIG) |> ignore
    tiff.SetField(TiffTag.ROWSPERSTRIP, int height) |> ignore
    tiff.SetField(TiffTag.COMPRESSION, Compression.NONE) |> ignore

    for row in 0 .. int height - 1 do
        if not (tiff.WriteScanline(pageBuffer, row * rowBytes, row, int16 0)) then
            invalidOp $"Failed to write TIFF scanline {row} to '{fileName}'."

    if not (tiff.WriteDirectory()) then
        invalidOp $"Failed to write TIFF directory to '{fileName}'."

let private readByteTiffSliceInto fileName expectedWidth expectedHeight rowBytes (pageBuffer: byte[]) =
    use tiff = Tiff.Open(fileName, "r")
    if isNull tiff then
        invalidOp $"Could not open '{fileName}' for TIFF reading."

    let width = uint (tiffFieldInt tiff TiffTag.IMAGEWIDTH 0)
    let height = uint (tiffFieldInt tiff TiffTag.IMAGELENGTH 0)
    if width <> expectedWidth || height <> expectedHeight then
        invalidOp $"Input slice '{fileName}' has shape {width}x{height}, expected {expectedWidth}x{expectedHeight}."

    for row in 0 .. int expectedHeight - 1 do
        if not (tiff.ReadScanline(pageBuffer, row * rowBytes, row, int16 0)) then
            invalidOp $"Failed to read TIFF scanline {row} from '{fileName}'."

let private readArrayPoolTiffStack<'T> inputDir =
    let expectedBits, expectedFormat, expectedBytesPerSample = scalarTiffLayout<'T> ()
    let files = stackTiffFiles inputDir
    if files.Length = 0 then
        invalidOp $"No TIFF files found in input stack directory: {inputDir}"

    let width, height, bits, sampleFormat, bytesPerSample, firstPage = readTiffPageBytes files[0]
    if bits <> expectedBits || sampleFormat <> expectedFormat || bytesPerSample <> expectedBytesPerSample then
        invalidOp $"Input pixel layout does not match {typeof<'T>.Name}: got {bits}-bit {sampleFormat}."

    let depth = uint files.Length
    let volume = rentVolume<'T> width height depth "arraypool.read"
    let sliceElements = int width * int height
    let sliceBytes = sliceElements * expectedBytesPerSample
    Buffer.BlockCopy(firstPage, 0, volume.Buffer, 0, sliceBytes)

    for z in 1 .. files.Length - 1 do
        let w, h, b, sf, bps, page = readTiffPageBytes files[z]
        if w <> width || h <> height then
            invalidOp $"Input slice '{files[z]}' has shape {w}x{h}, expected {width}x{height}."
        if b <> expectedBits || sf <> expectedFormat || bps <> expectedBytesPerSample then
            invalidOp $"Input slice '{files[z]}' has layout {b}-bit {sf}, expected {expectedBits}-bit {expectedFormat}."
        Buffer.BlockCopy(page, 0, volume.Buffer, z * sliceBytes, sliceBytes)

    volume

let private writeArrayPoolTiffPage<'T> fileName width height (buffer: 'T[]) elementOffset =
    let bitsPerSample, sampleFormat, bytesPerSample = scalarTiffLayout<'T> ()
    use tiff = Tiff.Open(fileName, ImageIO.tiffWriteMode fileName)
    if isNull tiff then
        invalidOp $"Could not open '{fileName}' for TIFF writing."

    tiff.SetField(TiffTag.IMAGEWIDTH, int width) |> ignore
    tiff.SetField(TiffTag.IMAGELENGTH, int height) |> ignore
    tiff.SetField(TiffTag.SAMPLESPERPIXEL, 1) |> ignore
    tiff.SetField(TiffTag.BITSPERSAMPLE, bitsPerSample) |> ignore
    tiff.SetField(TiffTag.SAMPLEFORMAT, sampleFormat) |> ignore
    tiff.SetField(TiffTag.PHOTOMETRIC, Photometric.MINISBLACK) |> ignore
    tiff.SetField(TiffTag.PLANARCONFIG, PlanarConfig.CONTIG) |> ignore
    tiff.SetField(TiffTag.ROWSPERSTRIP, int height) |> ignore
    tiff.SetField(TiffTag.COMPRESSION, Compression.NONE) |> ignore

    let rowBytes = int width * bytesPerSample
    let sliceBytes = rowBytes * int height
    let pageBytes = ArrayPool<byte>.Shared.Rent(sliceBytes)
    let rowBuffer = ArrayPool<byte>.Shared.Rent(rowBytes)
    try
        Buffer.BlockCopy(buffer, elementOffset * bytesPerSample, pageBytes, 0, sliceBytes)
        for row in 0 .. int height - 1 do
            Buffer.BlockCopy(pageBytes, row * rowBytes, rowBuffer, 0, rowBytes)
            if not (tiff.WriteScanline(rowBuffer, row)) then
                invalidOp $"Failed to write TIFF scanline {row} to '{fileName}'."

        if not (tiff.WriteDirectory()) then
            invalidOp $"Failed to write TIFF directory to '{fileName}'."
    finally
        ArrayPool<byte>.Shared.Return(rowBuffer)
        ArrayPool<byte>.Shared.Return(pageBytes)

let private writeArrayPoolTiffStack<'T> outputDir (volume: PooledVolume<'T>) =
    ensureCleanDirectory outputDir
    let sliceElements = int volume.Width * int volume.Height
    for z in 0 .. int volume.Depth - 1 do
        writeArrayPoolTiffPage (outputFile outputDir z) volume.Width volume.Height volume.Buffer (z * sliceElements)

let private copyArrayPoolVolume<'T> (volume: PooledVolume<'T>) =
    let output = rentVolume<'T> volume.Width volume.Height volume.Depth "arraypool.copy"
    volume.Span.CopyTo(output.Span)
    output

let private thresholdArrayPoolVolume<'T> thresholdValue (volume: PooledVolume<'T>) =
    let output = rentVolume<uint8> volume.Width volume.Height volume.Depth "arraypool.threshold"
    if typeof<'T> = typeof<uint8> then
        let input = box volume.Buffer :?> uint8[]
        let threshold8 = byte thresholdValue
        for i in 0 .. volume.Length - 1 do
            output.Buffer[i] <- if input[i] >= threshold8 then 255uy else 0uy
    elif typeof<'T> = typeof<uint16> then
        let input = box volume.Buffer :?> uint16[]
        let threshold16 = uint16 thresholdValue
        for i in 0 .. volume.Length - 1 do
            output.Buffer[i] <- if input[i] >= threshold16 then 255uy else 0uy
    elif typeof<'T> = typeof<float32> then
        let input = box volume.Buffer :?> float32[]
        let threshold32 = float32 thresholdValue
        for i in 0 .. volume.Length - 1 do
            output.Buffer[i] <- if input[i] >= threshold32 then 255uy else 0uy
    else
        invalidArg "T" $"Unsupported ArrayPool threshold type {typeof<'T>.Name}."
    output

let private pooledUInt8VolumeToImage (volume: PooledVolume<uint8>) name =
    use importer = new itk.simple.ImportImageFilter()
    importer.SetSize(new itk.simple.VectorUInt32([ volume.Width; volume.Height; volume.Depth ]))
    let handle = GCHandle.Alloc(volume.Buffer, GCHandleType.Pinned)
    try
        importer.SetBufferAsUInt8(handle.AddrOfPinnedObject())
        Image<uint8>.ofSimpleITKNDispose(importer.Execute(), name, 0)
    finally
        handle.Free()

let private labelImageToUInt8Volume (labels: Image<uint64>) width height depth =
    let labelArray = labels.toArray3D()
    let output = rentVolume<uint8> width height depth "arraypool.connectedComponents.output"
    let mutable offset = 0
    for z in 0 .. int depth - 1 do
        for y in 0 .. int height - 1 do
            for x in 0 .. int width - 1 do
                output.Buffer[offset] <- byte labelArray[x, y, z]
                offset <- offset + 1
    output

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

let private generateInt32 pattern shape outputDir =
    ensureCleanDirectory outputDir
    for z in 0u .. shape.Depth - 1u do
        let arr =
            Array2D.init (int shape.Width) (int shape.Height) (fun x y ->
                match pattern with
                | "binary" -> if (x + y + int z) % 7 = 0 then 255 else 0
                | _ -> (x * 3 + y * 5 + int z * 11) % 256)
        let img = Image<int32>.ofArray2D(arr, name = "input", index = int z)
        ImageIO.writeTiffSliceFile (outputFile outputDir (int z)) img
        img.decRefCount()

let private generateUInt32 pattern shape outputDir =
    ensureCleanDirectory outputDir
    for z in 0u .. shape.Depth - 1u do
        let arr =
            Array2D.init (int shape.Width) (int shape.Height) (fun x y ->
                match pattern with
                | "binary" -> if (x + y + int z) % 7 = 0 then 255u else 0u
                | _ -> uint32 ((x * 3 + y * 5 + int z * 11) % 256))
        let img = Image<uint32>.ofArray2D(arr, name = "input", index = int z)
        ImageIO.writeTiffSliceFile (outputFile outputDir (int z)) img
        img.decRefCount()

let private generate opts =
    let shape = require "shape" opts |> parseShape
    let pixelType = require "pixel-type" opts |> parsePixelType
    let output = require "output" opts
    let pattern = optional "pattern" "ramp" opts
    match pixelType with
    | UInt8 -> generateUInt8 pattern shape output
    | UInt16 -> generateUInt16 pattern shape output
    | UInt32 -> generateUInt32 pattern shape output
    | Int32 -> generateInt32 pattern shape output
    | Float32 -> generateFloat32 pattern shape output
    0

let private runChunkCopyTyped<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> input output availableMemory =
    ensureCleanDirectory output
    let src = benchmarkSource availableMemory
    src
    |> read<'T> input ".tiff"
    >=> write<'T> output ".tiff"
    |> sink
    0

let private runChunkReadWriteTyped<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> input output availableMemory debugLevel =
    let src = benchmarkSourceWithDebug debugLevel availableMemory
    src
    |> read<'T> input ".tiff"
    >=> write<'T> output ".tiff"
    |> sink
    0

let private runChunkReadWriteWithTiffOptionsTyped<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> input output availableMemory debugLevel options =
    let src = benchmarkSourceWithDebug debugLevel availableMemory
    src
    |> read<'T> input ".tiff"
    >=> writeTiffWithOptions<'T> options output ".tiff"
    |> sink
    0

let private runChunkThresholdTyped<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> input output thresholdValue availableMemory =
    ensureCleanDirectory output
    let src = benchmarkSource availableMemory
    src
    |> read<'T> input ".tiff"
    >=> ChunkFunctions.thresholdNative<'T> thresholdValue
    >=> ChunkFunctions.castToUInt8<'T>
    >=> write<uint8> output ".tiff"
    |> sink
    0

let private runChunkThresholdParallelCollectTyped<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> input output thresholdValue workers availableMemory =
    ensureCleanDirectory output
    let src = benchmarkSource availableMemory
    src
    |> read<'T> input ".tiff"
    >=> ChunkFunctions.thresholdNativeParallelCollect<'T> thresholdValue workers
    >=> ChunkFunctions.castToUInt8<'T>
    >=> write<uint8> output ".tiff"
    |> sink
    0

let private runBinaryDilateTyped<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> input output radius availableMemory =
    ensureCleanDirectory output
    let src = benchmarkSource availableMemory
    src
    |> read<'T> input ".tiff"
    >=> thresholdRange<'T> 128.0 infinity
    >=> binaryDilate radius
    >=> write output ".tiff"
    |> sink
    0

let private runChunkBinaryDilate input output radius thresholdValue workers availableMemory =
    ensureCleanDirectory output
    let thresholdByte =
        if thresholdValue < 0.0 || thresholdValue > 255.0 then
            invalidArg "threshold" $"Chunk binary dilation threshold must be in [0,255], got {thresholdValue}."
        uint8 thresholdValue
    if workers < 1 then
        invalidArg "workers" $"Chunk binary dilation expects at least one worker/window, got {workers}."
    let dilation =
        if workers = 1 then
            ChunkFunctions.binaryDilateZonohedral radius
        else
            ChunkFunctions.binaryDilateZonohedralParallel radius workers
    let src = benchmarkSource availableMemory
    src
    |> read<uint8> input ".tiff"
    >=> ChunkFunctions.thresholdBinary thresholdByte
    >=> dilation
    >=> write<uint8> output ".tiff"
    |> sink
    0

let private runChunkConnectedComponents input output thresholdValue windowSize workers availableMemory =
    let thresholdByte =
        if thresholdValue < 0.0 || thresholdValue > 255.0 then
            invalidArg "threshold" $"Chunk connected components threshold must be in [0,255], got {thresholdValue}."
        uint8 thresholdValue

    if workers < 1 then
        invalidArg "workers" $"Chunk connected components expects at least one worker, got {workers}."
    windowSize |> Option.iter (fun value ->
        if value < 1 then
            invalidArg "window" $"Chunk connected components expects a positive window size, got {value}.")

    let mutable checksum = 0UL
    let consumeLabels _debug _index (chunk: StackCore.Chunk<uint32>) =
        let labels = StackCore.Chunk.span<uint32> chunk
        let mutable i = 0
        while i < labels.Length do
            checksum <- checksum ^^^ uint64 labels[i]
            i <- i + 1
        StackCore.Chunk.decRef chunk

    let labelStage =
        match windowSize with
        | Some size -> ChunkFunctions.connectedComponentsSauf3DUInt8UInt32ParallelCollect size workers
        | None -> ChunkFunctions.connectedComponentsSauf3DUInt8UInt32 ()

    let src = benchmarkSource availableMemory
    let labeled =
        src
        |> read<uint8> input ".tiff"
        >=> ChunkFunctions.thresholdBinary thresholdByte
        >=> labelStage

    match output with
    | Some outputDir ->
        ensureCleanDirectory outputDir
        labeled
        >=> write<uint32> outputDir ".tiff"
        |> sink
    | None ->
        labeled
        >=> SlimPipeline.Stage.consumeWith "consumeChunkConnectedComponents" consumeLabels id
        |> sink

    if checksum = UInt64.MaxValue then
        printfn "unreachable checksum guard: %d" checksum
    0

let private uniformKernel3DFloat32 (kernelSize: uint) =
    let size = max 1u kernelSize
    let value = 1.0f / float32 (size * size * size)
    Array3D.create (int size) (int size) (int size) value

let private runChunkConvolveTyped<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    input
    output
    kernelSize
    workers
    native
    availableMemory
    debugLevel
    =
    ensureCleanDirectory output
    if workers < 1 then
        invalidArg "workers" $"Chunk convolution expects at least one worker/window, got {workers}."

    let src = benchmarkSourceWithDebug debugLevel availableMemory
    let kernel = uniformKernel3DFloat32 kernelSize
    if native then
        let convolution =
            if workers = 1 then
                ChunkFunctions.convolveFixedKernelNative<'T> kernel
            else
                ChunkFunctions.convolveFixedKernelNativeParallel<'T> kernel workers

        src
        |> read<'T> input ".tiff"
        >=> convolution
        >=> write<'T> output ".tiff"
        |> sink
    else
        let convolution =
            if workers = 1 then
                ChunkFunctions.convolveFixedKernel<float32> kernel
            else
                ChunkFunctions.convolveFixedKernelParallel<float32> kernel workers

        src
        |> read<'T> input ".tiff"
        >=> ChunkFunctions.castToFloat32<'T>
        >=> convolution
        >=> ChunkFunctions.castFromFloat32<'T>
        >=> write<'T> output ".tiff"
        |> sink
    0

let private runChunkMedianPhUInt8 input output radius availableMemory =
    ensureCleanDirectory output
    if radius > uint32 Int32.MaxValue then
        invalidArg "radius" $"PH median radius must fit in Int32, got {radius}."

    let src = benchmarkSource availableMemory
    src
    |> read<uint8> input ".tiff"
    >=> ChunkFunctions.medianPerreaultHebertUInt8DenseRolling (int radius)
    >=> write<uint8> output ".tiff"
    |> sink
    0

let private runChunkMedianPhYBandsUInt8 input output radius workers availableMemory =
    ensureCleanDirectory output
    if radius > uint32 Int32.MaxValue then
        invalidArg "radius" $"PH median radius must fit in Int32, got {radius}."
    if workers < 1 then
        invalidArg "workers" $"PH median y-band worker count must be at least 1, got {workers}."

    let src = benchmarkSource availableMemory
    src
    |> read<uint8> input ".tiff"
    >=> ChunkFunctions.medianPerreaultHebertUInt8DenseRollingYBands (int radius) workers
    >=> write<uint8> output ".tiff"
    |> sink
    0

let private runChunkMedianPhXFirstUInt8 input output radius availableMemory =
    ensureCleanDirectory output
    if radius > uint32 Int32.MaxValue then
        invalidArg "radius" $"PH median radius must fit in Int32, got {radius}."

    let src = benchmarkSource availableMemory
    src
    |> read<uint8> input ".tiff"
    >=> ChunkFunctions.medianPerreaultHebertUInt8DenseXFirstMaterialized (int radius)
    >=> write<uint8> output ".tiff"
    |> sink
    0

let private runChunkMedianPhXBlockUInt8 input output radius availableMemory =
    ensureCleanDirectory output
    if radius > uint32 Int32.MaxValue then
        invalidArg "radius" $"PH median radius must fit in Int32, got {radius}."

    let src = benchmarkSource availableMemory
    src
    |> read<uint8> input ".tiff"
    >=> ChunkFunctions.medianPerreaultHebertUInt8DenseXBlock (int radius)
    >=> write<uint8> output ".tiff"
    |> sink
    0

let private runChunkMedianPhXTransposeUInt8 input output radius availableMemory =
    ensureCleanDirectory output
    if radius > uint32 Int32.MaxValue then
        invalidArg "radius" $"PH median radius must fit in Int32, got {radius}."

    let src = benchmarkSource availableMemory
    src
    |> read<uint8> input ".tiff"
    >=> ChunkFunctions.medianPerreaultHebertUInt8DenseRollingTransposedXBlock (int radius)
    >=> write<uint8> output ".tiff"
    |> sink
    0

let private runChunkMedianPhTreeUInt8 input output radius availableMemory =
    ensureCleanDirectory output
    if radius > uint32 Int32.MaxValue then
        invalidArg "radius" $"PH median radius must fit in Int32, got {radius}."

    let src = benchmarkSource availableMemory
    src
    |> read<uint8> input ".tiff"
    >=> ChunkFunctions.medianPerreaultHebertUInt8DenseRollingTree (int radius)
    >=> write<uint8> output ".tiff"
    |> sink
    0

let private runChunkMedianPhBlockedZUInt8 input output radius availableMemory =
    ensureCleanDirectory output
    if radius > uint32 Int32.MaxValue then
        invalidArg "radius" $"PH median radius must fit in Int32, got {radius}."

    let src = benchmarkSource availableMemory
    src
    |> read<uint8> input ".tiff"
    >=> ChunkFunctions.medianPerreaultHebertUInt8DenseRollingBlockedZ (int radius)
    >=> write<uint8> output ".tiff"
    |> sink
    0

let private runChunkMedianQuickselectUInt16 input output radius workers availableMemory =
    ensureCleanDirectory output
    if radius > uint32 Int32.MaxValue then
        invalidArg "radius" $"Chunk quickselect median radius must fit in Int32, got {radius}."
    if workers < 1 then
        invalidArg "workers" $"Chunk quickselect median worker count must be at least 1, got {workers}."

    let src = benchmarkSource availableMemory
    src
    |> read<uint16> input ".tiff"
    >=> ChunkFunctions.medianQuickselectUInt16ParallelCollect (int radius) workers
    >=> write<uint16> output ".tiff"
    |> sink
    0

let private runChunkMedianSortUInt16 input output radius workers availableMemory =
    ensureCleanDirectory output
    if radius > uint32 Int32.MaxValue then
        invalidArg "radius" $"Chunk sort median radius must fit in Int32, got {radius}."
    if workers < 1 then
        invalidArg "workers" $"Chunk sort median worker count must be at least 1, got {workers}."

    let src = benchmarkSource availableMemory
    src
    |> read<uint16> input ".tiff"
    >=> ChunkFunctions.medianSortUInt16ParallelCollect (int radius) workers
    >=> write<uint16> output ".tiff"
    |> sink
    0

let private runChunkMedianNativeNthUInt16 input output radius workers availableMemory =
    ensureCleanDirectory output
    if radius > uint32 Int32.MaxValue then
        invalidArg "radius" $"Chunk native nth_element median radius must fit in Int32, got {radius}."
    if workers < 1 then
        invalidArg "workers" $"Chunk native nth_element median worker count must be at least 1, got {workers}."

    let src = benchmarkSource availableMemory
    src
    |> read<uint16> input ".tiff"
    >=> ChunkFunctions.medianNativeNthElementUInt16ParallelCollect (int radius) workers
    >=> write<uint16> output ".tiff"
    |> sink
    0

let private runChunkMedianNativeNthUInt8 input output radius workers availableMemory =
    ensureCleanDirectory output
    if radius > uint32 Int32.MaxValue then
        invalidArg "radius" $"Chunk native nth_element median radius must fit in Int32, got {radius}."
    if workers < 1 then
        invalidArg "workers" $"Chunk native nth_element median worker count must be at least 1, got {workers}."

    let src = benchmarkSource availableMemory
    src
    |> read<uint8> input ".tiff"
    >=> ChunkFunctions.medianNativeNthElementUInt8ParallelCollect (int radius) workers
    >=> write<uint8> output ".tiff"
    |> sink
    0

let private runChunkMedianNativeNthInt32 input output radius workers availableMemory =
    ensureCleanDirectory output
    if radius > uint32 Int32.MaxValue then
        invalidArg "radius" $"Chunk native nth_element median radius must fit in Int32, got {radius}."
    if workers < 1 then
        invalidArg "workers" $"Chunk native nth_element median worker count must be at least 1, got {workers}."

    let src = benchmarkSource availableMemory
    src
    |> read<int32> input ".tiff"
    >=> ChunkFunctions.medianNativeNthElementInt32ParallelCollect (int radius) workers
    >=> write<int32> output ".tiff"
    |> sink
    0

let private runChunkMedianNativeNthFloat32 input output radius workers availableMemory =
    ensureCleanDirectory output
    if radius > uint32 Int32.MaxValue then
        invalidArg "radius" $"Chunk native nth_element median radius must fit in Int32, got {radius}."
    if workers < 1 then
        invalidArg "workers" $"Chunk native nth_element median worker count must be at least 1, got {workers}."

    let src = benchmarkSource availableMemory
    src
    |> read<float32> input ".tiff"
    >=> ChunkFunctions.medianNativeNthElementFloat32ParallelCollect (int radius) workers
    >=> write<float32> output ".tiff"
    |> sink
    0

let private runChunkMedianStandardUInt8 input output radius availableMemory =
    if radius < 2u || radius > 40u then
        runChunkMedianNativeNthUInt8 input output radius 3 availableMemory
    else
        runChunkMedianPhYBandsUInt8 input output radius 3 availableMemory

let private splitChunkThick<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> : Stage<Chunk<'T>, Chunk<'T>> =
    let name = $"splitChunkThick.{typeof<'T>.Name}"

    let apply (debug: bool) (input: AsyncSeq<Chunk<'T>>) =
        asyncSeq {
            for chunk in input do
                try
                    let width, height, depth = chunk.Size
                    if depth = 0UL then
                        invalidArg "chunk" $"splitChunkThick cannot split an empty-depth chunk: {chunk.Size}."
                    let depthInt = int depth
                    let sliceBytes = chunk.ByteLength / depthInt
                    if sliceBytes * depthInt <> chunk.ByteLength then
                        invalidArg "chunk" $"splitChunkThick cannot evenly split {chunk.ByteLength} bytes into {depth} slices."

                    if debug && depth > 1UL then
                        printfn $"[{name}] Splitting thick chunk with depth {depth}."

                    for localZ in 0 .. depthInt - 1 do
                        let output = Chunk.create<'T> (width, height, 1UL)
                        Buffer.BlockCopy(chunk.Bytes, localZ * sliceBytes, output.Bytes, 0, sliceBytes)
                        yield output
                finally
                    Chunk.decRef chunk
        }

    let transition = SlimPipeline.ProfileTransition.create SlimPipeline.Streaming SlimPipeline.Streaming
    let memoryModel = SlimPipeline.StageMemoryModel.fromSinglePeak SlimPipeline.Map id
    SlimPipeline.Stage.fromAsyncSeq name apply transition memoryModel id

let private collectChunkSlicesThick<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (thickDepth: uint)
    : Stage<Chunk<'T>, Chunk<'T>> =

    let name = $"collectChunkSlicesThick.{typeof<'T>.Name}"
    let thickDepth = max 1u thickDepth |> int

    let apply (debug: bool) (input: AsyncSeq<Chunk<'T>>) =
        asyncSeq {
            let mutable slices = ResizeArray<Chunk<'T>>()
            let mutable width = 0UL
            let mutable height = 0UL

            let flush () =
                if slices.Count = 0 then
                    None
                else
                    let depth = slices.Count
                    let output = Chunk.create<'T> (width, height, uint64 depth)
                    try
                        let sliceBytes = slices[0].ByteLength
                        for i in 0 .. depth - 1 do
                            Buffer.BlockCopy(slices[i].Bytes, 0, output.Bytes, i * sliceBytes, sliceBytes)
                            Chunk.decRef slices[i]
                        if debug && depth > 1 then
                            printfn $"[{name}] Collected {depth} slices into a thick chunk."
                        slices <- ResizeArray<Chunk<'T>>()
                        Some output
                    with
                    | ex ->
                        Chunk.decRef output
                        for slice in slices do
                            Chunk.decRef slice
                        slices <- ResizeArray<Chunk<'T>>()
                        raise ex

            try
                for chunk in input do
                    let chunkWidth, chunkHeight, chunkDepth = chunk.Size
                    if chunkDepth <> 1UL then
                        Chunk.decRef chunk
                        invalidArg "chunk" $"collectChunkSlicesThick expects 2D chunks with depth 1, got {chunk.Size}."

                    if slices.Count = 0 then
                        width <- chunkWidth
                        height <- chunkHeight
                    elif chunkWidth <> width || chunkHeight <> height then
                        Chunk.decRef chunk
                        invalidArg "chunk" $"collectChunkSlicesThick expected slices of size {width}x{height}, got {chunkWidth}x{chunkHeight}."

                    slices.Add chunk
                    if slices.Count = thickDepth then
                        match flush () with
                        | Some output -> yield output
                        | None -> ()

                match flush () with
                | Some output -> yield output
                | None -> ()
            with
            | ex ->
                for slice in slices do
                    Chunk.decRef slice
                raise ex
        }

    let transition = SlimPipeline.ProfileTransition.create SlimPipeline.Streaming SlimPipeline.Streaming
    let memoryModel = SlimPipeline.StageMemoryModel.fromSinglePeak SlimPipeline.Map id
    SlimPipeline.Stage.fromAsyncSeq name apply transition memoryModel id

let private runZarrTyped<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> operation shape input output radius kernelSize thresholdValue workers chunkSize availableMemory =
    let src = benchmarkSource availableMemory
    let chunkDepth, chunkY, chunkX =
        if operation = "tiffToZarr" then
            let chunkSize = max 1u chunkSize
            chunkSize, chunkSize, chunkSize
        else
            let zarrInfo = getZarrInfo input 0 0
            match zarrInfo.chunks with
            | x :: y :: z :: _ -> max 1u (uint z), max 1u (uint y), max 1u (uint x)
            | _ -> 16u, 256u, 256u
    let depth = max 1u shape.Depth
    let readInputThick () =
        src
        |> readZarrThick<'T> 0u (depth - 1u) chunkDepth input 0 0 0 0 0

    let readInputSlices () =
        readInputThick ()
        >=> splitChunkThick<'T>

    match operation with
    | "copy" ->
        src
        |> readZarrEncodedChunks input 0 0 0 0
        >=> writeZarrEncodedChunksWithCompression ZarrCompression.None output "benchmark" 1.0 1.0 1.0 0
        |> sink
    | "copyThickThin" ->
        readInputSlices ()
        >=> collectChunkSlicesThick<'T> chunkDepth
        >=> writeZarrThickWithCompression<'T> ZarrCompression.None output "benchmark" depth chunkX chunkY chunkDepth 1.0 1.0 1.0 0
        |> sink
    | "zarrToTiff" ->
        readInputThick ()
        >=> writeThick<'T> output ".tiff"
        |> sink
    | "tiffToZarr" ->
        src
        |> readThick<'T> chunkDepth input ".tiff"
        >=> writeZarrThickWithCompression<'T> ZarrCompression.None output "benchmark" depth chunkX chunkY chunkDepth 1.0 1.0 1.0 0
        |> sink
    | "threshold" ->
        if typeof<'T> = typeof<uint8> then
            src
            |> readZarrChunks<uint8> input 0 0 0 0
            >=> thresholdZarrChunksUInt8 thresholdValue
            >=> writeZarrChunksWithCompression<uint8> ZarrCompression.None output "benchmark" 1.0 1.0 1.0 0
            |> sink
        else
            readInputThick ()
            >=> thresholdRange<'T> thresholdValue infinity
            >=> writeZarrThickWithCompression<uint8> ZarrCompression.None output "benchmark" depth chunkX chunkY chunkDepth 1.0 1.0 1.0 0
            |> sink
    | "median" ->
        failwith "run-zarr median used a retired Image stage bridge; use the TIFF Chunk median benchmarks or add a Chunk-native Zarr median benchmark."
    | "convolve" ->
        if workers < 1 then
            invalidArg "workers" $"Zarr convolution expects at least one worker/window, got {workers}."

        let kernel = uniformKernel3DFloat32 kernelSize
        let convolution =
            if workers = 1 then
                ChunkFunctions.convolveFixedKernelNative<'T> kernel
            else
                ChunkFunctions.convolveFixedKernelNativeParallel<'T> kernel workers

        src
        |> readZarrAlignedSlices<'T> 0u (depth - 1u) input 0 0 0 0
        >=> convolution
        >=> writeZarrAlignedSlicesWithCompression<'T> ZarrCompression.None output "benchmark" depth chunkX chunkY chunkDepth 1.0 1.0 1.0 0
        |> sink
    | "dilate" ->
        readInputSlices ()
        >=> thresholdRange<'T> 128.0 infinity
        >=> binaryDilate radius
        >=> writeZarrWithCompression<uint8> ZarrCompression.None output "benchmark" depth chunkX chunkY chunkDepth 1.0 1.0 1.0 0
        |> sink
    | _ -> failwith $"unsupported Zarr operation '{operation}'"
    0

let private runZarr opts =
    let operation = require "operation" opts
    let pixelType = require "pixel-type" opts |> parsePixelType
    let shape = require "shape" opts |> parseShape
    let input = require "input" opts
    let output = require "output" opts
    let radius = optional "radius" "1" opts |> UInt32.Parse
    let kernelSize = optional "kernel-size" "3" opts |> UInt32.Parse
    let thresholdValue = optional "threshold" "128" opts |> fun s -> Double.Parse(s, invariant)
    let workers = optional "workers" "3" opts |> Int32.Parse
    let chunkSize = optional "chunk-size" "128" opts |> UInt32.Parse
    let availableMemory = optional "available-memory" (string (1024UL * 1024UL * 1024UL * 1024UL)) opts |> UInt64.Parse
    ensureCleanDirectory output
    let stopwatch = Stopwatch.StartNew()
    let exitCode =
        match pixelType with
        | UInt8 -> runZarrTyped<uint8> operation shape input output radius kernelSize thresholdValue workers chunkSize availableMemory
        | UInt16 -> runZarrTyped<uint16> operation shape input output radius kernelSize thresholdValue workers chunkSize availableMemory
        | Float32 -> runZarrTyped<float32> operation shape input output radius kernelSize thresholdValue workers chunkSize availableMemory
        | _ -> unsupportedPixelType "run-zarr benchmark" "UInt8, UInt16, and Float32" pixelType
    stopwatch.Stop()
    writeInternalSeconds stopwatch.Elapsed
    exitCode

let private releaseChunkStage<'T when 'T: equality> =
    SlimPipeline.Stage.consumeWith
        $"releaseChunk.{typeof<'T>.Name}"
        (fun _debug _index (chunk: Chunk<'T>) -> Chunk.decRef chunk)
        (fun _ -> 0UL)

let private runZarrConvolveBreakdownTyped<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    variant
    shape
    input
    output
    kernelSize
    workers
    availableMemory =

    if workers < 1 then
        invalidArg "workers" $"Zarr convolution breakdown expects at least one worker/window, got {workers}."

    let zarrInfo = getZarrInfo input 0 0
    let chunkDepth, chunkY, chunkX =
        match zarrInfo.chunks with
        | x :: y :: z :: _ -> max 1u (uint z), max 1u (uint y), max 1u (uint x)
        | _ -> 16u, 256u, 256u

    let depth = max 1u shape.Depth
    let src = benchmarkSource availableMemory
    let reader =
        readZarrAlignedSlices<'T> 0u (depth - 1u) input 0 0 0 0
    let kernel = uniformKernel3DFloat32 kernelSize
    let convolution =
        if workers = 1 then
            ChunkFunctions.convolveFixedKernelNative<'T> kernel
        else
            ChunkFunctions.convolveFixedKernelNativeParallel<'T> kernel workers

    match variant with
    | "read" ->
        src
        |> reader
        >=> releaseChunkStage<'T>
        |> sink
    | "readWrite" ->
        src
        |> reader
        >=> writeZarrAlignedSlicesWithCompression<'T> ZarrCompression.None output "benchmark" depth chunkX chunkY chunkDepth 1.0 1.0 1.0 0
        |> sink
    | "readConvolve" ->
        src
        |> reader
        >=> convolution
        >=> releaseChunkStage<'T>
        |> sink
    | "convolve" ->
        src
        |> reader
        >=> convolution
        >=> writeZarrAlignedSlicesWithCompression<'T> ZarrCompression.None output "benchmark" depth chunkX chunkY chunkDepth 1.0 1.0 1.0 0
        |> sink
    | _ ->
        failwith $"unsupported Zarr convolve breakdown variant '{variant}'"

    0

let private runZarrConvolveBreakdown opts =
    let variant = require "variant" opts
    let pixelType = require "pixel-type" opts |> parsePixelType
    let shape = require "shape" opts |> parseShape
    let input = require "input" opts
    let output = optional "output" "" opts
    let kernelSize = optional "kernel-size" "3" opts |> UInt32.Parse
    let workers = optional "workers" "3" opts |> Int32.Parse
    let availableMemory = optional "available-memory" (string (1024UL * 1024UL * 1024UL * 1024UL)) opts |> UInt64.Parse

    match variant with
    | "readWrite"
    | "convolve" ->
        if String.IsNullOrWhiteSpace output then
            invalidArg "output" $"Zarr convolve breakdown variant '{variant}' requires --output."
        ensureCleanDirectory output
    | "read"
    | "readConvolve" -> ()
    | _ -> failwith $"unsupported Zarr convolve breakdown variant '{variant}'"

    let stopwatch = Stopwatch.StartNew()
    let exitCode =
        match pixelType with
        | UInt8 -> runZarrConvolveBreakdownTyped<uint8> variant shape input output kernelSize workers availableMemory
        | UInt16 -> runZarrConvolveBreakdownTyped<uint16> variant shape input output kernelSize workers availableMemory
        | Float32 -> runZarrConvolveBreakdownTyped<float32> variant shape input output kernelSize workers availableMemory
        | _ -> unsupportedPixelType "run-zarr-convolve-breakdown" "UInt8, UInt16, and Float32" pixelType
    stopwatch.Stop()
    writeInternalSeconds stopwatch.Elapsed
    exitCode

let private openZarrLevel (path: string) =
    let reader: OmeZarrReader =
        OmeZarrReader.OpenAsync(path, ct = Threading.CancellationToken.None)
        |> runTask
    reader.AsMultiscaleImage().OpenResolutionLevelAsync(0, 0, Threading.CancellationToken.None)
    |> runTask

let private openZarrArray (path: string) =
    let store = new LocalFileSystemStore(path)
    let group = ZarrGroup.OpenRootAsync(store, Threading.CancellationToken.None) |> runTask
    group.OpenArrayAsync("0", Threading.CancellationToken.None) |> runTask

let private collectZarrChunks (array: ZarrArray) =
    let chunks = ResizeArray<ZarrChunkRef>()
    let enumerator = array.EnumerateChunksAsync(Threading.CancellationToken.None).GetAsyncEnumerator()
    try
        let mutable more = true
        while more do
            more <- enumerator.MoveNextAsync().AsTask() |> runTask
            if more then
                chunks.Add(enumerator.Current)
    finally
        enumerator.DisposeAsync().AsTask() |> runUnitTask
    chunks.ToArray()

let private zarrChunkCoordKey (chunk: ZarrChunkRef) =
    String.Join(",", chunk.ChunkCoord)

let private zarrChunkLookup (array: ZarrArray) =
    collectZarrChunks array
    |> Array.map (fun chunk -> zarrChunkCoordKey chunk, chunk)
    |> dict

let private zarrXyzTriple (values: int64[]) =
    if values.Length >= 5 then
        uint64 values[4], uint64 values[3], uint64 values[2]
    elif values.Length = 3 then
        uint64 values[2], uint64 values[1], uint64 values[0]
    else
        failwith $"Expected a 3D or TCZYX Zarr chunk coordinate/shape, got rank {values.Length}."

let private zarrChunkIndex (chunk: ZarrChunkRef) : ChunkIndex =
    let x, y, z = zarrXyzTriple chunk.ChunkCoord
    int x, int y, int z

let private zarrChunkBufferSize (array: ZarrArray) =
    array.Metadata.ChunkShape
    |> Array.map int64
    |> zarrXyzTriple

let private bytesPerPixelType pixelType =
    match pixelType with
    | UInt8 -> 1
    | UInt16 -> 2
    | Int32 -> 4
    | Float32 -> 4
    | _ -> unsupportedPixelType "bytesPerPixelType" "UInt8, UInt16, Int32, and Float32" pixelType

let private chunkElementBytes<'T>() =
    match typeof<'T> with
    | t when t = typeof<byte> -> 1UL
    | t when t = typeof<uint8> -> 1UL
    | t when t = typeof<int8> -> 1UL
    | t when t = typeof<uint16> -> 2UL
    | t when t = typeof<int16> -> 2UL
    | t when t = typeof<int32> -> 4UL
    | t when t = typeof<float32> -> 4UL
    | t -> invalidArg "T" $"Unsupported benchmark chunk type {t.Name}."

let private createBenchmarkByteChunk<'T when 'T: equality> size (bytes: byte[]) release : Chunk<'T> =
    let width, height, depth = size
    let expected = width * height * depth * chunkElementBytes<'T>()
    if expected > uint64 Int32.MaxValue then
        invalidArg "size" $"Benchmark chunk byte length must fit in Int32; got {expected}."
    if uint64 bytes.LongLength < expected then
        invalidArg "bytes" $"Benchmark chunk byte buffer length was {bytes.LongLength}, expected at least {expected}."
    { Size = size
      Bytes = bytes
      ByteLength = int expected
      Release = release
      RefCount = ref 1 }

let private chunkFromDecodedBytes<'T when 'T: equality> (array: ZarrArray) (chunkRef: ZarrChunkRef) (decoded: byte[]) : Chunk<'T> =
    createBenchmarkByteChunk<'T>
        (zarrChunkBufferSize array)
        decoded
        ignore

let private decodedBytesFromChunk<'T when 'T: equality> (chunk: Chunk<'T>) =
    if chunk.ByteLength = chunk.Bytes.Length then
        chunk.Bytes
    else
        chunk.Bytes[0 .. chunk.ByteLength - 1]

let private decodedByteChunk (array: ZarrArray) (chunkRef: ZarrChunkRef) (decoded: byte[]) : Chunk<byte> =
    createBenchmarkByteChunk<byte>
        (zarrChunkBufferSize array)
        decoded
        ignore

let private validateZarrArrayType (array: ZarrArray) dataType =
    if not (String.Equals(array.Metadata.DataType.TypeString, dataType, StringComparison.OrdinalIgnoreCase)) then
        failwith $"Input Zarr type was {array.Metadata.DataType.TypeString}, but benchmark requested {dataType}."

let private zarrChunkRegions (shape: Shape) =
    seq {
        let chunkX = 256
        let chunkY = 256
        let chunkZ = 16
        let width = int shape.Width
        let height = int shape.Height
        let depth = int shape.Depth
        for zStart in 0 .. chunkZ .. depth - 1 do
            let zStop = min depth (zStart + chunkZ)
            for yStart in 0 .. chunkY .. height - 1 do
                let yStop = min height (yStart + chunkY)
                for xStart in 0 .. chunkX .. width - 1 do
                    let xStop = min width (xStart + chunkX)
                    PixelRegion(
                        [| 0L; 0L; int64 zStart; int64 yStart; int64 xStart |],
                        [| 1L; 1L; int64 zStop; int64 yStop; int64 xStop |])
    }

let private createOmeZarrWriterWithChunks output dataType (shape: Shape) chunkX chunkY chunkZ =
    let descriptor =
        BioImageDescriptor(
            int shape.Width,
            int shape.Height,
            ZCT(int shape.Depth, 1, 1),
            Name = "benchmark",
            DataType = dataType,
            ChunkX = chunkX,
            ChunkY = chunkY,
            ChunkZ = chunkZ,
            ChunkC = 1,
            ChunkT = 1,
            PhysicalSizeX = 1.0,
            PhysicalSizeY = 1.0,
            PhysicalSizeZ = 1.0,
            Compression = ZarrCompression.None)

    OmeZarrWriter.CreateAsync(output, descriptor, Threading.CancellationToken.None)
    |> runTask

let private createOmeZarrWriter output dataType (shape: Shape) =
    createOmeZarrWriterWithChunks output dataType shape 256 256 16

let private alignOutputZarrCodecsWithInput input output =
    let inputMetadataPath = Path.Combine(input, "0", "zarr.json")
    let outputMetadataPath = Path.Combine(output, "0", "zarr.json")
    let inputMetadata = JsonNode.Parse(File.ReadAllText(inputMetadataPath))
    let outputMetadata = JsonNode.Parse(File.ReadAllText(outputMetadataPath))

    match inputMetadata, outputMetadata with
    | null, _ -> failwith $"Could not parse input Zarr metadata at {inputMetadataPath}."
    | _, null -> failwith $"Could not parse output Zarr metadata at {outputMetadataPath}."
    | inputJson, outputJson ->
        let codecs = inputJson["codecs"]
        if isNull codecs then
            outputJson["codecs"] <- JsonArray()
        else
            outputJson["codecs"] <- JsonNode.Parse(codecs.ToJsonString())

        let options = JsonSerializerOptions(WriteIndented = true)
        File.WriteAllText(outputMetadataPath, outputJson.ToJsonString(options))

let private runZarrDirectCopy opts =
    let pixelType = require "pixel-type" opts |> parsePixelType
    let shape = require "shape" opts |> parseShape
    let input = require "input" opts
    let output = require "output" opts
    let chunkSize = optional "chunk-size" "0" opts |> Int32.Parse

    ensureCleanDirectory output

    let stopwatch = Stopwatch.StartNew()
    let inputArray = openZarrArray input
    let dataType = zarrDataType pixelType
    validateZarrArrayType inputArray dataType

    let writer =
        if chunkSize > 0 then
            createOmeZarrWriterWithChunks output dataType shape chunkSize chunkSize chunkSize
        else
            createOmeZarrWriter output dataType shape
    alignOutputZarrCodecsWithInput input output

    try
        let outputArray = openZarrArray output
        let outputChunks = zarrChunkLookup outputArray
        for inputChunk in collectZarrChunks inputArray do
            let outputChunk = outputChunks[zarrChunkCoordKey inputChunk]
            let encoded =
                inputArray.ReadChunkEncodedAsync(inputChunk, Threading.CancellationToken.None)
                |> runTask

            if isNull encoded then
                let decoded =
                    inputArray.ReadChunkDecodedAsync(inputChunk, Threading.CancellationToken.None)
                    |> runTask
                outputArray.WriteChunkDecodedAsync(outputChunk, decoded, Threading.CancellationToken.None)
                |> runUnitTask
            else
                outputArray.WriteChunkEncodedAsync(outputChunk, encoded, Threading.CancellationToken.None)
                |> runUnitTask
    finally
        writer.DisposeAsync().AsTask()
        |> runUnitTask

    stopwatch.Stop()
    writeInternalSeconds stopwatch.Elapsed
    0

let private runZarrDirectCopySameGrid opts =
    let input = require "input" opts
    let output = require "output" opts

    ensureCleanDirectory output

    let stopwatch = Stopwatch.StartNew()
    let inputArray = openZarrArray input
    let shapeValues = inputArray.Metadata.Shape
    let chunkValues = inputArray.Metadata.ChunkShape
    if shapeValues.Length <> 5 || chunkValues.Length <> 5 then
        failwith $"run-zarr-direct-copy-same-grid expects TCZYX rank 5 input, got shape rank {shapeValues.Length} and chunk rank {chunkValues.Length}."

    let shape =
        { Width = uint32 shapeValues[4]
          Height = uint32 shapeValues[3]
          Depth = uint32 shapeValues[2] }

    let dataType = inputArray.Metadata.DataType.TypeString
    let writer =
        createOmeZarrWriterWithChunks
            output
            dataType
            shape
            (int chunkValues[4])
            (int chunkValues[3])
            (int chunkValues[2])

    alignOutputZarrCodecsWithInput input output

    try
        let outputArray = openZarrArray output
        let outputChunks = zarrChunkLookup outputArray
        for inputChunk in collectZarrChunks inputArray do
            let outputChunk = outputChunks[zarrChunkCoordKey inputChunk]
            let encoded =
                inputArray.ReadChunkEncodedAsync(inputChunk, Threading.CancellationToken.None)
                |> runTask

            if isNull encoded then
                let decoded =
                    inputArray.ReadChunkDecodedAsync(inputChunk, Threading.CancellationToken.None)
                    |> runTask
                outputArray.WriteChunkDecodedAsync(outputChunk, decoded, Threading.CancellationToken.None)
                |> runUnitTask
            else
                outputArray.WriteChunkEncodedAsync(outputChunk, encoded, Threading.CancellationToken.None)
                |> runUnitTask
    finally
        writer.DisposeAsync().AsTask()
        |> runUnitTask

    stopwatch.Stop()
    writeInternalSeconds stopwatch.Elapsed
    0

let private thresholdByteChunkSimdInto (thresholdValue: double) (input: Chunk<byte>) (output: byte[]) =
    let inputData = input.Bytes
    let inputLength = input.ByteLength
    let threshold = byte (Math.Clamp(Math.Ceiling(thresholdValue), 0.0, 255.0))
    let thresholdVector = Vector<byte>(threshold)
    let onVector = Vector<byte>(1uy)
    let width = Vector<byte>.Count
    let mutable i = 0
    let vectorLimit = inputLength - (inputLength % width)
    while i < vectorLimit do
        let values = Vector<byte>(inputData, i)
        let mask = Vector.GreaterThanOrEqual(values, thresholdVector)
        Vector.BitwiseAnd(mask, onVector).CopyTo(output, i)
        i <- i + width

    while i < inputLength do
        output[i] <- if inputData[i] >= threshold then 1uy else 0uy
        i <- i + 1
    inputLength

let private thresholdUInt16ChunkToByteInto (thresholdValue: double) (input: Chunk<uint16>) (output: byte[]) =
    let values: Span<uint16> = StackCore.Chunk.span input
    let threshold = uint16 (Math.Clamp(Math.Ceiling(thresholdValue), 0.0, float UInt16.MaxValue))
    let mutable i = 0
    while i < values.Length do
        output[i] <- if values[i] >= threshold then 1uy else 0uy
        i <- i + 1
    values.Length

let private thresholdFloat32ChunkToByteInto (thresholdValue: double) (input: Chunk<float32>) (output: byte[]) =
    let values: Span<float32> = StackCore.Chunk.span input
    let threshold = float32 thresholdValue
    let mutable i = 0
    while i < values.Length do
        output[i] <- if values[i] >= threshold then 1uy else 0uy
        i <- i + 1
    values.Length

let private thresholdDecodedChunkToPooledByteChunk (pixelType: PixelType) (thresholdValue: double) (array: ZarrArray) (chunkRef: ZarrChunkRef) (decoded: byte[]) =
    let outputLength =
        match pixelType with
        | UInt8 -> decoded.Length
        | UInt16 -> decoded.Length / sizeof<uint16>
        | Float32 -> decoded.Length / sizeof<float32>
        | _ -> unsupportedPixelType "Zarr direct threshold" "UInt8, UInt16, and Float32" pixelType
    let output = ArrayPool<byte>.Shared.Rent(outputLength)
    let outputChunk written =
        if written <> outputLength then
            failwith $"Threshold wrote {written} bytes, expected {outputLength}."
        createBenchmarkByteChunk<byte>
            (zarrChunkBufferSize array)
            output
            (fun () -> ArrayPool<byte>.Shared.Return(output))
    match pixelType with
    | UInt8 ->
        let chunk = decodedByteChunk array chunkRef decoded
        let written = thresholdByteChunkSimdInto thresholdValue chunk output
        outputChunk written
    | UInt16 ->
        let chunk = chunkFromDecodedBytes<uint16> array chunkRef decoded
        let written = thresholdUInt16ChunkToByteInto thresholdValue chunk output
        outputChunk written
    | Float32 ->
        let chunk = chunkFromDecodedBytes<float32> array chunkRef decoded
        let written = thresholdFloat32ChunkToByteInto thresholdValue chunk output
        outputChunk written
    | _ -> unsupportedPixelType "Zarr direct threshold" "UInt8, UInt16, and Float32" pixelType

let private thresholdUInt8DecodedBytesSimdInto (thresholdValue: double) (input: byte[]) (output: byte[]) =
    let threshold = byte (Math.Clamp(Math.Ceiling(thresholdValue), 0.0, 255.0))
    let thresholdVector = Vector<byte>(threshold)
    let oneVector = Vector<byte>(1uy)
    let width = Vector<byte>.Count
    let mutable i = 0
    let vectorLimit = input.Length - (input.Length % width)
    while i < vectorLimit do
        let values = Vector<byte>(input, i)
        let mask = Vector.GreaterThanOrEqual(values, thresholdVector)
        Vector.BitwiseAnd(mask, oneVector).CopyTo(output, i)
        i <- i + width

    while i < input.Length do
        output[i] <- if input[i] >= threshold then 1uy else 0uy
        i <- i + 1
    input.Length

let private thresholdUInt8PageMaxSimdInto (thresholdValue: double) inputLength (input: byte[]) (output: byte[]) =
    let threshold = byte (Math.Clamp(Math.Ceiling(thresholdValue), 0.0, 255.0))
    let thresholdVector = Vector<byte>(threshold)
    let width = Vector<byte>.Count
    let mutable i = 0
    let vectorLimit = inputLength - (inputLength % width)
    while i < vectorLimit do
        let values = Vector<byte>(input, i)
        Vector.GreaterThanOrEqual(values, thresholdVector).CopyTo(output, i)
        i <- i + width

    while i < inputLength do
        output[i] <- if input[i] >= threshold then 255uy else 0uy
        i <- i + 1

[<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
let private thresholdUInt8PageSimdInto (thresholdValue: double) inputLength (input: byte[]) (output: byte[]) =
    let threshold = byte (Math.Clamp(Math.Ceiling(thresholdValue), 0.0, 255.0))
    let thresholdVector = Vector<byte>(threshold)
    let onVector = Vector<byte>(1uy)
    let width = Vector<byte>.Count
    let mutable i = 0
    let vectorLimit = inputLength - (inputLength % width)
    while i < vectorLimit do
        let values = Vector<byte>(input, i)
        let mask = Vector.GreaterThanOrEqual(values, thresholdVector)
        Vector.BitwiseAnd(mask, onVector).CopyTo(output, i)
        i <- i + width

    while i < inputLength do
        output[i] <- if input[i] >= threshold then 1uy else 0uy
        i <- i + 1

[<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
let private thresholdUInt16PageSimdInto (thresholdValue: double) inputLength (input: byte[]) (output: byte[]) =
    let pixels = inputLength / sizeof<uint16>
    let values = MemoryMarshal.Cast<byte, uint16>(input.AsSpan(0, inputLength))
    let threshold = uint16 (Math.Clamp(Math.Ceiling(thresholdValue), 0.0, float UInt16.MaxValue))
    let thresholdVector = Vector<uint16>(threshold)
    let onVector = Vector<uint16>(1us)
    let vectorWidth = Vector<uint16>.Count
    let mutable i = 0
    while i <= pixels - (2 * vectorWidth) do
        let loMask =
            Vector.BitwiseAnd(
                Vector.GreaterThanOrEqual(Vector<uint16>(values.Slice(i, vectorWidth)), thresholdVector),
                onVector)
        let hiMask =
            Vector.BitwiseAnd(
                Vector.GreaterThanOrEqual(Vector<uint16>(values.Slice(i + vectorWidth, vectorWidth)), thresholdVector),
                onVector)
        Vector.Narrow(loMask, hiMask).CopyTo(output, i)
        i <- i + (2 * vectorWidth)

    while i < pixels do
        output[i] <- if values[i] >= threshold then 1uy else 0uy
        i <- i + 1

[<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
let private thresholdFloat32PageSimdInto (thresholdValue: double) inputLength (input: byte[]) (output: byte[]) =
    let pixels = inputLength / sizeof<float32>
    let values = MemoryMarshal.Cast<byte, float32>(input.AsSpan(0, inputLength))
    let threshold = float32 thresholdValue
    let mutable i = 0
    while i < pixels do
        output[i] <- if values[i] >= threshold then 1uy else 0uy
        i <- i + 1

let private thresholdDirectPageSimdInto pixelType thresholdValue inputLength input output =
    match pixelType with
    | UInt8 -> thresholdUInt8PageSimdInto thresholdValue inputLength input output
    | UInt16 -> thresholdUInt16PageSimdInto thresholdValue inputLength input output
    | Float32 -> thresholdFloat32PageSimdInto thresholdValue inputLength input output
    | _ -> unsupportedPixelType "Direct threshold page SIMD" "UInt8, UInt16, and Float32" pixelType

let private thresholdUInt8PageInTypeSimdInto (thresholdValue: double) inputLength (input: byte[]) (output: byte[]) =
    thresholdUInt8PageSimdInto thresholdValue inputLength input output

[<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
let private thresholdUInt16PageInTypeMaxSimdInto (thresholdValue: double) inputLength (input: byte[]) (output: byte[]) =
    let pixels = inputLength / sizeof<uint16>
    let values = MemoryMarshal.Cast<byte, uint16>(input.AsSpan(0, inputLength))
    let result = MemoryMarshal.Cast<byte, uint16>(output.AsSpan(0, inputLength))
    let threshold = uint16 (Math.Clamp(Math.Ceiling(thresholdValue), 0.0, float UInt16.MaxValue))
    let thresholdVector = Vector<uint16>(threshold)
    let width = Vector<uint16>.Count
    let mutable i = 0
    let vectorLimit = pixels - (pixels % width)
    while i < vectorLimit do
        let mask = Vector.GreaterThanOrEqual(Vector<uint16>(values.Slice(i, width)), thresholdVector)
        mask.CopyTo(result.Slice(i, width))
        i <- i + width

    while i < pixels do
        result[i] <- if values[i] >= threshold then UInt16.MaxValue else 0us
        i <- i + 1

[<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
let private thresholdUInt16PageInTypeSimdInto (thresholdValue: double) inputLength (input: byte[]) (output: byte[]) =
    let pixels = inputLength / sizeof<uint16>
    let values = MemoryMarshal.Cast<byte, uint16>(input.AsSpan(0, inputLength))
    let result = MemoryMarshal.Cast<byte, uint16>(output.AsSpan(0, inputLength))
    let threshold = uint16 (Math.Clamp(Math.Ceiling(thresholdValue), 0.0, float UInt16.MaxValue))
    let thresholdVector = Vector<uint16>(threshold)
    let onVector = Vector<uint16>(1us)
    let width = Vector<uint16>.Count
    let mutable i = 0
    let vectorLimit = pixels - (pixels % width)
    while i < vectorLimit do
        let mask =
            Vector.BitwiseAnd(
                Vector.GreaterThanOrEqual(Vector<uint16>(values.Slice(i, width)), thresholdVector),
                onVector)
        mask.CopyTo(result.Slice(i, width))
        i <- i + width

    while i < pixels do
        result[i] <- if values[i] >= threshold then 1us else 0us
        i <- i + 1

[<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
let private thresholdFloat32PageInTypeMaxVectorInto (thresholdValue: double) inputLength (input: byte[]) (output: byte[]) =
    let pixels = inputLength / sizeof<float32>
    let values = MemoryMarshal.Cast<byte, float32>(input.AsSpan(0, inputLength))
    let result = MemoryMarshal.Cast<byte, float32>(output.AsSpan(0, inputLength))
    let threshold = float32 thresholdValue
    let thresholdVector = Vector<float32>(threshold)
    let trueVector = Vector<float32>(-1.0f)
    let falseVector = Vector<float32>(0.0f)
    let width = Vector<float32>.Count
    let mutable i = 0
    let vectorLimit = pixels - (pixels % width)
    while i < vectorLimit do
        let valuesVector = Vector<float32>(values.Slice(i, width))
        let mask = Vector.GreaterThanOrEqual(valuesVector, thresholdVector)
        Vector.ConditionalSelect(mask, trueVector, falseVector).CopyTo(result.Slice(i, width))
        i <- i + width

    while i < pixels do
        result[i] <- if values[i] >= threshold then -1.0f else 0.0f
        i <- i + 1

[<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
let private thresholdFloat32PageInTypeOneVectorInto (thresholdValue: double) inputLength (input: byte[]) (output: byte[]) =
    let pixels = inputLength / sizeof<float32>
    let values = MemoryMarshal.Cast<byte, float32>(input.AsSpan(0, inputLength))
    let result = MemoryMarshal.Cast<byte, float32>(output.AsSpan(0, inputLength))
    let threshold = float32 thresholdValue
    let thresholdVector = Vector<float32>(threshold)
    let trueVector = Vector<float32>(1.0f)
    let falseVector = Vector<float32>(0.0f)
    let width = Vector<float32>.Count
    let mutable i = 0
    let vectorLimit = pixels - (pixels % width)
    while i < vectorLimit do
        let valuesVector = Vector<float32>(values.Slice(i, width))
        let mask = Vector.GreaterThanOrEqual(valuesVector, thresholdVector)
        Vector.ConditionalSelect(mask, trueVector, falseVector).CopyTo(result.Slice(i, width))
        i <- i + width

    while i < pixels do
        result[i] <- if values[i] >= threshold then 1.0f else 0.0f
        i <- i + 1

[<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
let private thresholdFloat32PageInTypeInto (thresholdValue: double) inputLength (input: byte[]) (output: byte[]) =
    let pixels = inputLength / sizeof<float32>
    let values = MemoryMarshal.Cast<byte, float32>(input.AsSpan(0, inputLength))
    let result = MemoryMarshal.Cast<byte, float32>(output.AsSpan(0, inputLength))
    let threshold = float32 thresholdValue
    let mutable i = 0
    while i < pixels do
        result[i] <- if values[i] >= threshold then 1.0f else 0.0f
        i <- i + 1

let private thresholdDirectPageInTypeInto pixelType thresholdValue inputLength input output =
    match pixelType with
    | UInt8 -> thresholdUInt8PageInTypeSimdInto thresholdValue inputLength input output
    | UInt16 -> thresholdUInt16PageInTypeSimdInto thresholdValue inputLength input output
    | Float32 -> thresholdFloat32PageInTypeInto thresholdValue inputLength input output
    | _ -> unsupportedPixelType "Direct in-type threshold page SIMD" "UInt8, UInt16, and Float32" pixelType

[<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
let private thresholdTypedUInt8MaxVector (thresholdValue: double) length (input: byte[]) (output: byte[]) =
    thresholdUInt8PageMaxSimdInto thresholdValue length input output

[<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
let private thresholdTypedUInt8OneVector (thresholdValue: double) length (input: byte[]) (output: byte[]) =
    thresholdUInt8PageSimdInto thresholdValue length input output

[<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
let private thresholdTypedUInt16MaxVector (thresholdValue: double) length (input: uint16[]) (output: uint16[]) =
    let threshold = uint16 (Math.Clamp(Math.Ceiling(thresholdValue), 0.0, float UInt16.MaxValue))
    let thresholdVector = Vector<uint16>(threshold)
    let width = Vector<uint16>.Count
    let mutable i = 0
    let vectorLimit = length - (length % width)
    while i < vectorLimit do
        Vector.GreaterThanOrEqual(Vector<uint16>(input, i), thresholdVector).CopyTo(output, i)
        i <- i + width
    while i < length do
        output[i] <- if input[i] >= threshold then UInt16.MaxValue else 0us
        i <- i + 1

[<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
let private thresholdTypedUInt16OneVector (thresholdValue: double) length (input: uint16[]) (output: uint16[]) =
    let threshold = uint16 (Math.Clamp(Math.Ceiling(thresholdValue), 0.0, float UInt16.MaxValue))
    let thresholdVector = Vector<uint16>(threshold)
    let oneVector = Vector<uint16>(1us)
    let width = Vector<uint16>.Count
    let mutable i = 0
    let vectorLimit = length - (length % width)
    while i < vectorLimit do
        let mask = Vector.GreaterThanOrEqual(Vector<uint16>(input, i), thresholdVector)
        Vector.BitwiseAnd(mask, oneVector).CopyTo(output, i)
        i <- i + width
    while i < length do
        output[i] <- if input[i] >= threshold then 1us else 0us
        i <- i + 1

[<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
let private thresholdTypedFloat32MaxVector (thresholdValue: double) length (input: float32[]) (output: float32[]) =
    let threshold = float32 thresholdValue
    let thresholdVector = Vector<float32>(threshold)
    let trueVector = Vector<float32>(-1.0f)
    let falseVector = Vector<float32>(0.0f)
    let width = Vector<float32>.Count
    let mutable i = 0
    let vectorLimit = length - (length % width)
    while i < vectorLimit do
        let mask = Vector.GreaterThanOrEqual(Vector<float32>(input, i), thresholdVector)
        Vector.ConditionalSelect(mask, trueVector, falseVector).CopyTo(output, i)
        i <- i + width
    while i < length do
        output[i] <- if input[i] >= threshold then -1.0f else 0.0f
        i <- i + 1

[<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
let private thresholdTypedFloat32OneVector (thresholdValue: double) length (input: float32[]) (output: float32[]) =
    let threshold = float32 thresholdValue
    let thresholdVector = Vector<float32>(threshold)
    let trueVector = Vector<float32>(1.0f)
    let falseVector = Vector<float32>(0.0f)
    let width = Vector<float32>.Count
    let mutable i = 0
    let vectorLimit = length - (length % width)
    while i < vectorLimit do
        let mask = Vector.GreaterThanOrEqual(Vector<float32>(input, i), thresholdVector)
        Vector.ConditionalSelect(mask, trueVector, falseVector).CopyTo(output, i)
        i <- i + width
    while i < length do
        output[i] <- if input[i] >= threshold then 1.0f else 0.0f
        i <- i + 1

let private fillThresholdKernelInput pixelType inputLength (input: byte[]) =
    match pixelType with
    | UInt8 ->
        for i in 0 .. inputLength - 1 do
            input[i] <- byte (i &&& 0xFF)
    | UInt16 ->
        let values = MemoryMarshal.Cast<byte, uint16>(input.AsSpan(0, inputLength))
        for i in 0 .. values.Length - 1 do
            values[i] <- uint16 (i &&& 0xFFFF)
    | Float32 ->
        let values = MemoryMarshal.Cast<byte, float32>(input.AsSpan(0, inputLength))
        for i in 0 .. values.Length - 1 do
            values[i] <- float32 (i &&& 0xFF)
    | _ -> unsupportedPixelType "Threshold kernel microbenchmark" "UInt8, UInt16, and Float32" pixelType

let private runThresholdKernel opts =
    let pixelType = require "pixel-type" opts |> parsePixelType
    let shape = require "shape" opts |> parseShape
    let outputType = optional "output-type" "mask" opts
    let thresholdValue = optional "threshold" "128" opts |> fun s -> Double.Parse(s, invariant)
    let pixels = int shape.Width * int shape.Height * int shape.Depth
    let _bitsPerSample, _sampleFormat, bytesPerSample = scalarTiffLayoutForPixelType pixelType
    let inputLength = pixels * bytesPerSample
    let outputLength =
        match outputType with
        | "mask" -> pixels
        | "intype" -> inputLength
        | other -> invalidArg "output-type" $"Expected output-type mask or intype; got '{other}'."

    let input = ArrayPool<byte>.Shared.Rent(inputLength)
    let output = ArrayPool<byte>.Shared.Rent(outputLength)
    try
        fillThresholdKernelInput pixelType inputLength input
        let stopwatch = Stopwatch.StartNew()
        match outputType with
        | "mask" -> thresholdDirectPageSimdInto pixelType thresholdValue inputLength input output
        | "intype" -> thresholdDirectPageInTypeInto pixelType thresholdValue inputLength input output
        | _ -> ()
        stopwatch.Stop()
        writeInternalSeconds stopwatch.Elapsed
        let mutable checksum = 0
        if outputLength > 0 then
            checksum <- checksum + int output[0] + int output[outputLength / 2] + int output[outputLength - 1]
        if checksum = Int32.MinValue then
            printfn "%d" checksum
        0
    finally
        ArrayPool<byte>.Shared.Return(output)
        ArrayPool<byte>.Shared.Return(input)

let private thresholdUInt16DecodedBytesInto (thresholdValue: double) (input: byte[]) (output: byte[]) =
    let values = MemoryMarshal.Cast<byte, uint16>(input.AsSpan())
    let threshold = uint16 (Math.Clamp(Math.Ceiling(thresholdValue), 0.0, float UInt16.MaxValue))
    let mutable i = 0
    while i < values.Length do
        output[i] <- if values[i] >= threshold then 1uy else 0uy
        i <- i + 1
    values.Length

let private thresholdFloat32DecodedBytesInto (thresholdValue: double) (input: byte[]) (output: byte[]) =
    let values = MemoryMarshal.Cast<byte, float32>(input.AsSpan())
    let threshold = float32 thresholdValue
    let mutable i = 0
    while i < values.Length do
        output[i] <- if values[i] >= threshold then 1uy else 0uy
        i <- i + 1
    values.Length

let private thresholdDecodedBytesToPooledByteMemoryRaw (pixelType: PixelType) (thresholdValue: double) (decoded: byte[]) =
    let outputLength =
        match pixelType with
        | UInt8 -> decoded.Length
        | UInt16 -> decoded.Length / sizeof<uint16>
        | Float32 -> decoded.Length / sizeof<float32>
        | _ -> unsupportedPixelType "Raw Zarr threshold" "UInt8, UInt16, and Float32" pixelType
    let output = ArrayPool<byte>.Shared.Rent(outputLength)
    let written =
        match pixelType with
        | UInt8 -> thresholdUInt8DecodedBytesSimdInto thresholdValue decoded output
        | UInt16 -> thresholdUInt16DecodedBytesInto thresholdValue decoded output
        | Float32 -> thresholdFloat32DecodedBytesInto thresholdValue decoded output
        | _ -> unsupportedPixelType "Raw Zarr threshold" "UInt8, UInt16, and Float32" pixelType
    if written <> outputLength then
        failwith $"Raw threshold wrote {written} bytes, expected {outputLength}."
    output, outputLength

let private thresholdDecodedBytesToPooledInTypeMemory (pixelType: PixelType) (thresholdValue: double) (decoded: byte[]) =
    let output = ArrayPool<byte>.Shared.Rent(decoded.Length)
    match pixelType with
    | UInt8 -> thresholdUInt8PageSimdInto thresholdValue decoded.Length decoded output
    | UInt16 -> thresholdUInt16PageInTypeSimdInto thresholdValue decoded.Length decoded output
    | Float32 -> thresholdFloat32PageInTypeOneVectorInto thresholdValue decoded.Length decoded output
    | _ -> unsupportedPixelType "Zarr in-type threshold" "UInt8, UInt16, and Float32" pixelType
    output, decoded.Length

let private runZarrDirectThreshold opts =
    let pixelType = require "pixel-type" opts |> parsePixelType
    let shape = require "shape" opts |> parseShape
    let input = require "input" opts
    let output = require "output" opts
    let thresholdValue = optional "threshold" "128" opts |> fun s -> Double.Parse(s, invariant)

    ensureCleanDirectory output

    let stopwatch = Stopwatch.StartNew()
    let inputArray = openZarrArray input
    let dataType = zarrDataType pixelType
    validateZarrArrayType inputArray dataType

    let writer = createOmeZarrWriter output "uint8" shape
    alignOutputZarrCodecsWithInput input output

    try
        let outputArray = openZarrArray output
        let outputChunks = zarrChunkLookup outputArray
        for inputChunk in collectZarrChunks inputArray do
            let outputChunk = outputChunks[zarrChunkCoordKey inputChunk]
            let decoded =
                inputArray.ReadChunkDecodedAsync(inputChunk, Threading.CancellationToken.None)
                |> runTask
            let thresholded =
                thresholdDecodedChunkToPooledByteChunk pixelType thresholdValue inputArray inputChunk decoded
            try
                outputArray.WriteChunkDecodedAsync(outputChunk, thresholded.Bytes.AsMemory(0, thresholded.ByteLength), Threading.CancellationToken.None)
                |> runUnitTask
            finally
                StackCore.Chunk.decRef thresholded
    finally
        writer.DisposeAsync().AsTask()
        |> runUnitTask

    stopwatch.Stop()
    writeInternalSeconds stopwatch.Elapsed
    0

let private runZarrDirectThresholdInType opts =
    let pixelType = require "pixel-type" opts |> parsePixelType
    let shape = require "shape" opts |> parseShape
    let input = require "input" opts
    let output = require "output" opts
    let thresholdValue = optional "threshold" "128" opts |> fun s -> Double.Parse(s, invariant)

    ensureCleanDirectory output

    let stopwatch = Stopwatch.StartNew()
    let inputArray = openZarrArray input
    let dataType = zarrDataType pixelType
    validateZarrArrayType inputArray dataType

    let writer = createOmeZarrWriter output dataType shape
    alignOutputZarrCodecsWithInput input output

    try
        let outputArray = openZarrArray output
        let outputChunks = zarrChunkLookup outputArray
        for inputChunk in collectZarrChunks inputArray do
            let outputChunk = outputChunks[zarrChunkCoordKey inputChunk]
            let decoded =
                inputArray.ReadChunkDecodedAsync(inputChunk, Threading.CancellationToken.None)
                |> runTask
            let thresholded, thresholdedLength =
                thresholdDecodedBytesToPooledInTypeMemory pixelType thresholdValue decoded
            try
                outputArray.WriteChunkDecodedAsync(outputChunk, thresholded.AsMemory(0, thresholdedLength), Threading.CancellationToken.None)
                |> runUnitTask
            finally
                ArrayPool<byte>.Shared.Return(thresholded)
    finally
        writer.DisposeAsync().AsTask()
        |> runUnitTask

    stopwatch.Stop()
    writeInternalSeconds stopwatch.Elapsed
    0

let private runZarrDirectThresholdRaw opts =
    let pixelType = require "pixel-type" opts |> parsePixelType
    let shape = require "shape" opts |> parseShape
    let input = require "input" opts
    let output = require "output" opts
    let thresholdValue = optional "threshold" "128" opts |> fun s -> Double.Parse(s, invariant)

    ensureCleanDirectory output

    let stopwatch = Stopwatch.StartNew()
    let inputArray = openZarrArray input
    let dataType = zarrDataType pixelType
    validateZarrArrayType inputArray dataType

    let writer = createOmeZarrWriter output "uint8" shape
    alignOutputZarrCodecsWithInput input output

    try
        let outputArray = openZarrArray output
        let outputChunks = zarrChunkLookup outputArray
        for inputChunk in collectZarrChunks inputArray do
            let outputChunk = outputChunks[zarrChunkCoordKey inputChunk]
            let decoded =
                inputArray.ReadChunkDecodedAsync(inputChunk, Threading.CancellationToken.None)
                |> runTask
            let thresholded, thresholdedLength =
                thresholdDecodedBytesToPooledByteMemoryRaw pixelType thresholdValue decoded
            try
                outputArray.WriteChunkDecodedAsync(outputChunk, thresholded.AsMemory(0, thresholdedLength), Threading.CancellationToken.None)
                |> runUnitTask
            finally
                ArrayPool<byte>.Shared.Return(thresholded)
    finally
        writer.DisposeAsync().AsTask()
        |> runUnitTask

    stopwatch.Stop()
    writeInternalSeconds stopwatch.Elapsed
    0

let private runZarrChunkCopyTyped<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> shape input output availableMemory =
    ensureCleanDirectory output
    let zarrInfo = getZarrInfo input 0 0
    let depth = max 1u (uint zarrInfo.size[2])
    let chunkDepth, chunkY, chunkX =
        match zarrInfo.chunks with
        | x :: y :: z :: _ -> max 1u (uint z), max 1u (uint y), max 1u (uint x)
        | _ -> 16u, 256u, 256u
    let src = benchmarkSource availableMemory
    src
    |> readZarrRange<'T> 0u 1 (depth - 1u) input 0 0 0 0 0
    >=> writeZarr<'T> output "benchmark" depth chunkX chunkY chunkDepth 1.0 1.0 1.0 0
    |> sink
    0

let private runZarrChunkCopy opts =
    let pixelType = require "pixel-type" opts |> parsePixelType
    let shape = require "shape" opts |> parseShape
    let input = require "input" opts
    let output = require "output" opts
    let availableMemory = optional "available-memory" (string (1024UL * 1024UL * 1024UL * 1024UL)) opts |> UInt64.Parse
    precleanDirectoryForTimedRun output
    let stopwatch = Stopwatch.StartNew()
    let exitCode =
        match pixelType with
        | UInt8 -> runZarrChunkCopyTyped<uint8> shape input output availableMemory
        | UInt16 -> runZarrChunkCopyTyped<uint16> shape input output availableMemory
        | Int32 -> runZarrChunkCopyTyped<int32> shape input output availableMemory
        | Float32 -> runZarrChunkCopyTyped<float32> shape input output availableMemory
        | _ -> unsupportedPixelType "Zarr chunk copy benchmark" "UInt8, UInt16, Int32, and Float32" pixelType
    stopwatch.Stop()
    writeInternalSeconds stopwatch.Elapsed
    exitCode

let private runZarrReadOnlyTyped<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> input availableMemory =
    let zarrInfo = getZarrInfo input 0 0
    let depth = max 1u (uint zarrInfo.size[2])
    let src = benchmarkSource availableMemory
    src
    |> readZarrRange<'T> 0u 1 (depth - 1u) input 0 0 0 0 0
    |> sink
    0

let private runZarrReadOnly opts =
    let pixelType = require "pixel-type" opts |> parsePixelType
    let input = require "input" opts
    let availableMemory = optional "available-memory" (string (1024UL * 1024UL * 1024UL * 1024UL)) opts |> UInt64.Parse
    let stopwatch = Stopwatch.StartNew()
    let exitCode =
        match pixelType with
        | UInt8 -> runZarrReadOnlyTyped<uint8> input availableMemory
        | UInt16 -> runZarrReadOnlyTyped<uint16> input availableMemory
        | Int32 -> runZarrReadOnlyTyped<int32> input availableMemory
        | Float32 -> runZarrReadOnlyTyped<float32> input availableMemory
        | _ -> unsupportedPixelType "Zarr read-only benchmark" "UInt8, UInt16, Int32, and Float32" pixelType
    stopwatch.Stop()
    writeInternalSeconds stopwatch.Elapsed
    exitCode

let private runZarrWriteOnlyTyped<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (shape: Shape) output chunkSize availableMemory =
    ensureCleanDirectory output
    let src = benchmarkSource availableMemory
    let chunkSize = max 1u chunkSize
    src
    |> zero<'T> shape.Width shape.Height shape.Depth
    >=> writeZarrWithCompression ZarrCompression.None output "benchmark" shape.Depth chunkSize chunkSize chunkSize 1.0 1.0 1.0 0
    |> sink
    0

let private runZarrWriteOnly opts =
    let pixelType = require "pixel-type" opts |> parsePixelType
    let shape = require "shape" opts |> parseShape
    let output = require "output" opts
    let chunkSize = optional "chunk-size" "128" opts |> UInt32.Parse
    let availableMemory = optional "available-memory" (string (1024UL * 1024UL * 1024UL * 1024UL)) opts |> UInt64.Parse
    precleanDirectoryForTimedRun output
    let stopwatch = Stopwatch.StartNew()
    let exitCode =
        match pixelType with
        | UInt8 -> runZarrWriteOnlyTyped<uint8> shape output chunkSize availableMemory
        | UInt16 -> runZarrWriteOnlyTyped<uint16> shape output chunkSize availableMemory
        | Int32 -> runZarrWriteOnlyTyped<int32> shape output chunkSize availableMemory
        | Float32 -> runZarrWriteOnlyTyped<float32> shape output chunkSize availableMemory
        | _ -> unsupportedPixelType "Zarr write-only benchmark" "UInt8, UInt16, Int32, and Float32" pixelType
    stopwatch.Stop()
    writeInternalSeconds stopwatch.Elapsed
    exitCode

let private runTiffThickReadOnlyTyped<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> input chunkSize availableMemory =
    let src = benchmarkSource availableMemory
    let chunkDepth = max 1u chunkSize
    src
    |> readThick<'T> chunkDepth input ".tiff"
    |> sink
    0

let private runTiffThickReadOnly opts =
    let pixelType = require "pixel-type" opts |> parsePixelType
    let input = require "input" opts
    let chunkSize = optional "chunk-size" "128" opts |> UInt32.Parse
    let availableMemory = optional "available-memory" (string (1024UL * 1024UL * 1024UL * 1024UL)) opts |> UInt64.Parse
    let stopwatch = Stopwatch.StartNew()
    let exitCode =
        match pixelType with
        | UInt8 -> runTiffThickReadOnlyTyped<uint8> input chunkSize availableMemory
        | UInt16 -> runTiffThickReadOnlyTyped<uint16> input chunkSize availableMemory
        | Float32 -> runTiffThickReadOnlyTyped<float32> input chunkSize availableMemory
        | _ -> unsupportedPixelType "TIFF thick read-only benchmark" "UInt8, UInt16, and Float32" pixelType
    stopwatch.Stop()
    writeInternalSeconds stopwatch.Elapsed
    exitCode

let private zarrChunkBufferDrain<'T when 'T: equality> chunkSize : Stage<Chunk<'T>, unit> =
    let name = $"zarrChunkBufferDrain.{typeof<'T>.Name}"
    let chunkSize = max 1u chunkSize |> int
    let elementBytes = int (chunkElementBytes<'T>())
    let bufferBytes = chunkSize * chunkSize * chunkSize * elementBytes
    let mutable checksum = 0

    let consume (debug: bool) (_index: int) (chunk: Chunk<'T>) =
        try
            let width64, height64, depth64 = chunk.Size
            let width = int width64
            let height = int height64
            let depth = int depth64

            if width % chunkSize <> 0 || height % chunkSize <> 0 || depth % chunkSize <> 0 then
                invalidArg "chunk" $"{name} expects aligned thick chunks; got {chunk.Size} for chunk size {chunkSize}."

            let zChunkCount = depth / chunkSize
            let yChunkCount = height / chunkSize
            let xChunkCount = width / chunkSize
            let rowBytes = chunkSize * elementBytes

            for localZChunk in 0 .. zChunkCount - 1 do
                let localZStart = localZChunk * chunkSize
                for yChunk in 0 .. yChunkCount - 1 do
                    let yStart = yChunk * chunkSize
                    for xChunk in 0 .. xChunkCount - 1 do
                        let xStart = xChunk * chunkSize
                        let buffer = ArrayPool<byte>.Shared.Rent(bufferBytes)
                        try
                            for z in 0 .. chunkSize - 1 do
                                let sourceZ = localZStart + z
                                for y in 0 .. chunkSize - 1 do
                                    let sourceOffset =
                                        ((sourceZ * height + (yStart + y)) * width + xStart) * elementBytes
                                    let destinationOffset =
                                        ((z * chunkSize + y) * chunkSize) * elementBytes
                                    Buffer.BlockCopy(chunk.Bytes, sourceOffset, buffer, destinationOffset, rowBytes)
                            checksum <- checksum ^^^ int buffer[0]
                        finally
                            ArrayPool<byte>.Shared.Return(buffer)

            if debug then
                printfn $"[{name}] Split and drained {zChunkCount * yChunkCount * xChunkCount} Zarr chunk buffers; checksum {checksum}."
        finally
            Chunk.decRef chunk

    SlimPipeline.Stage.consumeWith name consume (fun _ -> uint64 bufferBytes)

let private runTiffThickSplitDrainTyped<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> input chunkSize availableMemory =
    let src = benchmarkSource availableMemory
    let chunkDepth = max 1u chunkSize
    src
    |> readThick<'T> chunkDepth input ".tiff"
    >=> zarrChunkBufferDrain<'T> chunkSize
    |> sink
    0

let private runTiffThickSplitDrain opts =
    let pixelType = require "pixel-type" opts |> parsePixelType
    let input = require "input" opts
    let chunkSize = optional "chunk-size" "128" opts |> UInt32.Parse
    let availableMemory = optional "available-memory" (string (1024UL * 1024UL * 1024UL * 1024UL)) opts |> UInt64.Parse
    let stopwatch = Stopwatch.StartNew()
    let exitCode =
        match pixelType with
        | UInt8 -> runTiffThickSplitDrainTyped<uint8> input chunkSize availableMemory
        | UInt16 -> runTiffThickSplitDrainTyped<uint16> input chunkSize availableMemory
        | Float32 -> runTiffThickSplitDrainTyped<float32> input chunkSize availableMemory
        | _ -> unsupportedPixelType "TIFF thick split-drain benchmark" "UInt8, UInt16, and Float32" pixelType
    stopwatch.Stop()
    writeInternalSeconds stopwatch.Elapsed
    exitCode

let private runZarrThickWriteOnlyTyped<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (shape: Shape) output chunkSize availableMemory =
    precleanDirectoryForTimedRun output
    let src = benchmarkSource availableMemory
    let chunkSize = max 1u chunkSize
    src
    |> zero<'T> shape.Width shape.Height shape.Depth
    >=> collectChunkSlicesThick<'T> chunkSize
    >=> writeZarrThickWithCompression<'T> ZarrCompression.None output "benchmark" shape.Depth chunkSize chunkSize chunkSize 1.0 1.0 1.0 0
    |> sink
    0

let private runZarrThickWriteOnly opts =
    let pixelType = require "pixel-type" opts |> parsePixelType
    let shape = require "shape" opts |> parseShape
    let output = require "output" opts
    let chunkSize = optional "chunk-size" "128" opts |> UInt32.Parse
    let availableMemory = optional "available-memory" (string (1024UL * 1024UL * 1024UL * 1024UL)) opts |> UInt64.Parse
    let stopwatch = Stopwatch.StartNew()
    let exitCode =
        match pixelType with
        | UInt8 -> runZarrThickWriteOnlyTyped<uint8> shape output chunkSize availableMemory
        | UInt16 -> runZarrThickWriteOnlyTyped<uint16> shape output chunkSize availableMemory
        | Float32 -> runZarrThickWriteOnlyTyped<float32> shape output chunkSize availableMemory
        | _ -> unsupportedPixelType "Zarr thick write-only benchmark" "UInt8, UInt16, and Float32" pixelType
    stopwatch.Stop()
    writeInternalSeconds stopwatch.Elapsed
    exitCode

let private fillUInt8PatternSlices : Stage<Chunk<uint8>, Chunk<uint8>> =
    SlimPipeline.Stage.mapi
        "fillUInt8PatternSlices"
        (fun _debug z (chunk: Chunk<uint8>) ->
            let width64, height64, depth64 = chunk.Size
            if depth64 <> 1UL then
                invalidArg "chunk" $"fillUInt8PatternSlices expects 2D slices, got {chunk.Size}."
            let width = int width64
            let height = int height64
            let z = int z
            let bytes = chunk.Bytes
            for y in 0 .. height - 1 do
                let rowOffset = y * width
                for x in 0 .. width - 1 do
                    bytes[rowOffset + x] <- byte ((x * 3 + y * 5 + z * 11) &&& 0xFF)
            chunk)
        id
        id

let private runZarrThickWriteOnlyPatternUInt8 (shape: Shape) output chunkSize availableMemory =
    precleanDirectoryForTimedRun output
    let src = benchmarkSource availableMemory
    let chunkSize = max 1u chunkSize
    src
    |> zero<uint8> shape.Width shape.Height shape.Depth
    >=> fillUInt8PatternSlices
    >=> collectChunkSlicesThick<uint8> chunkSize
    >=> writeZarrThickWithCompression<uint8> ZarrCompression.None output "benchmark" shape.Depth chunkSize chunkSize chunkSize 1.0 1.0 1.0 0
    |> sink
    0

let private runZarrThickWriteOnlyPattern opts =
    let pixelType = require "pixel-type" opts |> parsePixelType
    let shape = require "shape" opts |> parseShape
    let output = require "output" opts
    let chunkSize = optional "chunk-size" "128" opts |> UInt32.Parse
    let availableMemory = optional "available-memory" (string (1024UL * 1024UL * 1024UL * 1024UL)) opts |> UInt64.Parse
    let stopwatch = Stopwatch.StartNew()
    let exitCode =
        match pixelType with
        | UInt8 -> runZarrThickWriteOnlyPatternUInt8 shape output chunkSize availableMemory
        | _ -> unsupportedPixelType "Zarr thick pattern write-only benchmark" "UInt8" pixelType
    stopwatch.Stop()
    writeInternalSeconds stopwatch.Elapsed
    exitCode

let private directLocalZarrThickWriter<'T when 'T: equality>
    (shape: Shape)
    output
    chunkSize
    workers
    : Stage<Chunk<'T>, unit> =

    let name = $"directLocalZarrThickWriter.{typeof<'T>.Name}"
    let elementBytes = int (chunkElementBytes<'T>())
    let chunkSize = max 1u chunkSize |> int
    let workers = max 1 workers
    let root = Path.GetFullPath output
    let chunkRoot = Path.Combine(root, "0", "c", "0", "0")
    let mutable nextZ = 0

    let copyChunkRegionToBuffer (source: Chunk<'T>) sourceWidth sourceHeight localZStart yStart xStart (buffer: byte[]) =
        let rowBytes = chunkSize * elementBytes
        for z in 0 .. chunkSize - 1 do
            let sourceZ = localZStart + z
            for y in 0 .. chunkSize - 1 do
                let sourceOffset =
                    ((sourceZ * sourceHeight + (yStart + y)) * sourceWidth + xStart) * elementBytes
                let destinationOffset =
                    ((z * chunkSize + y) * chunkSize) * elementBytes
                Buffer.BlockCopy(source.Bytes, sourceOffset, buffer, destinationOffset, rowBytes)

    let writeOneChunk (source: Chunk<'T>) sourceWidth sourceHeight localZStart zChunk yChunk xChunk =
        let yStart = yChunk * chunkSize
        let xStart = xChunk * chunkSize
        let bufferBytes = chunkSize * chunkSize * chunkSize * elementBytes
        let buffer = ArrayPool<byte>.Shared.Rent(bufferBytes)
        try
            copyChunkRegionToBuffer source sourceWidth sourceHeight localZStart yStart xStart buffer
            let directory = Path.Combine(chunkRoot, string zChunk, string yChunk)
            Directory.CreateDirectory(directory) |> ignore
            let path = Path.Combine(directory, string xChunk)
            use stream =
                new FileStream(
                    path,
                    FileMode.Create,
                    FileAccess.Write,
                    FileShare.None,
                    bufferSize = 1024 * 1024,
                    options = FileOptions.SequentialScan)
            stream.Write(buffer, 0, bufferBytes)
        finally
            ArrayPool<byte>.Shared.Return(buffer)

    let apply (debug: bool) (input: AsyncSeq<Chunk<'T>>) =
        asyncSeq {
            try
                for chunk in input do
                    try
                        let width64, height64, depth64 = chunk.Size
                        let width = int width64
                        let height = int height64
                        let depth = int depth64
                        if width % chunkSize <> 0 || height % chunkSize <> 0 || depth % chunkSize <> 0 then
                            invalidArg "chunk" $"{name} expects full aligned thick chunks; got {chunk.Size} for chunk size {chunkSize}."
                        if nextZ % chunkSize <> 0 then
                            invalidOp $"{name} expected z cursor {nextZ} to be aligned to chunk size {chunkSize}."
                        if nextZ + depth > int shape.Depth then
                            invalidOp $"{name} received chunk ending at z={nextZ + depth}, beyond declared depth {shape.Depth}."

                        let zChunkBase = nextZ / chunkSize
                        let zChunkCount = depth / chunkSize
                        let yChunkCount = height / chunkSize
                        let xChunkCount = width / chunkSize
                        let options = ParallelOptions(MaxDegreeOfParallelism = workers)

                        for localZChunk in 0 .. zChunkCount - 1 do
                            let zChunk = zChunkBase + localZChunk
                            for yChunk in 0 .. yChunkCount - 1 do
                                Directory.CreateDirectory(Path.Combine(chunkRoot, string zChunk, string yChunk)) |> ignore

                            let totalXY = yChunkCount * xChunkCount
                            Parallel.For(
                                0,
                                totalXY,
                                options,
                                fun index ->
                                    let yChunk = index / xChunkCount
                                    let xChunk = index % xChunkCount
                                    writeOneChunk chunk width height (localZChunk * chunkSize) zChunk yChunk xChunk)
                            |> ignore

                        if debug then
                            printfn $"[{name}] Wrote z {nextZ}..{nextZ + depth - 1} to {output}."
                        nextZ <- nextZ + depth
                        yield ()
                    finally
                        Chunk.decRef chunk
            finally
                if nextZ <> int shape.Depth then
                    invalidOp $"{name} wrote {nextZ} slices, expected {shape.Depth}."
        }

    let memoryNeed nPixels =
        nPixels * uint64 (elementBytes * 2)

    let transition = SlimPipeline.ProfileTransition.create SlimPipeline.Streaming SlimPipeline.Streaming
    let memoryModel = SlimPipeline.StageMemoryModel.fromSinglePeak SlimPipeline.Map memoryNeed
    SlimPipeline.Stage.fromAsyncSeq name apply transition memoryModel id

let private runZarrThickWriteOnlyDirectLocalTyped<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (shape: Shape) output chunkSize workers availableMemory pixelType =
    precleanDirectoryForTimedRun output
    let dataType = zarrDataType pixelType
    let chunkSizeInt = max 1u chunkSize |> int
    let writer = createOmeZarrWriterWithChunks output dataType shape chunkSizeInt chunkSizeInt chunkSizeInt
    writer.DisposeAsync().AsTask() |> runUnitTask

    let src = benchmarkSource availableMemory
    src
    |> zero<'T> shape.Width shape.Height shape.Depth
    >=> collectChunkSlicesThick<'T> chunkSize
    >=> directLocalZarrThickWriter<'T> shape output chunkSize workers
    |> sink
    0

let private runZarrThickWriteOnlyDirectLocal opts =
    let pixelType = require "pixel-type" opts |> parsePixelType
    let shape = require "shape" opts |> parseShape
    let output = require "output" opts
    let chunkSize = optional "chunk-size" "128" opts |> UInt32.Parse
    let workers = optional "workers" "16" opts |> Int32.Parse
    let availableMemory = optional "available-memory" (string (1024UL * 1024UL * 1024UL * 1024UL)) opts |> UInt64.Parse
    let stopwatch = Stopwatch.StartNew()
    let exitCode =
        match pixelType with
        | UInt8 -> runZarrThickWriteOnlyDirectLocalTyped<uint8> shape output chunkSize workers availableMemory pixelType
        | UInt16 -> runZarrThickWriteOnlyDirectLocalTyped<uint16> shape output chunkSize workers availableMemory pixelType
        | Float32 -> runZarrThickWriteOnlyDirectLocalTyped<float32> shape output chunkSize workers availableMemory pixelType
        | _ -> unsupportedPixelType "Direct local Zarr thick write-only benchmark" "UInt8, UInt16, and Float32" pixelType
    stopwatch.Stop()
    writeInternalSeconds stopwatch.Elapsed
    exitCode

let private runChunkFftFloat32Zarr opts =
    let shape = require "shape" opts |> parseShape
    let input = require "input" opts
    let output = require "output" opts
    let chunkSize = optional "chunk-size" "64" opts |> UInt32.Parse
    let compression = optional "compression" "none" opts |> parseZarrCompression
    let availableMemory = optional "available-memory" (string (1024UL * 1024UL * 1024UL * 1024UL)) opts |> UInt64.Parse

    ensureCleanDirectory output
    let src = benchmarkSource availableMemory
    let stopwatch = Stopwatch.StartNew()

    src
    |> read<float32> input ".tiff"
    >=> fft
    >=> writeZarrWithCompression compression output "fft" shape.Depth chunkSize chunkSize chunkSize 1.0 1.0 1.0 0
    |> sink

    stopwatch.Stop()
    writeInternalSeconds stopwatch.Elapsed
    0

let private runChunkFftXYFloat32Zarr opts =
    let shape = require "shape" opts |> parseShape
    let input = require "input" opts
    let output = require "output" opts
    let chunkSize = optional "chunk-size" "64" opts |> UInt32.Parse
    let workers = optional "workers" "1" opts |> Int32.Parse
    let availableMemory = optional "available-memory" (string (1024UL * 1024UL * 1024UL * 1024UL)) opts |> UInt64.Parse

    ensureCleanDirectory output
    let src = benchmarkSource availableMemory
    let stopwatch = Stopwatch.StartNew()

    src
    |> read<float32> input ".tiff"
    >=> ChunkFunctions.fftXYFloat32ToComplex64InterleavedParallelCollect workers
    >=> writeZarrComplex64InterleavedFloat32 output "fft_xy" shape.Width shape.Height shape.Depth chunkSize chunkSize chunkSize 1.0 1.0 1.0 0
    |> sink

    stopwatch.Stop()
    writeInternalSeconds stopwatch.Elapsed
    0

let private runChunkFftZComplex64Zarr opts =
    let shape = require "shape" opts |> parseShape
    let input = require "input" opts
    let output = require "output" opts
    let chunkSize = optional "chunk-size" "64" opts |> UInt32.Parse

    ensureCleanDirectory output
    let stopwatch = Stopwatch.StartNew()

    fftZComplex64InterleavedZarrTiles
        input
        output
        "fft"
        shape.Width
        shape.Height
        shape.Depth
        chunkSize
        chunkSize
        chunkSize
        chunkSize
        chunkSize
        1.0
        1.0
        1.0
        0

    stopwatch.Stop()
    writeInternalSeconds stopwatch.Elapsed
    0

let private runChunkFftNativeFloat32Zarr opts =
    let shape = require "shape" opts |> parseShape
    let input = require "input" opts
    let output = require "output" opts
    let chunkSize = optional "chunk-size" "64" opts |> UInt32.Parse
    let workers = optional "workers" "1" opts |> Int32.Parse
    let availableMemory = optional "available-memory" (string (1024UL * 1024UL * 1024UL * 1024UL)) opts |> UInt64.Parse
    let tempXY = output.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + ".xy.tmp.zarr"

    ensureCleanDirectory output
    ensureCleanDirectory tempXY

    let stopwatch = Stopwatch.StartNew()
    try
        let src = benchmarkSource availableMemory
        src
        |> read<float32> input ".tiff"
        >=> ChunkFunctions.fftXYFloat32ToComplex64InterleavedParallelCollect workers
        >=> writeZarrComplex64InterleavedFloat32 tempXY "fft_xy" shape.Width shape.Height shape.Depth chunkSize chunkSize chunkSize 1.0 1.0 1.0 0
        |> sink

        fftZComplex64InterleavedZarrTiles
            tempXY
            output
            "fft"
            shape.Width
            shape.Height
            shape.Depth
            chunkSize
            chunkSize
            chunkSize
            chunkSize
            chunkSize
            1.0
            1.0
            1.0
            0
    finally
        if Directory.Exists tempXY then
            Directory.Delete(tempXY, true)

    stopwatch.Stop()
    writeInternalSeconds stopwatch.Elapsed
    0

type Fft3DKernelVariant =
    | Fft3DSimpleItk
    | Fft3DLowlevel
    | Fft3DLowlevelXYPlan
    | Fft3DLowlevelXYZPlan
    | Fft3DLowlevelFull
    | Fft3DLowlevelRealToComplexFull

type ChunkFft3DStageVariant =
    | ChunkFft3DComplexXY
    | ChunkFft3DRealXY
    | ChunkFft3DRealXYRoundTrip

type ChunkFft3DSpectralZarrVariant =
    | ChunkFft3DSpectralZarrWrite
    | ChunkFft3DSpectralZarrWritePrecomputed
    | ChunkFft3DSpectralZarrRead
    | ChunkFft3DSpectralZarrRoundTrip

let private parseFft3DKernelVariant value =
    match value with
    | "simpleitk" | "SimpleITK" | "sitk" -> [ Fft3DSimpleItk ]
    | "lowlevel" | "native" | "fftw" | "xy-z" -> [ Fft3DLowlevel ]
    | "lowlevel-xy-plan-z" | "xy-plan-z" | "xyplan-z" -> [ Fft3DLowlevelXYPlan ]
    | "lowlevel-xy-z-plan" | "xy-z-plan" | "xyz-plan" -> [ Fft3DLowlevelXYZPlan ]
    | "lowlevel-3d" | "native-3d" | "fftw-3d" -> [ Fft3DLowlevelFull ]
    | "lowlevel-r2c-3d" | "native-r2c-3d" | "fftw-r2c-3d" | "r2c-3d" -> [ Fft3DLowlevelRealToComplexFull ]
    | "all" | "All" | "ALL" -> [ Fft3DSimpleItk; Fft3DLowlevel; Fft3DLowlevelXYPlan; Fft3DLowlevelXYZPlan; Fft3DLowlevelFull; Fft3DLowlevelRealToComplexFull ]
    | _ -> failwith $"unsupported FFT3D kernel variant '{value}'"

let private fft3DKernelVariantName variant =
    match variant with
    | Fft3DSimpleItk -> "simpleitk"
    | Fft3DLowlevel -> "lowlevel-xy-z"
    | Fft3DLowlevelXYPlan -> "lowlevel-xy-plan-z"
    | Fft3DLowlevelXYZPlan -> "lowlevel-xy-z-plan"
    | Fft3DLowlevelFull -> "lowlevel-3d"
    | Fft3DLowlevelRealToComplexFull -> "lowlevel-r2c-3d"

let private parseChunkFft3DStageVariant value =
    match value with
    | "complex-xy" | "complexxy" | "c2c" | "complex" -> [ ChunkFft3DComplexXY ]
    | "real-xy" | "realxy" | "r2c" | "real" -> [ ChunkFft3DRealXY ]
    | "real-xy-roundtrip" | "realxy-roundtrip" | "r2c-c2r" | "roundtrip" -> [ ChunkFft3DRealXYRoundTrip ]
    | "all" | "All" | "ALL" -> [ ChunkFft3DComplexXY; ChunkFft3DRealXY; ChunkFft3DRealXYRoundTrip ]
    | _ -> failwith $"unsupported Chunk FFT3D stage variant '{value}'"

let private chunkFft3DStageVariantName = function
    | ChunkFft3DComplexXY -> "chunk-stage-fft3d-complex-xy-z-plan"
    | ChunkFft3DRealXY -> "chunk-stage-fft3d-real-xy-z-plan"
    | ChunkFft3DRealXYRoundTrip -> "chunk-stage-fft3d-real-xy-z-roundtrip"

let private parseChunkFft3DSpectralZarrVariant value =
    match value with
    | "write" -> ChunkFft3DSpectralZarrWrite
    | "write-precomputed" | "write-only" -> ChunkFft3DSpectralZarrWritePrecomputed
    | "read" -> ChunkFft3DSpectralZarrRead
    | "roundtrip" -> ChunkFft3DSpectralZarrRoundTrip
    | _ -> failwith $"unsupported Chunk FFT3D spectral Zarr variant '{value}'"

let private chunkFft3DSpectralZarrVariantName = function
    | ChunkFft3DSpectralZarrWrite -> "chunk-stage-fft3d-spectral-zarr-write"
    | ChunkFft3DSpectralZarrWritePrecomputed -> "chunk-stage-fft3d-spectral-zarr-write-precomputed"
    | ChunkFft3DSpectralZarrRead -> "chunk-stage-fft3d-spectral-zarr-read"
    | ChunkFft3DSpectralZarrRoundTrip -> "chunk-stage-fft3d-spectral-zarr-roundtrip"

let private fft3DInputValue x y z =
    let value = (x * 17 + y * 31 + z * 43 + (x ^^^ y ^^^ z) * 7) &&& 1023
    (float32 value - 512.0f) / 512.0f

let private fft3DInputArray (shape: Shape) =
    let width = int shape.Width
    let height = int shape.Height
    let depth = int shape.Depth
    Array3D.init width height depth fft3DInputValue

let private fillInterleavedComplex64FromReal (source: float32[,,]) (target: float32[]) =
    let width = source.GetLength 0
    let height = source.GetLength 1
    let depth = source.GetLength 2
    let mutable offset = 0
    for z in 0 .. depth - 1 do
        for y in 0 .. height - 1 do
            for x in 0 .. width - 1 do
                target[offset] <- source[x, y, z]
                target[offset + 1] <- 0.0f
                offset <- offset + 2

let private checksumComplex64Interleaved (values: float32[]) =
    let stride = max 2 ((values.Length / 4096) &&& ~~~1)
    let mutable checksum = 2166136261u
    let mutable i = 0
    while i < values.Length do
        checksum <- (checksum * 16777619u) ^^^ uint32 (BitConverter.SingleToInt32Bits values[i])
        checksum <- (checksum * 16777619u) ^^^ uint32 (BitConverter.SingleToInt32Bits values[i + 1])
        i <- i + stride
    checksum <- (checksum * 16777619u) ^^^ uint32 values.Length
    checksum

let private checksumComplex64InterleavedChunks (chunks: StackCore.Chunk<float32>[]) =
    let totalValues = chunks |> Array.sumBy (fun chunk -> (StackCore.Chunk.span<float32> chunk).Length)
    let stride = max 2 ((totalValues / 4096) &&& ~~~1)
    let mutable checksum = 2166136261u
    let mutable linear = 0
    let mutable nextSample = 0

    for chunk in chunks do
        let values = StackCore.Chunk.span<float32> chunk
        let mutable i = 0
        while i + 1 < values.Length do
            if linear >= nextSample then
                checksum <- (checksum * 16777619u) ^^^ uint32 (BitConverter.SingleToInt32Bits values[i])
                checksum <- (checksum * 16777619u) ^^^ uint32 (BitConverter.SingleToInt32Bits values[i + 1])
                nextSample <- nextSample + stride
            i <- i + 2
            linear <- linear + 2

    checksum <- (checksum * 16777619u) ^^^ uint32 totalValues
    checksum

let private checksumSpectralChunks (chunks: StackCore.SpectralChunk[]) =
    chunks |> Array.map (fun spectral -> spectral.Chunk) |> checksumComplex64InterleavedChunks

let private checksumFloat32Chunks (chunks: StackCore.Chunk<float32>[]) =
    let totalValues = chunks |> Array.sumBy (fun chunk -> (StackCore.Chunk.span<float32> chunk).Length)
    let stride = max 1 (totalValues / 4096)
    let mutable checksum = 2166136261u
    let mutable linear = 0
    let mutable nextSample = 0

    for chunk in chunks do
        let values = StackCore.Chunk.span<float32> chunk
        let mutable i = 0
        while i < values.Length do
            if linear >= nextSample then
                checksum <- (checksum * 16777619u) ^^^ uint32 (BitConverter.SingleToInt32Bits values[i])
                nextSample <- nextSample + stride
            i <- i + 1
            linear <- linear + 1

    checksum <- (checksum * 16777619u) ^^^ uint32 totalValues
    checksum

let private checksumComplexFloat32Array3D (values: ComplexFloat32[,,]) =
    let width = values.GetLength 0
    let height = values.GetLength 1
    let depth = values.GetLength 2
    let total = width * height * depth
    let stride = max 1 (total / 4096)
    let mutable checksum = 2166136261u
    let mutable linear = 0
    while linear < total do
        let x = linear % width
        let y = (linear / width) % height
        let z = linear / (width * height)
        let value = values[x, y, z]
        checksum <- (checksum * 16777619u) ^^^ uint32 (BitConverter.SingleToInt32Bits value.Real)
        checksum <- (checksum * 16777619u) ^^^ uint32 (BitConverter.SingleToInt32Bits value.Imaginary)
        linear <- linear + stride
    checksum <- (checksum * 16777619u) ^^^ uint32 total
    checksum

let private runSimpleItkFft3DKernel iterations (input: Image<float32>) =
    let mutable last = Unchecked.defaultof<Image<ComplexFloat32>>
    let stopwatch = Stopwatch.StartNew()
    try
        for _ in 1 .. iterations do
            if not (isNull (box last)) then
                last.decRefCount()
            use fft = new itk.simple.ForwardFFTImageFilter()
            use inputItk = new itk.simple.Image(input.toSimpleITK())
            let transformed = fft.Execute(inputItk)
            last <- Image<ComplexFloat32>.ofSimpleITKNDispose(transformed, "fft3d.simpleitk", 0)
        stopwatch.Stop()
        let checksum = last.toComplexFloat32Array3D() |> checksumComplexFloat32Array3D
        stopwatch.Elapsed, checksum
    finally
        if not (isNull (box last)) then
            last.decRefCount()

let private runLowlevelFft3DKernel iterations (source: float32[,,]) =
    let width = source.GetLength 0
    let height = source.GetLength 1
    let depth = source.GetLength 2
    let planeComplex = width * height
    let planeValues = planeComplex * 2
    let values = Array.zeroCreate<float32> (planeValues * depth)
    let handle = GCHandle.Alloc(values, GCHandleType.Pinned)
    try
        let basePointer = handle.AddrOfPinnedObject()
        let stopwatch = Stopwatch.StartNew()
        for _ in 1 .. iterations do
            fillInterleavedComplex64FromReal source values
            for z in 0 .. depth - 1 do
                let planePointer = IntPtr.Add(basePointer, z * planeValues * sizeof<float32>)
                Chunk.NativeSp.fftwfComplexXYInplace(planePointer, width, height, 0)
                |> Chunk.NativeSp.checkStatus "fft3d.lowlevel.xy"
            Chunk.NativeSp.fftwfComplexZInplace(basePointer, width, height, depth, 0)
            |> Chunk.NativeSp.checkStatus "fft3d.lowlevel.z"
        stopwatch.Stop()
        stopwatch.Elapsed, checksumComplex64Interleaved values
    finally
        handle.Free()

let private runLowlevelXYPlanFft3DKernel iterations (source: float32[,,]) =
    let width = source.GetLength 0
    let height = source.GetLength 1
    let depth = source.GetLength 2
    let planeComplex = width * height
    let planeValues = planeComplex * 2
    let values = Array.zeroCreate<float32> (planeValues * depth)
    let handle = GCHandle.Alloc(values, GCHandleType.Pinned)
    let mutable xyPlan = nativeint 0
    try
        let basePointer = handle.AddrOfPinnedObject()
        fillInterleavedComplex64FromReal source values
        xyPlan <- Chunk.NativeSp.fftwfComplexXYPlanCreate(basePointer, width, height, 0)
        if xyPlan = nativeint 0 then
            invalidOp "fft3d.lowlevel.xy.plan create failed in native helper."

        let stopwatch = Stopwatch.StartNew()
        for _ in 1 .. iterations do
            fillInterleavedComplex64FromReal source values
            for z in 0 .. depth - 1 do
                let planePointer = IntPtr.Add(basePointer, z * planeValues * sizeof<float32>)
                Chunk.NativeSp.fftwfComplexXYPlanExecute(xyPlan, planePointer)
                |> Chunk.NativeSp.checkStatus "fft3d.lowlevel.xy.plan.execute"
            Chunk.NativeSp.fftwfComplexZInplace(basePointer, width, height, depth, 0)
            |> Chunk.NativeSp.checkStatus "fft3d.lowlevel.z"
        stopwatch.Stop()
        stopwatch.Elapsed, checksumComplex64Interleaved values
    finally
        if xyPlan <> nativeint 0 then
            Chunk.NativeSp.fftwfComplexXYPlanDestroy(xyPlan)
        handle.Free()

let private runLowlevelXYZPlanFft3DKernel iterations (source: float32[,,]) =
    let width = source.GetLength 0
    let height = source.GetLength 1
    let depth = source.GetLength 2
    let planeComplex = width * height
    let planeValues = planeComplex * 2
    let values = Array.zeroCreate<float32> (planeValues * depth)
    let handle = GCHandle.Alloc(values, GCHandleType.Pinned)
    let mutable xyPlan = nativeint 0
    let mutable zPlan = nativeint 0
    try
        let basePointer = handle.AddrOfPinnedObject()
        fillInterleavedComplex64FromReal source values
        xyPlan <- Chunk.NativeSp.fftwfComplexXYPlanCreate(basePointer, width, height, 0)
        if xyPlan = nativeint 0 then
            invalidOp "fft3d.lowlevel.xy.plan create failed in native helper."
        zPlan <- Chunk.NativeSp.fftwfComplexZPlanCreate(basePointer, width, height, depth, 0)
        if zPlan = nativeint 0 then
            invalidOp "fft3d.lowlevel.z.plan create failed in native helper."

        let stopwatch = Stopwatch.StartNew()
        for _ in 1 .. iterations do
            fillInterleavedComplex64FromReal source values
            for z in 0 .. depth - 1 do
                let planePointer = IntPtr.Add(basePointer, z * planeValues * sizeof<float32>)
                Chunk.NativeSp.fftwfComplexXYPlanExecute(xyPlan, planePointer)
                |> Chunk.NativeSp.checkStatus "fft3d.lowlevel.xy.plan.execute"
            Chunk.NativeSp.fftwfComplexZPlanExecute(zPlan, basePointer)
            |> Chunk.NativeSp.checkStatus "fft3d.lowlevel.z.plan.execute"
        stopwatch.Stop()
        stopwatch.Elapsed, checksumComplex64Interleaved values
    finally
        if zPlan <> nativeint 0 then
            Chunk.NativeSp.fftwfComplexZPlanDestroy(zPlan)
        if xyPlan <> nativeint 0 then
            Chunk.NativeSp.fftwfComplexXYPlanDestroy(xyPlan)
        handle.Free()

let private runLowlevelFullFft3DKernel iterations (source: float32[,,]) =
    let width = source.GetLength 0
    let height = source.GetLength 1
    let depth = source.GetLength 2
    let complexCount = width * height * depth
    let values = Array.zeroCreate<float32> (complexCount * 2)
    let handle = GCHandle.Alloc(values, GCHandleType.Pinned)
    try
        let basePointer = handle.AddrOfPinnedObject()
        let stopwatch = Stopwatch.StartNew()
        for _ in 1 .. iterations do
            fillInterleavedComplex64FromReal source values
            Chunk.NativeSp.fftwfComplex3DInplace(basePointer, width, height, depth, 0)
            |> Chunk.NativeSp.checkStatus "fft3d.lowlevel.full"
        stopwatch.Stop()
        stopwatch.Elapsed, checksumComplex64Interleaved values
    finally
        handle.Free()

let private fillRealVolumeFromReal (source: float32[,,]) (target: float32[]) =
    let width = source.GetLength 0
    let height = source.GetLength 1
    let depth = source.GetLength 2
    let mutable offset = 0
    for z in 0 .. depth - 1 do
        for y in 0 .. height - 1 do
            for x in 0 .. width - 1 do
                target[offset] <- source[x, y, z]
                offset <- offset + 1

let private runLowlevelRealToComplexFullFft3DKernel iterations (source: float32[,,]) =
    let width = source.GetLength 0
    let height = source.GetLength 1
    let depth = source.GetLength 2
    let realCount = width * height * depth
    let complexWidth = width / 2 + 1
    let complexCount = complexWidth * height * depth
    let real = Array.zeroCreate<float32> realCount
    let values = Array.zeroCreate<float32> (complexCount * 2)
    let realHandle = GCHandle.Alloc(real, GCHandleType.Pinned)
    let complexHandle = GCHandle.Alloc(values, GCHandleType.Pinned)
    try
        let realPointer = realHandle.AddrOfPinnedObject()
        let complexPointer = complexHandle.AddrOfPinnedObject()
        let stopwatch = Stopwatch.StartNew()
        for _ in 1 .. iterations do
            fillRealVolumeFromReal source real
            Chunk.NativeSp.fftwfReal3DToComplex(realPointer, complexPointer, width, height, depth)
            |> Chunk.NativeSp.checkStatus "fft3d.lowlevel.r2c.full"
        stopwatch.Stop()
        stopwatch.Elapsed, checksumComplex64Interleaved values
    finally
        complexHandle.Free()
        realHandle.Free()

let private runFft3DKernel opts =
    let shape = require "shape" opts |> parseShape
    let iterations = optional "iterations" "1" opts |> Int32.Parse
    if iterations < 1 then
        invalidArg "iterations" $"FFT3D kernel benchmark expects positive iterations, got {iterations}."
    if shape.Width > uint Int32.MaxValue || shape.Height > uint Int32.MaxValue || shape.Depth > uint Int32.MaxValue then
        invalidArg "shape" $"FFT3D kernel benchmark dimensions must fit Int32, got {shape.Width}x{shape.Height}x{shape.Depth}."

    let variants = optional "variant" "all" opts |> parseFft3DKernelVariant
    let source = fft3DInputArray shape
    let inputImage =
        if variants |> List.contains Fft3DSimpleItk then
            Some (Image<float32>.ofArray3D(source, "fft3d.input", 0))
        else
            None

    try
        let mutable total = TimeSpan.Zero
        for variant in variants do
            let elapsed, checksum =
                match variant with
                | Fft3DSimpleItk -> runSimpleItkFft3DKernel iterations inputImage.Value
                | Fft3DLowlevel -> runLowlevelFft3DKernel iterations source
                | Fft3DLowlevelXYPlan -> runLowlevelXYPlanFft3DKernel iterations source
                | Fft3DLowlevelXYZPlan -> runLowlevelXYZPlanFft3DKernel iterations source
                | Fft3DLowlevelFull -> runLowlevelFullFft3DKernel iterations source
                | Fft3DLowlevelRealToComplexFull -> runLowlevelRealToComplexFullFft3DKernel iterations source
            total <- total + elapsed
            printfn
                "variant=%s shape=%ux%ux%u iterations=%d totalSeconds=%s perIterationSeconds=%s checksum=%u"
                (fft3DKernelVariantName variant)
                shape.Width
                shape.Height
                shape.Depth
                iterations
                (elapsed.TotalSeconds.ToString("F9", invariant))
                ((elapsed.TotalSeconds / float iterations).ToString("F9", invariant))
                checksum

        writeInternalSeconds total
        0
    finally
        inputImage |> Option.iter (fun image -> image.decRefCount())

let private fft3DInputChunks (shape: Shape) =
    if shape.Width > uint Int32.MaxValue || shape.Height > uint Int32.MaxValue || shape.Depth > uint Int32.MaxValue then
        invalidArg "shape" $"Chunk FFT3D kernel benchmark dimensions must fit Int32, got {shape.Width}x{shape.Height}x{shape.Depth}."

    let width = int shape.Width
    let height = int shape.Height
    let depth = int shape.Depth
    let plane = width * height

    Array.init depth (fun z ->
        let chunk = StackCore.Chunk.create<float32> (uint64 width, uint64 height, 1UL)
        let values = StackCore.Chunk.span<float32> chunk
        if values.Length <> plane then
            invalidOp $"Created slice has {values.Length} pixels, expected {plane}."

        for y in 0 .. height - 1 do
            let row = y * width
            for x in 0 .. width - 1 do
                values[row + x] <- fft3DInputValue x y z

        chunk)

let private cloneFloat32Chunks (input: StackCore.Chunk<float32>[]) =
    input
    |> Array.map (fun source ->
        let target = StackCore.Chunk.create<float32> source.Size
        (StackCore.Chunk.span<float32> source).CopyTo(StackCore.Chunk.span<float32> target)
        target)

let private runDirectChunkRealXYRoundTripWithPlans
    (forwardPlans: ChunkCore.ChunkFunctions.FftRealXYAndZPlanCache)
    (inversePlans: ChunkCore.ChunkFunctions.InvFftRealXYAndZPlanCache)
    (input: StackCore.Chunk<float32>[])
    =
    let ownedInput = cloneFloat32Chunks input
    let mutable spectral: StackCore.SpectralChunk[] = [||]
    let mutable output: StackCore.Chunk<float32>[] = [||]
    try
        spectral <- forwardPlans.ForwardFloat32SlicesToComplex64Interleaved(ownedInput)
        output <- inversePlans.InverseHermitianPackedToFloat32Slices(spectral)
        checksumFloat32Chunks output
    finally
        output |> Array.iter StackCore.Chunk.decRef
        spectral |> Array.iter (fun item -> StackCore.Chunk.decRef item.Chunk)
        ownedInput |> Array.iter StackCore.Chunk.decRef

let private runDirectChunkRealXYRoundTrip (input: StackCore.Chunk<float32>[]) =
    use forwardPlans = new ChunkCore.ChunkFunctions.FftRealXYAndZPlanCache()
    use inversePlans = new ChunkCore.ChunkFunctions.InvFftRealXYAndZPlanCache()
    runDirectChunkRealXYRoundTripWithPlans forwardPlans inversePlans input

let private runStageChunkRealXYRoundTrip (depth: int) (input: StackCore.Chunk<float32>[]) =
    let ownedInput = cloneFloat32Chunks input
    let spectral =
        (fft3DRealXY depth).Build().Apply false (AsyncSeq.ofSeq ownedInput)
    let output =
        (invFft3DRealXY depth).Build().Apply false spectral
        |> AsyncSeq.toListAsync
        |> Async.RunSynchronously
        |> List.toArray

    try
        checksumFloat32Chunks output
    finally
        output |> Array.iter StackCore.Chunk.decRef

let private runChunkFft3DStageOverhead opts =
    let shape = require "shape" opts |> parseShape
    let iterations = optional "iterations" "1" opts |> Int32.Parse
    if iterations < 1 then
        invalidArg "iterations" $"Chunk FFT3D stage overhead benchmark expects positive iterations, got {iterations}."

    let depth = int shape.Depth
    let source = fft3DInputChunks shape

    try
        let runTimed variant runner =
            StackCore.Chunk.resetStats()
            let stopwatch = Stopwatch.StartNew()
            let mutable checksum = 0u

            for _ in 1 .. iterations do
                checksum <- runner source

            stopwatch.Stop()
            printfn
                "variant=%s shape=%ux%ux%u iterations=%d totalSeconds=%s perIterationSeconds=%s checksum=%u chunkStats=%s"
                variant
                shape.Width
                shape.Height
                shape.Depth
                iterations
                (stopwatch.Elapsed.TotalSeconds.ToString("F9", invariant))
                ((stopwatch.Elapsed.TotalSeconds / float iterations).ToString("F9", invariant))
                checksum
                (StackCore.Chunk.stats() |> StackCore.Chunk.formatStats)

        runTimed "chunk-core-direct" runDirectChunkRealXYRoundTrip

        use forwardPlans = new ChunkCore.ChunkFunctions.FftRealXYAndZPlanCache()
        use inversePlans = new ChunkCore.ChunkFunctions.InvFftRealXYAndZPlanCache()
        runTimed
            "chunk-core-reused-plans"
            (runDirectChunkRealXYRoundTripWithPlans forwardPlans inversePlans)

        runTimed "stack-stage" (runStageChunkRealXYRoundTrip depth)

        0
    finally
        source |> Array.iter StackCore.Chunk.decRef

let private runChunkFft3DStageIo opts =
    let shape = require "shape" opts |> parseShape
    let input = require "input" opts
    let output = require "output" opts
    let availableMemory = optional "available-memory" (string (1024UL * 1024UL * 1024UL * 1024UL)) opts |> UInt64.Parse
    let depth = int shape.Depth

    ensureCleanDirectory output
    StackCore.Chunk.resetStats()

    let stopwatch = Stopwatch.StartNew()
    benchmarkSource availableMemory
    |> read<float32> input ".tiff"
    >=> fft3DRealXY depth
    >=> invFft3DRealXY depth
    >=> write<float32> output ".tiff"
    |> sink
    stopwatch.Stop()

    printfn
        "variant=chunk-fft3d-stage-io shape=%ux%ux%u totalSeconds=%s chunkStats=%s"
        shape.Width
        shape.Height
        shape.Depth
        (stopwatch.Elapsed.TotalSeconds.ToString("F9", invariant))
        (StackCore.Chunk.stats() |> StackCore.Chunk.formatStats)

    writeInternalSeconds stopwatch.Elapsed
    0

let private runChunkFft3DKernel opts =
    let shape = require "shape" opts |> parseShape
    let iterations = optional "iterations" "1" opts |> Int32.Parse
    if iterations < 1 then
        invalidArg "iterations" $"Chunk FFT3D kernel benchmark expects positive iterations, got {iterations}."

    let variants = optional "variant" "all" opts |> parseChunkFft3DStageVariant
    let depth = int shape.Depth
    let mutable total = TimeSpan.Zero

    for variant in variants do
        StackCore.Chunk.resetStats()
        let stopwatch = Stopwatch.StartNew()
        let mutable checksum = 0u

        for _ in 1 .. iterations do
            let input = fft3DInputChunks shape
            match variant with
            | ChunkFft3DComplexXY ->
                let output =
                    (fft3DComplexXY depth).Build().Apply false (AsyncSeq.ofSeq input)
                    |> AsyncSeq.toListAsync
                    |> Async.RunSynchronously
                    |> List.toArray

                try
                    checksum <- checksumComplex64InterleavedChunks output
                finally
                    output |> Array.iter StackCore.Chunk.decRef
            | ChunkFft3DRealXY ->
                let output =
                    (fft3DRealXY depth).Build().Apply false (AsyncSeq.ofSeq input)
                    |> AsyncSeq.toListAsync
                    |> Async.RunSynchronously
                    |> List.toArray

                try
                    checksum <- checksumSpectralChunks output
                finally
                    output |> Array.iter (fun spectral -> StackCore.Chunk.decRef spectral.Chunk)
            | ChunkFft3DRealXYRoundTrip ->
                let spectral =
                    (fft3DRealXY depth).Build().Apply false (AsyncSeq.ofSeq input)
                    |> AsyncSeq.toListAsync
                    |> Async.RunSynchronously
                    |> List.toArray

                let output =
                    (invFft3DRealXY depth).Build().Apply false (AsyncSeq.ofSeq spectral)
                    |> AsyncSeq.toListAsync
                    |> Async.RunSynchronously
                    |> List.toArray

                try
                    checksum <- checksumFloat32Chunks output
                finally
                    output |> Array.iter StackCore.Chunk.decRef

        stopwatch.Stop()
        total <- total + stopwatch.Elapsed
        printfn
            "variant=%s shape=%ux%ux%u iterations=%d totalSeconds=%s perIterationSeconds=%s checksum=%u chunkStats=%s"
            (chunkFft3DStageVariantName variant)
            shape.Width
            shape.Height
            shape.Depth
            iterations
            (stopwatch.Elapsed.TotalSeconds.ToString("F9", invariant))
            ((stopwatch.Elapsed.TotalSeconds / float iterations).ToString("F9", invariant))
            checksum
            (StackCore.Chunk.stats() |> StackCore.Chunk.formatStats)

    writeInternalSeconds total
    0

let private runChunkFft3DSpectralZarr opts =
    let shape = require "shape" opts |> parseShape
    let output = require "output" opts
    let variant = optional "variant" "write" opts |> parseChunkFft3DSpectralZarrVariant
    let chunkSize = optional "chunk-size" "64" opts |> UInt32.Parse
    let iterations = optional "iterations" "1" opts |> Int32.Parse
    let availableMemory = optional "available-memory" (string (1024UL * 1024UL * 1024UL * 1024UL)) opts |> UInt64.Parse
    if iterations < 1 then
        invalidArg "iterations" $"Chunk FFT3D spectral Zarr benchmark expects positive iterations, got {iterations}."

    let depth = int shape.Depth

    let writeSpectralStore () =
        ensureCleanDirectory output
        let input = fft3DInputChunks shape
        let spectral =
            (fft3DRealXY depth).Build().Apply false (AsyncSeq.ofSeq input)
        let writer =
            writeZarrSpectralComplex64InterleavedFloat32
                output
                "fft_real_xy"
                shape.Width
                shape.Height
                shape.Depth
                chunkSize
                chunkSize
                chunkSize
                1.0
                1.0
                1.0
                0

        writer.Build().Apply false spectral
        |> AsyncSeq.toListAsync
        |> Async.RunSynchronously
        |> ignore

    let releaseSpectral =
        SlimPipeline.Stage.consumeWith
            "releaseSpectralChunk"
            (fun _debug _index (spectral: StackCore.SpectralChunk) -> StackCore.Chunk.decRef spectral.Chunk)
            (fun _ -> 0UL)

    let releaseFloat32 =
        SlimPipeline.Stage.consumeWith
            "releaseFloat32Chunk"
            (fun _debug _index (chunk: StackCore.Chunk<float32>) -> StackCore.Chunk.decRef chunk)
            (fun _ -> 0UL)

    match variant with
    | ChunkFft3DSpectralZarrRead
    | ChunkFft3DSpectralZarrRoundTrip ->
        if not (File.Exists(Path.Combine(output, "0", "zarr.json"))) then
            writeSpectralStore ()
    | ChunkFft3DSpectralZarrWrite
    | ChunkFft3DSpectralZarrWritePrecomputed -> ()

    StackCore.Chunk.resetStats()
    let stopwatch = Stopwatch()

    for _ in 1 .. iterations do
        match variant with
        | ChunkFft3DSpectralZarrWrite ->
            stopwatch.Start()
            writeSpectralStore ()
            stopwatch.Stop()
        | ChunkFft3DSpectralZarrWritePrecomputed ->
            ensureCleanDirectory output
            let input = fft3DInputChunks shape
            let spectral =
                (fft3DRealXY depth).Build().Apply false (AsyncSeq.ofSeq input)
                |> AsyncSeq.toListAsync
                |> Async.RunSynchronously
                |> List.toArray

            let writer =
                writeZarrSpectralComplex64InterleavedFloat32
                    output
                    "fft_real_xy"
                    shape.Width
                    shape.Height
                    shape.Depth
                    chunkSize
                    chunkSize
                    chunkSize
                    1.0
                    1.0
                    1.0
                    0

            stopwatch.Start()
            writer.Build().Apply false (AsyncSeq.ofSeq spectral)
            |> AsyncSeq.toListAsync
            |> Async.RunSynchronously
            |> ignore
            stopwatch.Stop()
        | ChunkFft3DSpectralZarrRead ->
            stopwatch.Start()
            benchmarkSource availableMemory
            |> readZarrSpectralComplex64InterleavedFloat32Range 0u 1 (shape.Depth - 1u) output 0 0 0 0 0
            >=> releaseSpectral
            |> sink
            stopwatch.Stop()
        | ChunkFft3DSpectralZarrRoundTrip ->
            stopwatch.Start()
            benchmarkSource availableMemory
            |> readZarrSpectralComplex64InterleavedFloat32Range 0u 1 (shape.Depth - 1u) output 0 0 0 0 0
            >=> invFft3DRealXY depth
            >=> releaseFloat32
            |> sink
            stopwatch.Stop()

    printfn
        "variant=%s shape=%ux%ux%u iterations=%d totalSeconds=%s perIterationSeconds=%s chunkStats=%s"
        (chunkFft3DSpectralZarrVariantName variant)
        shape.Width
        shape.Height
        shape.Depth
        iterations
        (stopwatch.Elapsed.TotalSeconds.ToString("F9", invariant))
        ((stopwatch.Elapsed.TotalSeconds / float iterations).ToString("F9", invariant))
        (StackCore.Chunk.stats() |> StackCore.Chunk.formatStats)

    writeInternalSeconds stopwatch.Elapsed
    0

let private runChunkFft3DZarrRoundtripIo opts =
    let shape = require "shape" opts |> parseShape
    let input = require "input" opts
    let output = require "output" opts
    let chunkSize = optional "chunk-size" "64" opts |> UInt32.Parse
    let availableMemory = optional "available-memory" (string (1024UL * 1024UL * 1024UL * 1024UL)) opts |> UInt64.Parse
    let tempBase =
        optional
            "temp-zarr"
            (output.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + ".spectral.tmp.zarr")
            opts
    let tempXY = tempBase.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + ".xy"
    let tempZ = tempBase.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + ".z"
    let tempInvZ = tempBase.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + ".invz"

    let depth = int shape.Depth
    let packedComplexWidth = shape.Width / 2u + 1u
    let setSpectralDepth =
        SlimPipeline.Stage.map
            "setSpectralLogicalDepth"
            (fun _debug (spectral: StackCore.SpectralChunk) ->
                { spectral with LogicalSize = (shape.Width |> uint64, shape.Height |> uint64, shape.Depth |> uint64) })
            id
            id
    let setSpectralSliceDepth =
        SlimPipeline.Stage.map
            "setSpectralLogicalSliceDepth"
            (fun _debug (spectral: StackCore.SpectralChunk) ->
                { spectral with LogicalSize = (shape.Width |> uint64, shape.Height |> uint64, 1UL) })
            id
            id

    ensureCleanDirectory output
    ensureCleanDirectory tempXY
    ensureCleanDirectory tempZ
    ensureCleanDirectory tempInvZ

    StackCore.Chunk.resetStats()
    let stopwatch = Stopwatch.StartNew()
    try
        benchmarkSource availableMemory
        |> read<float32> input ".tiff"
        >=> fftRealXY
        >=> setSpectralDepth
        >=> writeZarrSpectralComplex64InterleavedFloat32
                tempXY
                "fft_real_xy"
                shape.Width
                shape.Height
                shape.Depth
                chunkSize
                chunkSize
                chunkSize
                1.0
                1.0
                1.0
                0
        |> sink

        fftZComplex64InterleavedZarrRawChunks
            tempXY
            tempZ
            "fft_z"
            packedComplexWidth
            shape.Height
            shape.Depth
            chunkSize
            1.0
            1.0
            1.0
            0

        invFftZComplex64InterleavedZarrRawChunks
            tempZ
            tempInvZ
            "inv_fft_z"
            packedComplexWidth
            shape.Height
            shape.Depth
            chunkSize
            1.0
            1.0
            1.0
            0

        benchmarkSource availableMemory
        |> readZarrSpectralComplex64InterleavedFloat32Range 0u 1 (shape.Depth - 1u) tempInvZ 0 0 0 0 0
        >=> setSpectralSliceDepth
        >=> invFftRealXY
        >=> write<float32> output ".tiff"
        |> sink

        stopwatch.Stop()
        printfn
            "variant=chunk-fft3d-zarr-roundtrip-io shape=%ux%ux%u totalSeconds=%s chunkStats=%s"
            shape.Width
            shape.Height
            shape.Depth
            (stopwatch.Elapsed.TotalSeconds.ToString("F9", invariant))
            (StackCore.Chunk.stats() |> StackCore.Chunk.formatStats)

        writeInternalSeconds stopwatch.Elapsed
        0
    finally
        if stopwatch.IsRunning then
            stopwatch.Stop()
        for tempPath in [ tempXY; tempZ; tempInvZ ] do
            if Directory.Exists tempPath then
                Directory.Delete(tempPath, true)

let private runChunkFft3DZarrSubchunkedRoundtripIo opts =
    let shape = require "shape" opts |> parseShape
    let input = require "input" opts
    let output = require "output" opts
    let chunkSize = optional "chunk-size" "64" opts |> UInt32.Parse
    let availableMemory = optional "available-memory" (string (1024UL * 1024UL * 1024UL * 1024UL)) opts |> UInt64.Parse
    let tempBase =
        optional
            "temp-zarr"
            (output.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + ".spectral-subchunked.tmp.zarr")
            opts
    let tempXY = tempBase.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + ".xy"
    let tempInvZ = tempBase.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + ".invz"

    let packedComplexWidth = shape.Width / 2u + 1u
    let setSpectralDepth =
        SlimPipeline.Stage.map
            "setSpectralLogicalDepth"
            (fun _debug (spectral: StackCore.SpectralChunk) ->
                { spectral with LogicalSize = (shape.Width |> uint64, shape.Height |> uint64, shape.Depth |> uint64) })
            id
            id
    let setSpectralSliceDepth =
        SlimPipeline.Stage.map
            "setSpectralLogicalSliceDepth"
            (fun _debug (spectral: StackCore.SpectralChunk) ->
                { spectral with LogicalSize = (shape.Width |> uint64, shape.Height |> uint64, 1UL) })
            id
            id

    ensureCleanDirectory output
    ensureCleanDirectory tempXY
    ensureCleanDirectory tempInvZ

    StackCore.Chunk.resetStats()
    let stopwatch = Stopwatch.StartNew()
    try
        benchmarkSource availableMemory
        |> read<float32> input ".tiff"
        >=> fftRealXY
        >=> setSpectralDepth
        >=> writeZarrSpectralComplex64InterleavedFloat32Tiled
                tempXY
                "fft_real_xy_tiled"
                shape.Width
                shape.Height
                shape.Depth
                chunkSize
                chunkSize
                chunkSize
                1.0
                1.0
                1.0
                0
        |> sink

        fftRoundtripZComplex64InterleavedZarrSubchunks
            tempXY
            tempInvZ
            "fft_z_roundtrip_subchunked"
            packedComplexWidth
            shape.Height
            shape.Depth
            chunkSize
            chunkSize
            chunkSize
            1.0
            1.0
            1.0
            0

        benchmarkSource availableMemory
        |> readZarrSpectralComplex64InterleavedFloat32TiledRange 0u 1 (shape.Depth - 1u) tempInvZ 0 0 0 0 0
        >=> setSpectralSliceDepth
        >=> invFftRealXY
        >=> write<float32> output ".tiff"
        |> sink

        stopwatch.Stop()
        printfn
            "variant=chunk-fft3d-zarr-subchunked-roundtrip-io shape=%ux%ux%u chunkSize=%u totalSeconds=%s chunkStats=%s"
            shape.Width
            shape.Height
            shape.Depth
            chunkSize
            (stopwatch.Elapsed.TotalSeconds.ToString("F9", invariant))
            (StackCore.Chunk.stats() |> StackCore.Chunk.formatStats)

        writeInternalSeconds stopwatch.Elapsed
        0
    finally
        if stopwatch.IsRunning then
            stopwatch.Stop()
        for tempPath in [ tempXY; tempInvZ ] do
            if Directory.Exists tempPath then
                Directory.Delete(tempPath, true)

let private runConnectedComponents input output windowSize availableMemory =
    let window = max 1u windowSize
    runChunkConnectedComponents input (Some output) 128.0 (Some (int window)) 3 availableMemory

let private run opts =
    let operation = require "operation" opts
    let pixelType = require "pixel-type" opts |> parsePixelType
    let input = require "input" opts
    let output = require "output" opts
    let radius = optional "radius" "1" opts |> UInt32.Parse
    let windowSize = optional "window" "16" opts |> UInt32.Parse
    let workerCount = optional "workers" "3" opts |> Int32.Parse
    let kernelSize = optional "kernel-size" "3" opts |> UInt32.Parse
    let thresholdValue = optional "threshold" "128" opts |> fun s -> Double.Parse(s, invariant)
    let availableMemory = optional "available-memory" (string (1024UL * 1024UL * 1024UL * 1024UL)) opts |> UInt64.Parse
    precleanDirectoryForTimedRun output
    let stopwatch = Stopwatch.StartNew()
    let exitCode =
        match operation, pixelType with
        | "copy", UInt8 -> runChunkCopyTyped<uint8> input output availableMemory
        | "copy", UInt16 -> runChunkCopyTyped<uint16> input output availableMemory
        | "copy", Float32 -> runChunkCopyTyped<float32> input output availableMemory
        | "threshold", UInt8 -> runChunkThresholdTyped<uint8> input output thresholdValue availableMemory
        | "threshold", UInt16 -> runChunkThresholdTyped<uint16> input output thresholdValue availableMemory
        | "threshold", Float32 -> runChunkThresholdTyped<float32> input output thresholdValue availableMemory
        | "median", UInt8 -> runChunkMedianStandardUInt8 input output radius availableMemory
        | "median", UInt16 -> runChunkMedianNativeNthUInt16 input output radius workerCount availableMemory
        | "median", Float32 -> runChunkMedianNativeNthFloat32 input output radius workerCount availableMemory
        | "median-ph", UInt8 -> runChunkMedianPhUInt8 input output radius availableMemory
        | "median-ph", _ -> failwith "median-ph benchmark is currently defined for UInt8 chunks only"
        | "median-ph-ybands", UInt8 -> runChunkMedianPhYBandsUInt8 input output radius (int windowSize) availableMemory
        | "median-ph-ybands", _ -> failwith "median-ph-ybands benchmark is currently defined for UInt8 chunks only"
        | "median-ph-xfirst", UInt8 -> runChunkMedianPhXFirstUInt8 input output radius availableMemory
        | "median-ph-xfirst", _ -> failwith "median-ph-xfirst benchmark is currently defined for UInt8 chunks only"
        | "median-ph-xblock", UInt8 -> runChunkMedianPhXBlockUInt8 input output radius availableMemory
        | "median-ph-xblock", _ -> failwith "median-ph-xblock benchmark is currently defined for UInt8 chunks only"
        | "median-ph-xtranspose", UInt8 -> runChunkMedianPhXTransposeUInt8 input output radius availableMemory
        | "median-ph-xtranspose", _ -> failwith "median-ph-xtranspose benchmark is currently defined for UInt8 chunks only"
        | "median-ph-tree", UInt8 -> runChunkMedianPhTreeUInt8 input output radius availableMemory
        | "median-ph-tree", _ -> failwith "median-ph-tree benchmark is currently defined for UInt8 chunks only"
        | "median-ph-blockedz", UInt8 -> runChunkMedianPhBlockedZUInt8 input output radius availableMemory
        | "median-ph-blockedz", _ -> failwith "median-ph-blockedz benchmark is currently defined for UInt8 chunks only"
        | "median-itk-chunk", _ -> failwith "median-itk-chunk used the retired ITK-wrapped median path; use median-native-nth or the direct Image benchmark instead."
        | "median-quickselect", UInt16 -> runChunkMedianQuickselectUInt16 input output radius (int windowSize) availableMemory
        | "median-quickselect", _ -> failwith "median-quickselect benchmark is currently defined for UInt16 chunks only"
        | "median-sort", UInt16 -> runChunkMedianSortUInt16 input output radius (int windowSize) availableMemory
        | "median-sort", _ -> failwith "median-sort benchmark is currently defined for UInt16 chunks only"
        | "median-native-nth", UInt8 -> runChunkMedianNativeNthUInt8 input output radius (int windowSize) availableMemory
        | "median-native-nth", UInt16 -> runChunkMedianNativeNthUInt16 input output radius (int windowSize) availableMemory
        | "median-native-nth", Int32 -> runChunkMedianNativeNthInt32 input output radius (int windowSize) availableMemory
        | "median-native-nth", Float32 -> runChunkMedianNativeNthFloat32 input output radius (int windowSize) availableMemory
        | "convolve", UInt8 -> runChunkConvolveTyped<uint8> input output kernelSize workerCount true availableMemory 0u
        | "convolve", UInt16 -> runChunkConvolveTyped<uint16> input output kernelSize workerCount true availableMemory 0u
        | "convolve", Float32 -> runChunkConvolveTyped<float32> input output kernelSize workerCount true availableMemory 0u
        | "dilate", UInt8 -> runBinaryDilateTyped<uint8> input output radius availableMemory
        | "dilate", UInt16 -> runBinaryDilateTyped<uint16> input output radius availableMemory
        | "dilate", Int32 -> runBinaryDilateTyped<int32> input output radius availableMemory
        | "dilate", Float32 -> runBinaryDilateTyped<float32> input output radius availableMemory
        | "connectedComponents", UInt8 -> runConnectedComponents input output windowSize availableMemory
        | "connectedComponents", _ -> failwith "connectedComponents benchmark is currently defined for UInt8 masks only"
        | _ -> failwith $"unsupported benchmark operation '{operation}' for {pixelType}"
    stopwatch.Stop()
    writeInternalSeconds stopwatch.Elapsed
    exitCode

let private ensureNonEmptyHistogram<'T when 'T: comparison> label (histogram: StackCore.Histogram<'T>) =
    if Map.isEmpty histogram.Counts then
        invalidOp $"{label} produced an empty histogram."

let private runChunkHistogramDenseTyped<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> input availableMemory windowSize =
    let histogram =
        benchmarkSource availableMemory
        |> read<'T> input ".tiff"
        >=> if windowSize > 1 then ChunkFunctions.histogramDenseReducerParallel<'T> windowSize else ChunkFunctions.histogramDenseReducer<'T> ()
        |> drain

    ensureNonEmptyHistogram "chunk dense histogram" histogram
    0

let private runChunkHistogramSparseTyped<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> input availableMemory windowSize =
    let histogram =
        benchmarkSource availableMemory
        |> read<'T> input ".tiff"
        >=> if windowSize > 1 then ChunkFunctions.histogramReducerParallel<'T> windowSize else ChunkFunctions.histogramReducer<'T> ()
        |> drain

    ensureNonEmptyHistogram "chunk sparse histogram" histogram
    0

let private runChunkHistogramLeftEdgesTyped<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> input availableMemory bins =
    let leftEdges = [ for bin in 0 .. max 1 bins - 1 -> float bin ]
    let histogram =
        benchmarkSource availableMemory
        |> read<'T> input ".tiff"
        >=> ChunkFunctions.histogramLeftEdgesReducer<'T> leftEdges
        |> drain

    ensureNonEmptyHistogram "chunk left-edge histogram" histogram
    0

let private runChunkHistogram opts =
    let pixelType = require "pixel-type" opts |> parsePixelType
    let input = require "input" opts
    let variant = optional "variant" "dense" opts
    let bins = optional "bins" "256" opts |> Int32.Parse
    let windowSize =
        if opts.ContainsKey "window-size" then
            optional "window-size" "1" opts |> Int32.Parse
        else
            optional "workers" "1" opts |> Int32.Parse
    let availableMemory = optional "available-memory" (string (1024UL * 1024UL * 1024UL * 1024UL)) opts |> UInt64.Parse

    let stopwatch = Stopwatch.StartNew()
    let exitCode =
        match variant, pixelType with
        | "dense", UInt8 -> runChunkHistogramDenseTyped<uint8> input availableMemory windowSize
        | "dense", UInt16 -> runChunkHistogramDenseTyped<uint16> input availableMemory windowSize
        | "dense", Float32 -> failwith "Dense chunk histograms are defined for integer pixel types up to 16 bits; use --variant sparse or --variant leftedges for Float32."
        | "sparse", UInt8 -> runChunkHistogramSparseTyped<uint8> input availableMemory windowSize
        | "sparse", UInt16 -> runChunkHistogramSparseTyped<uint16> input availableMemory windowSize
        | "sparse", Float32 -> runChunkHistogramSparseTyped<float32> input availableMemory windowSize
        | "leftedges", UInt8 -> runChunkHistogramLeftEdgesTyped<uint8> input availableMemory bins
        | "leftedges", UInt16 -> runChunkHistogramLeftEdgesTyped<uint16> input availableMemory bins
        | "leftedges", Float32 -> runChunkHistogramLeftEdgesTyped<float32> input availableMemory bins
        | other, _ -> failwith $"Unsupported chunk histogram variant '{other}'. Expected dense, sparse, or leftedges."

    stopwatch.Stop()
    writeInternalSeconds stopwatch.Elapsed
    exitCode

let private runChunkThresholdParallelCollect opts =
    let pixelType = require "pixel-type" opts |> parsePixelType
    let input = require "input" opts
    let output = require "output" opts
    let thresholdValue = optional "threshold" "128" opts |> fun s -> Double.Parse(s, invariant)
    let workers = optional "workers" "1" opts |> Int32.Parse
    let availableMemory = optional "available-memory" (string (1024UL * 1024UL * 1024UL * 1024UL)) opts |> UInt64.Parse

    if workers < 1 then
        invalidArg "workers" $"Chunk threshold parallelCollect expects at least one worker/window, got {workers}."

    let stopwatch = Stopwatch.StartNew()
    let exitCode =
        match pixelType with
        | UInt8 -> runChunkThresholdParallelCollectTyped<uint8> input output thresholdValue workers availableMemory
        | UInt16 -> runChunkThresholdParallelCollectTyped<uint16> input output thresholdValue workers availableMemory
        | Int32 -> runChunkThresholdParallelCollectTyped<int32> input output thresholdValue workers availableMemory
        | Float32 -> runChunkThresholdParallelCollectTyped<float32> input output thresholdValue workers availableMemory
        | _ -> unsupportedPixelType "Chunk threshold parallelCollect benchmark" "UInt8, UInt16, Int32, and Float32" pixelType

    stopwatch.Stop()
    writeInternalSeconds stopwatch.Elapsed
    exitCode

let private runChunkDilate opts =
    let input = require "input" opts
    let output = require "output" opts
    let radius = optional "radius" "1" opts |> UInt32.Parse
    let thresholdValue = optional "threshold" "128" opts |> fun s -> Double.Parse(s, invariant)
    let workers = optional "workers" "1" opts |> Int32.Parse
    let availableMemory = optional "available-memory" (string (1024UL * 1024UL * 1024UL * 1024UL)) opts |> UInt64.Parse

    let stopwatch = Stopwatch.StartNew()
    let exitCode = runChunkBinaryDilate input output radius thresholdValue workers availableMemory
    stopwatch.Stop()
    writeInternalSeconds stopwatch.Elapsed
    exitCode

let private fillStructureTensorAosGradient (chunk: StackCore.Chunk<float32>) =
    let values = StackCore.Chunk.span<float32> chunk
    let spatialCount = values.Length / 3
    let mutable i = 0
    while i < spatialCount do
        let baseIndex = i * 3
        values[baseIndex] <- float32 ((i * 37 + 11) &&& 0x3FF) / 257.0f
        values[baseIndex + 1] <- float32 ((i * 53 + 17) &&& 0x3FF) / 263.0f
        values[baseIndex + 2] <- float32 ((i * 71 + 23) &&& 0x3FF) / 269.0f
        i <- i + 1

let private fillStructureTensorSoaGradient (chunk: StackCore.Chunk<float32>) =
    let values = StackCore.Chunk.span<float32> chunk
    let spatialCount = values.Length / 3
    let mutable i = 0
    while i < spatialCount do
        values[i] <- float32 ((i * 37 + 11) &&& 0x3FF) / 257.0f
        values[spatialCount + i] <- float32 ((i * 53 + 17) &&& 0x3FF) / 263.0f
        values[2 * spatialCount + i] <- float32 ((i * 71 + 23) &&& 0x3FF) / 269.0f
        i <- i + 1

let private fillStructureTensorComponentGradient (window: StackCore.Window<StackCore.Chunk<float32>>) =
    match window.Items with
    | [ gxChunk; gyChunk; gzChunk ] ->
        let gxValues = StackCore.Chunk.span<float32> gxChunk
        let gyValues = StackCore.Chunk.span<float32> gyChunk
        let gzValues = StackCore.Chunk.span<float32> gzChunk
        let mutable i = 0
        while i < gxValues.Length do
            gxValues[i] <- float32 ((i * 37 + 11) &&& 0x3FF) / 257.0f
            gyValues[i] <- float32 ((i * 53 + 17) &&& 0x3FF) / 263.0f
            gzValues[i] <- float32 ((i * 71 + 23) &&& 0x3FF) / 269.0f
            i <- i + 1
    | _ ->
        invalidArg "window" "Expected a 3-component gradient window."

let private structureTensorOuterAos (input: StackCore.Chunk<float32>) (output: StackCore.Chunk<float32>) =
    let inputValues = StackCore.Chunk.span<float32> input
    let outputValues = StackCore.Chunk.span<float32> output
    let spatialCount = inputValues.Length / 3
    let mutable i = 0
    while i < spatialCount do
        let inputBase = i * 3
        let gx = inputValues[inputBase]
        let gy = inputValues[inputBase + 1]
        let gz = inputValues[inputBase + 2]
        let outputBase = i * 6
        outputValues[outputBase] <- gx * gx
        outputValues[outputBase + 1] <- gx * gy
        outputValues[outputBase + 2] <- gx * gz
        outputValues[outputBase + 3] <- gy * gy
        outputValues[outputBase + 4] <- gy * gz
        outputValues[outputBase + 5] <- gz * gz
        i <- i + 1

let private structureTensorOuterSoaScalar (input: StackCore.Chunk<float32>) (output: StackCore.Chunk<float32>) =
    let inputValues = StackCore.Chunk.span<float32> input
    let outputValues = StackCore.Chunk.span<float32> output
    let spatialCount = inputValues.Length / 3
    let mutable i = 0
    while i < spatialCount do
        let gx = inputValues[i]
        let gy = inputValues[spatialCount + i]
        let gz = inputValues[2 * spatialCount + i]
        outputValues[i] <- gx * gx
        outputValues[spatialCount + i] <- gx * gy
        outputValues[2 * spatialCount + i] <- gx * gz
        outputValues[3 * spatialCount + i] <- gy * gy
        outputValues[4 * spatialCount + i] <- gy * gz
        outputValues[5 * spatialCount + i] <- gz * gz
        i <- i + 1

let private structureTensorOuterSoaVector (input: StackCore.Chunk<float32>) (output: StackCore.Chunk<float32>) =
    let inputValues = StackCore.Chunk.span<float32> input
    let outputValues = StackCore.Chunk.span<float32> output
    let spatialCount = inputValues.Length / 3
    let width = Vector<float32>.Count
    let vectorEnd = spatialCount - spatialCount % width
    let mutable i = 0
    while i < vectorEnd do
        let gx = Vector<float32>(inputValues.Slice(i, width))
        let gy = Vector<float32>(inputValues.Slice(spatialCount + i, width))
        let gz = Vector<float32>(inputValues.Slice(2 * spatialCount + i, width))
        (gx * gx).CopyTo(outputValues.Slice(i, width))
        (gx * gy).CopyTo(outputValues.Slice(spatialCount + i, width))
        (gx * gz).CopyTo(outputValues.Slice(2 * spatialCount + i, width))
        (gy * gy).CopyTo(outputValues.Slice(3 * spatialCount + i, width))
        (gy * gz).CopyTo(outputValues.Slice(4 * spatialCount + i, width))
        (gz * gz).CopyTo(outputValues.Slice(5 * spatialCount + i, width))
        i <- i + width
    while i < spatialCount do
        let gx = inputValues[i]
        let gy = inputValues[spatialCount + i]
        let gz = inputValues[2 * spatialCount + i]
        outputValues[i] <- gx * gx
        outputValues[spatialCount + i] <- gx * gy
        outputValues[2 * spatialCount + i] <- gx * gz
        outputValues[3 * spatialCount + i] <- gy * gy
        outputValues[4 * spatialCount + i] <- gy * gz
        outputValues[5 * spatialCount + i] <- gz * gz
        i <- i + 1

let private structureTensorOuterComponentsVector
    (input: StackCore.Window<StackCore.Chunk<float32>>)
    (output: StackCore.Window<StackCore.Chunk<float32>>)
    =
    match input.Items, output.Items with
    | [ gxChunk; gyChunk; gzChunk ], [ xxChunk; xyChunk; xzChunk; yyChunk; yzChunk; zzChunk ] ->
        let gxValues = StackCore.Chunk.span<float32> gxChunk
        let gyValues = StackCore.Chunk.span<float32> gyChunk
        let gzValues = StackCore.Chunk.span<float32> gzChunk
        let xxValues = StackCore.Chunk.span<float32> xxChunk
        let xyValues = StackCore.Chunk.span<float32> xyChunk
        let xzValues = StackCore.Chunk.span<float32> xzChunk
        let yyValues = StackCore.Chunk.span<float32> yyChunk
        let yzValues = StackCore.Chunk.span<float32> yzChunk
        let zzValues = StackCore.Chunk.span<float32> zzChunk
        let width = Vector<float32>.Count
        let vectorEnd = gxValues.Length - gxValues.Length % width
        let mutable i = 0
        while i < vectorEnd do
            let gx = Vector<float32>(gxValues.Slice(i, width))
            let gy = Vector<float32>(gyValues.Slice(i, width))
            let gz = Vector<float32>(gzValues.Slice(i, width))
            (gx * gx).CopyTo(xxValues.Slice(i, width))
            (gx * gy).CopyTo(xyValues.Slice(i, width))
            (gx * gz).CopyTo(xzValues.Slice(i, width))
            (gy * gy).CopyTo(yyValues.Slice(i, width))
            (gy * gz).CopyTo(yzValues.Slice(i, width))
            (gz * gz).CopyTo(zzValues.Slice(i, width))
            i <- i + width
        while i < gxValues.Length do
            let gx = gxValues[i]
            let gy = gyValues[i]
            let gz = gzValues[i]
            xxValues[i] <- gx * gx
            xyValues[i] <- gx * gy
            xzValues[i] <- gx * gz
            yyValues[i] <- gy * gy
            yzValues[i] <- gy * gz
            zzValues[i] <- gz * gz
            i <- i + 1
    | _ ->
        invalidArg "window" "Expected 3 input components and 6 output tensor components."

let private smoothTensorAos3Point (input: StackCore.Chunk<float32>) (output: StackCore.Chunk<float32>) =
    let inputValues = StackCore.Chunk.span<float32> input
    let outputValues = StackCore.Chunk.span<float32> output
    let spatialCount = inputValues.Length / 6
    let last = spatialCount - 1
    let mutable i = 0
    while i < spatialCount do
        let prevBase = (if i = 0 then 0 else i - 1) * 6
        let currentBase = i * 6
        let nextBase = (if i = last then last else i + 1) * 6
        let mutable c = 0
        while c < 6 do
            outputValues[currentBase + c] <-
                0.25f * inputValues[prevBase + c]
                + 0.5f * inputValues[currentBase + c]
                + 0.25f * inputValues[nextBase + c]
            c <- c + 1
        i <- i + 1

let private smoothTensorSoa3PointVector (input: StackCore.Chunk<float32>) (output: StackCore.Chunk<float32>) =
    let inputValues = StackCore.Chunk.span<float32> input
    let outputValues = StackCore.Chunk.span<float32> output
    let spatialCount = inputValues.Length / 6
    let width = Vector<float32>.Count
    let quarter = Vector<float32>(0.25f)
    let half = Vector<float32>(0.5f)
    let mutable comp = 0
    while comp < 6 do
        let offset = comp * spatialCount
        outputValues[offset] <- 0.75f * inputValues[offset] + 0.25f * inputValues[offset + 1]
        let mutable i = 1
        let vectorEnd = spatialCount - 1 - ((spatialCount - 1) % width)
        while i < vectorEnd do
            let prev = Vector<float32>(inputValues.Slice(offset + i - 1, width))
            let current = Vector<float32>(inputValues.Slice(offset + i, width))
            let next = Vector<float32>(inputValues.Slice(offset + i + 1, width))
            (quarter * prev + half * current + quarter * next).CopyTo(outputValues.Slice(offset + i, width))
            i <- i + width
        while i < spatialCount - 1 do
            outputValues[offset + i] <-
                0.25f * inputValues[offset + i - 1]
                + 0.5f * inputValues[offset + i]
                + 0.25f * inputValues[offset + i + 1]
            i <- i + 1
        outputValues[offset + spatialCount - 1] <-
            0.25f * inputValues[offset + spatialCount - 2]
            + 0.75f * inputValues[offset + spatialCount - 1]
        comp <- comp + 1

let private smoothTensorComponents3PointVector
    (input: StackCore.Window<StackCore.Chunk<float32>>)
    (output: StackCore.Window<StackCore.Chunk<float32>>)
    =
    let smoothComponent (inputChunk: StackCore.Chunk<float32>) (outputChunk: StackCore.Chunk<float32>) =
        let inputValues = StackCore.Chunk.span<float32> inputChunk
        let outputValues = StackCore.Chunk.span<float32> outputChunk
        let width = Vector<float32>.Count
        let quarter = Vector<float32>(0.25f)
        let half = Vector<float32>(0.5f)
        outputValues[0] <- 0.75f * inputValues[0] + 0.25f * inputValues[1]
        let mutable i = 1
        let vectorEnd = inputValues.Length - 1 - ((inputValues.Length - 1) % width)
        while i < vectorEnd do
            let prev = Vector<float32>(inputValues.Slice(i - 1, width))
            let current = Vector<float32>(inputValues.Slice(i, width))
            let next = Vector<float32>(inputValues.Slice(i + 1, width))
            (quarter * prev + half * current + quarter * next).CopyTo(outputValues.Slice(i, width))
            i <- i + width
        while i < inputValues.Length - 1 do
            outputValues[i] <-
                0.25f * inputValues[i - 1]
                + 0.5f * inputValues[i]
                + 0.25f * inputValues[i + 1]
            i <- i + 1
        outputValues[inputValues.Length - 1] <-
            0.25f * inputValues[inputValues.Length - 2]
            + 0.75f * inputValues[inputValues.Length - 1]

    match input.Items, output.Items with
    | [ i0; i1; i2; i3; i4; i5 ], [ o0; o1; o2; o3; o4; o5 ] ->
        smoothComponent i0 o0
        smoothComponent i1 o1
        smoothComponent i2 o2
        smoothComponent i3 o3
        smoothComponent i4 o4
        smoothComponent i5 o5
    | _ ->
        invalidArg "window" "Expected 6 input and output tensor components."

let private symmetricEigenvalues3x3Float32 xx xy xz yy yz zz =
    let xx = float xx
    let xy = float xy
    let xz = float xz
    let yy = float yy
    let yz = float yz
    let zz = float zz
    let p1 = xy * xy + xz * xz + yz * yz
    if p1 = 0.0 then
        let mutable a = xx
        let mutable b = yy
        let mutable c = zz
        if a < b then
            let t = a
            a <- b
            b <- t
        if b < c then
            let t = b
            b <- c
            c <- t
        if a < b then
            let t = a
            a <- b
            b <- t
        struct (float32 a, float32 b, float32 c)
    else
        let q = (xx + yy + zz) / 3.0
        let axx = xx - q
        let ayy = yy - q
        let azz = zz - q
        let p2 = axx * axx + ayy * ayy + azz * azz + 2.0 * p1
        let p = sqrt (p2 / 6.0)
        let bxx = axx / p
        let bxy = xy / p
        let bxz = xz / p
        let byy = ayy / p
        let byz = yz / p
        let bzz = azz / p
        let detB =
            bxx * (byy * bzz - byz * byz)
            - bxy * (bxy * bzz - byz * bxz)
            + bxz * (bxy * byz - byy * bxz)
        let r = detB / 2.0
        let phi =
            if r <= -1.0 then Math.PI / 3.0
            elif r >= 1.0 then 0.0
            else Math.Acos(r) / 3.0
        let eig0 = q + 2.0 * p * Math.Cos(phi)
        let eig2 = q + 2.0 * p * Math.Cos(phi + 2.0 * Math.PI / 3.0)
        let eig1 = 3.0 * q - eig0 - eig2
        struct (float32 eig0, float32 eig1, float32 eig2)

let private symmetricEigenvalues3x3JacobiMaxFloat32 rotations xx xy xz yy yz zz =
    let mutable a00 = float xx
    let mutable a01 = float xy
    let mutable a02 = float xz
    let mutable a11 = float yy
    let mutable a12 = float yz
    let mutable a22 = float zz

    let inline rotationCoefficients app aqq apq =
        let tau = (aqq - app) / (2.0 * apq)
        let sign = if tau >= 0.0 then 1.0 else -1.0
        let t = sign / (abs tau + sqrt (1.0 + tau * tau))
        let c = 1.0 / sqrt (1.0 + t * t)
        let s = t * c
        struct (t, c, s)

    let inline rotate01 () =
        if abs a01 > 1e-14 then
            let app = a00
            let aqq = a11
            let apq = a01
            let struct (t, c, s) = rotationCoefficients app aqq apq
            a00 <- app - t * apq
            a11 <- aqq + t * apq
            a01 <- 0.0
            let ar0 = a02
            let ar1 = a12
            a02 <- c * ar0 - s * ar1
            a12 <- s * ar0 + c * ar1

    let inline rotate02 () =
        if abs a02 > 1e-14 then
            let app = a00
            let aqq = a22
            let apq = a02
            let struct (t, c, s) = rotationCoefficients app aqq apq
            a00 <- app - t * apq
            a22 <- aqq + t * apq
            a02 <- 0.0
            let ar0 = a01
            let ar2 = a12
            a01 <- c * ar0 - s * ar2
            a12 <- s * ar0 + c * ar2

    let inline rotate12 () =
        if abs a12 > 1e-14 then
            let app = a11
            let aqq = a22
            let apq = a12
            let struct (t, c, s) = rotationCoefficients app aqq apq
            a11 <- app - t * apq
            a22 <- aqq + t * apq
            a12 <- 0.0
            let ar1 = a01
            let ar2 = a02
            a01 <- c * ar1 - s * ar2
            a02 <- s * ar1 + c * ar2

    let mutable r = 0
    while r < rotations do
        let abs01 = abs a01
        let abs02 = abs a02
        let abs12 = abs a12
        if abs01 >= abs02 && abs01 >= abs12 then
            rotate01()
        elif abs02 >= abs12 then
            rotate02()
        else
            rotate12()
        r <- r + 1

    let mutable l0 = a00
    let mutable l1 = a11
    let mutable l2 = a22
    if l0 < l1 then
        let t = l0
        l0 <- l1
        l1 <- t
    if l1 < l2 then
        let t = l1
        l1 <- l2
        l2 <- t
    if l0 < l1 then
        let t = l0
        l0 <- l1
        l1 <- t
    struct (float32 l0, float32 l1, float32 l2)

let private structureTensorEigenvaluesAos (input: StackCore.Chunk<float32>) (output: StackCore.Chunk<float32>) =
    let inputValues = StackCore.Chunk.span<float32> input
    let outputValues = StackCore.Chunk.span<float32> output
    let spatialCount = inputValues.Length / 6
    let mutable i = 0
    while i < spatialCount do
        let p = i * 6
        let struct (l0, l1, l2) =
            symmetricEigenvalues3x3Float32
                inputValues[p]
                inputValues[p + 1]
                inputValues[p + 2]
                inputValues[p + 3]
                inputValues[p + 4]
                inputValues[p + 5]
        let o = i * 3
        outputValues[o] <- l0
        outputValues[o + 1] <- l1
        outputValues[o + 2] <- l2
        i <- i + 1

let private structureTensorEigenvaluesSoa (input: StackCore.Chunk<float32>) (output: StackCore.Chunk<float32>) =
    let inputValues = StackCore.Chunk.span<float32> input
    let outputValues = StackCore.Chunk.span<float32> output
    let spatialCount = inputValues.Length / 6
    let mutable i = 0
    while i < spatialCount do
        let struct (l0, l1, l2) =
            symmetricEigenvalues3x3Float32
                inputValues[i]
                inputValues[spatialCount + i]
                inputValues[2 * spatialCount + i]
                inputValues[3 * spatialCount + i]
                inputValues[4 * spatialCount + i]
                inputValues[5 * spatialCount + i]
        outputValues[i] <- l0
        outputValues[spatialCount + i] <- l1
        outputValues[2 * spatialCount + i] <- l2
        i <- i + 1

let private structureTensorEigenvaluesComponents
    (input: StackCore.Window<StackCore.Chunk<float32>>)
    (output: StackCore.Window<StackCore.Chunk<float32>>)
    =
    match input.Items, output.Items with
    | [ xxChunk; xyChunk; xzChunk; yyChunk; yzChunk; zzChunk ], [ l0Chunk; l1Chunk; l2Chunk ] ->
        let xxValues = StackCore.Chunk.span<float32> xxChunk
        let xyValues = StackCore.Chunk.span<float32> xyChunk
        let xzValues = StackCore.Chunk.span<float32> xzChunk
        let yyValues = StackCore.Chunk.span<float32> yyChunk
        let yzValues = StackCore.Chunk.span<float32> yzChunk
        let zzValues = StackCore.Chunk.span<float32> zzChunk
        let l0Values = StackCore.Chunk.span<float32> l0Chunk
        let l1Values = StackCore.Chunk.span<float32> l1Chunk
        let l2Values = StackCore.Chunk.span<float32> l2Chunk
        let mutable i = 0
        while i < xxValues.Length do
            let struct (l0, l1, l2) =
                symmetricEigenvalues3x3Float32
                    xxValues[i]
                    xyValues[i]
                    xzValues[i]
                    yyValues[i]
                    yzValues[i]
                    zzValues[i]
            l0Values[i] <- l0
            l1Values[i] <- l1
            l2Values[i] <- l2
            i <- i + 1
    | _ ->
        invalidArg "window" "Expected 6 tensor components and 3 eigenvalue components."

let private structureTensorEigenvaluesComponentsJacobi rotations
    (input: StackCore.Window<StackCore.Chunk<float32>>)
    (output: StackCore.Window<StackCore.Chunk<float32>>)
    =
    match input.Items, output.Items with
    | [ xxChunk; xyChunk; xzChunk; yyChunk; yzChunk; zzChunk ], [ l0Chunk; l1Chunk; l2Chunk ] ->
        let xxValues = StackCore.Chunk.span<float32> xxChunk
        let xyValues = StackCore.Chunk.span<float32> xyChunk
        let xzValues = StackCore.Chunk.span<float32> xzChunk
        let yyValues = StackCore.Chunk.span<float32> yyChunk
        let yzValues = StackCore.Chunk.span<float32> yzChunk
        let zzValues = StackCore.Chunk.span<float32> zzChunk
        let l0Values = StackCore.Chunk.span<float32> l0Chunk
        let l1Values = StackCore.Chunk.span<float32> l1Chunk
        let l2Values = StackCore.Chunk.span<float32> l2Chunk
        let mutable i = 0
        while i < xxValues.Length do
            let struct (l0, l1, l2) =
                symmetricEigenvalues3x3JacobiMaxFloat32
                    rotations
                    xxValues[i]
                    xyValues[i]
                    xzValues[i]
                    yyValues[i]
                    yzValues[i]
                    zzValues[i]
            l0Values[i] <- l0
            l1Values[i] <- l1
            l2Values[i] <- l2
            i <- i + 1
    | _ ->
        invalidArg "window" "Expected 6 tensor components and 3 eigenvalue components."

let private symmetricEigenJacobiAllocLocal xx xy xz yy yz zz =
    let a = Array2D.zeroCreate<float> 3 3
    let v = Array2D.zeroCreate<float> 3 3

    a[0, 0] <- float xx; a[0, 1] <- float xy; a[0, 2] <- float xz
    a[1, 0] <- float xy; a[1, 1] <- float yy; a[1, 2] <- float yz
    a[2, 0] <- float xz; a[2, 1] <- float yz; a[2, 2] <- float zz

    for i in 0 .. 2 do
        for j in 0 .. 2 do
            v[i, j] <- if i = j then 1.0 else 0.0

    let rotate p q =
        if abs a[p, q] > 1e-14 then
            let tau = (a[q, q] - a[p, p]) / (2.0 * a[p, q])
            let sign = if tau >= 0.0 then 1.0 else -1.0
            let t = sign / (abs tau + sqrt (1.0 + tau * tau))
            let c = 1.0 / sqrt (1.0 + t * t)
            let s = t * c
            let app = a[p, p]
            let aqq = a[q, q]
            let apq = a[p, q]

            a[p, p] <- app - t * apq
            a[q, q] <- aqq + t * apq
            a[p, q] <- 0.0
            a[q, p] <- 0.0

            for r in 0 .. 2 do
                if r <> p && r <> q then
                    let arp = a[r, p]
                    let arq = a[r, q]
                    a[r, p] <- c * arp - s * arq
                    a[p, r] <- a[r, p]
                    a[r, q] <- s * arp + c * arq
                    a[q, r] <- a[r, q]

            for r in 0 .. 2 do
                let vrp = v[r, p]
                let vrq = v[r, q]
                v[r, p] <- c * vrp - s * vrq
                v[r, q] <- s * vrp + c * vrq

    for _ in 1 .. 32 do
        rotate 0 1
        rotate 0 2
        rotate 1 2

    [ for i in 0 .. 2 ->
        let vector = TinyLinAlg.normalize (TinyLinAlg.v3 v[0, i] v[1, i] v[2, i])
        a[i, i], vector ]
    |> List.sortByDescending fst

let private structureTensorEigensystemComponents
    (input: StackCore.Window<StackCore.Chunk<float32>>)
    (output: StackCore.Window<StackCore.Chunk<float32>>)
    =
    match input.Items, output.Items with
    | [ xxChunk; xyChunk; xzChunk; yyChunk; yzChunk; zzChunk ],
      [ l0Chunk; l1Chunk; l2Chunk; v00Chunk; v01Chunk; v02Chunk; v10Chunk; v11Chunk; v12Chunk; v20Chunk; v21Chunk; v22Chunk ] ->
        let xxValues = StackCore.Chunk.span<float32> xxChunk
        let xyValues = StackCore.Chunk.span<float32> xyChunk
        let xzValues = StackCore.Chunk.span<float32> xzChunk
        let yyValues = StackCore.Chunk.span<float32> yyChunk
        let yzValues = StackCore.Chunk.span<float32> yzChunk
        let zzValues = StackCore.Chunk.span<float32> zzChunk
        let l0Values = StackCore.Chunk.span<float32> l0Chunk
        let l1Values = StackCore.Chunk.span<float32> l1Chunk
        let l2Values = StackCore.Chunk.span<float32> l2Chunk
        let v00Values = StackCore.Chunk.span<float32> v00Chunk
        let v01Values = StackCore.Chunk.span<float32> v01Chunk
        let v02Values = StackCore.Chunk.span<float32> v02Chunk
        let v10Values = StackCore.Chunk.span<float32> v10Chunk
        let v11Values = StackCore.Chunk.span<float32> v11Chunk
        let v12Values = StackCore.Chunk.span<float32> v12Chunk
        let v20Values = StackCore.Chunk.span<float32> v20Chunk
        let v21Values = StackCore.Chunk.span<float32> v21Chunk
        let v22Values = StackCore.Chunk.span<float32> v22Chunk
        let mutable i = 0
        while i < xxValues.Length do
            let eigen =
                TinyLinAlg.symmetricEigen3
                    { m00 = float xxValues[i]; m01 = float xyValues[i]; m02 = float xzValues[i]
                      m10 = float xyValues[i]; m11 = float yyValues[i]; m12 = float yzValues[i]
                      m20 = float xzValues[i]; m21 = float yzValues[i]; m22 = float zzValues[i] }
            l0Values[i] <- float32 eigen.Value0
            l1Values[i] <- float32 eigen.Value1
            l2Values[i] <- float32 eigen.Value2
            v00Values[i] <- float32 eigen.Vector0.x
            v01Values[i] <- float32 eigen.Vector0.y
            v02Values[i] <- float32 eigen.Vector0.z
            v10Values[i] <- float32 eigen.Vector1.x
            v11Values[i] <- float32 eigen.Vector1.y
            v12Values[i] <- float32 eigen.Vector1.z
            v20Values[i] <- float32 eigen.Vector2.x
            v21Values[i] <- float32 eigen.Vector2.y
            v22Values[i] <- float32 eigen.Vector2.z
            i <- i + 1
    | _ ->
        invalidArg "input" "Expected 6 tensor input components and 12 eigensystem output components."

let private structureTensorEigensystemComponentsChunkAlgebra
    (input: StackCore.Window<StackCore.Chunk<float32>>)
    (output: StackCore.Window<StackCore.Chunk<float32>>)
    =
    let inline sortValuesDescending a b c =
        let mutable x = a
        let mutable y = b
        let mutable z = c
        if x < y then
            let t = x
            x <- y
            y <- t
        if y < z then
            let t = y
            y <- z
            z <- t
        if x < y then
            let t = x
            x <- y
            y <- t
        struct (x, y, z)

    let inline normalize3 (x: float) (y: float) (z: float) =
        let n = sqrt (x * x + y * y + z * z)
        if n <= 1e-18 then
            struct (1.0, 0.0, 0.0)
        else
            let inv = 1.0 / n
            let mutable x = x * inv
            let mutable y = y * inv
            let mutable z = z * inv
            let ax = abs x
            let ay = abs y
            let az = abs z
            let sign =
                if ax >= ay && ax >= az then x
                elif ay >= az then y
                else z
            if sign < 0.0 then
                x <- -x
                y <- -y
                z <- -z
            struct (x, y, z)

    let inline cross ax ay az bx by bz =
        struct (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)

    let inline tryEigenvector xx xy xz yy yz zz lambda tolerance =
        let r00 = xx - lambda
        let r01 = xy
        let r02 = xz
        let r10 = xy
        let r11 = yy - lambda
        let r12 = yz
        let r20 = xz
        let r21 = yz
        let r22 = zz - lambda
        let struct (c01x, c01y, c01z) = cross r00 r01 r02 r10 r11 r12
        let struct (c02x, c02y, c02z) = cross r00 r01 r02 r20 r21 r22
        let struct (c12x, c12y, c12z) = cross r10 r11 r12 r20 r21 r22
        let n01 = c01x * c01x + c01y * c01y + c01z * c01z
        let n02 = c02x * c02x + c02y * c02y + c02z * c02z
        let n12 = c12x * c12x + c12y * c12y + c12z * c12z
        let struct (cx, cy, cz, n) =
            if n01 >= n02 && n01 >= n12 then struct (c01x, c01y, c01z, n01)
            elif n02 >= n12 then struct (c02x, c02y, c02z, n02)
            else struct (c12x, c12y, c12z, n12)
        if n <= tolerance * tolerance then
            struct (false, 1.0, 0.0, 0.0)
        else
            let struct (x, y, z) = normalize3 cx cy cz
            struct (true, x, y, z)

    let inline orthonormalComplement ax ay az =
        let aax = abs ax
        let aay = abs ay
        let aaz = abs az
        let struct (rx, ry, rz) =
            if aax <= aay && aax <= aaz then struct (1.0, 0.0, 0.0)
            elif aay <= aaz then struct (0.0, 1.0, 0.0)
            else struct (0.0, 0.0, 1.0)
        let struct (u0, u1, u2) = cross ax ay az rx ry rz
        let struct (u0, u1, u2) = normalize3 u0 u1 u2
        let struct (v0, v1, v2) = cross ax ay az u0 u1 u2
        let struct (v0, v1, v2) = normalize3 v0 v1 v2
        struct (u0, u1, u2, v0, v1, v2)

    let inline eigenvalues xx xy xz yy yz zz =
        let p1 = xy * xy + xz * xz + yz * yz
        if p1 = 0.0 then
            sortValuesDescending xx yy zz
        else
            let q = (xx + yy + zz) / 3.0
            let axx = xx - q
            let ayy = yy - q
            let azz = zz - q
            let p2 = axx * axx + ayy * ayy + azz * azz + 2.0 * p1
            let p = sqrt (p2 / 6.0)
            let bxx = axx / p
            let bxy = xy / p
            let bxz = xz / p
            let byy = ayy / p
            let byz = yz / p
            let bzz = azz / p
            let detB =
                bxx * (byy * bzz - byz * byz)
                - bxy * (bxy * bzz - byz * bxz)
                + bxz * (bxy * byz - byy * bxz)
            let r = detB / 2.0
            let phi =
                if r <= -1.0 then Math.PI / 3.0
                elif r >= 1.0 then 0.0
                else Math.Acos(r) / 3.0
            let e0 = q + 2.0 * p * Math.Cos(phi)
            let e2 = q + 2.0 * p * Math.Cos(phi + 2.0 * Math.PI / 3.0)
            let e1 = 3.0 * q - e0 - e2
            sortValuesDescending e0 e1 e2

    match input.Items, output.Items with
    | [ xxChunk; xyChunk; xzChunk; yyChunk; yzChunk; zzChunk ],
      [ l0Chunk; l1Chunk; l2Chunk; v00Chunk; v01Chunk; v02Chunk; v10Chunk; v11Chunk; v12Chunk; v20Chunk; v21Chunk; v22Chunk ] ->
        let xxValues = StackCore.Chunk.span<float32> xxChunk
        let xyValues = StackCore.Chunk.span<float32> xyChunk
        let xzValues = StackCore.Chunk.span<float32> xzChunk
        let yyValues = StackCore.Chunk.span<float32> yyChunk
        let yzValues = StackCore.Chunk.span<float32> yzChunk
        let zzValues = StackCore.Chunk.span<float32> zzChunk
        let l0Values = StackCore.Chunk.span<float32> l0Chunk
        let l1Values = StackCore.Chunk.span<float32> l1Chunk
        let l2Values = StackCore.Chunk.span<float32> l2Chunk
        let v00Values = StackCore.Chunk.span<float32> v00Chunk
        let v01Values = StackCore.Chunk.span<float32> v01Chunk
        let v02Values = StackCore.Chunk.span<float32> v02Chunk
        let v10Values = StackCore.Chunk.span<float32> v10Chunk
        let v11Values = StackCore.Chunk.span<float32> v11Chunk
        let v12Values = StackCore.Chunk.span<float32> v12Chunk
        let v20Values = StackCore.Chunk.span<float32> v20Chunk
        let v21Values = StackCore.Chunk.span<float32> v21Chunk
        let v22Values = StackCore.Chunk.span<float32> v22Chunk
        let mutable i = 0
        while i < xxValues.Length do
            let xx = float xxValues[i]
            let xy = float xyValues[i]
            let xz = float xzValues[i]
            let yy = float yyValues[i]
            let yz = float yzValues[i]
            let zz = float zzValues[i]
            let scale =
                max 1.0
                    (max (abs xx)
                        (max (abs xy)
                            (max (abs xz)
                                (max (abs yy)
                                    (max (abs yz) (abs zz))))))
            let struct (e0, e1, e2) = eigenvalues xx xy xz yy yz zz
            let gap = 1e-10 * scale
            let vectorTol = 1e-12 * scale
            let mutable x0 = 1.0
            let mutable y0 = 0.0
            let mutable z0 = 0.0
            let mutable x1 = 0.0
            let mutable y1 = 1.0
            let mutable z1 = 0.0
            let mutable x2 = 0.0
            let mutable y2 = 0.0
            let mutable z2 = 1.0
            if abs (e0 - e1) <= gap && abs (e1 - e2) <= gap then
                ()
            elif abs (e0 - e1) <= gap then
                let struct (ok, tx, ty, tz) = tryEigenvector xx xy xz yy yz zz e2 vectorTol
                if ok then
                    x2 <- tx; y2 <- ty; z2 <- tz
                    let struct (u0, u1, u2, w0, w1, w2) = orthonormalComplement x2 y2 z2
                    x0 <- u0; y0 <- u1; z0 <- u2
                    x1 <- w0; y1 <- w1; z1 <- w2
            elif abs (e1 - e2) <= gap then
                let struct (ok, tx, ty, tz) = tryEigenvector xx xy xz yy yz zz e0 vectorTol
                if ok then
                    x0 <- tx; y0 <- ty; z0 <- tz
                    let struct (u0, u1, u2, w0, w1, w2) = orthonormalComplement x0 y0 z0
                    x1 <- u0; y1 <- u1; z1 <- u2
                    x2 <- w0; y2 <- w1; z2 <- w2
            else
                let struct (ok0, tx0, ty0, tz0) = tryEigenvector xx xy xz yy yz zz e0 vectorTol
                if ok0 then
                    x0 <- tx0; y0 <- ty0; z0 <- tz0
                let struct (ok1, tx1, ty1, tz1) = tryEigenvector xx xy xz yy yz zz e1 vectorTol
                if ok1 then
                    x1 <- tx1; y1 <- ty1; z1 <- tz1
                let struct (ok2, tx2, ty2, tz2) = tryEigenvector xx xy xz yy yz zz e2 vectorTol
                if ok2 then
                    x2 <- tx2; y2 <- ty2; z2 <- tz2
            l0Values[i] <- float32 e0
            l1Values[i] <- float32 e1
            l2Values[i] <- float32 e2
            v00Values[i] <- float32 x0
            v01Values[i] <- float32 y0
            v02Values[i] <- float32 z0
            v10Values[i] <- float32 x1
            v11Values[i] <- float32 y1
            v12Values[i] <- float32 z1
            v20Values[i] <- float32 x2
            v21Values[i] <- float32 y2
            v22Values[i] <- float32 z2
            i <- i + 1
    | _ ->
        invalidArg "input" "Expected 6 tensor input components and 12 eigensystem output components."

let private structureTensorEigensystemComponentsJacobiAlloc
    (input: StackCore.Window<StackCore.Chunk<float32>>)
    (output: StackCore.Window<StackCore.Chunk<float32>>)
    =
    match input.Items, output.Items with
    | [ xxChunk; xyChunk; xzChunk; yyChunk; yzChunk; zzChunk ],
      [ l0Chunk; l1Chunk; l2Chunk; v00Chunk; v01Chunk; v02Chunk; v10Chunk; v11Chunk; v12Chunk; v20Chunk; v21Chunk; v22Chunk ] ->
        let xxValues = StackCore.Chunk.span<float32> xxChunk
        let xyValues = StackCore.Chunk.span<float32> xyChunk
        let xzValues = StackCore.Chunk.span<float32> xzChunk
        let yyValues = StackCore.Chunk.span<float32> yyChunk
        let yzValues = StackCore.Chunk.span<float32> yzChunk
        let zzValues = StackCore.Chunk.span<float32> zzChunk
        let l0Values = StackCore.Chunk.span<float32> l0Chunk
        let l1Values = StackCore.Chunk.span<float32> l1Chunk
        let l2Values = StackCore.Chunk.span<float32> l2Chunk
        let v00Values = StackCore.Chunk.span<float32> v00Chunk
        let v01Values = StackCore.Chunk.span<float32> v01Chunk
        let v02Values = StackCore.Chunk.span<float32> v02Chunk
        let v10Values = StackCore.Chunk.span<float32> v10Chunk
        let v11Values = StackCore.Chunk.span<float32> v11Chunk
        let v12Values = StackCore.Chunk.span<float32> v12Chunk
        let v20Values = StackCore.Chunk.span<float32> v20Chunk
        let v21Values = StackCore.Chunk.span<float32> v21Chunk
        let v22Values = StackCore.Chunk.span<float32> v22Chunk
        let mutable i = 0
        while i < xxValues.Length do
            let eigen =
                symmetricEigenJacobiAllocLocal
                    xxValues[i]
                    xyValues[i]
                    xzValues[i]
                    yyValues[i]
                    yzValues[i]
                    zzValues[i]
            let e0, vec0 = eigen[0]
            let e1, vec1 = eigen[1]
            let e2, vec2 = eigen[2]
            l0Values[i] <- float32 e0
            l1Values[i] <- float32 e1
            l2Values[i] <- float32 e2
            v00Values[i] <- float32 vec0.x
            v01Values[i] <- float32 vec0.y
            v02Values[i] <- float32 vec0.z
            v10Values[i] <- float32 vec1.x
            v11Values[i] <- float32 vec1.y
            v12Values[i] <- float32 vec1.z
            v20Values[i] <- float32 vec2.x
            v21Values[i] <- float32 vec2.y
            v22Values[i] <- float32 vec2.z
            i <- i + 1
    | _ ->
        invalidArg "input" "Expected 6 tensor input components and 12 eigensystem output components."

let private convertAos3ToSoa3 (input: StackCore.Chunk<float32>) (output: StackCore.Chunk<float32>) =
    let inputValues = StackCore.Chunk.span<float32> input
    let outputValues = StackCore.Chunk.span<float32> output
    let spatialCount = inputValues.Length / 3
    let mutable i = 0
    while i < spatialCount do
        let inputBase = i * 3
        outputValues[i] <- inputValues[inputBase]
        outputValues[spatialCount + i] <- inputValues[inputBase + 1]
        outputValues[2 * spatialCount + i] <- inputValues[inputBase + 2]
        i <- i + 1

let private convertSoa6ToAos6 (input: StackCore.Chunk<float32>) (output: StackCore.Chunk<float32>) =
    let inputValues = StackCore.Chunk.span<float32> input
    let outputValues = StackCore.Chunk.span<float32> output
    let spatialCount = outputValues.Length / 6
    let mutable i = 0
    while i < spatialCount do
        let outputBase = i * 6
        outputValues[outputBase] <- inputValues[i]
        outputValues[outputBase + 1] <- inputValues[spatialCount + i]
        outputValues[outputBase + 2] <- inputValues[2 * spatialCount + i]
        outputValues[outputBase + 3] <- inputValues[3 * spatialCount + i]
        outputValues[outputBase + 4] <- inputValues[4 * spatialCount + i]
        outputValues[outputBase + 5] <- inputValues[5 * spatialCount + i]
        i <- i + 1

let private checksumStructureTensorChunkFloat32 (chunk: StackCore.Chunk<float32>) =
    let values = StackCore.Chunk.span<float32> chunk
    let stride = max 1 (values.Length / 1024)
    let mutable checksum = 2166136261u
    let mutable i = 0
    while i < values.Length do
        let bits = uint32 (BitConverter.SingleToInt32Bits(values[i]))
        checksum <- (checksum ^^^ bits) * 16777619u
        i <- i + stride
    int checksum

let private checksumStructureTensorComponentWindowFloat32 (window: StackCore.Window<StackCore.Chunk<float32>>) =
    let mutable checksum = 2166136261u
    for chunk in window.Items do
        let values = StackCore.Chunk.span<float32> chunk
        let stride = max 1 (values.Length / 1024)
        let mutable i = 0
        while i < values.Length do
            let bits = uint32 (BitConverter.SingleToInt32Bits(values[i]))
            checksum <- (checksum ^^^ bits) * 16777619u
            i <- i + stride
    int checksum

let private createStructureTensorComponentWindow shape componentCount : StackCore.Window<StackCore.Chunk<float32>> =
    let chunks =
        [ for _ in 1 .. componentCount ->
            StackCore.Chunk.create<float32> (uint64 shape.Width, uint64 shape.Height, uint64 shape.Depth) ]
    { Items = chunks
      EmitRange = 0u, uint componentCount
      ReleaseCount = uint componentCount }

let private releaseStructureTensorComponentWindow (window: StackCore.Window<StackCore.Chunk<float32>>) =
    window.Items |> List.iter StackCore.Chunk.decRef

let private runStructureTensorTimedMeasured bytesPerIteration iterations (action: unit -> unit) (checksum: unit -> int) =
    GC.Collect()
    GC.WaitForPendingFinalizers()
    GC.Collect()
    let before = GC.GetAllocatedBytesForCurrentThread()
    let stopwatch = Stopwatch.StartNew()
    let mutable i = 0
    while i < iterations do
        action()
        i <- i + 1
    stopwatch.Stop()
    let after = GC.GetAllocatedBytesForCurrentThread()
    let checksumValue = checksum()
    let bytes = float bytesPerIteration * float iterations
    let gib = bytes / (1024.0 * 1024.0 * 1024.0)
    let gibPerSecond = gib / stopwatch.Elapsed.TotalSeconds
    printfn "checksum=%d elapsed=%.6fs throughput=%.3f GiB/s allocated=%d" checksumValue stopwatch.Elapsed.TotalSeconds gibPerSecond (after - before)
    writeInternalSeconds stopwatch.Elapsed
    0

let private runStructureTensorLayoutVariant shape iterations variant =
    let aosGradient = StackCore.Chunk.create<float32> (uint64 shape.Width * 3UL, uint64 shape.Height, uint64 shape.Depth)
    let soaGradient = StackCore.Chunk.create<float32> (uint64 shape.Width, uint64 shape.Height, uint64 shape.Depth * 3UL)
    let aosTensor = StackCore.Chunk.create<float32> (uint64 shape.Width * 6UL, uint64 shape.Height, uint64 shape.Depth)
    let soaTensor = StackCore.Chunk.create<float32> (uint64 shape.Width, uint64 shape.Height, uint64 shape.Depth * 6UL)
    let aosTensorScratch = StackCore.Chunk.create<float32> (uint64 shape.Width * 6UL, uint64 shape.Height, uint64 shape.Depth)
    let soaTensorScratch = StackCore.Chunk.create<float32> (uint64 shape.Width, uint64 shape.Height, uint64 shape.Depth * 6UL)
    let aosEigenvalues = StackCore.Chunk.create<float32> (uint64 shape.Width * 3UL, uint64 shape.Height, uint64 shape.Depth)
    let soaEigenvalues = StackCore.Chunk.create<float32> (uint64 shape.Width, uint64 shape.Height, uint64 shape.Depth * 3UL)
    let componentGradient = createStructureTensorComponentWindow shape 3
    let componentTensor = createStructureTensorComponentWindow shape 6
    let componentTensorScratch = createStructureTensorComponentWindow shape 6
    let componentEigenvalues = createStructureTensorComponentWindow shape 3
    let componentEigensystem = createStructureTensorComponentWindow shape 12
    try
        fillStructureTensorAosGradient aosGradient
        fillStructureTensorSoaGradient soaGradient
        fillStructureTensorComponentGradient componentGradient
        structureTensorOuterAos aosGradient aosTensor
        structureTensorOuterSoaVector soaGradient soaTensor
        structureTensorOuterComponentsVector componentGradient componentTensor
        let spatialCount = int64 shape.Width * int64 shape.Height * int64 shape.Depth
        let outerBytes = spatialCount * int64 sizeof<float32> * int64 (3 + 6)
        let smoothBytes = spatialCount * int64 sizeof<float32> * int64 (6 + 6)
        let eigenBytes = spatialCount * int64 sizeof<float32> * int64 (6 + 3)
        let eigensystemBytes = spatialCount * int64 sizeof<float32> * int64 (6 + 12)
        let pipelineBytes = outerBytes + smoothBytes + eigenBytes
        let eigensystemPipelineBytes = outerBytes + smoothBytes + eigensystemBytes
        let convert3Bytes = spatialCount * int64 sizeof<float32> * int64 (3 + 3)
        let convert6Bytes = spatialCount * int64 sizeof<float32> * int64 (6 + 6)
        printfn "variant=%s shape=%ux%ux%u spatial=%d iterations=%d" variant shape.Width shape.Height shape.Depth spatialCount iterations
        match variant with
        | "aos-outer" ->
            runStructureTensorTimedMeasured outerBytes iterations (fun () -> structureTensorOuterAos aosGradient aosTensor) (fun () -> checksumStructureTensorChunkFloat32 aosTensor)
        | "soa-outer" ->
            runStructureTensorTimedMeasured outerBytes iterations (fun () -> structureTensorOuterSoaScalar soaGradient soaTensor) (fun () -> checksumStructureTensorChunkFloat32 soaTensor)
        | "soa-outer-vector" ->
            runStructureTensorTimedMeasured outerBytes iterations (fun () -> structureTensorOuterSoaVector soaGradient soaTensor) (fun () -> checksumStructureTensorChunkFloat32 soaTensor)
        | "components-outer-vector" ->
            runStructureTensorTimedMeasured outerBytes iterations (fun () -> structureTensorOuterComponentsVector componentGradient componentTensor) (fun () -> checksumStructureTensorComponentWindowFloat32 componentTensor)
        | "aos-smooth" ->
            runStructureTensorTimedMeasured smoothBytes iterations (fun () -> smoothTensorAos3Point aosTensor aosTensorScratch) (fun () -> checksumStructureTensorChunkFloat32 aosTensorScratch)
        | "soa-smooth-vector" ->
            runStructureTensorTimedMeasured smoothBytes iterations (fun () -> smoothTensorSoa3PointVector soaTensor soaTensorScratch) (fun () -> checksumStructureTensorChunkFloat32 soaTensorScratch)
        | "components-smooth-vector" ->
            runStructureTensorTimedMeasured smoothBytes iterations (fun () -> smoothTensorComponents3PointVector componentTensor componentTensorScratch) (fun () -> checksumStructureTensorComponentWindowFloat32 componentTensorScratch)
        | "aos-eigenvalues" ->
            runStructureTensorTimedMeasured eigenBytes iterations (fun () -> structureTensorEigenvaluesAos aosTensor aosEigenvalues) (fun () -> checksumStructureTensorChunkFloat32 aosEigenvalues)
        | "soa-eigenvalues" ->
            runStructureTensorTimedMeasured eigenBytes iterations (fun () -> structureTensorEigenvaluesSoa soaTensor soaEigenvalues) (fun () -> checksumStructureTensorChunkFloat32 soaEigenvalues)
        | "components-eigenvalues" ->
            runStructureTensorTimedMeasured eigenBytes iterations (fun () -> structureTensorEigenvaluesComponents componentTensor componentEigenvalues) (fun () -> checksumStructureTensorComponentWindowFloat32 componentEigenvalues)
        | "components-eigenvalues-jacobi6" ->
            runStructureTensorTimedMeasured eigenBytes iterations (fun () -> structureTensorEigenvaluesComponentsJacobi 6 componentTensor componentEigenvalues) (fun () -> checksumStructureTensorComponentWindowFloat32 componentEigenvalues)
        | "components-eigenvalues-jacobi8" ->
            runStructureTensorTimedMeasured eigenBytes iterations (fun () -> structureTensorEigenvaluesComponentsJacobi 8 componentTensor componentEigenvalues) (fun () -> checksumStructureTensorComponentWindowFloat32 componentEigenvalues)
        | "components-eigensystem" ->
            runStructureTensorTimedMeasured eigensystemBytes iterations (fun () -> structureTensorEigensystemComponents componentTensor componentEigensystem) (fun () -> checksumStructureTensorComponentWindowFloat32 componentEigensystem)
        | "components-eigensystem-chunk-algebra" ->
            runStructureTensorTimedMeasured eigensystemBytes iterations (fun () -> structureTensorEigensystemComponentsChunkAlgebra componentTensor componentEigensystem) (fun () -> checksumStructureTensorComponentWindowFloat32 componentEigensystem)
        | "components-eigensystem-jacobi-alloc" ->
            runStructureTensorTimedMeasured eigensystemBytes iterations (fun () -> structureTensorEigensystemComponentsJacobiAlloc componentTensor componentEigensystem) (fun () -> checksumStructureTensorComponentWindowFloat32 componentEigensystem)
        | "aos-pipeline" ->
            runStructureTensorTimedMeasured
                pipelineBytes
                iterations
                (fun () ->
                    structureTensorOuterAos aosGradient aosTensor
                    smoothTensorAos3Point aosTensor aosTensorScratch
                    structureTensorEigenvaluesAos aosTensorScratch aosEigenvalues)
                (fun () -> checksumStructureTensorChunkFloat32 aosEigenvalues)
        | "soa-pipeline-vector" ->
            runStructureTensorTimedMeasured
                pipelineBytes
                iterations
                (fun () ->
                    structureTensorOuterSoaVector soaGradient soaTensor
                    smoothTensorSoa3PointVector soaTensor soaTensorScratch
                    structureTensorEigenvaluesSoa soaTensorScratch soaEigenvalues)
                (fun () -> checksumStructureTensorChunkFloat32 soaEigenvalues)
        | "components-pipeline-vector" ->
            runStructureTensorTimedMeasured
                pipelineBytes
                iterations
                (fun () ->
                    structureTensorOuterComponentsVector componentGradient componentTensor
                    smoothTensorComponents3PointVector componentTensor componentTensorScratch
                    structureTensorEigenvaluesComponents componentTensorScratch componentEigenvalues)
                (fun () -> checksumStructureTensorComponentWindowFloat32 componentEigenvalues)
        | "components-pipeline-jacobi6" ->
            runStructureTensorTimedMeasured
                pipelineBytes
                iterations
                (fun () ->
                    structureTensorOuterComponentsVector componentGradient componentTensor
                    smoothTensorComponents3PointVector componentTensor componentTensorScratch
                    structureTensorEigenvaluesComponentsJacobi 6 componentTensorScratch componentEigenvalues)
                (fun () -> checksumStructureTensorComponentWindowFloat32 componentEigenvalues)
        | "components-pipeline-jacobi8" ->
            runStructureTensorTimedMeasured
                pipelineBytes
                iterations
                (fun () ->
                    structureTensorOuterComponentsVector componentGradient componentTensor
                    smoothTensorComponents3PointVector componentTensor componentTensorScratch
                    structureTensorEigenvaluesComponentsJacobi 8 componentTensorScratch componentEigenvalues)
                (fun () -> checksumStructureTensorComponentWindowFloat32 componentEigenvalues)
        | "components-pipeline-eigensystem" ->
            runStructureTensorTimedMeasured
                eigensystemPipelineBytes
                iterations
                (fun () ->
                    structureTensorOuterComponentsVector componentGradient componentTensor
                    smoothTensorComponents3PointVector componentTensor componentTensorScratch
                    structureTensorEigensystemComponents componentTensorScratch componentEigensystem)
                (fun () -> checksumStructureTensorComponentWindowFloat32 componentEigensystem)
        | "components-pipeline-eigensystem-chunk-algebra" ->
            runStructureTensorTimedMeasured
                eigensystemPipelineBytes
                iterations
                (fun () ->
                    structureTensorOuterComponentsVector componentGradient componentTensor
                    smoothTensorComponents3PointVector componentTensor componentTensorScratch
                    structureTensorEigensystemComponentsChunkAlgebra componentTensorScratch componentEigensystem)
                (fun () -> checksumStructureTensorComponentWindowFloat32 componentEigensystem)
        | "components-pipeline-eigensystem-jacobi-alloc" ->
            runStructureTensorTimedMeasured
                eigensystemPipelineBytes
                iterations
                (fun () ->
                    structureTensorOuterComponentsVector componentGradient componentTensor
                    smoothTensorComponents3PointVector componentTensor componentTensorScratch
                    structureTensorEigensystemComponentsJacobiAlloc componentTensorScratch componentEigensystem)
                (fun () -> checksumStructureTensorComponentWindowFloat32 componentEigensystem)
        | "aos-to-soa" ->
            runStructureTensorTimedMeasured convert3Bytes iterations (fun () -> convertAos3ToSoa3 aosGradient soaGradient) (fun () -> checksumStructureTensorChunkFloat32 soaGradient)
        | "soa-to-aos" ->
            structureTensorOuterSoaVector soaGradient soaTensor
            runStructureTensorTimedMeasured convert6Bytes iterations (fun () -> convertSoa6ToAos6 soaTensor aosTensor) (fun () -> checksumStructureTensorChunkFloat32 aosTensor)
        | other ->
            invalidArg "variant" $"Unsupported structure tensor layout variant '{other}'."
    finally
        releaseStructureTensorComponentWindow componentEigensystem
        releaseStructureTensorComponentWindow componentEigenvalues
        releaseStructureTensorComponentWindow componentTensorScratch
        releaseStructureTensorComponentWindow componentTensor
        releaseStructureTensorComponentWindow componentGradient
        StackCore.Chunk.decRef soaEigenvalues
        StackCore.Chunk.decRef aosEigenvalues
        StackCore.Chunk.decRef soaTensorScratch
        StackCore.Chunk.decRef aosTensorScratch
        StackCore.Chunk.decRef soaTensor
        StackCore.Chunk.decRef aosTensor
        StackCore.Chunk.decRef soaGradient
        StackCore.Chunk.decRef aosGradient

let private runChunkStructureTensorLayout opts =
    let shape = require "shape" opts |> parseShape
    let variant = optional "variant" "all" opts
    let iterations = optional "iterations" "10" opts |> int
    if iterations < 1 then invalidArg "iterations" "Expected at least one iteration."
    let variants =
        if variant = "all" then
            [ "aos-outer"
              "soa-outer"
              "soa-outer-vector"
              "components-outer-vector"
              "aos-smooth"
              "soa-smooth-vector"
              "components-smooth-vector"
              "aos-eigenvalues"
              "soa-eigenvalues"
              "components-eigenvalues"
              "components-eigenvalues-jacobi6"
              "components-eigenvalues-jacobi8"
              "components-eigensystem"
              "components-eigensystem-chunk-algebra"
              "components-eigensystem-jacobi-alloc"
              "aos-pipeline"
              "soa-pipeline-vector"
              "components-pipeline-vector"
              "components-pipeline-jacobi6"
              "components-pipeline-jacobi8"
              "components-pipeline-eigensystem"
              "components-pipeline-eigensystem-chunk-algebra"
              "components-pipeline-eigensystem-jacobi-alloc"
              "aos-to-soa"
              "soa-to-aos" ]
        else
            [ variant ]
    variants
    |> List.fold
        (fun exitCode current ->
            if exitCode <> 0 then exitCode
            else runStructureTensorLayoutVariant shape iterations current)
        0

let private runChunkConnectedComponentsCommand opts =
    let input = require "input" opts
    let output = opts |> Map.tryFind "output"
    let thresholdValue = optional "threshold" "128" opts |> fun s -> Double.Parse(s, invariant)
    let windowSize = opts |> Map.tryFind "window" |> Option.map Int32.Parse
    let workers = optional "workers" "1" opts |> Int32.Parse
    let availableMemory = optional "available-memory" (string (1024UL * 1024UL * 1024UL * 1024UL)) opts |> UInt64.Parse

    let stopwatch = Stopwatch.StartNew()
    let exitCode = runChunkConnectedComponents input output thresholdValue windowSize workers availableMemory
    stopwatch.Stop()
    writeInternalSeconds stopwatch.Elapsed
    exitCode

let private runChunkConvolve opts =
    let pixelType = require "pixel-type" opts |> parseChunkConvolvePixelType
    let input = require "input" opts
    let output = require "output" opts
    let kernelSize = optional "kernel-size" "3" opts |> UInt32.Parse
    let workers = optional "workers" "1" opts |> Int32.Parse
    let native = optional "native" "false" opts |> Boolean.Parse
    let availableMemory = optional "available-memory" (string (1024UL * 1024UL * 1024UL * 1024UL)) opts |> UInt64.Parse
    let debugLevel = optional "debug-level" "0" opts |> UInt32.Parse

    let stopwatch = Stopwatch.StartNew()
    let exitCode =
        match pixelType with
        | ChunkUInt8 -> runChunkConvolveTyped<uint8> input output kernelSize workers native availableMemory debugLevel
        | ChunkInt8 -> runChunkConvolveTyped<int8> input output kernelSize workers native availableMemory debugLevel
        | ChunkUInt16 -> runChunkConvolveTyped<uint16> input output kernelSize workers native availableMemory debugLevel
        | ChunkInt16 -> runChunkConvolveTyped<int16> input output kernelSize workers native availableMemory debugLevel
        | ChunkFloat32 -> runChunkConvolveTyped<float32> input output kernelSize workers native availableMemory debugLevel

    stopwatch.Stop()
    writeInternalSeconds stopwatch.Elapsed
    exitCode

let private runArrayPoolTyped<'T when 'T: equality> operation input output thresholdValue =
    let volume = readArrayPoolTiffStack<'T> input
    try
        match operation with
        | "copy" ->
            let copied = copyArrayPoolVolume volume
            try
                writeArrayPoolTiffStack output copied
            finally
                copied.decRefCount()
        | "threshold" ->
            let mask = thresholdArrayPoolVolume thresholdValue volume
            try
                writeArrayPoolTiffStack output mask
            finally
                mask.decRefCount()
        | _ -> failwith $"unsupported ArrayPool operation '{operation}' for {typeof<'T>.Name}"
        0
    finally
        volume.decRefCount()

let private runArrayPoolConnectedComponents input output thresholdValue =
    let volume = readArrayPoolTiffStack<uint8> input
    try
        let mask = thresholdArrayPoolVolume thresholdValue volume
        try
            let maskImage = pooledUInt8VolumeToImage mask "arraypool.connectedComponents.input"
            try
                let connected = ImageFunctions.connectedComponents maskImage
                try
                    let labels = labelImageToUInt8Volume connected.Labels mask.Width mask.Height mask.Depth
                    try
                        writeArrayPoolTiffStack output labels
                    finally
                        labels.decRefCount()
                finally
                    connected.Labels.decRefCount()
            finally
                maskImage.decRefCount()
        finally
            mask.decRefCount()
        0
    finally
        volume.decRefCount()

let private runArrayPool opts =
    let operation = require "operation" opts
    let pixelType = require "pixel-type" opts |> parsePixelType
    let input = require "input" opts
    let output = require "output" opts
    let thresholdValue = optional "threshold" "128" opts |> fun s -> Double.Parse(s, invariant)
    precleanDirectoryForTimedRun output
    let stopwatch = Stopwatch.StartNew()
    let exitCode =
        match operation, pixelType with
        | "connectedComponents", UInt8 -> runArrayPoolConnectedComponents input output thresholdValue
        | "connectedComponents", _ -> failwith "ArrayPool connectedComponents benchmark is currently defined for UInt8 masks only"
        | _, UInt8 -> runArrayPoolTyped<uint8> operation input output thresholdValue
        | _, UInt16 -> runArrayPoolTyped<uint16> operation input output thresholdValue
        | _, Int32 -> runArrayPoolTyped<int32> operation input output thresholdValue
        | _, Float32 -> runArrayPoolTyped<float32> operation input output thresholdValue
        | _, _ -> unsupportedPixelType "ArrayPool benchmark" "UInt8, UInt16, Int32, and Float32" pixelType
    stopwatch.Stop()
    writeInternalSeconds stopwatch.Elapsed
    exitCode

let private runArrayPoolSliceTyped<'T when 'T: equality> operation input output thresholdValue =
    let files = stackTiffFiles input
    if files.Length = 0 then
        invalidOp $"No TIFF files found in input stack directory: {input}"

    ensureCleanDirectory output
    for i in 0 .. files.Length - 1 do
        let slice = readArrayPoolTiffSlice<'T> files[i] $"arraypool.slice.read[{i}]"
        try
            match operation with
            | "copy" ->
                writeArrayPoolTiffPage (outputFile output i) slice.Width slice.Height slice.Buffer 0
            | "threshold" ->
                let mask = thresholdArrayPoolVolume thresholdValue slice
                try
                    writeArrayPoolTiffPage (outputFile output i) mask.Width mask.Height mask.Buffer 0
                finally
                    mask.decRefCount()
            | _ -> failwith $"unsupported slice ArrayPool operation '{operation}' for {typeof<'T>.Name}"
        finally
            slice.decRefCount()
    0

let private runArrayPoolSlice opts =
    let operation = require "operation" opts
    let pixelType = require "pixel-type" opts |> parsePixelType
    let input = require "input" opts
    let output = require "output" opts
    let thresholdValue = optional "threshold" "128" opts |> fun s -> Double.Parse(s, invariant)
    precleanDirectoryForTimedRun output
    let stopwatch = Stopwatch.StartNew()
    let exitCode =
        match operation, pixelType with
        | "connectedComponents", _ -> failwith "Slice ArrayPool backend is intended for copy/threshold allocation experiments, not connected components."
        | _, UInt8 -> runArrayPoolSliceTyped<uint8> operation input output thresholdValue
        | _, UInt16 -> runArrayPoolSliceTyped<uint16> operation input output thresholdValue
        | _, Int32 -> runArrayPoolSliceTyped<int32> operation input output thresholdValue
        | _, Float32 -> runArrayPoolSliceTyped<float32> operation input output thresholdValue
        | _, _ -> unsupportedPixelType "Slice ArrayPool benchmark" "UInt8, UInt16, Int32, and Float32" pixelType
    stopwatch.Stop()
    writeInternalSeconds stopwatch.Elapsed
    exitCode

let private runArrayPoolSliceReuseTyped<'T when 'T: equality> operation input output thresholdValue =
    let files = stackTiffFiles input
    if files.Length = 0 then
        invalidOp $"No TIFF files found in input stack directory: {input}"

    let width, height, rowBytes, scanlineSize = inspectTiffSlice<'T> files[0]
    let inputSlice = rentVolume<'T> width height 1u "arraypool.slice-reuse.input"
    let outputMask =
        if operation = "threshold" then
            Some(rentVolume<uint8> width height 1u "arraypool.slice-reuse.threshold")
        else
            None
    let readScanline = ArrayPool<byte>.Shared.Rent(scanlineSize)
    let writeRow = ArrayPool<byte>.Shared.Rent(rowBytes)
    let writeMaskRow =
        if operation = "threshold" && typeof<'T> <> typeof<uint8> then
            Some(ArrayPool<byte>.Shared.Rent(int width))
        else
            None

    ensureCleanDirectory output
    try
        for i in 0 .. files.Length - 1 do
            readArrayPoolTiffSliceInto<'T> files[i] width height rowBytes readScanline inputSlice
            match operation with
            | "copy" ->
                writeArrayPoolTiffPageWithRowBuffer (outputFile output i) width height rowBytes writeRow inputSlice.Buffer 0
            | "threshold" ->
                let mask = outputMask.Value
                if typeof<'T> = typeof<uint8> then
                    let inputBuffer = box inputSlice.Buffer :?> uint8[]
                    let maskBuffer = mask.Buffer
                    let threshold8 = byte thresholdValue
                    for p in 0 .. inputSlice.Length - 1 do
                        maskBuffer[p] <- if inputBuffer[p] >= threshold8 then 255uy else 0uy
                    writeArrayPoolTiffPageWithRowBuffer (outputFile output i) width height (int width) writeRow mask.Buffer 0
                elif typeof<'T> = typeof<uint16> then
                    let inputBuffer = box inputSlice.Buffer :?> uint16[]
                    let maskBuffer = mask.Buffer
                    let threshold16 = uint16 thresholdValue
                    for p in 0 .. inputSlice.Length - 1 do
                        maskBuffer[p] <- if inputBuffer[p] >= threshold16 then 255uy else 0uy
                    writeArrayPoolTiffPageWithRowBuffer (outputFile output i) width height (int width) writeMaskRow.Value mask.Buffer 0
                elif typeof<'T> = typeof<float32> then
                    let inputBuffer = box inputSlice.Buffer :?> float32[]
                    let maskBuffer = mask.Buffer
                    let threshold32 = float32 thresholdValue
                    for p in 0 .. inputSlice.Length - 1 do
                        maskBuffer[p] <- if inputBuffer[p] >= threshold32 then 255uy else 0uy
                    writeArrayPoolTiffPageWithRowBuffer (outputFile output i) width height (int width) writeMaskRow.Value mask.Buffer 0
                else
                    invalidArg "T" $"Unsupported ArrayPool threshold type {typeof<'T>.Name}."
            | _ -> failwith $"unsupported reusable slice ArrayPool operation '{operation}' for {typeof<'T>.Name}"
        0
    finally
        inputSlice.decRefCount()
        outputMask |> Option.iter (fun volume -> volume.decRefCount())
        writeMaskRow |> Option.iter ArrayPool<byte>.Shared.Return
        ArrayPool<byte>.Shared.Return(writeRow)
        ArrayPool<byte>.Shared.Return(readScanline)

let private runArrayPoolSliceReuse opts =
    let operation = require "operation" opts
    let pixelType = require "pixel-type" opts |> parsePixelType
    let input = require "input" opts
    let output = require "output" opts
    let thresholdValue = optional "threshold" "128" opts |> fun s -> Double.Parse(s, invariant)
    precleanDirectoryForTimedRun output
    let stopwatch = Stopwatch.StartNew()
    let exitCode =
        match operation, pixelType with
        | "connectedComponents", _ -> failwith "Reusable slice ArrayPool backend is intended for copy/threshold allocation experiments, not connected components."
        | _, UInt8 -> runArrayPoolSliceReuseTyped<uint8> operation input output thresholdValue
        | _, UInt16 -> runArrayPoolSliceReuseTyped<uint16> operation input output thresholdValue
        | _, Int32 -> runArrayPoolSliceReuseTyped<int32> operation input output thresholdValue
        | _, Float32 -> runArrayPoolSliceReuseTyped<float32> operation input output thresholdValue
        | _, _ -> unsupportedPixelType "Reusable slice ArrayPool benchmark" "UInt8, UInt16, Int32, and Float32" pixelType
    stopwatch.Stop()
    writeInternalSeconds stopwatch.Elapsed
    exitCode

let private runByteSliceReuse opts =
    let operation = require "operation" opts
    let pixelType = require "pixel-type" opts |> parsePixelType
    let input = require "input" opts
    let output = require "output" opts
    let thresholdValue = optional "threshold" "128" opts |> fun s -> Double.Parse(s, invariant)

    if pixelType <> UInt8 then
        failwith "Byte-slice reuse backend is only defined for UInt8."

    let files = stackTiffFiles input
    if files.Length = 0 then
        invalidOp $"No TIFF files found in input stack directory: {input}"

    let width, height, rowBytes, _scanlineSize = inspectTiffSlice<uint8> files[0]
    let pageBytes = rowBytes * int height
    let inputPage = ArrayPool<byte>.Shared.Rent(pageBytes)
    let outputPage =
        if operation = "threshold" then
            Some(ArrayPool<byte>.Shared.Rent(pageBytes))
        else
            None

    ensureCleanDirectory output
    let stopwatch = Stopwatch.StartNew()
    try
        for i in 0 .. files.Length - 1 do
            readByteTiffSliceInto files[i] width height rowBytes inputPage
            match operation with
            | "copy" ->
                writeByteTiffPageFor<uint8> (outputFile output i) width height rowBytes inputPage
            | "threshold" ->
                let out = outputPage.Value
                let threshold8 = byte thresholdValue
                for p in 0 .. pageBytes - 1 do
                    out[p] <- if inputPage[p] >= threshold8 then 255uy else 0uy
                writeByteTiffPageFor<uint8> (outputFile output i) width height rowBytes out
            | _ -> failwith $"unsupported byte-slice operation '{operation}'"
        stopwatch.Stop()
        writeInternalSeconds stopwatch.Elapsed
        0
    finally
        outputPage |> Option.iter ArrayPool<byte>.Shared.Return
        ArrayPool<byte>.Shared.Return(inputPage)

let private runByteFloat32SliceReuse opts =
    let operation = require "operation" opts
    let pixelType = require "pixel-type" opts |> parsePixelType
    let input = require "input" opts
    let output = require "output" opts
    let thresholdValue = optional "threshold" "128" opts |> fun s -> Double.Parse(s, invariant)

    if pixelType <> Float32 then
        failwith "Byte-float32 slice reuse backend is only defined for Float32."

    let files = stackTiffFiles input
    if files.Length = 0 then
        invalidOp $"No TIFF files found in input stack directory: {input}"

    let width, height, rowBytes, _scanlineSize = inspectTiffSlice<float32> files[0]
    let pageBytes = rowBytes * int height
    let inputPage = ArrayPool<byte>.Shared.Rent(pageBytes)
    let outputPage =
        if operation = "threshold" then
            Some(ArrayPool<byte>.Shared.Rent(int width * int height))
        else
            None

    ensureCleanDirectory output
    let stopwatch = Stopwatch.StartNew()
    try
        for i in 0 .. files.Length - 1 do
            readByteTiffSliceInto files[i] width height rowBytes inputPage
            match operation with
            | "copy" ->
                writeByteTiffPageFor<float32> (outputFile output i) width height rowBytes inputPage
            | "threshold" ->
                let out = outputPage.Value
                let inputSpan = MemoryMarshal.Cast<byte, float32>(inputPage.AsSpan(0, pageBytes))
                let threshold32 = float32 thresholdValue
                for p in 0 .. inputSpan.Length - 1 do
                    out[p] <- if inputSpan[p] >= threshold32 then 255uy else 0uy
                writeByteTiffPageFor<uint8> (outputFile output i) width height (int width) out
            | _ -> failwith $"unsupported byte-float32-slice operation '{operation}'"
        stopwatch.Stop()
        writeInternalSeconds stopwatch.Elapsed
        0
    finally
        outputPage |> Option.iter ArrayPool<byte>.Shared.Return
        ArrayPool<byte>.Shared.Return(inputPage)

let private runLibTiffDirectCopy opts =
    let operation = require "operation" opts
    let pixelType = require "pixel-type" opts |> parsePixelType
    let input = require "input" opts
    let output = require "output" opts

    if operation <> "copy" then
        failwith $"Direct LibTiff backend is intentionally copy-only; got '{operation}'."

    let files = stackTiffFiles input
    if files.Length = 0 then
        invalidOp $"No TIFF files found in input stack directory: {input}"

    let width, height, rowBytes, scanlineSize = inspectDirectTiffSlice pixelType files[0]
    let pageBytes = rowBytes * int height
    let page = ArrayPool<byte>.Shared.Rent(pageBytes)
    let scratch = ArrayPool<byte>.Shared.Rent(scanlineSize)

    ensureCleanDirectory output
    let stopwatch = Stopwatch.StartNew()
    try
        for i in 0 .. files.Length - 1 do
            readDirectByteTiffSliceInto pixelType files[i] width height rowBytes page scratch
            writeDirectByteTiffPage pixelType (outputFile output i) width height rowBytes page

        stopwatch.Stop()
        writeInternalSeconds stopwatch.Elapsed
        0
    finally
        ArrayPool<byte>.Shared.Return(scratch)
        ArrayPool<byte>.Shared.Return(page)

let private runLibTiffDirectThreshold opts =
    let operation = require "operation" opts
    let pixelType = require "pixel-type" opts |> parsePixelType
    let input = require "input" opts
    let output = require "output" opts
    let thresholdValue = optional "threshold" "128" opts |> fun s -> Double.Parse(s, invariant)

    if operation <> "threshold" then
        failwith $"Direct LibTiff threshold backend is intentionally threshold-only; got '{operation}'."

    let files = stackTiffFiles input
    if files.Length = 0 then
        invalidOp $"No TIFF files found in input stack directory: {input}"

    let width, height, rowBytes, scanlineSize = inspectDirectTiffSlice pixelType files[0]
    let pageBytes = rowBytes * int height
    let maskBytes = int width * int height
    let page = ArrayPool<byte>.Shared.Rent(pageBytes)
    let scratch = ArrayPool<byte>.Shared.Rent(scanlineSize)
    let mask = ArrayPool<byte>.Shared.Rent(maskBytes)

    ensureCleanDirectory output
    let stopwatch = Stopwatch.StartNew()
    try
        for i in 0 .. files.Length - 1 do
            readDirectByteTiffSliceInto pixelType files[i] width height rowBytes page scratch
            thresholdDirectPageSimdInto pixelType thresholdValue pageBytes page mask
            writeDirectByteTiffPage UInt8 (outputFile output i) width height (int width) mask

        stopwatch.Stop()
        writeInternalSeconds stopwatch.Elapsed
        0
    finally
        ArrayPool<byte>.Shared.Return(mask)
        ArrayPool<byte>.Shared.Return(scratch)
        ArrayPool<byte>.Shared.Return(page)

let private runLibTiffDirectThresholdInType opts =
    let operation = require "operation" opts
    let pixelType = require "pixel-type" opts |> parsePixelType
    let input = require "input" opts
    let output = require "output" opts
    let thresholdValue = optional "threshold" "128" opts |> fun s -> Double.Parse(s, invariant)

    if operation <> "threshold" then
        failwith $"Direct LibTiff in-type threshold backend is intentionally threshold-only; got '{operation}'."

    let files = stackTiffFiles input
    if files.Length = 0 then
        invalidOp $"No TIFF files found in input stack directory: {input}"

    let width, height, rowBytes, scanlineSize = inspectDirectTiffSlice pixelType files[0]
    let pageBytes = rowBytes * int height
    let page = ArrayPool<byte>.Shared.Rent(pageBytes)
    let outputPage = ArrayPool<byte>.Shared.Rent(pageBytes)
    let scratch = ArrayPool<byte>.Shared.Rent(scanlineSize)

    ensureCleanDirectory output
    let stopwatch = Stopwatch.StartNew()
    try
        for i in 0 .. files.Length - 1 do
            readDirectByteTiffSliceInto pixelType files[i] width height rowBytes page scratch
            thresholdDirectPageInTypeInto pixelType thresholdValue pageBytes page outputPage
            writeDirectByteTiffPage pixelType (outputFile output i) width height rowBytes outputPage

        stopwatch.Stop()
        writeInternalSeconds stopwatch.Elapsed
        0
    finally
        ArrayPool<byte>.Shared.Return(scratch)
        ArrayPool<byte>.Shared.Return(outputPage)
        ArrayPool<byte>.Shared.Return(page)

let private runTimedHotLoop iterations (action: unit -> unit) (checksum: unit -> int) =
    action()
    GC.Collect()
    GC.WaitForPendingFinalizers()
    GC.Collect()
    let stopwatch = Stopwatch.StartNew()
    for _ in 1 .. iterations do
        action()
    stopwatch.Stop()
    let checksumValue = checksum()
    if checksumValue = Int32.MinValue then
        printfn "%d" checksumValue
    writeInternalSeconds stopwatch.Elapsed
    printfn "totalSeconds=%s perIterationSeconds=%s checksum=%d" (stopwatch.Elapsed.TotalSeconds.ToString("F9", invariant)) ((stopwatch.Elapsed.TotalSeconds / float iterations).ToString("F9", invariant)) checksumValue
    0

let private runTimedHotLoopMeasured bytesPerIteration iterations (action: unit -> unit) (checksum: unit -> int) =
    action()
    GC.Collect()
    GC.WaitForPendingFinalizers()
    GC.Collect()
    let allocatedBefore = GC.GetAllocatedBytesForCurrentThread()
    let stopwatch = Stopwatch.StartNew()
    for _ in 1 .. iterations do
        action()
    stopwatch.Stop()
    let allocatedAfter = GC.GetAllocatedBytesForCurrentThread()
    let checksumValue = checksum()
    if checksumValue = Int32.MinValue then
        printfn "%d" checksumValue
    writeInternalSeconds stopwatch.Elapsed
    let perIteration = stopwatch.Elapsed.TotalSeconds / float iterations
    let gbps =
        if perIteration <= 0.0 then
            Double.PositiveInfinity
        else
            (float bytesPerIteration / perIteration) / 1.0e9
    let allocatedPerIteration = float (allocatedAfter - allocatedBefore) / float iterations
    printfn
        "totalSeconds=%s perIterationSeconds=%s effectiveGBps=%s allocatedBytesPerIteration=%s checksum=%d"
        (stopwatch.Elapsed.TotalSeconds.ToString("F9", invariant))
        (perIteration.ToString("F9", invariant))
        (gbps.ToString("F3", invariant))
        (allocatedPerIteration.ToString("F1", invariant))
        checksumValue
    0

type ReductionPixelType =
    | ReductionUInt8
    | ReductionUInt16
    | ReductionFloat32
    | ReductionFloat64

let private parseReductionPixelType value =
    match value with
    | "UInt8" | "uint8" -> ReductionUInt8
    | "UInt16" | "uint16" -> ReductionUInt16
    | "Float32" | "float32" -> ReductionFloat32
    | "Float64" | "float64" | "Double" | "double" -> ReductionFloat64
    | _ -> invalidArg "pixel-type" $"Unsupported reduction pixel type '{value}'. Use UInt8, UInt16, Float32, or Float64."

let private reductionPixelName = function
    | ReductionUInt8 -> "UInt8"
    | ReductionUInt16 -> "UInt16"
    | ReductionFloat32 -> "Float32"
    | ReductionFloat64 -> "Float64"

let private fillReductionChunkUInt8 (chunk: StackCore.Chunk<uint8>) =
    let values = StackCore.Chunk.span<uint8> chunk
    let mutable i = 0
    while i < values.Length do
        values[i] <- uint8 ((i * 37 + i / 17) &&& 0xFF)
        i <- i + 1

let private fillReductionChunkUInt16 (chunk: StackCore.Chunk<uint16>) =
    let values = StackCore.Chunk.span<uint16> chunk
    let mutable i = 0
    while i < values.Length do
        values[i] <- uint16 ((i * 257 + i / 13) &&& 0xFFFF)
        i <- i + 1

let private fillReductionChunkFloat32 (chunk: StackCore.Chunk<float32>) =
    let values = StackCore.Chunk.span<float32> chunk
    let mutable i = 0
    while i < values.Length do
        values[i] <- float32 ((i * 37 + i / 17) &&& 0xFFFF) / 257.0f
        i <- i + 1

let private fillReductionChunkFloat64 (chunk: StackCore.Chunk<float>) =
    let values = StackCore.Chunk.span<float> chunk
    let mutable i = 0
    while i < values.Length do
        values[i] <- float ((i * 37 + i / 17) &&& 0xFFFF) / 257.0
        i <- i + 1

let private sumScalarUInt8 (values: Span<uint8>) =
    let mutable sum = 0.0
    let mutable i = 0
    while i < values.Length do
        sum <- sum + float values[i]
        i <- i + 1
    sum

let private sumScalarUInt16 (values: Span<uint16>) =
    let mutable sum = 0.0
    let mutable i = 0
    while i < values.Length do
        sum <- sum + float values[i]
        i <- i + 1
    sum

let private sumScalarFloat32 (values: Span<float32>) =
    let mutable sum = 0.0
    let mutable i = 0
    while i < values.Length do
        sum <- sum + float values[i]
        i <- i + 1
    sum

let private sumScalarFloat64 (values: Span<float>) =
    let mutable sum = 0.0
    let mutable i = 0
    while i < values.Length do
        sum <- sum + values[i]
        i <- i + 1
    sum

let private momentsScalarUInt8 (values: Span<uint8>) =
    let mutable sum = 0.0
    let mutable sumSq = 0.0
    let mutable minimum = Double.PositiveInfinity
    let mutable maximum = Double.NegativeInfinity
    let mutable i = 0
    while i < values.Length do
        let value = float values[i]
        sum <- sum + value
        sumSq <- sumSq + value * value
        if value < minimum then minimum <- value
        if value > maximum then maximum <- value
        i <- i + 1
    let count = float values.Length
    let var =
        if values.Length > 1 then
            max 0.0 ((sumSq - sum * sum / count) / (count - 1.0))
        else
            0.0
    struct (sum, sqrt var, minimum, maximum)

let private momentsScalarUInt16 (values: Span<uint16>) =
    let mutable sum = 0.0
    let mutable sumSq = 0.0
    let mutable minimum = Double.PositiveInfinity
    let mutable maximum = Double.NegativeInfinity
    let mutable i = 0
    while i < values.Length do
        let value = float values[i]
        sum <- sum + value
        sumSq <- sumSq + value * value
        if value < minimum then minimum <- value
        if value > maximum then maximum <- value
        i <- i + 1
    let count = float values.Length
    let var =
        if values.Length > 1 then
            max 0.0 ((sumSq - sum * sum / count) / (count - 1.0))
        else
            0.0
    struct (sum, sqrt var, minimum, maximum)

let private momentsScalarFloat32 (values: Span<float32>) =
    let mutable sum = 0.0
    let mutable sumSq = 0.0
    let mutable minimum = Double.PositiveInfinity
    let mutable maximum = Double.NegativeInfinity
    let mutable i = 0
    while i < values.Length do
        let value = float values[i]
        sum <- sum + value
        sumSq <- sumSq + value * value
        if value < minimum then minimum <- value
        if value > maximum then maximum <- value
        i <- i + 1
    let count = float values.Length
    let var =
        if values.Length > 1 then
            max 0.0 ((sumSq - sum * sum / count) / (count - 1.0))
        else
            0.0
    struct (sum, sqrt var, minimum, maximum)

let private momentsScalarFloat64 (values: Span<float>) =
    let mutable sum = 0.0
    let mutable sumSq = 0.0
    let mutable minimum = Double.PositiveInfinity
    let mutable maximum = Double.NegativeInfinity
    let mutable i = 0
    while i < values.Length do
        let value = values[i]
        sum <- sum + value
        sumSq <- sumSq + value * value
        if value < minimum then minimum <- value
        if value > maximum then maximum <- value
        i <- i + 1
    let count = float values.Length
    let var =
        if values.Length > 1 then
            max 0.0 ((sumSq - sum * sum / count) / (count - 1.0))
        else
            0.0
    struct (sum, sqrt var, minimum, maximum)

let private sumLanesUInt32 (v: Vector<uint32>) =
    let mutable sum = 0.0
    let mutable lane = 0
    while lane < Vector<uint32>.Count do
        sum <- sum + float v[lane]
        lane <- lane + 1
    sum

let private sumLanesUInt64 (v: Vector<uint64>) =
    let mutable sum = 0.0
    let mutable lane = 0
    while lane < Vector<uint64>.Count do
        sum <- sum + float v[lane]
        lane <- lane + 1
    sum

let private reduceMinByte (v: Vector<byte>) =
    let mutable minimum = Byte.MaxValue
    let mutable lane = 0
    while lane < Vector<byte>.Count do
        if v[lane] < minimum then minimum <- v[lane]
        lane <- lane + 1
    float minimum

let private reduceMaxByte (v: Vector<byte>) =
    let mutable maximum = Byte.MinValue
    let mutable lane = 0
    while lane < Vector<byte>.Count do
        if v[lane] > maximum then maximum <- v[lane]
        lane <- lane + 1
    float maximum

let private reduceMinUInt16 (v: Vector<uint16>) =
    let mutable minimum = UInt16.MaxValue
    let mutable lane = 0
    while lane < Vector<uint16>.Count do
        if v[lane] < minimum then minimum <- v[lane]
        lane <- lane + 1
    float minimum

let private reduceMaxUInt16 (v: Vector<uint16>) =
    let mutable maximum = UInt16.MinValue
    let mutable lane = 0
    while lane < Vector<uint16>.Count do
        if v[lane] > maximum then maximum <- v[lane]
        lane <- lane + 1
    float maximum

let private sumVectorWidenedUInt8 (values: Span<uint8>) =
    let width = Vector<byte>.Count
    let vectorEnd = values.Length - (values.Length % width)
    let flushEvery = 4096
    let mutable sum0 = Vector<uint32>.Zero
    let mutable sum1 = Vector<uint32>.Zero
    let mutable sum2 = Vector<uint32>.Zero
    let mutable sum3 = Vector<uint32>.Zero
    let mutable sum = 0.0
    let mutable vectorsSinceFlush = 0
    let flush () =
        sum <- sum + sumLanesUInt32 sum0 + sumLanesUInt32 sum1 + sumLanesUInt32 sum2 + sumLanesUInt32 sum3
        sum0 <- Vector<uint32>.Zero
        sum1 <- Vector<uint32>.Zero
        sum2 <- Vector<uint32>.Zero
        sum3 <- Vector<uint32>.Zero
        vectorsSinceFlush <- 0
    let mutable i = 0
    while i < vectorEnd do
        let v = Vector<byte>(values.Slice(i, width))
        let mutable lo16 = Vector<uint16>.Zero
        let mutable hi16 = Vector<uint16>.Zero
        Vector.Widen(v, &lo16, &hi16)
        let mutable lo0 = Vector<uint32>.Zero
        let mutable lo1 = Vector<uint32>.Zero
        let mutable hi0 = Vector<uint32>.Zero
        let mutable hi1 = Vector<uint32>.Zero
        Vector.Widen(lo16, &lo0, &lo1)
        Vector.Widen(hi16, &hi0, &hi1)
        sum0 <- sum0 + lo0
        sum1 <- sum1 + lo1
        sum2 <- sum2 + hi0
        sum3 <- sum3 + hi1
        vectorsSinceFlush <- vectorsSinceFlush + 1
        if vectorsSinceFlush = flushEvery then flush()
        i <- i + width
    flush()
    while i < values.Length do
        sum <- sum + float values[i]
        i <- i + 1
    sum

let private momentsVectorWidenedUInt8 (values: Span<uint8>) =
    if values.Length = 0 then
        struct (0.0, 0.0, Double.PositiveInfinity, Double.NegativeInfinity)
    else
        let width = Vector<byte>.Count
        let vectorEnd = values.Length - (values.Length % width)
        let flushEvery = 4096
        let mutable sum0 = Vector<uint32>.Zero
        let mutable sum1 = Vector<uint32>.Zero
        let mutable sum2 = Vector<uint32>.Zero
        let mutable sum3 = Vector<uint32>.Zero
        let mutable sq0 = Vector<uint32>.Zero
        let mutable sq1 = Vector<uint32>.Zero
        let mutable sq2 = Vector<uint32>.Zero
        let mutable sq3 = Vector<uint32>.Zero
        let mutable minAcc = Vector<byte>(Byte.MaxValue)
        let mutable maxAcc = Vector<byte>(Byte.MinValue)
        let mutable sum = 0.0
        let mutable sumSq = 0.0
        let mutable vectorsSinceFlush = 0
        let flush () =
            sum <- sum + sumLanesUInt32 sum0 + sumLanesUInt32 sum1 + sumLanesUInt32 sum2 + sumLanesUInt32 sum3
            sumSq <- sumSq + sumLanesUInt32 sq0 + sumLanesUInt32 sq1 + sumLanesUInt32 sq2 + sumLanesUInt32 sq3
            sum0 <- Vector<uint32>.Zero
            sum1 <- Vector<uint32>.Zero
            sum2 <- Vector<uint32>.Zero
            sum3 <- Vector<uint32>.Zero
            sq0 <- Vector<uint32>.Zero
            sq1 <- Vector<uint32>.Zero
            sq2 <- Vector<uint32>.Zero
            sq3 <- Vector<uint32>.Zero
            vectorsSinceFlush <- 0
        let mutable i = 0
        while i < vectorEnd do
            let v = Vector<byte>(values.Slice(i, width))
            minAcc <- Vector.Min(minAcc, v)
            maxAcc <- Vector.Max(maxAcc, v)
            let mutable lo16 = Vector<uint16>.Zero
            let mutable hi16 = Vector<uint16>.Zero
            Vector.Widen(v, &lo16, &hi16)
            let mutable lo0 = Vector<uint32>.Zero
            let mutable lo1 = Vector<uint32>.Zero
            let mutable hi0 = Vector<uint32>.Zero
            let mutable hi1 = Vector<uint32>.Zero
            Vector.Widen(lo16, &lo0, &lo1)
            Vector.Widen(hi16, &hi0, &hi1)
            sum0 <- sum0 + lo0
            sum1 <- sum1 + lo1
            sum2 <- sum2 + hi0
            sum3 <- sum3 + hi1
            sq0 <- sq0 + lo0 * lo0
            sq1 <- sq1 + lo1 * lo1
            sq2 <- sq2 + hi0 * hi0
            sq3 <- sq3 + hi1 * hi1
            vectorsSinceFlush <- vectorsSinceFlush + 1
            if vectorsSinceFlush = flushEvery then flush()
            i <- i + width
        flush()
        let mutable minimum = reduceMinByte minAcc
        let mutable maximum = reduceMaxByte maxAcc
        while i < values.Length do
            let value = float values[i]
            sum <- sum + value
            sumSq <- sumSq + value * value
            if value < minimum then minimum <- value
            if value > maximum then maximum <- value
            i <- i + 1
        let count = float values.Length
        let var = if values.Length > 1 then max 0.0 ((sumSq - sum * sum / count) / (count - 1.0)) else 0.0
        struct (sum, sqrt var, minimum, maximum)

let private sumVectorWidenedUInt16 (values: Span<uint16>) =
    let width = Vector<uint16>.Count
    let vectorEnd = values.Length - (values.Length % width)
    let flushEvery = 4096
    let mutable sum0 = Vector<uint32>.Zero
    let mutable sum1 = Vector<uint32>.Zero
    let mutable sum = 0.0
    let mutable vectorsSinceFlush = 0
    let flush () =
        sum <- sum + sumLanesUInt32 sum0 + sumLanesUInt32 sum1
        sum0 <- Vector<uint32>.Zero
        sum1 <- Vector<uint32>.Zero
        vectorsSinceFlush <- 0
    let mutable i = 0
    while i < vectorEnd do
        let v = Vector<uint16>(values.Slice(i, width))
        let mutable lo = Vector<uint32>.Zero
        let mutable hi = Vector<uint32>.Zero
        Vector.Widen(v, &lo, &hi)
        sum0 <- sum0 + lo
        sum1 <- sum1 + hi
        vectorsSinceFlush <- vectorsSinceFlush + 1
        if vectorsSinceFlush = flushEvery then flush()
        i <- i + width
    flush()
    while i < values.Length do
        sum <- sum + float values[i]
        i <- i + 1
    sum

let private momentsVectorWidenedUInt16 (values: Span<uint16>) =
    if values.Length = 0 then
        struct (0.0, 0.0, Double.PositiveInfinity, Double.NegativeInfinity)
    else
        let width = Vector<uint16>.Count
        let vectorEnd = values.Length - (values.Length % width)
        let flushEvery = 4096
        let mutable sum0 = Vector<uint32>.Zero
        let mutable sum1 = Vector<uint32>.Zero
        let mutable sq0 = Vector<uint64>.Zero
        let mutable sq1 = Vector<uint64>.Zero
        let mutable sq2 = Vector<uint64>.Zero
        let mutable sq3 = Vector<uint64>.Zero
        let mutable minAcc = Vector<uint16>(UInt16.MaxValue)
        let mutable maxAcc = Vector<uint16>(UInt16.MinValue)
        let mutable sum = 0.0
        let mutable sumSq = 0.0
        let mutable vectorsSinceFlush = 0
        let flush () =
            sum <- sum + sumLanesUInt32 sum0 + sumLanesUInt32 sum1
            sumSq <- sumSq + sumLanesUInt64 sq0 + sumLanesUInt64 sq1 + sumLanesUInt64 sq2 + sumLanesUInt64 sq3
            sum0 <- Vector<uint32>.Zero
            sum1 <- Vector<uint32>.Zero
            sq0 <- Vector<uint64>.Zero
            sq1 <- Vector<uint64>.Zero
            sq2 <- Vector<uint64>.Zero
            sq3 <- Vector<uint64>.Zero
            vectorsSinceFlush <- 0
        let mutable i = 0
        while i < vectorEnd do
            let v = Vector<uint16>(values.Slice(i, width))
            minAcc <- Vector.Min(minAcc, v)
            maxAcc <- Vector.Max(maxAcc, v)
            let mutable lo = Vector<uint32>.Zero
            let mutable hi = Vector<uint32>.Zero
            Vector.Widen(v, &lo, &hi)
            sum0 <- sum0 + lo
            sum1 <- sum1 + hi
            let loSq = lo * lo
            let hiSq = hi * hi
            let mutable loSq0 = Vector<uint64>.Zero
            let mutable loSq1 = Vector<uint64>.Zero
            let mutable hiSq0 = Vector<uint64>.Zero
            let mutable hiSq1 = Vector<uint64>.Zero
            Vector.Widen(loSq, &loSq0, &loSq1)
            Vector.Widen(hiSq, &hiSq0, &hiSq1)
            sq0 <- sq0 + loSq0
            sq1 <- sq1 + loSq1
            sq2 <- sq2 + hiSq0
            sq3 <- sq3 + hiSq1
            vectorsSinceFlush <- vectorsSinceFlush + 1
            if vectorsSinceFlush = flushEvery then flush()
            i <- i + width
        flush()
        let mutable minimum = reduceMinUInt16 minAcc
        let mutable maximum = reduceMaxUInt16 maxAcc
        while i < values.Length do
            let value = float values[i]
            sum <- sum + value
            sumSq <- sumSq + value * value
            if value < minimum then minimum <- value
            if value > maximum then maximum <- value
            i <- i + 1
        let count = float values.Length
        let var = if values.Length > 1 then max 0.0 ((sumSq - sum * sum / count) / (count - 1.0)) else 0.0
        struct (sum, sqrt var, minimum, maximum)

let private sumVectorFloat32 (values: Span<float32>) =
    let width = Vector<float32>.Count
    let vectorEnd = values.Length - (values.Length % width)
    let mutable acc = Vector<float32>.Zero
    let mutable i = 0
    while i < vectorEnd do
        acc <- acc + Vector<float32>(values.Slice(i, width))
        i <- i + width
    let mutable sum = 0.0
    let mutable lane = 0
    while lane < width do
        sum <- sum + float acc[lane]
        lane <- lane + 1
    while i < values.Length do
        sum <- sum + float values[i]
        i <- i + 1
    sum

let private sumVectorFloat64 (values: Span<float>) =
    let width = Vector<float>.Count
    let vectorEnd = values.Length - (values.Length % width)
    let mutable acc = Vector<float>.Zero
    let mutable i = 0
    while i < vectorEnd do
        acc <- acc + Vector<float>(values.Slice(i, width))
        i <- i + width
    let mutable sum = 0.0
    let mutable lane = 0
    while lane < width do
        sum <- sum + acc[lane]
        lane <- lane + 1
    while i < values.Length do
        sum <- sum + values[i]
        i <- i + 1
    sum

let private sumVectorAccurateFloat32 (values: Span<float32>) =
    let width = Vector<float32>.Count
    let vectorEnd = values.Length - (values.Length % width)
    let mutable acc0 = Vector<float>.Zero
    let mutable acc1 = Vector<float>.Zero
    let mutable i = 0
    while i < vectorEnd do
        let v = Vector<float32>(values.Slice(i, width))
        let mutable lo = Vector<float>.Zero
        let mutable hi = Vector<float>.Zero
        Vector.Widen(v, &lo, &hi)
        acc0 <- acc0 + lo
        acc1 <- acc1 + hi
        i <- i + width
    let mutable sum = 0.0
    let mutable lane = 0
    while lane < Vector<float>.Count do
        sum <- sum + acc0[lane] + acc1[lane]
        lane <- lane + 1
    while i < values.Length do
        sum <- sum + float values[i]
        i <- i + 1
    sum

let private momentsVectorFloat32 (values: Span<float32>) =
    if values.Length = 0 then
        struct (0.0, 0.0, Double.PositiveInfinity, Double.NegativeInfinity)
    else
        let width = Vector<float32>.Count
        let vectorEnd = values.Length - (values.Length % width)
        let mutable sumAcc = Vector<float32>.Zero
        let mutable sumSqAcc = Vector<float32>.Zero
        let mutable minAcc = Vector<float32>(Single.PositiveInfinity)
        let mutable maxAcc = Vector<float32>(Single.NegativeInfinity)
        let mutable i = 0
        while i < vectorEnd do
            let v = Vector<float32>(values.Slice(i, width))
            sumAcc <- sumAcc + v
            sumSqAcc <- sumSqAcc + v * v
            minAcc <- Vector.Min(minAcc, v)
            maxAcc <- Vector.Max(maxAcc, v)
            i <- i + width
        let mutable sum = 0.0
        let mutable sumSq = 0.0
        let mutable minimum = Double.PositiveInfinity
        let mutable maximum = Double.NegativeInfinity
        let mutable lane = 0
        while lane < width do
            let sumValue = float sumAcc[lane]
            let sumSqValue = float sumSqAcc[lane]
            let minValue = float minAcc[lane]
            let maxValue = float maxAcc[lane]
            sum <- sum + sumValue
            sumSq <- sumSq + sumSqValue
            if minValue < minimum then minimum <- minValue
            if maxValue > maximum then maximum <- maxValue
            lane <- lane + 1
        while i < values.Length do
            let value = float values[i]
            sum <- sum + value
            sumSq <- sumSq + value * value
            if value < minimum then minimum <- value
            if value > maximum then maximum <- value
            i <- i + 1
        let count = float values.Length
        let var = if values.Length > 1 then max 0.0 ((sumSq - sum * sum / count) / (count - 1.0)) else 0.0
        struct (sum, sqrt var, minimum, maximum)

let private momentsVectorAccurateFloat32 (values: Span<float32>) =
    if values.Length = 0 then
        struct (0.0, 0.0, Double.PositiveInfinity, Double.NegativeInfinity)
    else
        let width = Vector<float32>.Count
        let vectorEnd = values.Length - (values.Length % width)
        let mutable sum0 = Vector<float>.Zero
        let mutable sum1 = Vector<float>.Zero
        let mutable sumSq0 = Vector<float>.Zero
        let mutable sumSq1 = Vector<float>.Zero
        let mutable minAcc = Vector<float32>(Single.PositiveInfinity)
        let mutable maxAcc = Vector<float32>(Single.NegativeInfinity)
        let mutable i = 0
        while i < vectorEnd do
            let v = Vector<float32>(values.Slice(i, width))
            minAcc <- Vector.Min(minAcc, v)
            maxAcc <- Vector.Max(maxAcc, v)
            let mutable lo = Vector<float>.Zero
            let mutable hi = Vector<float>.Zero
            Vector.Widen(v, &lo, &hi)
            sum0 <- sum0 + lo
            sum1 <- sum1 + hi
            sumSq0 <- sumSq0 + lo * lo
            sumSq1 <- sumSq1 + hi * hi
            i <- i + width
        let mutable sum = 0.0
        let mutable sumSq = 0.0
        let mutable minimum = Double.PositiveInfinity
        let mutable maximum = Double.NegativeInfinity
        let mutable lane = 0
        while lane < Vector<float>.Count do
            sum <- sum + sum0[lane] + sum1[lane]
            sumSq <- sumSq + sumSq0[lane] + sumSq1[lane]
            lane <- lane + 1
        lane <- 0
        while lane < Vector<float32>.Count do
            let minValue = float minAcc[lane]
            let maxValue = float maxAcc[lane]
            if minValue < minimum then minimum <- minValue
            if maxValue > maximum then maximum <- maxValue
            lane <- lane + 1
        while i < values.Length do
            let value = float values[i]
            sum <- sum + value
            sumSq <- sumSq + value * value
            if value < minimum then minimum <- value
            if value > maximum then maximum <- value
            i <- i + 1
        let count = float values.Length
        let var = if values.Length > 1 then max 0.0 ((sumSq - sum * sum / count) / (count - 1.0)) else 0.0
        struct (sum, sqrt var, minimum, maximum)

let private momentsVectorFloat64 (values: Span<float>) =
    if values.Length = 0 then
        struct (0.0, 0.0, Double.PositiveInfinity, Double.NegativeInfinity)
    else
        let width = Vector<float>.Count
        let vectorEnd = values.Length - (values.Length % width)
        let mutable sumAcc = Vector<float>.Zero
        let mutable sumSqAcc = Vector<float>.Zero
        let mutable minAcc = Vector<float>(Double.PositiveInfinity)
        let mutable maxAcc = Vector<float>(Double.NegativeInfinity)
        let mutable i = 0
        while i < vectorEnd do
            let v = Vector<float>(values.Slice(i, width))
            sumAcc <- sumAcc + v
            sumSqAcc <- sumSqAcc + v * v
            minAcc <- Vector.Min(minAcc, v)
            maxAcc <- Vector.Max(maxAcc, v)
            i <- i + width
        let mutable sum = 0.0
        let mutable sumSq = 0.0
        let mutable minimum = Double.PositiveInfinity
        let mutable maximum = Double.NegativeInfinity
        let mutable lane = 0
        while lane < width do
            let sumValue = sumAcc[lane]
            let sumSqValue = sumSqAcc[lane]
            let minValue = minAcc[lane]
            let maxValue = maxAcc[lane]
            sum <- sum + sumValue
            sumSq <- sumSq + sumSqValue
            if minValue < minimum then minimum <- minValue
            if maxValue > maximum then maximum <- maxValue
            lane <- lane + 1
        while i < values.Length do
            let value = values[i]
            sum <- sum + value
            sumSq <- sumSq + value * value
            if value < minimum then minimum <- value
            if value > maximum then maximum <- value
            i <- i + 1
        let count = float values.Length
        let var = if values.Length > 1 then max 0.0 ((sumSq - sum * sum / count) / (count - 1.0)) else 0.0
        struct (sum, sqrt var, minimum, maximum)

let private sumScalarChunkUInt8 (chunk: StackCore.Chunk<uint8>) =
    sumScalarUInt8 (StackCore.Chunk.span<uint8> chunk)

let private sumScalarChunkUInt16 (chunk: StackCore.Chunk<uint16>) =
    sumScalarUInt16 (StackCore.Chunk.span<uint16> chunk)

let private sumScalarChunkFloat32 (chunk: StackCore.Chunk<float32>) =
    sumScalarFloat32 (StackCore.Chunk.span<float32> chunk)

let private sumScalarChunkFloat64 (chunk: StackCore.Chunk<float>) =
    sumScalarFloat64 (StackCore.Chunk.span<float> chunk)

let private sumVectorChunkUInt8 (chunk: StackCore.Chunk<uint8>) =
    sumVectorWidenedUInt8 (StackCore.Chunk.span<uint8> chunk)

let private sumVectorChunkUInt16 (chunk: StackCore.Chunk<uint16>) =
    sumVectorWidenedUInt16 (StackCore.Chunk.span<uint16> chunk)

let private momentsScalarChunkUInt8 (chunk: StackCore.Chunk<uint8>) =
    momentsScalarUInt8 (StackCore.Chunk.span<uint8> chunk)

let private momentsScalarChunkUInt16 (chunk: StackCore.Chunk<uint16>) =
    momentsScalarUInt16 (StackCore.Chunk.span<uint16> chunk)

let private momentsScalarChunkFloat32 (chunk: StackCore.Chunk<float32>) =
    momentsScalarFloat32 (StackCore.Chunk.span<float32> chunk)

let private momentsScalarChunkFloat64 (chunk: StackCore.Chunk<float>) =
    momentsScalarFloat64 (StackCore.Chunk.span<float> chunk)

let private momentsVectorChunkUInt8 (chunk: StackCore.Chunk<uint8>) =
    momentsVectorWidenedUInt8 (StackCore.Chunk.span<uint8> chunk)

let private momentsVectorChunkUInt16 (chunk: StackCore.Chunk<uint16>) =
    momentsVectorWidenedUInt16 (StackCore.Chunk.span<uint16> chunk)

let private sumVectorChunkFloat32 (chunk: StackCore.Chunk<float32>) =
    sumVectorFloat32 (StackCore.Chunk.span<float32> chunk)

let private sumVectorChunkFloat64 (chunk: StackCore.Chunk<float>) =
    sumVectorFloat64 (StackCore.Chunk.span<float> chunk)

let private sumVectorAccurateChunkFloat32 (chunk: StackCore.Chunk<float32>) =
    sumVectorAccurateFloat32 (StackCore.Chunk.span<float32> chunk)

let private momentsVectorChunkFloat32 (chunk: StackCore.Chunk<float32>) =
    momentsVectorFloat32 (StackCore.Chunk.span<float32> chunk)

let private momentsVectorChunkFloat64 (chunk: StackCore.Chunk<float>) =
    momentsVectorFloat64 (StackCore.Chunk.span<float> chunk)

let private momentsVectorAccurateChunkFloat32 (chunk: StackCore.Chunk<float32>) =
    momentsVectorAccurateFloat32 (StackCore.Chunk.span<float32> chunk)

let private runReductionHotLoop name iterations action checksum =
    printfn "variant=%s" name
    runTimedHotLoop iterations action checksum

let private runReductionHotLoopMeasured name bytesPerIteration iterations action checksum =
    printfn "variant=%s" name
    runTimedHotLoopMeasured bytesPerIteration iterations action checksum

let private runChunkSimdReductionsTyped<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    pixelName
    shape
    variant
    iterations
    fill
    sumScalarImpl
    momentsScalarImpl
    sumVectorImpl
    momentsVectorImpl
    sumVectorAccurateImpl
    momentsVectorAccurateImpl
    =
    let chunk = StackCore.Chunk.create<'T> (uint64 shape.Width, uint64 shape.Height, uint64 shape.Depth)
    try
        fill chunk
        let valueCount = int shape.Width * int shape.Height * int shape.Depth
        printfn "pixelType=%s shape=%ux%ux%u values=%d bytes=%d iterations=%d" pixelName shape.Width shape.Height shape.Depth valueCount chunk.ByteLength iterations
        let mutable statsResult = ChunkCore.ChunkFunctions.zeroStats
        let mutable sumResult = 0.0
        let mutable stdResult = 0.0
        let mutable minResult = 0.0
        let mutable maxResult = 0.0
        let measuredBytes = int64 chunk.ByteLength
        match variant with
        | "computeStats-current" ->
            runReductionHotLoopMeasured variant measuredBytes iterations
                (fun () -> statsResult <- ChunkCore.ChunkFunctions.computeStats chunk)
                (fun () -> int statsResult.NumPixels ^^^ int statsResult.Sum ^^^ int statsResult.Min ^^^ int statsResult.Max)
        | "sum-scalar" ->
            runReductionHotLoopMeasured variant measuredBytes iterations
                (fun () ->
                    sumResult <- sumScalarImpl chunk)
                (fun () -> int sumResult)
        | "moments-scalar" ->
            runReductionHotLoopMeasured variant measuredBytes iterations
                (fun () ->
                    let struct (sum, std, minimum, maximum) = momentsScalarImpl chunk
                    sumResult <- sum
                    stdResult <- std
                    minResult <- minimum
                    maxResult <- maximum)
                (fun () -> int sumResult ^^^ int stdResult ^^^ int minResult ^^^ int maxResult)
        | "sum-vector" ->
            match sumVectorImpl with
            | Some sumVector ->
                runReductionHotLoopMeasured variant measuredBytes iterations
                    (fun () -> sumResult <- sumVector chunk)
                    (fun () -> int sumResult)
            | None ->
                invalidArg "variant" $"Variant '{variant}' is not implemented for {pixelName} reductions."
        | "moments-vector" ->
            match momentsVectorImpl with
            | Some momentsVector ->
                runReductionHotLoopMeasured variant measuredBytes iterations
                    (fun () ->
                        let struct (sum, std, minimum, maximum) = momentsVector chunk
                        sumResult <- sum
                        stdResult <- std
                        minResult <- minimum
                        maxResult <- maximum)
                    (fun () -> int sumResult ^^^ int stdResult ^^^ int minResult ^^^ int maxResult)
            | None ->
                invalidArg "variant" $"Variant '{variant}' is not implemented for {pixelName} reductions."
        | "sum-vector-accurate" ->
            match sumVectorAccurateImpl with
            | Some sumVector ->
                runReductionHotLoopMeasured variant measuredBytes iterations
                    (fun () -> sumResult <- sumVector chunk)
                    (fun () -> int sumResult)
            | None ->
                invalidArg "variant" $"Variant '{variant}' is not implemented for {pixelName} reductions."
        | "moments-vector-accurate" ->
            match momentsVectorAccurateImpl with
            | Some momentsVector ->
                runReductionHotLoopMeasured variant measuredBytes iterations
                    (fun () ->
                        let struct (sum, std, minimum, maximum) = momentsVector chunk
                        sumResult <- sum
                        stdResult <- std
                        minResult <- minimum
                        maxResult <- maximum)
                    (fun () -> int sumResult ^^^ int stdResult ^^^ int minResult ^^^ int maxResult)
            | None ->
                invalidArg "variant" $"Variant '{variant}' is not implemented for {pixelName} reductions."
        | other ->
            invalidArg "variant" $"Unknown chunk SIMD reductions variant '{other}'."
    finally
        StackCore.Chunk.decRef chunk

let private runChunkSimdReductions opts =
    let pixelType = require "pixel-type" opts |> parseReductionPixelType
    let shape = require "shape" opts |> parseShape
    let variant = optional "variant" "computeStats-current" opts
    let iterations = optional "iterations" "10" opts |> int
    if iterations < 1 then invalidArg "iterations" "Expected at least one iteration."

    match pixelType with
    | ReductionUInt8 ->
        runChunkSimdReductionsTyped<uint8> (reductionPixelName pixelType) shape variant iterations fillReductionChunkUInt8 sumScalarChunkUInt8 momentsScalarChunkUInt8 (Some sumVectorChunkUInt8) (Some momentsVectorChunkUInt8) None None
    | ReductionUInt16 ->
        runChunkSimdReductionsTyped<uint16> (reductionPixelName pixelType) shape variant iterations fillReductionChunkUInt16 sumScalarChunkUInt16 momentsScalarChunkUInt16 (Some sumVectorChunkUInt16) (Some momentsVectorChunkUInt16) None None
    | ReductionFloat32 ->
        runChunkSimdReductionsTyped<float32> (reductionPixelName pixelType) shape variant iterations fillReductionChunkFloat32 sumScalarChunkFloat32 momentsScalarChunkFloat32 (Some sumVectorChunkFloat32) (Some momentsVectorChunkFloat32) (Some sumVectorAccurateChunkFloat32) (Some momentsVectorAccurateChunkFloat32)
    | ReductionFloat64 ->
        runChunkSimdReductionsTyped<float> (reductionPixelName pixelType) shape variant iterations fillReductionChunkFloat64 sumScalarChunkFloat64 momentsScalarChunkFloat64 (Some sumVectorChunkFloat64) (Some momentsVectorChunkFloat64) None None

let private fillCastUInt8Chunk (chunk: StackCore.Chunk<uint8>) =
    let values = StackCore.Chunk.span<uint8> chunk
    let mutable i = 0
    while i < values.Length do
        values[i] <- uint8 ((i * 37 + i / 17) &&& 0xFF)
        i <- i + 1

let private fillCastUInt16Chunk (chunk: StackCore.Chunk<uint16>) =
    let values = StackCore.Chunk.span<uint16> chunk
    let mutable i = 0
    while i < values.Length do
        values[i] <- uint16 ((i * 7919 + i / 11) &&& 0xFFFF)
        i <- i + 1

let private fillCastFloat32Chunk (chunk: StackCore.Chunk<float32>) =
    let values = StackCore.Chunk.span<float32> chunk
    let mutable i = 0
    while i < values.Length do
        values[i] <- float32 (((i * 37 + i / 13) &&& 0x1FFFF) - 32768) / 2.0f
        i <- i + 1

let private checksumFloat32Chunk (chunk: StackCore.Chunk<float32>) =
    let values = StackCore.Chunk.span<float32> chunk
    let stride = max 1 (values.Length / 4096)
    let mutable checksum = 2166136261u
    let mutable i = 0
    while i < values.Length do
        checksum <- (checksum * 16777619u) ^^^ uint32 (int values[i])
        i <- i + stride
    int checksum

let private checksumUInt8Chunk (chunk: StackCore.Chunk<uint8>) =
    let values = StackCore.Chunk.span<uint8> chunk
    let stride = max 1 (values.Length / 4096)
    let mutable checksum = 2166136261u
    let mutable i = 0
    while i < values.Length do
        checksum <- (checksum * 16777619u) ^^^ uint32 values[i]
        i <- i + stride
    int checksum

let private checksumUInt16Chunk (chunk: StackCore.Chunk<uint16>) =
    let values = StackCore.Chunk.span<uint16> chunk
    let stride = max 1 (values.Length / 4096)
    let mutable checksum = 2166136261u
    let mutable i = 0
    while i < values.Length do
        checksum <- (checksum * 16777619u) ^^^ uint32 values[i]
        i <- i + stride
    int checksum

let private runChunkCast opts =
    let shape = require "shape" opts |> parseShape
    let variant = optional "variant" "uint8-to-float32" opts
    let iterations = optional "iterations" "10" opts |> int
    if iterations < 1 then invalidArg "iterations" "Expected at least one iteration."

    let size = (uint64 shape.Width, uint64 shape.Height, uint64 shape.Depth)
    match variant with
    | "uint8-to-float32" ->
        let input = StackCore.Chunk.create<uint8> size
        let mutable lastOutput : StackCore.Chunk<float32> option = None
        let replaceOutput output =
            lastOutput |> Option.iter StackCore.Chunk.decRef
            lastOutput <- Some output
        try
            fillCastUInt8Chunk input
            printfn "shape=%ux%ux%u values=%d inputBytes=%d iterations=%d" shape.Width shape.Height shape.Depth (StackCore.Chunk.span<uint8> input).Length input.ByteLength iterations
            runTimedHotLoopMeasured
                (int64 input.ByteLength + int64 input.ByteLength * int64 sizeof<float32>)
                iterations
                (fun () -> ChunkCore.ChunkFunctions.castChunkToFloat32 input |> replaceOutput)
                (fun () -> lastOutput |> Option.map checksumFloat32Chunk |> Option.defaultValue 0)
        finally
            lastOutput |> Option.iter StackCore.Chunk.decRef
            StackCore.Chunk.decRef input
    | "uint16-to-float32" ->
        let input = StackCore.Chunk.create<uint16> size
        let mutable lastOutput : StackCore.Chunk<float32> option = None
        let replaceOutput output =
            lastOutput |> Option.iter StackCore.Chunk.decRef
            lastOutput <- Some output
        try
            fillCastUInt16Chunk input
            printfn "shape=%ux%ux%u values=%d inputBytes=%d iterations=%d" shape.Width shape.Height shape.Depth (StackCore.Chunk.span<uint16> input).Length input.ByteLength iterations
            runTimedHotLoopMeasured
                (int64 input.ByteLength + (int64 input.ByteLength / int64 sizeof<uint16>) * int64 sizeof<float32>)
                iterations
                (fun () -> ChunkCore.ChunkFunctions.castChunkToFloat32 input |> replaceOutput)
                (fun () -> lastOutput |> Option.map checksumFloat32Chunk |> Option.defaultValue 0)
        finally
            lastOutput |> Option.iter StackCore.Chunk.decRef
            StackCore.Chunk.decRef input
    | "float32-to-uint8" ->
        let input = StackCore.Chunk.create<float32> size
        let mutable lastOutput : StackCore.Chunk<uint8> option = None
        let replaceOutput output =
            lastOutput |> Option.iter StackCore.Chunk.decRef
            lastOutput <- Some output
        try
            fillCastFloat32Chunk input
            printfn "shape=%ux%ux%u values=%d inputBytes=%d iterations=%d" shape.Width shape.Height shape.Depth (StackCore.Chunk.span<float32> input).Length input.ByteLength iterations
            runTimedHotLoopMeasured
                (int64 input.ByteLength + int64 input.ByteLength / int64 sizeof<float32>)
                iterations
                (fun () -> ChunkCore.ChunkFunctions.castFloat32ToChunk<uint8> input |> replaceOutput)
                (fun () -> lastOutput |> Option.map checksumUInt8Chunk |> Option.defaultValue 0)
        finally
            lastOutput |> Option.iter StackCore.Chunk.decRef
            StackCore.Chunk.decRef input
    | "float32-to-uint16" ->
        let input = StackCore.Chunk.create<float32> size
        let mutable lastOutput : StackCore.Chunk<uint16> option = None
        let replaceOutput output =
            lastOutput |> Option.iter StackCore.Chunk.decRef
            lastOutput <- Some output
        try
            fillCastFloat32Chunk input
            printfn "shape=%ux%ux%u values=%d inputBytes=%d iterations=%d" shape.Width shape.Height shape.Depth (StackCore.Chunk.span<float32> input).Length input.ByteLength iterations
            runTimedHotLoopMeasured
                (int64 input.ByteLength + (int64 input.ByteLength / int64 sizeof<float32>) * int64 sizeof<uint16>)
                iterations
                (fun () -> ChunkCore.ChunkFunctions.castFloat32ToChunk<uint16> input |> replaceOutput)
                (fun () -> lastOutput |> Option.map checksumUInt16Chunk |> Option.defaultValue 0)
        finally
            lastOutput |> Option.iter StackCore.Chunk.decRef
            StackCore.Chunk.decRef input
    | other ->
        invalidArg "variant" $"Unsupported chunk cast variant '{other}'."

let private fillPixelwiseFloat32Chunk seed (chunk: StackCore.Chunk<float32>) =
    let values = StackCore.Chunk.span<float32> chunk
    let mutable i = 0
    while i < values.Length do
        values[i] <- float32 (((i + seed) * 37 + i / 11) &&& 0xFFFF) / 257.0f + 1.0f
        i <- i + 1

let private scalarMapFloat32 name scalarOp (chunk: StackCore.Chunk<float32>) =
    let output = StackCore.Chunk.create<float32> chunk.Size
    try
        let input = StackCore.Chunk.span<float32> chunk
        let outputSpan = StackCore.Chunk.span<float32> output
        let mutable i = 0
        while i < input.Length do
            outputSpan[i] <- scalarOp input[i]
            i <- i + 1
        output
    with
    | _ ->
        StackCore.Chunk.decRef output
        reraise()

let private scalarMap2Float32 name scalarOp (a: StackCore.Chunk<float32>) (b: StackCore.Chunk<float32>) =
    let output = StackCore.Chunk.create<float32> a.Size
    try
        let aSpan = StackCore.Chunk.span<float32> a
        let bSpan = StackCore.Chunk.span<float32> b
        let outputSpan = StackCore.Chunk.span<float32> output
        let mutable i = 0
        while i < aSpan.Length do
            outputSpan[i] <- scalarOp aSpan[i] bSpan[i]
            i <- i + 1
        output
    with
    | _ ->
        StackCore.Chunk.decRef output
        reraise()

let private runChunkPixelwiseFloat32 opts =
    let shape = require "shape" opts |> parseShape
    let variant = optional "variant" "vector-add" opts
    let iterations = optional "iterations" "10" opts |> int
    if iterations < 1 then invalidArg "iterations" "Expected at least one iteration."

    let a = StackCore.Chunk.create<float32> (uint64 shape.Width, uint64 shape.Height, uint64 shape.Depth)
    let b = StackCore.Chunk.create<float32> (uint64 shape.Width, uint64 shape.Height, uint64 shape.Depth)
    let mutable lastOutput : StackCore.Chunk<float32> option = None
    let replaceOutput output =
        match lastOutput with
        | Some previous -> StackCore.Chunk.decRef previous
        | None -> ()
        lastOutput <- Some output

    try
        fillPixelwiseFloat32Chunk 0 a
        fillPixelwiseFloat32Chunk 17 b
        printfn "shape=%ux%ux%u values=%d inputBytes=%d iterations=%d" shape.Width shape.Height shape.Depth (StackCore.Chunk.span<float32> a).Length a.ByteLength iterations
        let bytesPerIteration =
            match variant with
            | "scalar-pair-add" | "vector-pair-add"
            | "scalar-pair-mul" | "vector-pair-mul"
            | "scalar-absdiff" | "vector-absdiff"
            | "scalar-blend" | "vector-blend" -> int64 a.ByteLength * 3L
            | _ -> int64 a.ByteLength * 2L
        let action =
            match variant with
            | "scalar-add" -> fun () -> scalarMapFloat32 "scalarAddFloat32" (fun x -> x + 3.25f) a |> replaceOutput
            | "vector-add" -> fun () -> ChunkCore.ChunkFunctions.mapFloat32Vector "vectorAddFloat32" (fun x -> x + 3.25f) (fun v -> v + Vector<float32>(3.25f)) a |> replaceOutput
            | "scalar-mul" -> fun () -> scalarMapFloat32 "scalarMulFloat32" (fun x -> x * 1.25f) a |> replaceOutput
            | "vector-mul" -> fun () -> ChunkCore.ChunkFunctions.mapFloat32Vector "vectorMulFloat32" (fun x -> x * 1.25f) (fun v -> v * Vector<float32>(1.25f)) a |> replaceOutput
            | "scalar-threshold" -> fun () -> scalarMapFloat32 "scalarThresholdFloat32" (fun x -> if x >= 128.0f then 1.0f else 0.0f) a |> replaceOutput
            | "vector-threshold" ->
                let threshold = Vector<float32>(128.0f)
                let one = Vector<float32>(1.0f)
                let zero = Vector<float32>.Zero
                fun () ->
                    ChunkCore.ChunkFunctions.mapFloat32Vector
                        "vectorThresholdFloat32"
                        (fun x -> if x >= 128.0f then 1.0f else 0.0f)
                        (fun v -> Vector.ConditionalSelect(Vector.GreaterThanOrEqual(v, threshold), one, zero))
                        a
                    |> replaceOutput
            | "scalar-pair-add" -> fun () -> scalarMap2Float32 "scalarPairAddFloat32" (fun x y -> x + y) a b |> replaceOutput
            | "vector-pair-add" -> fun () -> ChunkCore.ChunkFunctions.map2Float32Vector "vectorPairAddFloat32" (fun x y -> x + y) (fun x y -> x + y) a b |> replaceOutput
            | "scalar-pair-mul" -> fun () -> scalarMap2Float32 "scalarPairMulFloat32" (fun x y -> x * y) a b |> replaceOutput
            | "vector-pair-mul" -> fun () -> ChunkCore.ChunkFunctions.map2Float32Vector "vectorPairMulFloat32" (fun x y -> x * y) (fun x y -> x * y) a b |> replaceOutput
            | "scalar-absdiff" -> fun () -> scalarMap2Float32 "scalarAbsDiffFloat32" (fun x y -> abs (x - y)) a b |> replaceOutput
            | "vector-absdiff" -> fun () -> ChunkCore.ChunkFunctions.map2Float32Vector "vectorAbsDiffFloat32" (fun x y -> abs (x - y)) (fun x y -> Vector.Abs(x - y)) a b |> replaceOutput
            | "scalar-blend" -> fun () -> scalarMap2Float32 "scalarBlendFloat32" (fun x y -> if x >= 128.0f then x else y) a b |> replaceOutput
            | "vector-blend" ->
                let threshold = Vector<float32>(128.0f)
                fun () ->
                    ChunkCore.ChunkFunctions.map2Float32Vector
                        "vectorBlendFloat32"
                        (fun x y -> if x >= 128.0f then x else y)
                        (fun x y -> Vector.ConditionalSelect(Vector.GreaterThanOrEqual(x, threshold), x, y))
                        a
                        b
                    |> replaceOutput
            | other -> invalidArg "variant" $"Unsupported pixelwise float32 variant '{other}'."
        runTimedHotLoopMeasured bytesPerIteration iterations action (fun () -> lastOutput |> Option.map checksumFloat32Chunk |> Option.defaultValue 0)
    finally
        match lastOutput with
        | Some output -> StackCore.Chunk.decRef output
        | None -> ()
        StackCore.Chunk.decRef b
        StackCore.Chunk.decRef a

let private fillAxisConvolveChunk z width height =
    let chunk = StackCore.Chunk.create<uint8> (uint64 width, uint64 height, 1UL)
    let values = StackCore.Chunk.span<uint8> chunk
    let mutable i = 0
    for y in 0 .. height - 1 do
        for x in 0 .. width - 1 do
            values[i] <- uint8 ((z * 43 + y * 17 + x * 9 + x * y) &&& 0xFF)
            i <- i + 1
    chunk

let private makeBoxKernel kernelSize =
    if kernelSize < 1 || kernelSize % 2 = 0 then
        failwith $"kernel-size must be a positive odd integer, got {kernelSize}"
    Array.create kernelSize (1.0f / float32 kernelSize)

let private updateUInt8Checksum (checksum: byref<int>) (chunk: StackCore.Chunk<uint8>) =
    let values = StackCore.Chunk.span<uint8> chunk
    let stride = max 1 (values.Length / 4096)
    let mutable i = 0
    while i < values.Length do
        checksum <- (checksum * 16777619) ^^^ int values[i]
        i <- i + stride
    checksum <- (checksum * 16777619) ^^^ values.Length

let private chunkAxisName axis =
    match axis with
    | AxisX -> "x"
    | AxisY -> "y"
    | AxisZ -> "z"

let private chunkAxisVariantName variant =
    match variant with
    | AxisNativeGeneric -> "native-generic"
    | AxisNative1D -> "native-1d"

let private runChunkAxisConvolveKernel opts =
    let shape = require "shape" opts |> parseShape
    let kernelSize = optional "kernel-size" "9" opts |> int
    let iterations = optional "iterations" "100" opts |> int
    if iterations < 1 then
        failwith $"iterations must be positive, got {iterations}"

    let axes = optional "axis" "all" opts |> parseChunkAxisConvolveAxis
    let variants = optional "variant" "all" opts |> parseChunkAxisConvolveVariant
    let kernel = makeBoxKernel kernelSize
    let width = int shape.Width
    let height = int shape.Height

    if width < 1 || height < 1 then
        failwith $"shape must have positive width and height, got {shape.Width}x{shape.Height}x{shape.Depth}"
    let depth = int shape.Depth
    if depth < 1 then
        failwith $"shape must have positive depth, got {shape.Width}x{shape.Height}x{shape.Depth}"

    let chunks = Array.init depth (fun z -> fillAxisConvolveChunk z width height)
    let zeroChunk = StackCore.Chunk.create<uint8> (uint64 width, uint64 height, 1UL)

    try
        for axis in axes do
            for variant in variants do
                let mutable checksum = 2166136261u |> int
                let action () =
                    match axis with
                    | AxisX ->
                        for z in 0 .. depth - 1 do
                            let output =
                                match variant with
                                | AxisNativeGeneric -> ChunkCore.ChunkFunctions.convolveNativeXUInt8 kernel chunks[z]
                                | AxisNative1D -> ChunkCore.ChunkFunctions.convolveNativeXUInt8Specialized kernel chunks[z]
                            try
                                updateUInt8Checksum &checksum output
                            finally
                                StackCore.Chunk.decRef output
                    | AxisY ->
                        for z in 0 .. depth - 1 do
                            let output =
                                match variant with
                                | AxisNativeGeneric -> ChunkCore.ChunkFunctions.convolveNativeYUInt8 kernel chunks[z]
                                | AxisNative1D -> ChunkCore.ChunkFunctions.convolveNativeYUInt8Specialized kernel chunks[z]
                            try
                                updateUInt8Checksum &checksum output
                            finally
                                StackCore.Chunk.decRef output
                    | AxisZ ->
                        let radius = kernel.Length / 2
                        let window = Array.zeroCreate<StackCore.Chunk<uint8>> kernel.Length
                        for z in 0 .. depth - 1 do
                            for k in 0 .. kernel.Length - 1 do
                                let sourceZ = z + k - radius
                                window[k] <- if sourceZ >= 0 && sourceZ < depth then chunks[sourceZ] else zeroChunk
                            let output =
                                match variant with
                                | AxisNativeGeneric -> ChunkCore.ChunkFunctions.convolveNativeZUInt8 kernel window
                                | AxisNative1D -> ChunkCore.ChunkFunctions.convolveNativeZUInt8Specialized kernel window
                            try
                                updateUInt8Checksum &checksum output
                            finally
                                StackCore.Chunk.decRef output

                action()
                GC.Collect()
                GC.WaitForPendingFinalizers()
                GC.Collect()

                let stopwatch = Stopwatch.StartNew()
                for _ in 1 .. iterations do
                    action()
                stopwatch.Stop()

                writeInternalSeconds stopwatch.Elapsed
                printfn
                    "variant=%s axis=%s shape=%ux%ux%u kernelSize=%d iterations=%d totalSeconds=%s perIterationSeconds=%s checksum=%d"
                    (chunkAxisVariantName variant)
                    (chunkAxisName axis)
                    shape.Width
                    shape.Height
                    shape.Depth
                    kernelSize
                    iterations
                    (stopwatch.Elapsed.TotalSeconds.ToString("F9", invariant))
                    ((stopwatch.Elapsed.TotalSeconds / float iterations).ToString("F9", invariant))
                    checksum
        0
    finally
        StackCore.Chunk.decRef zeroChunk
        chunks |> Array.iter StackCore.Chunk.decRef

let private runLibTiffDirectThresholdHotLoop opts =
    let pixelType = require "pixel-type" opts |> parsePixelType
    let input = require "input" opts
    let variant = optional "variant" "byte-intype-max" opts
    let thresholdValue = optional "threshold" "128" opts |> fun s -> Double.Parse(s, invariant)
    let iterations = optional "iterations" "1000" opts |> int
    if iterations < 1 then
        invalidArg "iterations" "Expected at least one iteration."

    let files = stackTiffFiles input
    if files.Length = 0 then
        invalidOp $"No TIFF files found in input stack directory: {input}"

    let width, height, rowBytes, scanlineSize = inspectDirectTiffSlice pixelType files[0]
    let pageBytes = rowBytes * int height
    let pixels = int width * int height
    let page = ArrayPool<byte>.Shared.Rent(pageBytes)
    let scratch = ArrayPool<byte>.Shared.Rent(scanlineSize)

    try
        readDirectByteTiffSliceInto pixelType files[0] width height rowBytes page scratch
        match pixelType, variant with
        | UInt8, "byte-mask-max" ->
            let output = ArrayPool<byte>.Shared.Rent(pixels)
            try
                runTimedHotLoop iterations
                    (fun () -> thresholdUInt8PageMaxSimdInto thresholdValue pixels page output)
                    (fun () -> int output[0] + int output[pixels / 2] + int output[pixels - 1])
            finally
                ArrayPool<byte>.Shared.Return(output)
        | UInt8, "byte-mask-one"
        | UInt8, "byte-intype-one"
        | UInt8, "typed-intype-one" ->
            let output = ArrayPool<byte>.Shared.Rent(pixels)
            try
                runTimedHotLoop iterations
                    (fun () -> thresholdUInt8PageSimdInto thresholdValue pixels page output)
                    (fun () -> int output[0] + int output[pixels / 2] + int output[pixels - 1])
            finally
                ArrayPool<byte>.Shared.Return(output)
        | UInt8, "byte-intype-max"
        | UInt8, "typed-intype-max" ->
            let output = ArrayPool<byte>.Shared.Rent(pixels)
            try
                runTimedHotLoop iterations
                    (fun () -> thresholdUInt8PageMaxSimdInto thresholdValue pixels page output)
                    (fun () -> int output[0] + int output[pixels / 2] + int output[pixels - 1])
            finally
                ArrayPool<byte>.Shared.Return(output)
        | UInt8, "typed-copy-intype-max" ->
            let typedInput = ArrayPool<byte>.Shared.Rent(pixels)
            let output = ArrayPool<byte>.Shared.Rent(pixels)
            try
                runTimedHotLoop iterations
                    (fun () ->
                        Buffer.BlockCopy(page, 0, typedInput, 0, pixels)
                        thresholdTypedUInt8MaxVector thresholdValue pixels typedInput output)
                    (fun () -> int output[0] + int output[pixels / 2] + int output[pixels - 1])
            finally
                ArrayPool<byte>.Shared.Return(output)
                ArrayPool<byte>.Shared.Return(typedInput)
        | UInt8, "typed-copy-intype-one" ->
            let typedInput = ArrayPool<byte>.Shared.Rent(pixels)
            let output = ArrayPool<byte>.Shared.Rent(pixels)
            try
                runTimedHotLoop iterations
                    (fun () ->
                        Buffer.BlockCopy(page, 0, typedInput, 0, pixels)
                        thresholdTypedUInt8OneVector thresholdValue pixels typedInput output)
                    (fun () -> int output[0] + int output[pixels / 2] + int output[pixels - 1])
            finally
                ArrayPool<byte>.Shared.Return(output)
                ArrayPool<byte>.Shared.Return(typedInput)
        | UInt16, "byte-mask-one" ->
            let output = ArrayPool<byte>.Shared.Rent(pixels)
            try
                runTimedHotLoop iterations
                    (fun () -> thresholdUInt16PageSimdInto thresholdValue pageBytes page output)
                    (fun () -> int output[0] + int output[pixels / 2] + int output[pixels - 1])
            finally
                ArrayPool<byte>.Shared.Return(output)
        | UInt16, "byte-intype-max" ->
            let output = ArrayPool<byte>.Shared.Rent(pageBytes)
            try
                runTimedHotLoop iterations
                    (fun () -> thresholdUInt16PageInTypeMaxSimdInto thresholdValue pageBytes page output)
                    (fun () ->
                        let values = MemoryMarshal.Cast<byte, uint16>(output.AsSpan(0, pageBytes))
                        int values[0] + int values[pixels / 2] + int values[pixels - 1])
            finally
                ArrayPool<byte>.Shared.Return(output)
        | UInt16, "byte-intype-one" ->
            let output = ArrayPool<byte>.Shared.Rent(pageBytes)
            try
                runTimedHotLoop iterations
                    (fun () -> thresholdUInt16PageInTypeSimdInto thresholdValue pageBytes page output)
                    (fun () ->
                        let values = MemoryMarshal.Cast<byte, uint16>(output.AsSpan(0, pageBytes))
                        int values[0] + int values[pixels / 2] + int values[pixels - 1])
            finally
                ArrayPool<byte>.Shared.Return(output)
        | UInt16, "typed-intype-max" ->
            let typedInput = ArrayPool<uint16>.Shared.Rent(pixels)
            let output = ArrayPool<uint16>.Shared.Rent(pixels)
            try
                Buffer.BlockCopy(page, 0, typedInput, 0, pageBytes)
                runTimedHotLoop iterations
                    (fun () -> thresholdTypedUInt16MaxVector thresholdValue pixels typedInput output)
                    (fun () -> int output[0] + int output[pixels / 2] + int output[pixels - 1])
            finally
                ArrayPool<uint16>.Shared.Return(output)
                ArrayPool<uint16>.Shared.Return(typedInput)
        | UInt16, "typed-intype-one" ->
            let typedInput = ArrayPool<uint16>.Shared.Rent(pixels)
            let output = ArrayPool<uint16>.Shared.Rent(pixels)
            try
                Buffer.BlockCopy(page, 0, typedInput, 0, pageBytes)
                runTimedHotLoop iterations
                    (fun () -> thresholdTypedUInt16OneVector thresholdValue pixels typedInput output)
                    (fun () -> int output[0] + int output[pixels / 2] + int output[pixels - 1])
            finally
                ArrayPool<uint16>.Shared.Return(output)
                ArrayPool<uint16>.Shared.Return(typedInput)
        | UInt16, "typed-copy-intype-max" ->
            let typedInput = ArrayPool<uint16>.Shared.Rent(pixels)
            let output = ArrayPool<uint16>.Shared.Rent(pixels)
            try
                runTimedHotLoop iterations
                    (fun () ->
                        Buffer.BlockCopy(page, 0, typedInput, 0, pageBytes)
                        thresholdTypedUInt16MaxVector thresholdValue pixels typedInput output)
                    (fun () -> int output[0] + int output[pixels / 2] + int output[pixels - 1])
            finally
                ArrayPool<uint16>.Shared.Return(output)
                ArrayPool<uint16>.Shared.Return(typedInput)
        | UInt16, "typed-copy-intype-one" ->
            let typedInput = ArrayPool<uint16>.Shared.Rent(pixels)
            let output = ArrayPool<uint16>.Shared.Rent(pixels)
            try
                runTimedHotLoop iterations
                    (fun () ->
                        Buffer.BlockCopy(page, 0, typedInput, 0, pageBytes)
                        thresholdTypedUInt16OneVector thresholdValue pixels typedInput output)
                    (fun () -> int output[0] + int output[pixels / 2] + int output[pixels - 1])
            finally
                ArrayPool<uint16>.Shared.Return(output)
                ArrayPool<uint16>.Shared.Return(typedInput)
        | Float32, "byte-mask-one" ->
            let output = ArrayPool<byte>.Shared.Rent(pixels)
            try
                runTimedHotLoop iterations
                    (fun () -> thresholdFloat32PageSimdInto thresholdValue pageBytes page output)
                    (fun () -> int output[0] + int output[pixels / 2] + int output[pixels - 1])
            finally
                ArrayPool<byte>.Shared.Return(output)
        | Float32, "byte-intype-max" ->
            let output = ArrayPool<byte>.Shared.Rent(pageBytes)
            try
                runTimedHotLoop iterations
                    (fun () -> thresholdFloat32PageInTypeMaxVectorInto thresholdValue pageBytes page output)
                    (fun () ->
                        let values = MemoryMarshal.Cast<byte, float32>(output.AsSpan(0, pageBytes))
                        int values[0] + int values[pixels / 2] + int values[pixels - 1])
            finally
                ArrayPool<byte>.Shared.Return(output)
        | Float32, "byte-intype-one" ->
            let output = ArrayPool<byte>.Shared.Rent(pageBytes)
            try
                runTimedHotLoop iterations
                    (fun () -> thresholdFloat32PageInTypeOneVectorInto thresholdValue pageBytes page output)
                    (fun () ->
                        let values = MemoryMarshal.Cast<byte, float32>(output.AsSpan(0, pageBytes))
                        int values[0] + int values[pixels / 2] + int values[pixels - 1])
            finally
                ArrayPool<byte>.Shared.Return(output)
        | Float32, "typed-intype-max" ->
            let typedInput = ArrayPool<float32>.Shared.Rent(pixels)
            let output = ArrayPool<float32>.Shared.Rent(pixels)
            try
                Buffer.BlockCopy(page, 0, typedInput, 0, pageBytes)
                runTimedHotLoop iterations
                    (fun () -> thresholdTypedFloat32MaxVector thresholdValue pixels typedInput output)
                    (fun () -> int output[0] + int output[pixels / 2] + int output[pixels - 1])
            finally
                ArrayPool<float32>.Shared.Return(output)
                ArrayPool<float32>.Shared.Return(typedInput)
        | Float32, "typed-intype-one" ->
            let typedInput = ArrayPool<float32>.Shared.Rent(pixels)
            let output = ArrayPool<float32>.Shared.Rent(pixels)
            try
                Buffer.BlockCopy(page, 0, typedInput, 0, pageBytes)
                runTimedHotLoop iterations
                    (fun () -> thresholdTypedFloat32OneVector thresholdValue pixels typedInput output)
                    (fun () -> int output[0] + int output[pixels / 2] + int output[pixels - 1])
            finally
                ArrayPool<float32>.Shared.Return(output)
                ArrayPool<float32>.Shared.Return(typedInput)
        | Float32, "typed-copy-intype-max" ->
            let typedInput = ArrayPool<float32>.Shared.Rent(pixels)
            let output = ArrayPool<float32>.Shared.Rent(pixels)
            try
                runTimedHotLoop iterations
                    (fun () ->
                        Buffer.BlockCopy(page, 0, typedInput, 0, pageBytes)
                        thresholdTypedFloat32MaxVector thresholdValue pixels typedInput output)
                    (fun () -> int output[0] + int output[pixels / 2] + int output[pixels - 1])
            finally
                ArrayPool<float32>.Shared.Return(output)
                ArrayPool<float32>.Shared.Return(typedInput)
        | Float32, "typed-copy-intype-one" ->
            let typedInput = ArrayPool<float32>.Shared.Rent(pixels)
            let output = ArrayPool<float32>.Shared.Rent(pixels)
            try
                runTimedHotLoop iterations
                    (fun () ->
                        Buffer.BlockCopy(page, 0, typedInput, 0, pageBytes)
                        thresholdTypedFloat32OneVector thresholdValue pixels typedInput output)
                    (fun () -> int output[0] + int output[pixels / 2] + int output[pixels - 1])
            finally
                ArrayPool<float32>.Shared.Return(output)
                ArrayPool<float32>.Shared.Return(typedInput)
        | _, unsupported ->
            invalidArg "variant" $"Unsupported hotloop variant '{unsupported}' for {pixelType}."
    finally
        ArrayPool<byte>.Shared.Return(scratch)
        ArrayPool<byte>.Shared.Return(page)

let private runZarrDirectThresholdHotLoop opts =
    let pixelType = require "pixel-type" opts |> parsePixelType
    let input = require "input" opts
    let variant = optional "variant" "byte-intype-max" opts
    let thresholdValue = optional "threshold" "128" opts |> fun s -> Double.Parse(s, invariant)
    let iterations = optional "iterations" "1000" opts |> int
    if iterations < 1 then
        invalidArg "iterations" "Expected at least one iteration."

    let inputArray = openZarrArray input
    let dataType = zarrDataType pixelType
    validateZarrArrayType inputArray dataType

    let chunks = collectZarrChunks inputArray
    if chunks.Length = 0 then
        invalidOp $"No Zarr chunks found in input array: {input}"

    let decoded =
        inputArray.ReadChunkDecodedAsync(chunks[0], Threading.CancellationToken.None)
        |> runTask

    let bytesPerSample = bytesPerPixelType pixelType
    let pageBytes = decoded.Length
    let pixels = pageBytes / bytesPerSample
    if pixels < 1 || pixels * bytesPerSample <> pageBytes then
        invalidOp $"Decoded Zarr chunk byte length {pageBytes} is not valid for {pixelType}."

    printfn "chunkCoord=%s chunkBytes=%d pixels=%d" (zarrChunkCoordKey chunks[0]) pageBytes pixels

    match pixelType, variant with
    | UInt8, "byte-mask-max"
    | UInt8, "byte-intype-max"
    | UInt8, "typed-intype-max" ->
        let output = ArrayPool<byte>.Shared.Rent(pixels)
        try
            runTimedHotLoop iterations
                (fun () -> thresholdUInt8PageMaxSimdInto thresholdValue pixels decoded output)
                (fun () -> int output[0] + int output[pixels / 2] + int output[pixels - 1])
        finally
            ArrayPool<byte>.Shared.Return(output)
    | UInt8, "byte-mask-one"
    | UInt8, "byte-intype-one"
    | UInt8, "typed-intype-one" ->
        let output = ArrayPool<byte>.Shared.Rent(pixels)
        try
            runTimedHotLoop iterations
                (fun () -> thresholdUInt8PageSimdInto thresholdValue pixels decoded output)
                (fun () -> int output[0] + int output[pixels / 2] + int output[pixels - 1])
        finally
            ArrayPool<byte>.Shared.Return(output)
    | UInt8, "typed-copy-intype-max" ->
        let typedInput = ArrayPool<byte>.Shared.Rent(pixels)
        let output = ArrayPool<byte>.Shared.Rent(pixels)
        try
            runTimedHotLoop iterations
                (fun () ->
                    Buffer.BlockCopy(decoded, 0, typedInput, 0, pixels)
                    thresholdTypedUInt8MaxVector thresholdValue pixels typedInput output)
                (fun () -> int output[0] + int output[pixels / 2] + int output[pixels - 1])
        finally
            ArrayPool<byte>.Shared.Return(output)
            ArrayPool<byte>.Shared.Return(typedInput)
    | UInt8, "typed-copy-intype-one" ->
        let typedInput = ArrayPool<byte>.Shared.Rent(pixels)
        let output = ArrayPool<byte>.Shared.Rent(pixels)
        try
            runTimedHotLoop iterations
                (fun () ->
                    Buffer.BlockCopy(decoded, 0, typedInput, 0, pixels)
                    thresholdTypedUInt8OneVector thresholdValue pixels typedInput output)
                (fun () -> int output[0] + int output[pixels / 2] + int output[pixels - 1])
        finally
            ArrayPool<byte>.Shared.Return(output)
            ArrayPool<byte>.Shared.Return(typedInput)
    | UInt16, "byte-mask-one" ->
        let output = ArrayPool<byte>.Shared.Rent(pixels)
        try
            runTimedHotLoop iterations
                (fun () -> thresholdUInt16PageSimdInto thresholdValue pageBytes decoded output)
                (fun () -> int output[0] + int output[pixels / 2] + int output[pixels - 1])
        finally
            ArrayPool<byte>.Shared.Return(output)
    | UInt16, "byte-intype-max" ->
        let output = ArrayPool<byte>.Shared.Rent(pageBytes)
        try
            runTimedHotLoop iterations
                (fun () -> thresholdUInt16PageInTypeMaxSimdInto thresholdValue pageBytes decoded output)
                (fun () ->
                    let values = MemoryMarshal.Cast<byte, uint16>(output.AsSpan(0, pageBytes))
                    int values[0] + int values[pixels / 2] + int values[pixels - 1])
        finally
            ArrayPool<byte>.Shared.Return(output)
    | UInt16, "byte-intype-one" ->
        let output = ArrayPool<byte>.Shared.Rent(pageBytes)
        try
            runTimedHotLoop iterations
                (fun () -> thresholdUInt16PageInTypeSimdInto thresholdValue pageBytes decoded output)
                (fun () ->
                    let values = MemoryMarshal.Cast<byte, uint16>(output.AsSpan(0, pageBytes))
                    int values[0] + int values[pixels / 2] + int values[pixels - 1])
        finally
            ArrayPool<byte>.Shared.Return(output)
    | UInt16, "typed-intype-max" ->
        let typedInput = ArrayPool<uint16>.Shared.Rent(pixels)
        let output = ArrayPool<uint16>.Shared.Rent(pixels)
        try
            Buffer.BlockCopy(decoded, 0, typedInput, 0, pageBytes)
            runTimedHotLoop iterations
                (fun () -> thresholdTypedUInt16MaxVector thresholdValue pixels typedInput output)
                (fun () -> int output[0] + int output[pixels / 2] + int output[pixels - 1])
        finally
            ArrayPool<uint16>.Shared.Return(output)
            ArrayPool<uint16>.Shared.Return(typedInput)
    | UInt16, "typed-intype-one" ->
        let typedInput = ArrayPool<uint16>.Shared.Rent(pixels)
        let output = ArrayPool<uint16>.Shared.Rent(pixels)
        try
            Buffer.BlockCopy(decoded, 0, typedInput, 0, pageBytes)
            runTimedHotLoop iterations
                (fun () -> thresholdTypedUInt16OneVector thresholdValue pixels typedInput output)
                (fun () -> int output[0] + int output[pixels / 2] + int output[pixels - 1])
        finally
            ArrayPool<uint16>.Shared.Return(output)
            ArrayPool<uint16>.Shared.Return(typedInput)
    | UInt16, "typed-copy-intype-max" ->
        let typedInput = ArrayPool<uint16>.Shared.Rent(pixels)
        let output = ArrayPool<uint16>.Shared.Rent(pixels)
        try
            runTimedHotLoop iterations
                (fun () ->
                    Buffer.BlockCopy(decoded, 0, typedInput, 0, pageBytes)
                    thresholdTypedUInt16MaxVector thresholdValue pixels typedInput output)
                (fun () -> int output[0] + int output[pixels / 2] + int output[pixels - 1])
        finally
            ArrayPool<uint16>.Shared.Return(output)
            ArrayPool<uint16>.Shared.Return(typedInput)
    | UInt16, "typed-copy-intype-one" ->
        let typedInput = ArrayPool<uint16>.Shared.Rent(pixels)
        let output = ArrayPool<uint16>.Shared.Rent(pixels)
        try
            runTimedHotLoop iterations
                (fun () ->
                    Buffer.BlockCopy(decoded, 0, typedInput, 0, pageBytes)
                    thresholdTypedUInt16OneVector thresholdValue pixels typedInput output)
                (fun () -> int output[0] + int output[pixels / 2] + int output[pixels - 1])
        finally
            ArrayPool<uint16>.Shared.Return(output)
            ArrayPool<uint16>.Shared.Return(typedInput)
    | Float32, "byte-mask-one" ->
        let output = ArrayPool<byte>.Shared.Rent(pixels)
        try
            runTimedHotLoop iterations
                (fun () -> thresholdFloat32PageSimdInto thresholdValue pageBytes decoded output)
                (fun () -> int output[0] + int output[pixels / 2] + int output[pixels - 1])
        finally
            ArrayPool<byte>.Shared.Return(output)
    | Float32, "byte-intype-max" ->
        let output = ArrayPool<byte>.Shared.Rent(pageBytes)
        try
            runTimedHotLoop iterations
                (fun () -> thresholdFloat32PageInTypeMaxVectorInto thresholdValue pageBytes decoded output)
                (fun () ->
                    let values = MemoryMarshal.Cast<byte, float32>(output.AsSpan(0, pageBytes))
                    int values[0] + int values[pixels / 2] + int values[pixels - 1])
        finally
            ArrayPool<byte>.Shared.Return(output)
    | Float32, "byte-intype-one" ->
        let output = ArrayPool<byte>.Shared.Rent(pageBytes)
        try
            runTimedHotLoop iterations
                (fun () -> thresholdFloat32PageInTypeOneVectorInto thresholdValue pageBytes decoded output)
                (fun () ->
                    let values = MemoryMarshal.Cast<byte, float32>(output.AsSpan(0, pageBytes))
                    int values[0] + int values[pixels / 2] + int values[pixels - 1])
        finally
            ArrayPool<byte>.Shared.Return(output)
    | Float32, "typed-intype-max" ->
        let typedInput = ArrayPool<float32>.Shared.Rent(pixels)
        let output = ArrayPool<float32>.Shared.Rent(pixels)
        try
            Buffer.BlockCopy(decoded, 0, typedInput, 0, pageBytes)
            runTimedHotLoop iterations
                (fun () -> thresholdTypedFloat32MaxVector thresholdValue pixels typedInput output)
                (fun () -> int output[0] + int output[pixels / 2] + int output[pixels - 1])
        finally
            ArrayPool<float32>.Shared.Return(output)
            ArrayPool<float32>.Shared.Return(typedInput)
    | Float32, "typed-intype-one" ->
        let typedInput = ArrayPool<float32>.Shared.Rent(pixels)
        let output = ArrayPool<float32>.Shared.Rent(pixels)
        try
            Buffer.BlockCopy(decoded, 0, typedInput, 0, pageBytes)
            runTimedHotLoop iterations
                (fun () -> thresholdTypedFloat32OneVector thresholdValue pixels typedInput output)
                (fun () -> int output[0] + int output[pixels / 2] + int output[pixels - 1])
        finally
            ArrayPool<float32>.Shared.Return(output)
            ArrayPool<float32>.Shared.Return(typedInput)
    | Float32, "typed-copy-intype-max" ->
        let typedInput = ArrayPool<float32>.Shared.Rent(pixels)
        let output = ArrayPool<float32>.Shared.Rent(pixels)
        try
            runTimedHotLoop iterations
                (fun () ->
                    Buffer.BlockCopy(decoded, 0, typedInput, 0, pageBytes)
                    thresholdTypedFloat32MaxVector thresholdValue pixels typedInput output)
                (fun () -> int output[0] + int output[pixels / 2] + int output[pixels - 1])
        finally
            ArrayPool<float32>.Shared.Return(output)
            ArrayPool<float32>.Shared.Return(typedInput)
    | Float32, "typed-copy-intype-one" ->
        let typedInput = ArrayPool<float32>.Shared.Rent(pixels)
        let output = ArrayPool<float32>.Shared.Rent(pixels)
        try
            runTimedHotLoop iterations
                (fun () ->
                    Buffer.BlockCopy(decoded, 0, typedInput, 0, pageBytes)
                    thresholdTypedFloat32OneVector thresholdValue pixels typedInput output)
                (fun () -> int output[0] + int output[pixels / 2] + int output[pixels - 1])
        finally
            ArrayPool<float32>.Shared.Return(output)
            ArrayPool<float32>.Shared.Return(typedInput)
    | _, unsupported ->
        invalidArg "variant" $"Unsupported Zarr hotloop variant '{unsupported}' for {pixelType}."

let private runLibTiffStripCopy opts =
    let operation = require "operation" opts
    let pixelType = require "pixel-type" opts |> parsePixelType
    let input = require "input" opts
    let output = require "output" opts

    if operation <> "copy" then
        failwith $"Direct LibTiff strip backend is intentionally copy-only; got '{operation}'."

    let files = stackTiffFiles input
    if files.Length = 0 then
        invalidOp $"No TIFF files found in input stack directory: {input}"

    use first = Tiff.Open(files[0], "r")
    if isNull first then
        invalidOp $"Could not open '{files[0]}' for TIFF reading."

    let width, height, rowBytes, pageBytes, strips, stripBytes, rowsPerStrip = inspectOpenStripTiffSlice pixelType files[0] first
    let page = ArrayPool<byte>.Shared.Rent(pageBytes)

    ensureCleanDirectory output
    let stopwatch = Stopwatch.StartNew()
    try
        for i in 0 .. files.Length - 1 do
            readEncodedStripTiffSliceInto pixelType files[i] width height rowBytes pageBytes strips page
            writeEncodedStripTiffPage pixelType (outputFile output i) width height rowBytes pageBytes strips stripBytes rowsPerStrip page

        stopwatch.Stop()
        writeInternalSeconds stopwatch.Elapsed
        0
    finally
        ArrayPool<byte>.Shared.Return(page)

let private runLibTiffRawStripCopy opts =
    let operation = require "operation" opts
    let pixelType = require "pixel-type" opts |> parsePixelType
    let input = require "input" opts
    let output = require "output" opts

    if operation <> "copy" then
        failwith $"Raw-strip LibTiff backend is intentionally copy-only; got '{operation}'."

    let files = stackTiffFiles input
    if files.Length = 0 then
        invalidOp $"No TIFF files found in input stack directory: {input}"

    use first = Tiff.Open(files[0], "r")
    if isNull first then
        invalidOp $"Could not open '{files[0]}' for TIFF reading."

    let width, height, rowBytes, pageBytes, strips, rawStripSizes, _rowsPerStrip = inspectOpenRawStripTiffSlice pixelType files[0] first
    let page = ArrayPool<byte>.Shared.Rent(pageBytes)

    ensureCleanDirectory output
    let stopwatch = Stopwatch.StartNew()
    try
        for i in 0 .. files.Length - 1 do
            readRawStripTiffSliceInto pixelType files[i] width height rowBytes pageBytes strips rawStripSizes page
            writeRawStripTiffPage pixelType (outputFile output i) width height rowBytes pageBytes page

        stopwatch.Stop()
        writeInternalSeconds stopwatch.Elapsed
        0
    finally
        ArrayPool<byte>.Shared.Return(page)

let private runNativeLibTiffRawStripCopy opts =
    let operation = require "operation" opts
    let pixelType = require "pixel-type" opts |> parsePixelType
    let input = require "input" opts
    let output = require "output" opts

    if operation <> "copy" then
        failwith $"Native raw-strip LibTiff backend is intentionally copy-only; got '{operation}'."

    try
        ensureCleanDirectory output
        let stopwatch = Stopwatch.StartNew()
        let files = stackTiffFiles input
        if files.Length = 0 then
            invalidOp $"No TIFF files found in input stack directory: {input}"

        let info, rowBytes, _bytesPerSample = readNativeTiffInfo pixelType files[0]
        let pageBytes = int info.PageBytes
        let page = ArrayPool<byte>.Shared.Rent(pageBytes)

        try
            for i in 0 .. files.Length - 1 do
                readNativeRawTiffSliceInto pixelType files[i] info.Width info.Height rowBytes pageBytes page
                writeNativeRawTiffPage pixelType (outputFile output i) info.Width info.Height pageBytes page

            stopwatch.Stop()
            writeInternalSeconds stopwatch.Elapsed
            0
        finally
            ArrayPool<byte>.Shared.Return(page)
    with
    | :? DllNotFoundException as ex ->
        invalidOp $"Could not load native libtiff shim 'sp_libtiff_shim'. Build it first, for example: bash benchmarks/native-libtiff-shim/build-unix.sh. Loader detail: {ex.Message}"
    | :? EntryPointNotFoundException as ex ->
        invalidOp $"Native libtiff shim is missing an expected entry point. Rebuild benchmarks/native-libtiff-shim. Loader detail: {ex.Message}"

let private runNativeLibTiffRawStripChunkCopy opts =
    let operation = require "operation" opts
    let pixelType = require "pixel-type" opts |> parsePixelType
    let input = require "input" opts
    let output = require "output" opts

    if operation <> "copy" then
        failwith $"Native raw-strip Chunk LibTiff backend is intentionally copy-only; got '{operation}'."
    if pixelType <> UInt8 then
        failwith $"Native raw-strip Chunk LibTiff backend is intentionally UInt8-only for now; got {pixelType}."

    try
        ensureCleanDirectory output
        let stopwatch = Stopwatch.StartNew()
        let files = stackTiffFiles input
        if files.Length = 0 then
            invalidOp $"No TIFF files found in input stack directory: {input}"

        let info, rowBytes, _bytesPerSample = readNativeTiffInfo pixelType files[0]
        let pageBytes = int info.PageBytes
        if uint64 pageBytes <> uint64 info.Width * uint64 info.Height then
            invalidOp $"Native raw-strip Chunk benchmark expected UInt8 scalar pages, got {pageBytes} bytes for {info.Width}x{info.Height}."

        let chunk = StackCore.Chunk.create<uint8> (uint64 info.Width, uint64 info.Height, 1UL)
        if chunk.ByteLength <> pageBytes then
            StackCore.Chunk.decRef chunk
            invalidOp $"Chunk byte length {chunk.ByteLength} differs from TIFF page bytes {pageBytes}."

        try
            for i in 0 .. files.Length - 1 do
                readNativeRawTiffSliceInto pixelType files[i] info.Width info.Height rowBytes pageBytes chunk.Bytes
                writeNativeRawTiffPage pixelType (outputFile output i) info.Width info.Height pageBytes chunk.Bytes

            stopwatch.Stop()
            writeInternalSeconds stopwatch.Elapsed
            0
        finally
            StackCore.Chunk.decRef chunk
    with
    | :? DllNotFoundException as ex ->
        invalidOp $"Could not load native libtiff shim 'sp_libtiff_shim'. Build it first, for example: bash benchmarks/native-libtiff-shim/build-unix.sh. Loader detail: {ex.Message}"
    | :? EntryPointNotFoundException as ex ->
        invalidOp $"Native libtiff shim is missing an expected entry point. Rebuild benchmarks/native-libtiff-shim. Loader detail: {ex.Message}"

let private runNativeLibTiffRawStripVolumeChunkCopy opts =
    let operation = require "operation" opts
    let pixelType = require "pixel-type" opts |> parsePixelType
    let input = require "input" opts
    let output = require "output" opts

    if operation <> "copy" then
        failwith $"Native raw-strip volume Chunk LibTiff backend is intentionally copy-only; got '{operation}'."
    if pixelType <> UInt8 then
        failwith $"Native raw-strip volume Chunk LibTiff backend is intentionally UInt8-only for now; got {pixelType}."

    try
        ensureCleanDirectory output
        let stopwatch = Stopwatch.StartNew()
        let files = stackTiffFiles input
        if files.Length = 0 then
            invalidOp $"No TIFF files found in input stack directory: {input}"

        let info, rowBytes, _bytesPerSample = readNativeTiffInfo pixelType files[0]
        let pageBytes = int info.PageBytes
        if uint64 pageBytes <> uint64 info.Width * uint64 info.Height then
            invalidOp $"Native raw-strip volume Chunk benchmark expected UInt8 scalar pages, got {pageBytes} bytes for {info.Width}x{info.Height}."

        let chunk = StackCore.Chunk.create<uint8> (uint64 info.Width, uint64 info.Height, uint64 files.Length)
        if chunk.ByteLength <> pageBytes * files.Length then
            StackCore.Chunk.decRef chunk
            invalidOp $"Chunk byte length {chunk.ByteLength} differs from expected volume bytes {pageBytes * files.Length}."

        try
            for i in 0 .. files.Length - 1 do
                readNativeRawTiffSliceIntoOffset pixelType files[i] info.Width info.Height rowBytes pageBytes (i * pageBytes) chunk.Bytes

            for i in 0 .. files.Length - 1 do
                writeNativeRawTiffPageFromOffset pixelType (outputFile output i) info.Width info.Height pageBytes (i * pageBytes) chunk.Bytes

            stopwatch.Stop()
            writeInternalSeconds stopwatch.Elapsed
            0
        finally
            StackCore.Chunk.decRef chunk
    with
    | :? DllNotFoundException as ex ->
        invalidOp $"Could not load native libtiff shim 'sp_libtiff_shim'. Build it first, for example: bash benchmarks/native-libtiff-shim/build-unix.sh. Loader detail: {ex.Message}"
    | :? EntryPointNotFoundException as ex ->
        invalidOp $"Native libtiff shim is missing an expected entry point. Rebuild benchmarks/native-libtiff-shim. Loader detail: {ex.Message}"

let private runStackReadWrite opts =
    let pixelType = require "pixel-type" opts |> parsePixelType
    let input = require "input" opts
    let output = require "output" opts
    let availableMemory = optional "available-memory" (string (1024UL * 1024UL * 1024UL * 1024UL)) opts |> UInt64.Parse
    let debugLevel = optional "debug-level" "0" opts |> UInt32.Parse

    ensureCleanDirectory output
    let stopwatch = Stopwatch.StartNew()
    let exitCode =
        match pixelType with
        | UInt8 -> runChunkReadWriteTyped<uint8> input output availableMemory debugLevel
        | UInt16 -> runChunkReadWriteTyped<uint16> input output availableMemory debugLevel
        | Float32 -> runChunkReadWriteTyped<float32> input output availableMemory debugLevel
        | UInt32
        | Int32 -> failwith $"Stack read/write benchmark currently supports UInt8, UInt16, and Float32; got {pixelType}."

    stopwatch.Stop()
    writeInternalSeconds stopwatch.Elapsed
    exitCode

let private runStackTiffPathCopy opts =
    let pixelType = require "pixel-type" opts |> parsePixelType
    let input = require "input" opts
    let output = require "output" opts
    let availableMemory = optional "available-memory" (string (1024UL * 1024UL * 1024UL * 1024UL)) opts |> UInt64.Parse
    let debugLevel = optional "debug-level" "0" opts |> UInt32.Parse
    let options: StackIO.TiffWriteOptions =
        { Compression = optional "write-compression" "none" opts |> parseTiffCompression
          ByteOrder = optional "write-byte-order" "native" opts |> parseTiffByteOrder }

    ensureCleanDirectory output
    let stopwatch = Stopwatch.StartNew()
    let exitCode =
        match pixelType with
        | UInt8 -> runChunkReadWriteWithTiffOptionsTyped<uint8> input output availableMemory debugLevel options
        | UInt16 -> runChunkReadWriteWithTiffOptionsTyped<uint16> input output availableMemory debugLevel options
        | Float32 -> runChunkReadWriteWithTiffOptionsTyped<float32> input output availableMemory debugLevel options
        | UInt32
        | Int32 -> failwith $"Stack TIFF path benchmark currently supports UInt8, UInt16, and Float32; got {pixelType}."

    stopwatch.Stop()
    writeInternalSeconds stopwatch.Elapsed
    exitCode

let private runNativeLibTiffRawStripReadOnly opts =
    let pixelType = require "pixel-type" opts |> parsePixelType
    let input = require "input" opts

    try
        let stopwatch = Stopwatch.StartNew()
        let files = stackTiffFiles input
        if files.Length = 0 then
            invalidOp $"No TIFF files found in input stack directory: {input}"

        let info, rowBytes, _bytesPerSample = readNativeTiffInfo pixelType files[0]
        let pageBytes = int info.PageBytes
        let page = ArrayPool<byte>.Shared.Rent(pageBytes)
        let mutable checksum = 0

        try
            for i in 0 .. files.Length - 1 do
                readNativeRawTiffSliceInto pixelType files[i] info.Width info.Height rowBytes pageBytes page
                checksum <- checksum ^^^ int page[0] ^^^ int page[pageBytes - 1] ^^^ i

            if checksum = Int32.MinValue then
                eprintfn "impossible checksum: %d" checksum

            stopwatch.Stop()
            writeInternalSeconds stopwatch.Elapsed
            0
        finally
            ArrayPool<byte>.Shared.Return(page)
    with
    | :? DllNotFoundException as ex ->
        invalidOp $"Could not load native libtiff shim 'sp_libtiff_shim'. Build it first, for example: bash benchmarks/native-libtiff-shim/build-unix.sh. Loader detail: {ex.Message}"
    | :? EntryPointNotFoundException as ex ->
        invalidOp $"Native libtiff shim is missing an expected entry point. Rebuild benchmarks/native-libtiff-shim. Loader detail: {ex.Message}"

let private runNativeLibTiffRawStripChunkReadOnly opts =
    let pixelType = require "pixel-type" opts |> parsePixelType
    let input = require "input" opts

    if pixelType <> UInt8 then
        failwith $"Native raw-strip Chunk read-only backend is intentionally UInt8-only for now; got {pixelType}."

    try
        let stopwatch = Stopwatch.StartNew()
        let files = stackTiffFiles input
        if files.Length = 0 then
            invalidOp $"No TIFF files found in input stack directory: {input}"

        let info, rowBytes, _bytesPerSample = readNativeTiffInfo pixelType files[0]
        let pageBytes = int info.PageBytes
        let chunk = StackCore.Chunk.create<uint8> (uint64 info.Width, uint64 info.Height, 1UL)
        let mutable checksum = 0

        try
            if chunk.ByteLength <> pageBytes then
                invalidOp $"Chunk byte length {chunk.ByteLength} differs from TIFF page bytes {pageBytes}."

            for i in 0 .. files.Length - 1 do
                readNativeRawTiffSliceInto pixelType files[i] info.Width info.Height rowBytes pageBytes chunk.Bytes
                checksum <- checksum ^^^ int chunk.Bytes[0] ^^^ int chunk.Bytes[pageBytes - 1] ^^^ i

            if checksum = Int32.MinValue then
                eprintfn "impossible checksum: %d" checksum

            stopwatch.Stop()
            writeInternalSeconds stopwatch.Elapsed
            0
        finally
            StackCore.Chunk.decRef chunk
    with
    | :? DllNotFoundException as ex ->
        invalidOp $"Could not load native libtiff shim 'sp_libtiff_shim'. Build it first, for example: bash benchmarks/native-libtiff-shim/build-unix.sh. Loader detail: {ex.Message}"
    | :? EntryPointNotFoundException as ex ->
        invalidOp $"Native libtiff shim is missing an expected entry point. Rebuild benchmarks/native-libtiff-shim. Loader detail: {ex.Message}"

let private fillPage (page: byte[]) pageBytes =
    for i in 0 .. pageBytes - 1 do
        page[i] <- byte (i &&& 0xFF)

let private runNativeLibTiffRawStripWriteOnly opts =
    let pixelType = require "pixel-type" opts |> parsePixelType
    let input = require "input" opts
    let output = require "output" opts

    try
        let files = stackTiffFiles input
        if files.Length = 0 then
            invalidOp $"No TIFF files found in input stack directory: {input}"

        let info, _rowBytes, _bytesPerSample = readNativeTiffInfo pixelType files[0]
        let pageBytes = int info.PageBytes
        let page = ArrayPool<byte>.Shared.Rent(pageBytes)
        fillPage page pageBytes

        ensureCleanDirectory output
        let stopwatch = Stopwatch.StartNew()
        try
            for i in 0 .. files.Length - 1 do
                writeNativeRawTiffPage pixelType (outputFile output i) info.Width info.Height pageBytes page

            stopwatch.Stop()
            writeInternalSeconds stopwatch.Elapsed
            0
        finally
            ArrayPool<byte>.Shared.Return(page)
    with
    | :? DllNotFoundException as ex ->
        invalidOp $"Could not load native libtiff shim 'sp_libtiff_shim'. Build it first, for example: bash benchmarks/native-libtiff-shim/build-unix.sh. Loader detail: {ex.Message}"
    | :? EntryPointNotFoundException as ex ->
        invalidOp $"Native libtiff shim is missing an expected entry point. Rebuild benchmarks/native-libtiff-shim. Loader detail: {ex.Message}"

let private runNativeLibTiffRawStripChunkWriteOnly opts =
    let pixelType = require "pixel-type" opts |> parsePixelType
    let input = require "input" opts
    let output = require "output" opts

    if pixelType <> UInt8 then
        failwith $"Native raw-strip Chunk write-only backend is intentionally UInt8-only for now; got {pixelType}."

    try
        let files = stackTiffFiles input
        if files.Length = 0 then
            invalidOp $"No TIFF files found in input stack directory: {input}"

        let info, _rowBytes, _bytesPerSample = readNativeTiffInfo pixelType files[0]
        let pageBytes = int info.PageBytes
        let chunk = StackCore.Chunk.create<uint8> (uint64 info.Width, uint64 info.Height, 1UL)
        if chunk.ByteLength <> pageBytes then
            StackCore.Chunk.decRef chunk
            invalidOp $"Chunk byte length {chunk.ByteLength} differs from TIFF page bytes {pageBytes}."
        fillPage chunk.Bytes pageBytes

        ensureCleanDirectory output
        let stopwatch = Stopwatch.StartNew()
        try
            for i in 0 .. files.Length - 1 do
                writeNativeRawTiffPage pixelType (outputFile output i) info.Width info.Height pageBytes chunk.Bytes

            stopwatch.Stop()
            writeInternalSeconds stopwatch.Elapsed
            0
        finally
            StackCore.Chunk.decRef chunk
    with
    | :? DllNotFoundException as ex ->
        invalidOp $"Could not load native libtiff shim 'sp_libtiff_shim'. Build it first, for example: bash benchmarks/native-libtiff-shim/build-unix.sh. Loader detail: {ex.Message}"
    | :? EntryPointNotFoundException as ex ->
        invalidOp $"Native libtiff shim is missing an expected entry point. Rebuild benchmarks/native-libtiff-shim. Loader detail: {ex.Message}"

let private runNativeLibTiffScanlineCopy opts =
    let operation = require "operation" opts
    let pixelType = require "pixel-type" opts |> parsePixelType
    let input = require "input" opts
    let output = require "output" opts

    if operation <> "copy" then
        failwith $"Native scanline LibTiff backend is intentionally copy-only; got '{operation}'."

    try
        ensureCleanDirectory output
        let stopwatch = Stopwatch.StartNew()
        let files = stackTiffFiles input
        if files.Length = 0 then
            invalidOp $"No TIFF files found in input stack directory: {input}"

        let info, rowBytes, _bytesPerSample = readNativeTiffInfo pixelType files[0]
        let pageBytes = int info.PageBytes
        let page = ArrayPool<byte>.Shared.Rent(pageBytes)

        try
            for i in 0 .. files.Length - 1 do
                readNativeScanlineTiffSliceInto pixelType files[i] info.Width info.Height rowBytes pageBytes page
                writeNativeScanlineTiffPage pixelType (outputFile output i) info.Width info.Height pageBytes page

            stopwatch.Stop()
            writeInternalSeconds stopwatch.Elapsed
            0
        finally
            ArrayPool<byte>.Shared.Return(page)
    with
    | :? DllNotFoundException as ex ->
        invalidOp $"Could not load native libtiff shim 'sp_libtiff_shim'. Build it first, for example: bash benchmarks/native-libtiff-shim/build-unix.sh. Loader detail: {ex.Message}"
    | :? EntryPointNotFoundException as ex ->
        invalidOp $"Native libtiff shim is missing an expected entry point. Rebuild benchmarks/native-libtiff-shim. Loader detail: {ex.Message}"

let private runTiffLibraryRawStripCopy opts =
    let operation = require "operation" opts
    let pixelType = require "pixel-type" opts |> parsePixelType
    let input = require "input" opts
    let output = require "output" opts

    if operation <> "copy" then
        failwith $"TiffLibrary raw-strip backend is intentionally copy-only; got '{operation}'."

    let files = stackTiffFiles input
    if files.Length = 0 then
        invalidOp $"No TIFF files found in input stack directory: {input}"

    let width, height, rowBytes, pageBytes, _stripOffsets, stripByteCounts = readTiffLibraryInfo pixelType files[0]
    let page = ArrayPool<byte>.Shared.Rent(pageBytes)

    ensureCleanDirectory output
    let stopwatch = Stopwatch.StartNew()
    try
        for i in 0 .. files.Length - 1 do
            readTiffLibraryRawSliceInto pixelType files[i] width height rowBytes pageBytes stripByteCounts.Count page
            writeTiffLibraryRawPage pixelType (outputFile output i) width height pageBytes page

        stopwatch.Stop()
        writeInternalSeconds stopwatch.Elapsed
        0
    finally
        ArrayPool<byte>.Shared.Return(page)

let private imageSharpDecoderOptions () =
    SixLabors.ImageSharp.Formats.DecoderOptions(
        SkipMetadata = true,
        MaxFrames = 1u)

let private imageSharpTiffEncoder bitsPerPixel =
    SixLabors.ImageSharp.Formats.Tiff.TiffEncoder(
        Compression = Nullable(SixLabors.ImageSharp.Formats.Tiff.Constants.TiffCompression.None),
        BitsPerPixel = Nullable(bitsPerPixel),
        PhotometricInterpretation = Nullable(SixLabors.ImageSharp.Formats.Tiff.Constants.TiffPhotometricInterpretation.BlackIsZero),
        SkipMetadata = true)

let private runImageSharpCopyAs<'TPixel when 'TPixel : unmanaged and 'TPixel :> SixLabors.ImageSharp.PixelFormats.IPixel<'TPixel>> bitsPerPixel (files: string array) output =
    let decoderOptions = imageSharpDecoderOptions ()
    let encoder = imageSharpTiffEncoder bitsPerPixel

    for i in 0 .. files.Length - 1 do
        use stream = File.OpenRead(files[i])
        use image = SixLabors.ImageSharp.Image.Load<'TPixel>(decoderOptions, stream)
        use outputStream = File.Create(outputFile output i)
        image.Save(outputStream, encoder)

let private runImageSharpCopy opts =
    let operation = require "operation" opts
    let pixelType = require "pixel-type" opts |> parsePixelType
    let input = require "input" opts
    let output = require "output" opts

    if operation <> "copy" then
        failwith $"ImageSharp backend is intentionally copy-only; got '{operation}'."

    let files = stackTiffFiles input
    if files.Length = 0 then
        invalidOp $"No TIFF files found in input stack directory: {input}"

    ensureCleanDirectory output
    let stopwatch = Stopwatch.StartNew()
    match pixelType with
    | UInt8 ->
        runImageSharpCopyAs<SixLabors.ImageSharp.PixelFormats.L8> SixLabors.ImageSharp.Formats.Tiff.TiffBitsPerPixel.Bit8 files output
    | UInt16 ->
        runImageSharpCopyAs<SixLabors.ImageSharp.PixelFormats.L16> SixLabors.ImageSharp.Formats.Tiff.TiffBitsPerPixel.Bit16 files output
    | _ -> unsupportedPixelType "ImageSharp copy benchmark" "UInt8 and UInt16 grayscale" pixelType

    stopwatch.Stop()
    writeInternalSeconds stopwatch.Elapsed
    0

let private runLibTiffDirectReadOnly opts =
    let pixelType = require "pixel-type" opts |> parsePixelType
    let input = require "input" opts

    let stopwatch = Stopwatch.StartNew()
    let files = stackTiffFiles input
    if files.Length = 0 then
        invalidOp $"No TIFF files found in input stack directory: {input}"

    let width, height, rowBytes, scanlineSize = inspectDirectTiffSlice pixelType files[0]
    let pageBytes = rowBytes * int height
    let page = ArrayPool<byte>.Shared.Rent(pageBytes)
    let scratch = ArrayPool<byte>.Shared.Rent(scanlineSize)

    try
        for i in 0 .. files.Length - 1 do
            readDirectByteTiffSliceInto pixelType files[i] width height rowBytes page scratch

        stopwatch.Stop()
        writeInternalSeconds stopwatch.Elapsed
        0
    finally
        ArrayPool<byte>.Shared.Return(scratch)
        ArrayPool<byte>.Shared.Return(page)

let private runLibTiffDirectWriteOnly opts =
    let pixelType = require "pixel-type" opts |> parsePixelType
    let input = require "input" opts
    let output = require "output" opts

    let files = stackTiffFiles input
    if files.Length = 0 then
        invalidOp $"No TIFF files found in input stack directory: {input}"

    let width, height, rowBytes, _scanlineSize = inspectDirectTiffSlice pixelType files[0]
    let pageBytes = rowBytes * int height
    let page = ArrayPool<byte>.Shared.Rent(pageBytes)

    ensureCleanDirectory output
    try
        for i in 0 .. pageBytes - 1 do
            page[i] <- byte (i &&& 0xFF)

        let stopwatch = Stopwatch.StartNew()
        for i in 0 .. files.Length - 1 do
            writeDirectByteTiffPage pixelType (outputFile output i) width height rowBytes page

        stopwatch.Stop()
        writeInternalSeconds stopwatch.Elapsed
        0
    finally
        ArrayPool<byte>.Shared.Return(page)

[<EntryPoint>]
let main args =
    try
        match args with
        | [| |] -> usage ()
        | _ when args[0] = "generate" -> args[1..] |> parseArgs |> generate
        | _ when args[0] = "run" -> args[1..] |> parseArgs |> run
        | _ when args[0] = "run-zarr" -> args[1..] |> parseArgs |> runZarr
        | _ when args[0] = "run-zarr-convolve-breakdown" -> args[1..] |> parseArgs |> runZarrConvolveBreakdown
        | _ when args[0] = "run-zarr-direct-copy" -> args[1..] |> parseArgs |> runZarrDirectCopy
        | _ when args[0] = "run-zarr-direct-copy-same-grid" -> args[1..] |> parseArgs |> runZarrDirectCopySameGrid
        | _ when args[0] = "run-zarr-direct-threshold" -> args[1..] |> parseArgs |> runZarrDirectThreshold
        | _ when args[0] = "run-zarr-direct-threshold-raw" -> args[1..] |> parseArgs |> runZarrDirectThresholdRaw
        | _ when args[0] = "run-zarr-direct-threshold-intype" -> args[1..] |> parseArgs |> runZarrDirectThresholdInType
        | _ when args[0] = "run-zarr-direct-threshold-hotloop" -> args[1..] |> parseArgs |> runZarrDirectThresholdHotLoop
        | _ when args[0] = "run-zarr-chunk-copy" -> args[1..] |> parseArgs |> runZarrChunkCopy
        | _ when args[0] = "run-zarr-readonly" -> args[1..] |> parseArgs |> runZarrReadOnly
        | _ when args[0] = "run-zarr-writeonly" -> args[1..] |> parseArgs |> runZarrWriteOnly
        | _ when args[0] = "run-tiff-thick-readonly" -> args[1..] |> parseArgs |> runTiffThickReadOnly
        | _ when args[0] = "run-tiff-thick-split-drain" -> args[1..] |> parseArgs |> runTiffThickSplitDrain
        | _ when args[0] = "run-zarr-thick-writeonly" -> args[1..] |> parseArgs |> runZarrThickWriteOnly
        | _ when args[0] = "run-zarr-thick-writeonly-pattern" -> args[1..] |> parseArgs |> runZarrThickWriteOnlyPattern
        | _ when args[0] = "run-zarr-thick-writeonly-directlocal" -> args[1..] |> parseArgs |> runZarrThickWriteOnlyDirectLocal
        | _ when args[0] = "run-chunk-fft-float32-zarr" -> args[1..] |> parseArgs |> runChunkFftFloat32Zarr
        | _ when args[0] = "run-chunk-fft-xy-float32-zarr" -> args[1..] |> parseArgs |> runChunkFftXYFloat32Zarr
        | _ when args[0] = "run-chunk-fft-z-complex64-zarr" -> args[1..] |> parseArgs |> runChunkFftZComplex64Zarr
        | _ when args[0] = "run-chunk-fft-native-float32-zarr" -> args[1..] |> parseArgs |> runChunkFftNativeFloat32Zarr
        | _ when args[0] = "run-fft3d-kernel" -> args[1..] |> parseArgs |> runFft3DKernel
        | _ when args[0] = "run-chunk-fft3d-stage" -> args[1..] |> parseArgs |> runChunkFft3DKernel
        | _ when args[0] = "run-chunk-fft3d-stage-io" -> args[1..] |> parseArgs |> runChunkFft3DStageIo
        | _ when args[0] = "run-chunk-fft3d-stage-overhead" -> args[1..] |> parseArgs |> runChunkFft3DStageOverhead
        | _ when args[0] = "run-chunk-fft3d-kernel" -> args[1..] |> parseArgs |> runChunkFft3DKernel
        | _ when args[0] = "run-chunk-fft3d-spectral-zarr" -> args[1..] |> parseArgs |> runChunkFft3DSpectralZarr
        | _ when args[0] = "run-chunk-fft3d-zarr-roundtrip-io" -> args[1..] |> parseArgs |> runChunkFft3DZarrRoundtripIo
        | _ when args[0] = "run-chunk-fft3d-zarr-subchunked-roundtrip-io" -> args[1..] |> parseArgs |> runChunkFft3DZarrSubchunkedRoundtripIo
        | _ when args[0] = "run-arraypool" -> args[1..] |> parseArgs |> runArrayPool
        | _ when args[0] = "run-arraypool-slice" -> args[1..] |> parseArgs |> runArrayPoolSlice
        | _ when args[0] = "run-arraypool-slice-reuse" -> args[1..] |> parseArgs |> runArrayPoolSliceReuse
        | _ when args[0] = "run-byte-slice-reuse" -> args[1..] |> parseArgs |> runByteSliceReuse
        | _ when args[0] = "run-byte-float32-slice-reuse" -> args[1..] |> parseArgs |> runByteFloat32SliceReuse
        | _ when args[0] = "run-libtiff-direct-copy" -> args[1..] |> parseArgs |> runLibTiffDirectCopy
        | _ when args[0] = "run-libtiff-direct-threshold" -> args[1..] |> parseArgs |> runLibTiffDirectThreshold
        | _ when args[0] = "run-libtiff-direct-threshold-intype" -> args[1..] |> parseArgs |> runLibTiffDirectThresholdInType
        | _ when args[0] = "run-libtiff-direct-threshold-hotloop" -> args[1..] |> parseArgs |> runLibTiffDirectThresholdHotLoop
        | _ when args[0] = "run-chunk-threshold-parallel" -> args[1..] |> parseArgs |> runChunkThresholdParallelCollect
        | _ when args[0] = "run-chunk-histogram" -> args[1..] |> parseArgs |> runChunkHistogram
        | _ when args[0] = "run-chunk-simd-reductions" -> args[1..] |> parseArgs |> runChunkSimdReductions
        | _ when args[0] = "run-chunk-cast" -> args[1..] |> parseArgs |> runChunkCast
        | _ when args[0] = "run-chunk-pixelwise-float32" -> args[1..] |> parseArgs |> runChunkPixelwiseFloat32
        | _ when args[0] = "run-chunk-structure-tensor-layout" -> args[1..] |> parseArgs |> runChunkStructureTensorLayout
        | _ when args[0] = "run-chunk-connected-components" -> args[1..] |> parseArgs |> runChunkConnectedComponentsCommand
        | _ when args[0] = "run-chunk-dilate" -> args[1..] |> parseArgs |> runChunkDilate
        | _ when args[0] = "run-chunk-convolve" -> args[1..] |> parseArgs |> runChunkConvolve
        | _ when args[0] = "run-chunk-axis-convolve-kernel" -> args[1..] |> parseArgs |> runChunkAxisConvolveKernel
        | _ when args[0] = "run-threshold-kernel" -> args[1..] |> parseArgs |> runThresholdKernel
        | _ when args[0] = "run-libtiff-strip-copy" -> args[1..] |> parseArgs |> runLibTiffStripCopy
        | _ when args[0] = "run-libtiff-raw-strip-copy" -> args[1..] |> parseArgs |> runLibTiffRawStripCopy
        | _ when args[0] = "run-bitmiracle-raw-strip-copy" -> args[1..] |> parseArgs |> runLibTiffRawStripCopy
        | _ when args[0] = "run-native-libtiff-raw-strip-copy" -> args[1..] |> parseArgs |> runNativeLibTiffRawStripCopy
        | _ when args[0] = "run-native-libtiff-raw-strip-chunk-copy" -> args[1..] |> parseArgs |> runNativeLibTiffRawStripChunkCopy
        | _ when args[0] = "run-native-libtiff-raw-strip-volume-chunk-copy" -> args[1..] |> parseArgs |> runNativeLibTiffRawStripVolumeChunkCopy
        | _ when args[0] = "run-stack-read-write" -> args[1..] |> parseArgs |> runStackReadWrite
        | _ when args[0] = "run-stack-tiff-path-copy" -> args[1..] |> parseArgs |> runStackTiffPathCopy
        | _ when args[0] = "run-native-libtiff-raw-strip-readonly" -> args[1..] |> parseArgs |> runNativeLibTiffRawStripReadOnly
        | _ when args[0] = "run-native-libtiff-raw-strip-chunk-readonly" -> args[1..] |> parseArgs |> runNativeLibTiffRawStripChunkReadOnly
        | _ when args[0] = "run-native-libtiff-raw-strip-writeonly" -> args[1..] |> parseArgs |> runNativeLibTiffRawStripWriteOnly
        | _ when args[0] = "run-native-libtiff-raw-strip-chunk-writeonly" -> args[1..] |> parseArgs |> runNativeLibTiffRawStripChunkWriteOnly
        | _ when args[0] = "run-native-libtiff-scanline-copy" -> args[1..] |> parseArgs |> runNativeLibTiffScanlineCopy
        | _ when args[0] = "run-tifflibrary-raw-strip-copy" -> args[1..] |> parseArgs |> runTiffLibraryRawStripCopy
        | _ when args[0] = "run-imagesharp-copy" -> args[1..] |> parseArgs |> runImageSharpCopy
        | _ when args[0] = "run-libtiff-direct-readonly" -> args[1..] |> parseArgs |> runLibTiffDirectReadOnly
        | _ when args[0] = "run-libtiff-direct-writeonly" -> args[1..] |> parseArgs |> runLibTiffDirectWriteOnly
        | _ -> usage ()
    with ex ->
        fail ex.Message
