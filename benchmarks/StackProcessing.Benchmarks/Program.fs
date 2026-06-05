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
open BitMiracle.LibTiff.Classic
open StackProcessing
open ZarrNET.Core
open ZarrNET.Core.Nodes
open ZarrNET.Core.OmeZarr.Coordinates
open ZarrNET.Core.Zarr
open ZarrNET.Core.Zarr.Store

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
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run --operation copy|threshold|convolve|median|dilate|connectedComponents --pixel-type UInt8|UInt16|Float32 --input DIR --output DIR [--radius N] [--kernel-size N] [--threshold X] [--window N] [--available-memory BYTES]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-zarr --operation copy|threshold|convolve|median|dilate --pixel-type UInt8|UInt16|Float32 --shape WxHxD --input ZARR --output ZARR [--radius N] [--kernel-size N] [--threshold X] [--available-memory BYTES]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-zarr-direct-copy --pixel-type UInt8|UInt16|Float32 --shape WxHxD --input ZARR --output ZARR
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-zarr-direct-threshold --pixel-type UInt8|UInt16|Float32 --shape WxHxD --input ZARR --output ZARR [--threshold X]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-zarr-direct-threshold-raw --pixel-type UInt8|UInt16|Float32 --shape WxHxD --input ZARR --output ZARR [--threshold X]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-zarr-chunk-copy --pixel-type UInt8|UInt16|Float32 --shape WxHxD --input ZARR --output ZARR [--available-memory BYTES]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-zarr-readonly --pixel-type UInt8|UInt16|Float32 --input ZARR [--available-memory BYTES]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-zarr-writeonly --pixel-type UInt8|UInt16|Float32 --shape WxHxD --output ZARR [--available-memory BYTES]

ArrayPool experiment:
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-arraypool --operation copy|threshold|connectedComponents --pixel-type UInt8|UInt16|Float32 --input DIR --output DIR [--threshold X]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-arraypool-slice --operation copy|threshold --pixel-type UInt8|UInt16|Float32 --input DIR --output DIR [--threshold X]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-arraypool-slice-reuse --operation copy|threshold --pixel-type UInt8|UInt16|Float32 --input DIR --output DIR [--threshold X]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-byte-slice-reuse --operation copy|threshold --pixel-type UInt8 --input DIR --output DIR [--threshold X]
  dotnet run --project benchmarks/StackProcessing.Benchmarks -- run-byte-float32-slice-reuse --operation copy|threshold --pixel-type Float32 --input DIR --output DIR [--threshold X]
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

let private runTask (task: Threading.Tasks.Task<'T>) : 'T =
    task.GetAwaiter().GetResult()

let private runUnitTask (task: Threading.Tasks.Task) : unit =
    task.GetAwaiter().GetResult()

let private parsePixelType value =
    match value with
    | "UInt8" | "uint8" -> UInt8
    | "UInt16" | "uint16" -> UInt16
    | "Float32" | "float32" -> Float32
    | _ -> failwith $"unsupported pixel type '{value}'"

let private zarrDataType pixelType =
    match pixelType with
    | UInt8 -> "uint8"
    | UInt16 -> "uint16"
    | Float32 -> "float32"

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
    elif t = typeof<uint16> then 16, SampleFormat.UINT, 2
    elif t = typeof<float32> then 32, SampleFormat.IEEEFP, 4
    else
        invalidArg "T" $"ArrayPool benchmark supports UInt8, UInt16, and Float32 TIFF stacks; got {t.Name}."

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
        | SampleFormat.UINT, 8 -> 1
        | SampleFormat.UINT, 16 -> 2
        | SampleFormat.IEEEFP, 32 -> 4
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
    >=> dilateZonohedral radius None
    >=> write output ".tiff"
    |> sink
    0

let private uniformKernel3D (kernelSize: uint) =
    let size = max 1u kernelSize
    let value = 1.0 / float (size * size * size)
    Array3D.create (int size) (int size) (int size) value
    |> fun values -> Image<float>.ofArray3D(values, name = $"uniformKernel{size}")

let private runConvolveTyped<'T when 'T: equality> input output kernelSize availableMemory =
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

let private zarrChunkMedianSlab<'T when 'T: equality> radius chunkZ : Stage<Image<'T>, Image<'T>> =
    let radius = max 0u radius
    let kernelDepth = 2u * radius + 1u
    let chunkDepth = max (max 1u chunkZ) kernelDepth
    let inputDepth = chunkDepth + 2u * radius
    let componentBytes =
        if typeof<'T> = typeof<uint8> then 1UL
        elif typeof<'T> = typeof<uint16> then 2UL
        elif typeof<'T> = typeof<float32> then 4UL
        else invalidArg "T" $"Zarr median benchmark supports UInt8, UInt16, and Float32; got {typeof<'T>.Name}."
    let medianMemoryNeed nPixels =
        2UL * nPixels * uint64 inputDepth * componentBytes

    let medianStage =
        SlimPipeline.Stage.map
            "zarrChunkMedianSlab.median"
            (fun _ (image: Image<'T>) ->
                try
                    ImageFunctions.median radius image
                finally
                    image.decRefCount())
            medianMemoryNeed
            id

    let cropMiddleSlab =
        let memoryNeed nPixels =
            2UL * nPixels * uint64 chunkDepth * componentBytes

        SlimPipeline.Stage.map
            "zarrChunkMedianSlab.cropMiddle"
            (fun _ (slab: Slab<'T>) ->
                let image = slab.Image
                try
                    if image.GetDimensions() <> 3u then
                        failwith $"zarrChunkMedianSlab expected a 3D slab, got {image.GetDimensions()}D."

                    let size = image.GetSize()
                    let outputDepth = min chunkDepth (image.GetDepth())
                    let startZ = radius
                    let stopZ = startZ + outputDepth - 1u
                    ImageFunctions.extractSub [ 0u; 0u; startZ ] [ size[0] - 1u; size[1] - 1u; stopZ ] image
                finally
                    image.decRefCount())
            memoryNeed
            id

    StackCore.window inputDepth radius chunkDepth
    --> StackCore.requireWindowSize<'T> kernelDepth
    --> StackCore.windowToSlabWithRange<'T>
    --> StackCore.mapSlabWithStage medianStage
    --> cropMiddleSlab

let private runZarrTyped<'T when 'T: equality> operation shape input output radius kernelSize thresholdValue availableMemory =
    ensureCleanDirectory output
    let src = benchmarkSource availableMemory
    let zarrInfo = getZarrInfo input 0 0
    let chunkDepth, chunkY, chunkX =
        match zarrInfo.chunks with
        | _t :: _c :: z :: y :: x :: _ -> max 1u (uint z), max 1u (uint y), max 1u (uint x)
        | z :: y :: x :: _ -> max 1u (uint z), max 1u (uint y), max 1u (uint x)
        | _ -> 16u, 256u, 256u
    let readInput () =
        src
        |> readZarrSlab<'T> input 0 0 0 0 0

    let readInputSlabStacked () =
        src
        |> readZarrSlabStacked<'T> input 0 0 0 0 0

    match operation with
    | "copy" ->
        readInputSlabStacked ()
        |> writeZarrSlab output chunkX chunkY 1.0 1.0 1.0 0
        |> sink
    | "threshold" ->
        readInputSlabStacked ()
        >=> threshold thresholdValue infinity
        |> writeZarrSlab output chunkX chunkY 1.0 1.0 1.0 0
        |> sink
    | "median" ->
        readInput ()
        >=> zarrChunkMedianSlab<'T> radius chunkDepth
        |> writeZarrSlab output chunkX chunkY 1.0 1.0 1.0 0
        |> sink
    | "convolve" ->
        let kernel = uniformKernel3D kernelSize
        try
            readInput ()
            >=> cast<'T, float>
            >=> convolve kernel None None None
            >=> cast<float, 'T>
            >=> writeZarr output "benchmark" shape.Depth chunkX chunkY chunkDepth 1.0 1.0 1.0 0
            |> sink
        finally
            kernel.decRefCount()
    | "dilate" ->
        readInput ()
        >=> threshold 128.0 infinity
        >=> dilateZonohedral radius None
        >=> writeZarr output "benchmark" shape.Depth chunkX chunkY chunkDepth 1.0 1.0 1.0 0
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
    let availableMemory = optional "available-memory" (string (1024UL * 1024UL * 1024UL * 1024UL)) opts |> UInt64.Parse
    let stopwatch = Stopwatch.StartNew()
    let exitCode =
        match pixelType with
        | UInt8 -> runZarrTyped<uint8> operation shape input output radius kernelSize thresholdValue availableMemory
        | UInt16 -> runZarrTyped<uint16> operation shape input output radius kernelSize thresholdValue availableMemory
        | Float32 -> runZarrTyped<float32> operation shape input output radius kernelSize thresholdValue availableMemory
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
    let store = LocalFileSystemStore(path)
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
    | Float32 -> 4

let private chunkFromDecodedBytes<'T when 'T: equality> (array: ZarrArray) (chunkRef: ZarrChunkRef) (decoded: byte[]) : Chunk<'T> =
    StackCore.Chunk.createBytes<'T>
        (zarrChunkIndex chunkRef)
        (zarrXyzTriple chunkRef.Origin)
        (zarrXyzTriple chunkRef.Shape)
        (zarrChunkBufferSize array)
        decoded

let private decodedBytesFromChunk<'T when 'T: equality> (chunk: Chunk<'T>) =
    StackCore.Chunk.bytes chunk

let private decodedByteChunk (array: ZarrArray) (chunkRef: ZarrChunkRef) (decoded: byte[]) : Chunk<byte> =
    StackCore.Chunk.createBytes<byte>
        (zarrChunkIndex chunkRef)
        (zarrXyzTriple chunkRef.Origin)
        (zarrXyzTriple chunkRef.Shape)
        (zarrChunkBufferSize array)
        decoded

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

let private createOmeZarrWriter output dataType (shape: Shape) =
    let descriptor =
        BioImageDescriptor(
            int shape.Width,
            int shape.Height,
            ZCT(int shape.Depth, 1, 1),
            Name = "benchmark",
            DataType = dataType,
            ChunkX = 256,
            ChunkY = 256,
            ChunkZ = 16,
            ChunkC = 1,
            ChunkT = 1,
            PhysicalSizeX = 1.0,
            PhysicalSizeY = 1.0,
            PhysicalSizeZ = 1.0)

    OmeZarrWriter.CreateAsync(output, descriptor, Threading.CancellationToken.None)
    |> runTask

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
    let inputData = StackCore.Chunk.bytes input
    let threshold = byte (Math.Clamp(Math.Ceiling(thresholdValue), 0.0, 255.0))
    let thresholdVector = Vector<byte>(threshold)
    let oneVector = Vector<byte>(1uy)
    let width = Vector<byte>.Count
    let mutable i = 0
    let vectorLimit = inputData.Length - (inputData.Length % width)
    while i < vectorLimit do
        let values = Vector<byte>(inputData, i)
        let mask = Vector.GreaterThanOrEqual(values, thresholdVector)
        Vector.BitwiseAnd(mask, oneVector).CopyTo(output, i)
        i <- i + width

    while i < inputData.Length do
        output[i] <- if inputData[i] >= threshold then 1uy else 0uy
        i <- i + 1
    inputData.Length

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
    let output = ArrayPool<byte>.Shared.Rent(outputLength)
    let outputChunk written =
        if written <> outputLength then
            failwith $"Threshold wrote {written} bytes, expected {outputLength}."
        StackCore.Chunk.createOwnedBytes<byte>
            (zarrChunkIndex chunkRef)
            (zarrXyzTriple chunkRef.Origin)
            (zarrXyzTriple chunkRef.Shape)
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
    let output = ArrayPool<byte>.Shared.Rent(outputLength)
    let written =
        match pixelType with
        | UInt8 -> thresholdUInt8DecodedBytesSimdInto thresholdValue decoded output
        | UInt16 -> thresholdUInt16DecodedBytesInto thresholdValue decoded output
        | Float32 -> thresholdFloat32DecodedBytesInto thresholdValue decoded output
    if written <> outputLength then
        failwith $"Raw threshold wrote {written} bytes, expected {outputLength}."
    output, outputLength

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
                outputArray.WriteChunkDecodedAsync(outputChunk, StackCore.Chunk.memory thresholded, Threading.CancellationToken.None)
                |> runUnitTask
            finally
                StackCore.Chunk.release thresholded
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

let private runZarrChunkCopyTyped<'T when 'T: equality> shape input output availableMemory =
    ensureCleanDirectory output
    let src = benchmarkSource availableMemory
    src
    |> readZarrSlabStacked<'T> input 0 0 0 0 0
    |> writeZarrSlab output 256u 256u 1.0 1.0 1.0 0
    |> sink
    0

let private runZarrChunkCopy opts =
    let pixelType = require "pixel-type" opts |> parsePixelType
    let shape = require "shape" opts |> parseShape
    let input = require "input" opts
    let output = require "output" opts
    let availableMemory = optional "available-memory" (string (1024UL * 1024UL * 1024UL * 1024UL)) opts |> UInt64.Parse
    let stopwatch = Stopwatch.StartNew()
    let exitCode =
        match pixelType with
        | UInt8 -> runZarrChunkCopyTyped<uint8> shape input output availableMemory
        | UInt16 -> runZarrChunkCopyTyped<uint16> shape input output availableMemory
        | Float32 -> runZarrChunkCopyTyped<float32> shape input output availableMemory
    stopwatch.Stop()
    writeInternalSeconds stopwatch.Elapsed
    exitCode

let private runZarrReadOnlyTyped<'T when 'T: equality> input availableMemory =
    let src = benchmarkSource availableMemory
    src
    |> readZarrSlab<'T> input 0 0 0 0 0
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
        | Float32 -> runZarrReadOnlyTyped<float32> input availableMemory
    stopwatch.Stop()
    writeInternalSeconds stopwatch.Elapsed
    exitCode

let private runZarrWriteOnlyTyped<'T when 'T: equality> (shape: Shape) output availableMemory =
    ensureCleanDirectory output
    let src = benchmarkSource availableMemory
    src
    |> zero<'T> shape.Width shape.Height shape.Depth
    >=> writeZarr output "benchmark" shape.Depth 256u 256u 16u 1.0 1.0 1.0 0
    |> sink
    0

let private runZarrWriteOnly opts =
    let pixelType = require "pixel-type" opts |> parsePixelType
    let shape = require "shape" opts |> parseShape
    let output = require "output" opts
    let availableMemory = optional "available-memory" (string (1024UL * 1024UL * 1024UL * 1024UL)) opts |> UInt64.Parse
    let stopwatch = Stopwatch.StartNew()
    let exitCode =
        match pixelType with
        | UInt8 -> runZarrWriteOnlyTyped<uint8> shape output availableMemory
        | UInt16 -> runZarrWriteOnlyTyped<uint16> shape output availableMemory
        | Float32 -> runZarrWriteOnlyTyped<float32> shape output availableMemory
    stopwatch.Stop()
    writeInternalSeconds stopwatch.Elapsed
    exitCode

let private runConnectedComponents input output windowSize availableMemory =
    ensureCleanDirectory output
    let window = max 1u windowSize
    let _, _, depth = getStackSize input ".tiff"
    let src = benchmarkSource availableMemory

    if depth <= window then
        src
        |> read<uint8> input ".tiff"
        >=> threshold 128.0 infinity
        >=> connectedComponentsLabels (Some depth)
        >=> cast<uint64, uint8>
        >=> writeSlabSlices output ".tiff" 1u
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
                >=> makeConnectedComponentLabelTranslationTable (Some window)
                |> drain

            src
            |> readRange<uint64> 0u 1 (depth - 1u) tmp tmpSuffix
            >=> updateConnectedComponents (Some window) table
            >=> cast<uint64, uint8>
            >=> writeSlabSlices output ".tiff" 1u
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
        | "convolve", UInt8 -> runConvolveTyped<uint8> input output kernelSize availableMemory
        | "convolve", UInt16 -> runConvolveTyped<uint16> input output kernelSize availableMemory
        | "convolve", Float32 -> runConvolveTyped<float32> input output kernelSize availableMemory
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
    let stopwatch = Stopwatch.StartNew()
    let exitCode =
        match operation, pixelType with
        | "connectedComponents", UInt8 -> runArrayPoolConnectedComponents input output thresholdValue
        | "connectedComponents", _ -> failwith "ArrayPool connectedComponents benchmark is currently defined for UInt8 masks only"
        | _, UInt8 -> runArrayPoolTyped<uint8> operation input output thresholdValue
        | _, UInt16 -> runArrayPoolTyped<uint16> operation input output thresholdValue
        | _, Float32 -> runArrayPoolTyped<float32> operation input output thresholdValue
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
    let stopwatch = Stopwatch.StartNew()
    let exitCode =
        match operation, pixelType with
        | "connectedComponents", _ -> failwith "Slice ArrayPool backend is intended for copy/threshold allocation experiments, not connected components."
        | _, UInt8 -> runArrayPoolSliceTyped<uint8> operation input output thresholdValue
        | _, UInt16 -> runArrayPoolSliceTyped<uint16> operation input output thresholdValue
        | _, Float32 -> runArrayPoolSliceTyped<float32> operation input output thresholdValue
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
    let stopwatch = Stopwatch.StartNew()
    let exitCode =
        match operation, pixelType with
        | "connectedComponents", _ -> failwith "Reusable slice ArrayPool backend is intended for copy/threshold allocation experiments, not connected components."
        | _, UInt8 -> runArrayPoolSliceReuseTyped<uint8> operation input output thresholdValue
        | _, UInt16 -> runArrayPoolSliceReuseTyped<uint16> operation input output thresholdValue
        | _, Float32 -> runArrayPoolSliceReuseTyped<float32> operation input output thresholdValue
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

    let stopwatch = Stopwatch.StartNew()
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

    let stopwatch = Stopwatch.StartNew()
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

[<EntryPoint>]
let main args =
    try
        match args with
        | [| |] -> usage ()
        | _ when args[0] = "generate" -> args[1..] |> parseArgs |> generate
        | _ when args[0] = "run" -> args[1..] |> parseArgs |> run
        | _ when args[0] = "run-zarr" -> args[1..] |> parseArgs |> runZarr
        | _ when args[0] = "run-zarr-direct-copy" -> args[1..] |> parseArgs |> runZarrDirectCopy
        | _ when args[0] = "run-zarr-direct-threshold" -> args[1..] |> parseArgs |> runZarrDirectThreshold
        | _ when args[0] = "run-zarr-direct-threshold-raw" -> args[1..] |> parseArgs |> runZarrDirectThresholdRaw
        | _ when args[0] = "run-zarr-chunk-copy" -> args[1..] |> parseArgs |> runZarrChunkCopy
        | _ when args[0] = "run-zarr-readonly" -> args[1..] |> parseArgs |> runZarrReadOnly
        | _ when args[0] = "run-zarr-writeonly" -> args[1..] |> parseArgs |> runZarrWriteOnly
        | _ when args[0] = "run-arraypool" -> args[1..] |> parseArgs |> runArrayPool
        | _ when args[0] = "run-arraypool-slice" -> args[1..] |> parseArgs |> runArrayPoolSlice
        | _ when args[0] = "run-arraypool-slice-reuse" -> args[1..] |> parseArgs |> runArrayPoolSliceReuse
        | _ when args[0] = "run-byte-slice-reuse" -> args[1..] |> parseArgs |> runByteSliceReuse
        | _ when args[0] = "run-byte-float32-slice-reuse" -> args[1..] |> parseArgs |> runByteFloat32SliceReuse
        | _ -> usage ()
    with ex ->
        fail ex.Message
