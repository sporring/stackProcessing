module StackIO

open SlimPipeline // Core processing model
open System
open System.Buffers
open System.Collections.Generic
open System.IO
open System.Reflection
open System.Runtime.InteropServices
open System.Text.Json
open System.Text.Json.Nodes
open System.Text.RegularExpressions
open System.Threading
open System.Threading.Tasks
open BitMiracle.LibTiff.Classic
open FSharp.Control
open StackCore
open PureHDF
open PureHDF.Selections
open ZarrNET.Core
open ZarrNET.Core.Nodes
open ZarrNET.Core.OmeZarr.Coordinates
open ZarrNET.Core.Zarr
open ZarrNET.Core.Zarr.Store

module ChunkKernel = ChunkCore.ChunkFunctions

type FileInfo =
    { dimensions: uint
      size: uint64 list
      componentType: string
      numberOfComponents: uint }

type ChunkInfo = { chunks: int list; size: uint64 list; topLeftInfo: FileInfo}

type ImageInfo =
    { format: string
      dimensions: uint
      size: uint64 list
      componentType: string
      numberOfComponents: uint
      chunks: int list }

let private inputValue input =
    input |> SingleOrPair.sum |> SingleOrPair.fst

let private imageBytes<'T> nPixels =
    StackProcessingCost.imageBytes<'T> nPixels

let private friendlyScalarTypeName (t: Type) =
    if t = typeof<uint8> then "UInt8"
    elif t = typeof<int8> then "Int8"
    elif t = typeof<uint16> then "UInt16"
    elif t = typeof<int16> then "Int16"
    elif t = typeof<uint32> then "UInt32"
    elif t = typeof<int32> then "Int32"
    elif t = typeof<uint64> then "UInt64"
    elif t = typeof<int64> then "Int64"
    elif t = typeof<float32> then "Float32"
    elif t = typeof<float> then "Float64"
    elif t = typeof<System.Numerics.Complex> then "ComplexFloat64"
    else t.Name

let private imageIoCost<'T> kind evaluation calibrationKey bytes ops : StageTimeCostModel =
    StackProcessingCost.imageIoCost<'T> kind evaluation calibrationKey bytes ops

let private fixedImageOperatorTimeCost<'T> operator evaluation voxels fallback =
    let pixelType = StackProcessingCost.pixelTypeName<'T>
    let context _ =
        StackProcessingCost.Fitting.OperatorEstimateContext.create
            operator
            (Some pixelType)
            (Some voxels)
            (Some(StackProcessingCost.imageBytes<'T> voxels))
            None
            None
            None
            None

    StackProcessingCost.Fitting.OperatorCostRuntime.timeCostModel evaluation context fallback

let private suffixCostLabel (suffix: string) =
    if String.IsNullOrWhiteSpace suffix then
        "unknown"
    else
        suffix.Trim().TrimStart('.').ToLowerInvariant()

let private imageStackOperatorName operator suffix =
    $"{operator}.{suffixCostLabel suffix}"

let private fixedImageStackOperatorTimeCost<'T> operator evaluation suffix voxels fallback =
    let stackOperator = imageStackOperatorName operator suffix
    let pixelType = StackProcessingCost.pixelTypeName<'T>
    let context _ =
        StackProcessingCost.Fitting.OperatorEstimateContext.create
            stackOperator
            (Some pixelType)
            (Some voxels)
            (Some(StackProcessingCost.imageBytes<'T> voxels))
            None
            None
            None
            None

    StackProcessingCost.Fitting.OperatorCostRuntime.timeCostModel evaluation context fallback

let private imageStackOperatorTimeCost<'T> operator evaluation suffix fallback =
    let stackOperator = imageStackOperatorName operator suffix
    let pixelType = StackProcessingCost.pixelTypeName<'T>
    let context input =
        let voxels = inputValue input
        StackProcessingCost.Fitting.OperatorEstimateContext.create
            stackOperator
            (Some pixelType)
            (Some voxels)
            (Some(StackProcessingCost.imageBytes<'T> voxels))
            None
            None
            None
            None

    StackProcessingCost.Fitting.OperatorCostRuntime.timeCostModel evaluation context fallback

let private withCostModel costModel stage =
    StackProcessingCost.withCostModel costModel stage

let private cleanStage name cleanup =
    { identityStage name with Cleaning = [ cleanup ] }

let private isTiffStackSuffix (suffix: string) =
    let trimmed = suffix.Trim()
    let normalized =
        if trimmed.StartsWith(".", StringComparison.Ordinal) then
            trimmed
        else
            "." + trimmed

    String.Equals(normalized, ".tif", StringComparison.OrdinalIgnoreCase)
    || String.Equals(normalized, ".tiff", StringComparison.OrdinalIgnoreCase)

let private canReadDirectTiffStack<'T> suffix =
    let t = typeof<'T>
    isTiffStackSuffix suffix
    && (t = typeof<uint8>
        || t = typeof<int8>
        || t = typeof<uint16>
        || t = typeof<int16>
        || t = typeof<uint32>
        || t = typeof<int32>
        || t = typeof<float32>
        || t = typeof<float>)

let private canWriteDirectTiffStack<'T> suffix =
    let t = typeof<'T>
    isTiffStackSuffix suffix
    && (t = typeof<uint8>
        || t = typeof<int8>
        || t = typeof<uint16>
        || t = typeof<int16>
        || t = typeof<uint32>
        || t = typeof<int32>
        || t = typeof<float32>
        || t = typeof<float>)

let private tiffPixelLayout<'T> () =
    let t = typeof<'T>
    if t = typeof<uint8> then 8, 1us, 1
    elif t = typeof<int8> then 8, 2us, 1
    elif t = typeof<uint16> then 16, 1us, 2
    elif t = typeof<int16> then 16, 2us, 2
    elif t = typeof<uint32> then 32, 1us, 4
    elif t = typeof<int32> then 32, 2us, 4
    elif t = typeof<float32> then 32, 3us, 4
    elif t = typeof<float> then 64, 3us, 8
    else
        invalidArg "T" $"TIFF scalar IO currently supports UInt8, Int8, UInt16, Int16, UInt32, Int32, Float32, and Float64 chunks; got {t.Name}."

[<Literal>]
let private TiffSampleFormatUInt = 1us

[<Literal>]
let private TiffSampleFormatInt = 2us

[<Literal>]
let private TiffSampleFormatIeeeFp = 3us

type private TiffPageInfo =
    { Width: uint32
      Height: uint32
      RowsPerStrip: uint32
      Strips: uint32
      BitsPerSample: uint16
      SampleFormat: uint16
      SamplesPerPixel: uint16
      PlanarConfig: uint16
      Compression: uint16
      IsTiled: int32
      IsByteSwapped: int32
      PageBytes: uint64
      RawPageBytes: uint64 }

module private TiffConstants =
    [<Literal>]
    let CompressionNone = 1us

    [<Literal>]
    let PlanarConfigContig = 1us

type TiffCompression =
    | None = 1
    | Lzw = 5
    | Deflate = 32946
    | PackBits = 32773

type TiffByteOrder =
    | Native = 0
    | Opposite = 1

type TiffWriteOptions =
    { Compression: TiffCompression
      ByteOrder: TiffByteOrder }

let defaultTiffWriteOptions =
    { Compression = TiffCompression.None
      ByteOrder = TiffByteOrder.Native }

type private TiffReadStrategy =
    | RawFastPath
    | GeneralBitMiraclePath

let private tiffReadStrategyName strategy =
    match strategy with
    | RawFastPath -> "raw-fast-path"
    | GeneralBitMiraclePath -> "general-bitmiracle-path"

let private tiffFieldInt (tiff: Tiff) tag fallback =
    let field = tiff.GetField(tag)
    if isNull field || field.Length = 0 then fallback else field[0].ToInt()

let private tiffFieldIntDefaulted (tiff: Tiff) tag fallback =
    let field = tiff.GetFieldDefaulted(tag)
    if isNull field || field.Length = 0 then fallback else field[0].ToInt()

let private openTiffOrFail fileName mode =
    let tiff = Tiff.Open(fileName, mode)
    if isNull tiff then
        invalidOp $"Could not open TIFF file '{fileName}' with mode '{mode}'."
    tiff

let private setTiffDirectoryOrFail (tiff: Tiff) fileName pageIndex =
    if pageIndex < 0 || pageIndex > int Int16.MaxValue then
        invalidArg "pageIndex" $"TIFF page index {pageIndex} is outside the supported BitMiracle directory range."
    if pageIndex <> 0 && not (tiff.SetDirectory(int16 pageIndex)) then
        invalidOp $"Could not select TIFF page {pageIndex} in '{fileName}'."

let private inspectOpenTiffPage fileName pageIndex (tiff: Tiff) =
    setTiffDirectoryOrFail tiff fileName pageIndex

    let width = uint32 (tiffFieldInt tiff TiffTag.IMAGEWIDTH 0)
    let height = uint32 (tiffFieldInt tiff TiffTag.IMAGELENGTH 0)
    let bitsPerSample = uint16 (tiffFieldIntDefaulted tiff TiffTag.BITSPERSAMPLE 1)
    let samplesPerPixel = uint16 (tiffFieldIntDefaulted tiff TiffTag.SAMPLESPERPIXEL 1)
    let sampleFormat = uint16 (tiffFieldIntDefaulted tiff TiffTag.SAMPLEFORMAT (int SampleFormat.UINT))
    let planarConfig = uint16 (tiffFieldIntDefaulted tiff TiffTag.PLANARCONFIG (int PlanarConfig.CONTIG))
    let compression = uint16 (tiffFieldIntDefaulted tiff TiffTag.COMPRESSION (int Compression.NONE))
    let rowsPerStrip = uint32 (tiffFieldIntDefaulted tiff TiffTag.ROWSPERSTRIP (int height))

    if width = 0u || height = 0u || bitsPerSample = 0us || samplesPerPixel = 0us || bitsPerSample % 8us <> 0us then
        invalidOp $"TIFF page {pageIndex} in '{fileName}' has unsupported dimensions or sample layout."

    let bytesPerPixel = uint64 bitsPerSample / 8UL * uint64 samplesPerPixel
    let pageBytes = uint64 width * uint64 height * bytesPerPixel
    if pageBytes > uint64 Int32.MaxValue then
        invalidOp $"TIFF page {pageIndex} in '{fileName}' has {pageBytes} bytes, which exceeds the current Chunk byte[] limit."

    let strips = tiff.NumberOfStrips()
    if strips < 1 then
        invalidOp $"TIFF page {pageIndex} in '{fileName}' has no strips."

    let mutable rawPageBytes = 0UL
    for strip in 0 .. strips - 1 do
        let stripBytes = tiff.RawStripSize(strip)
        if stripBytes <= 0L then
            invalidOp $"TIFF page {pageIndex} in '{fileName}' has invalid raw strip {strip} byte count {stripBytes}."
        rawPageBytes <- rawPageBytes + uint64 stripBytes

    { Width = width
      Height = height
      RowsPerStrip = rowsPerStrip
      Strips = uint32 strips
      BitsPerSample = bitsPerSample
      SampleFormat = sampleFormat
      SamplesPerPixel = samplesPerPixel
      PlanarConfig = planarConfig
      Compression = compression
      IsTiled = if tiff.IsTiled() then 1 else 0
      IsByteSwapped = if tiff.IsByteSwapped() then 1 else 0
      PageBytes = pageBytes
      RawPageBytes = rawPageBytes }

let private tryBitMiracleTiffInfoPage fileName pageIndex =
    try
        use tiff = openTiffOrFail fileName "r"
        Some(inspectOpenTiffPage fileName pageIndex tiff)
    with
    | _ -> None

let private bitMiracleTiffInfoPageOrFail fileName pageIndex =
    use tiff = openTiffOrFail fileName "r"
    inspectOpenTiffPage fileName pageIndex tiff

let private tryBitMiracleTiffDirectoryCount fileName =
    try
        use tiff = openTiffOrFail fileName "r"
        let count = tiff.NumberOfDirectories()
        if count > 0s then Some(uint32 count) else None
    with
    | _ -> None

let private bitMiracleTiffFastPathLayout<'T> () =
    let bitsPerSample, sampleFormat, bytesPerSample = tiffPixelLayout<'T> ()
    let supported =
        typeof<'T> = typeof<uint8>
        || typeof<'T> = typeof<int8>
        || typeof<'T> = typeof<uint16>
        || typeof<'T> = typeof<int16>
        || typeof<'T> = typeof<uint32>
        || typeof<'T> = typeof<int32>
        || typeof<'T> = typeof<float32>
        || typeof<'T> = typeof<float>
    if supported then
        Some(uint16 bitsPerSample, sampleFormat, bytesPerSample)
    else
        None

let private isBitMiracleScalarTiffPage expectedWidth expectedHeight expectedBits expectedSampleFormat expectedBytesPerSample (info: TiffPageInfo) =
    info.Width = expectedWidth &&
    info.Height = expectedHeight &&
    info.BitsPerSample = expectedBits &&
    info.SampleFormat = expectedSampleFormat &&
    info.SamplesPerPixel = 1us &&
    info.PlanarConfig = TiffConstants.PlanarConfigContig &&
    info.Compression = TiffConstants.CompressionNone &&
    info.IsTiled = 0 &&
    info.IsByteSwapped = 0 &&
    info.PageBytes = info.RawPageBytes &&
    info.PageBytes = uint64 expectedWidth * uint64 expectedHeight * uint64 expectedBytesPerSample

let private isBitMiracleComplexTiffPage expectedWidth expectedHeight bitsPerSample bytesPerSample (info: TiffPageInfo) =
    info.Width = expectedWidth &&
    info.Height = expectedHeight &&
    info.BitsPerSample = bitsPerSample &&
    info.SampleFormat = TiffSampleFormatIeeeFp &&
    info.SamplesPerPixel = 2us &&
    info.PlanarConfig = TiffConstants.PlanarConfigContig &&
    info.Compression = TiffConstants.CompressionNone &&
    info.IsTiled = 0 &&
    info.IsByteSwapped = 0 &&
    info.PageBytes = info.RawPageBytes &&
    info.PageBytes = uint64 expectedWidth * uint64 expectedHeight * 2UL * uint64 bytesPerSample

let private isBitMiracleRgbTiffPage expectedWidth expectedHeight (info: TiffPageInfo) =
    info.Width = expectedWidth &&
    info.Height = expectedHeight &&
    info.BitsPerSample = 8us &&
    info.SampleFormat = TiffSampleFormatUInt &&
    info.SamplesPerPixel = 3us &&
    info.PlanarConfig = TiffConstants.PlanarConfigContig &&
    info.Compression = TiffConstants.CompressionNone &&
    info.IsTiled = 0 &&
    info.PageBytes = info.RawPageBytes &&
    info.PageBytes = uint64 expectedWidth * uint64 expectedHeight * 3UL

let private bitMiracleCompression compression =
    match compression with
    | TiffCompression.None -> Compression.NONE
    | TiffCompression.Lzw -> Compression.LZW
    | TiffCompression.Deflate -> Compression.DEFLATE
    | TiffCompression.PackBits -> Compression.PACKBITS
    | other -> invalidArg "compression" $"Unsupported TIFF compression option {other}."

let private bitMiracleWriteMode byteOrder =
    match byteOrder with
    | TiffByteOrder.Native ->
        if BitConverter.IsLittleEndian then "wl" else "wb"
    | TiffByteOrder.Opposite ->
        if BitConverter.IsLittleEndian then "wb" else "wl"
    | other ->
        invalidArg "byteOrder" $"Unsupported TIFF byte order option {other}."

let private bitMiracleSampleFormat sampleFormat =
    enum<SampleFormat> (int sampleFormat)

let private bitMiracleSetCommonFields (tiff: Tiff) width height bitsPerSample sampleFormat samplesPerPixel compression =
    tiff.SetField(TiffTag.IMAGEWIDTH, [| box (int width) |]) |> ignore
    tiff.SetField(TiffTag.IMAGELENGTH, [| box (int height) |]) |> ignore
    tiff.SetField(TiffTag.SAMPLESPERPIXEL, [| box (int samplesPerPixel) |]) |> ignore
    tiff.SetField(TiffTag.BITSPERSAMPLE, [| box (int bitsPerSample) |]) |> ignore
    tiff.SetField(TiffTag.SAMPLEFORMAT, [| box (bitMiracleSampleFormat sampleFormat) |]) |> ignore
    tiff.SetField(TiffTag.PHOTOMETRIC, [| box (if samplesPerPixel = 3us then Photometric.RGB else Photometric.MINISBLACK) |]) |> ignore
    tiff.SetField(TiffTag.PLANARCONFIG, [| box PlanarConfig.CONTIG |]) |> ignore
    tiff.SetField(TiffTag.ROWSPERSTRIP, [| box (int height) |]) |> ignore
    tiff.SetField(TiffTag.COMPRESSION, [| box compression |]) |> ignore

let private validateBitMiracleWriteBuffer pageBytes sliceOffsetBytes (chunk: Chunk<'T>) =
    if pageBytes > uint64 Int32.MaxValue ||
       sliceOffsetBytes < 0 ||
       chunk.ByteLength < sliceOffsetBytes + int pageBytes then
        invalidArg "chunk" $"Chunk byte length {chunk.ByteLength} is too small for TIFF page payload {pageBytes} at offset {sliceOffsetBytes}."

let private writeBitMiracleRawTiffPageFrom fileName width height bitsPerSample sampleFormat samplesPerPixel pageBytes sliceOffsetBytes (chunk: Chunk<'T>) =
    validateBitMiracleWriteBuffer pageBytes sliceOffsetBytes chunk
    use tiff = openTiffOrFail fileName (bitMiracleWriteMode TiffByteOrder.Native)
    bitMiracleSetCommonFields tiff width height bitsPerSample sampleFormat samplesPerPixel Compression.NONE
    let written = tiff.WriteRawStrip(0, chunk.Bytes, sliceOffsetBytes, int pageBytes)
    if written <> int pageBytes then
        invalidOp $"BitMiracle raw TIFF write wrote {written} bytes to '{fileName}', expected {pageBytes}."
    if not (tiff.WriteDirectory()) then
        invalidOp $"BitMiracle could not write TIFF directory to '{fileName}'."

let private writeBitMiracleEncodedTiffPageFrom options fileName width height bitsPerSample sampleFormat samplesPerPixel pageBytes sliceOffsetBytes (chunk: Chunk<'T>) =
    validateBitMiracleWriteBuffer pageBytes sliceOffsetBytes chunk
    use tiff = openTiffOrFail fileName (bitMiracleWriteMode options.ByteOrder)
    bitMiracleSetCommonFields tiff width height bitsPerSample sampleFormat samplesPerPixel (bitMiracleCompression options.Compression)
    let written = tiff.WriteEncodedStrip(0, chunk.Bytes, sliceOffsetBytes, int pageBytes)
    if written <> int pageBytes then
        invalidOp $"BitMiracle encoded TIFF write wrote {written} bytes to '{fileName}', expected {pageBytes}."
    if not (tiff.WriteDirectory()) then
        invalidOp $"BitMiracle could not write TIFF directory to '{fileName}'."

let private tiffDirectoryCount (filename: string) =
    match tryBitMiracleTiffDirectoryCount filename with
    | Some count -> count
    | None ->
        invalidOp $"Could not count TIFF directories in '{filename}' through BitMiracle. This TIFF layout is currently unsupported."

let private tiffBytesPerSample bitsPerSample sampleFormat =
    match sampleFormat, bitsPerSample with
    | TiffSampleFormatUInt, 8
    | TiffSampleFormatInt, 8 -> 1
    | TiffSampleFormatUInt, 16
    | TiffSampleFormatInt, 16 -> 2
    | TiffSampleFormatUInt, 32
    | TiffSampleFormatInt, 32
    | TiffSampleFormatIeeeFp, 32 -> 4
    | TiffSampleFormatIeeeFp, 64 -> 8
    | _ ->
        invalidOp $"TIFF scalar IO currently supports UInt8, Int8, UInt16, Int16, UInt32, Int32, Float32, and Float64 pages; got {bitsPerSample}-bit sample format {sampleFormat}."

let private tiffComponentType bitsPerSample sampleFormat =
    match sampleFormat, bitsPerSample with
    | TiffSampleFormatUInt, 8 -> "UInt8"
    | TiffSampleFormatInt, 8 -> "Int8"
    | TiffSampleFormatUInt, 16 -> "UInt16"
    | TiffSampleFormatInt, 16 -> "Int16"
    | TiffSampleFormatUInt, 32 -> "UInt32"
    | TiffSampleFormatInt, 32 -> "Int32"
    | TiffSampleFormatIeeeFp, 32 -> "Float32"
    | TiffSampleFormatIeeeFp, 64 -> "Float64"
    | _ ->
        invalidOp $"TIFF scalar IO currently supports UInt8, Int8, UInt16, Int16, UInt32, Int32, Float32, and Float64 pages; got {bitsPerSample}-bit sample format {sampleFormat}."

let getFileInfo (filename: string) : FileInfo =
    let extension = Path.GetExtension(filename).ToLowerInvariant()
    let isTiffVolume =
        extension = ".tif"
        || extension = ".tiff"
        || extension = ".btf"
        || extension = ".bigtiff"

    if not isTiffVolume then
        invalidArg "filename" $"getFileInfo currently supports TIFF/BigTIFF files only: {filename}"

    let info = bitMiracleTiffInfoPageOrFail filename 0
    let depth = tiffDirectoryCount filename |> uint64
    let dimensions = if depth > 1UL then 3u else 2u
    let size =
        if depth > 1UL then
            [ uint64 info.Width; uint64 info.Height; depth ]
        else
            [ uint64 info.Width; uint64 info.Height ]

    { dimensions = dimensions
      size = size
      componentType = tiffComponentType (int info.BitsPerSample) info.SampleFormat
      numberOfComponents = uint info.SamplesPerPixel }

let private validateTiffSamples samplesPerPixel =
    if samplesPerPixel <> 1 then
        invalidOp $"TIFF scalar IO expects one sample per pixel; got {samplesPerPixel} samples per pixel."

let private ensureDirectTiffChunkRead<'T> suffix =
    if not (canReadDirectTiffStack<'T> suffix) then
        invalidArg "suffix" $"readChunkSlices currently supports direct scalar TIFF stacks for UInt8, Int8, UInt16, Int16, UInt32, Int32, Float32, and Float64; got suffix '{suffix}' and type {typeof<'T>.Name}."

let private ensureDirectTiffChunkWrite<'T> suffix =
    if not (canWriteDirectTiffStack<'T> suffix) then
        invalidArg "suffix" $"writeChunkSlices currently supports direct scalar TIFF stack output for UInt8, Int8, UInt16, Int16, UInt32, Int32, and Float32; got suffix '{suffix}' and type {typeof<'T>.Name}."

let private castChunkStage<'S, 'T when 'S: equality and 'S: (new: unit -> 'S) and 'S: struct and 'S :> ValueType
                                  and 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> : Stage<Chunk<'S>, Chunk<'T>> =
    if typeof<'S> = typeof<'T> then
        unbox (box (identityStage "cast.identity"))
    elif typeof<'T> = typeof<float32> then
        unbox (box (ChunkFunctions.castToFloat32<'S>))
    elif typeof<'T> = typeof<uint8> then
        unbox (box (ChunkFunctions.castToUInt8<'S>))
    elif typeof<'S> = typeof<float32> then
        unbox (box (ChunkFunctions.castFromFloat32<'T>))
    else
        ChunkFunctions.castChunk<'S, 'T>

type private ChunkTiffReadPlan =
    { Width: uint
      Height: uint
      RowBytes: int
      Strategy: TiffReadStrategy
      ConvertUInt8ToFloat32: bool }

let private inspectChunkTiffSliceForRead<'T> fileName =
    let expectedBits, expectedFormat, expectedBytesPerSample = tiffPixelLayout<'T> ()
    let info = bitMiracleTiffInfoPageOrFail fileName 0
    let width = uint info.Width
    let height = uint info.Height
    if width = 0u || height = 0u then
        invalidOp $"TIFF slice '{fileName}' has invalid page dimensions {width}x{height}."
    validateTiffSamples (int info.SamplesPerPixel)

    let bitsPerSample = int info.BitsPerSample
    let sampleFormat = info.SampleFormat
    if bitsPerSample = expectedBits && sampleFormat = expectedFormat then
        let rowBytes = int width * expectedBytesPerSample
        let pageBytes = uint64 rowBytes * uint64 height
        let strategy =
            if info.IsTiled <> 0 || info.PlanarConfig <> TiffConstants.PlanarConfigContig then
                invalidOp $"TIFF slice '{fileName}' uses an unsupported tiled or planar layout for Chunk reading."
            elif info.PageBytes <> pageBytes then
                invalidOp $"TIFF slice '{fileName}' has page byte count {info.PageBytes}, expected {pageBytes}."
            elif info.Compression = TiffConstants.CompressionNone && info.RawPageBytes = pageBytes && info.IsByteSwapped = 0 then
                RawFastPath
            else
                GeneralBitMiraclePath

        { Width = width
          Height = height
          RowBytes = rowBytes
          Strategy = strategy
          ConvertUInt8ToFloat32 = false }
    elif typeof<'T> = typeof<float32> && bitsPerSample = 8 && sampleFormat = TiffSampleFormatUInt then
        if info.IsTiled <> 0 || info.PlanarConfig <> TiffConstants.PlanarConfigContig then
            invalidOp $"TIFF slice '{fileName}' uses an unsupported tiled or planar layout for UInt8-to-Float32 reading."
        { Width = width
          Height = height
          RowBytes = int width * sizeof<float32>
          Strategy = GeneralBitMiraclePath
          ConvertUInt8ToFloat32 = true }
    else
        invalidOp $"Input slice '{fileName}' does not match {typeof<'T>.Name}: got {bitsPerSample}-bit sample format {sampleFormat}."

let private validateBitMiracleReadBuffer fileName expectedBytes sliceOffsetBytes (chunk: Chunk<'T>) =
    if expectedBytes > uint64 Int32.MaxValue ||
       sliceOffsetBytes < 0 ||
       chunk.ByteLength < sliceOffsetBytes + int expectedBytes then
        invalidArg "chunk" $"Chunk byte length {chunk.ByteLength} is too small to read TIFF page '{fileName}' at offset {sliceOffsetBytes} with length {expectedBytes}."

let private readBitMiracleRawTiffPageIntoOffsetNoMetadata fileName pageIndex expectedBytes sliceOffsetBytes (chunk: Chunk<'T>) =
    validateBitMiracleReadBuffer fileName expectedBytes sliceOffsetBytes chunk
    use tiff = openTiffOrFail fileName "r"
    setTiffDirectoryOrFail tiff fileName pageIndex

    let mutable offset = sliceOffsetBytes
    let mutable bytesRead = 0UL
    let strips = tiff.NumberOfStrips()
    for strip in 0 .. strips - 1 do
        let stripBytes64 = tiff.RawStripSize(strip)
        if stripBytes64 <= 0L || stripBytes64 > int64 Int32.MaxValue then
            invalidOp $"BitMiracle raw read saw invalid strip {strip} byte count {stripBytes64} in '{fileName}'."
        let stripBytes = int stripBytes64
        if bytesRead + uint64 stripBytes > expectedBytes then
            invalidOp $"BitMiracle raw read for page {pageIndex} in '{fileName}' exceeded expected payload {expectedBytes} bytes."
        let got = tiff.ReadRawStrip(strip, chunk.Bytes, offset, stripBytes)
        if got <> stripBytes then
            invalidOp $"BitMiracle raw read got {got} bytes from strip {strip} in '{fileName}', expected {stripBytes}."
        offset <- offset + stripBytes
        bytesRead <- bytesRead + uint64 stripBytes

    if bytesRead <> expectedBytes then
        invalidOp $"BitMiracle raw read produced {bytesRead} bytes for page {pageIndex} in '{fileName}', expected {expectedBytes}."

let private readBitMiracleDecodedTiffPageIntoOffsetNoMetadata fileName pageIndex expectedBytes sliceOffsetBytes (chunk: Chunk<'T>) =
    validateBitMiracleReadBuffer fileName expectedBytes sliceOffsetBytes chunk
    use tiff = openTiffOrFail fileName "r"
    setTiffDirectoryOrFail tiff fileName pageIndex

    let stripBytes = tiff.StripSize()
    if stripBytes <= 0 then
        invalidOp $"BitMiracle decoded read saw invalid decoded strip size {stripBytes} in '{fileName}'."

    let mutable offset = sliceOffsetBytes
    let mutable bytesRead = 0UL
    let strips = tiff.NumberOfStrips()
    for strip in 0 .. strips - 1 do
        let remaining = int expectedBytes - int bytesRead
        let requested = min stripBytes remaining
        let got = tiff.ReadEncodedStrip(strip, chunk.Bytes, offset, requested)
        if got < 0 || got > requested then
            invalidOp $"BitMiracle decoded read got {got} bytes from strip {strip} in '{fileName}', expected at most {requested}."
        offset <- offset + got
        bytesRead <- bytesRead + uint64 got

    if bytesRead <> expectedBytes then
        invalidOp $"BitMiracle decoded read produced {bytesRead} bytes for page {pageIndex} in '{fileName}', expected {expectedBytes}."

let private readBitMiracleUInt8TiffPageAsFloat32IntoOffset fileName pageIndex expectedWidth expectedHeight sliceOffsetBytes (chunk: Chunk<float32>) =
    let widthI = int expectedWidth
    let heightI = int expectedHeight
    let sourceBytes = widthI * heightI
    let targetBytes = sourceBytes * sizeof<float32>
    if sliceOffsetBytes < 0 || chunk.ByteLength < sliceOffsetBytes + targetBytes then
        invalidArg "chunk" $"Chunk byte length {chunk.ByteLength} is too small to read UInt8 TIFF page '{fileName}' as Float32 at offset {sliceOffsetBytes} with length {targetBytes}."

    let scratch = ArrayPool<byte>.Shared.Rent(sourceBytes)
    try
        use tiff = openTiffOrFail fileName "r"
        setTiffDirectoryOrFail tiff fileName pageIndex
        let stripBytes = tiff.StripSize()
        if stripBytes <= 0 then
            invalidOp $"BitMiracle UInt8-to-Float32 read saw invalid decoded strip size {stripBytes} in '{fileName}'."

        let mutable offset = 0
        let strips = tiff.NumberOfStrips()
        for strip in 0 .. strips - 1 do
            let remaining = sourceBytes - offset
            let requested = min stripBytes remaining
            let got = tiff.ReadEncodedStrip(strip, scratch, offset, requested)
            if got < 0 || got > requested then
                invalidOp $"BitMiracle UInt8-to-Float32 read got {got} bytes from strip {strip} in '{fileName}', expected at most {requested}."
            offset <- offset + got

        if offset <> sourceBytes then
            invalidOp $"BitMiracle UInt8-to-Float32 read produced {offset} bytes for page {pageIndex} in '{fileName}', expected {sourceBytes}."

        let outputBytes = chunk.Bytes.AsSpan(sliceOffsetBytes, targetBytes)
        let output = MemoryMarshal.Cast<byte, float32>(outputBytes)
        ChunkKernel.castUInt8SpanToFloat32 (ReadOnlySpan<byte>(scratch, 0, sourceBytes)) output
    finally
        ArrayPool<byte>.Shared.Return(scratch)

let private scalarTiffReaderForPlan<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (plan: ChunkTiffReadPlan)
    : string -> int -> Chunk<'T> -> int -> unit =

    if plan.ConvertUInt8ToFloat32 then
        fun fileName pageIndex chunk sliceOffsetBytes ->
            let floatChunk = box chunk :?> Chunk<float32>
            readBitMiracleUInt8TiffPageAsFloat32IntoOffset fileName pageIndex plan.Width plan.Height sliceOffsetBytes floatChunk
    else
        let sliceBytes = uint64 plan.RowBytes * uint64 plan.Height
        match plan.Strategy with
        | RawFastPath ->
            fun fileName pageIndex chunk sliceOffsetBytes ->
                readBitMiracleRawTiffPageIntoOffsetNoMetadata fileName pageIndex sliceBytes sliceOffsetBytes chunk
        | GeneralBitMiraclePath ->
            fun fileName pageIndex chunk sliceOffsetBytes ->
                readBitMiracleDecodedTiffPageIntoOffsetNoMetadata fileName pageIndex sliceBytes sliceOffsetBytes chunk

type private ComplexTiffReadPlan =
    { LogicalWidth: uint
      Height: uint
      RowBytes: int
      Strategy: TiffReadStrategy }

let private inspectComplexTiffSlice bytesPerSample typeLabel fileName =
    let info = bitMiracleTiffInfoPageOrFail fileName 0
    let width = uint info.Width
    let height = uint info.Height
    if width = 0u || height = 0u then
        invalidOp $"{typeLabel} TIFF slice '{fileName}' has invalid page dimensions {width}x{height}."
    if info.Width <> uint32 width ||
       info.Height <> uint32 height ||
       info.BitsPerSample <> uint16 (bytesPerSample * 8) ||
       info.SampleFormat <> TiffSampleFormatIeeeFp ||
       info.SamplesPerPixel <> 2us ||
       info.PlanarConfig <> TiffConstants.PlanarConfigContig ||
       info.IsTiled <> 0 then
        invalidOp $"{typeLabel} TIFF slices must be contiguous two-sample Float{bytesPerSample * 8} pixels, got bits={info.BitsPerSample}, format={info.SampleFormat}, samples={info.SamplesPerPixel}, planar={info.PlanarConfig} in '{fileName}'."

    let rowBytes = int width * 2 * bytesPerSample
    let pageBytes = uint64 rowBytes * uint64 height
    if info.PageBytes <> pageBytes then
        invalidOp $"{typeLabel} TIFF slice '{fileName}' has page byte count {info.PageBytes}, expected {pageBytes}."
    let strategy =
        if info.Compression = TiffConstants.CompressionNone && info.RawPageBytes = pageBytes && info.IsByteSwapped = 0 then
            RawFastPath
        else
            GeneralBitMiraclePath
    { LogicalWidth = width
      Height = height
      RowBytes = rowBytes
      Strategy = strategy }

let private inspectComplex64TiffSlice fileName =
    inspectComplexTiffSlice sizeof<float32> "Complex64" fileName

let private inspectComplex128TiffSlice fileName =
    inspectComplexTiffSlice sizeof<float> "Complex128" fileName

let private readComplexTiffSliceIntoOffset
    strategy
    fileName
    pageIndex
    (chunk: Chunk<'T>)
    bytesPerSample
    typeLabel
    expectedWidth
    expectedHeight
    sliceOffsetBytes
    =

    let pageBytes = uint64 expectedWidth * uint64 expectedHeight * 2UL * uint64 bytesPerSample

    match strategy with
    | RawFastPath ->
        readBitMiracleRawTiffPageIntoOffsetNoMetadata fileName pageIndex pageBytes sliceOffsetBytes chunk
    | GeneralBitMiraclePath ->
        readBitMiracleDecodedTiffPageIntoOffsetNoMetadata fileName pageIndex pageBytes sliceOffsetBytes chunk

let private readComplex64TiffSliceIntoOffset strategy fileName pageIndex (chunk: Chunk<float32>) expectedWidth expectedHeight sliceOffsetBytes =
    readComplexTiffSliceIntoOffset strategy fileName pageIndex chunk sizeof<float32> "Complex64" expectedWidth expectedHeight sliceOffsetBytes

let private readComplex128TiffSliceIntoOffset strategy fileName pageIndex (chunk: Chunk<float>) expectedWidth expectedHeight sliceOffsetBytes =
    readComplexTiffSliceIntoOffset strategy fileName pageIndex chunk sizeof<float> "Complex128" expectedWidth expectedHeight sliceOffsetBytes

let private writeComplex64TiffSliceFromOffset fileName (chunk: Chunk<float32>) logicalWidth height sliceOffsetBytes =
    let pageBytes = uint64 logicalWidth * uint64 height * 2UL * uint64 sizeof<float32>
    writeBitMiracleRawTiffPageFrom fileName logicalWidth height 32us TiffSampleFormatIeeeFp 2us pageBytes sliceOffsetBytes chunk

let private writeComplex128TiffSliceFromOffset fileName (chunk: Chunk<float>) logicalWidth height sliceOffsetBytes =
    let pageBytes = uint64 logicalWidth * uint64 height * 2UL * uint64 sizeof<float>
    writeBitMiracleRawTiffPageFrom fileName logicalWidth height 64us TiffSampleFormatIeeeFp 2us pageBytes sliceOffsetBytes chunk

let private writeChunkTiffSliceFromOffset<'T when 'T: equality> fileName (chunk: Chunk<'T>) width height sliceOffsetBytes =
    match bitMiracleTiffFastPathLayout<'T> () with
    | None -> invalidArg "T" $"BitMiracle TIFF raw write does not support chunk type {typeof<'T>.Name}."
    | Some(bitsPerSample, sampleFormat, bytesPerSample) ->
        let pageBytes = uint64 width * uint64 height * uint64 bytesPerSample
        writeBitMiracleRawTiffPageFrom fileName width height bitsPerSample sampleFormat 1us pageBytes sliceOffsetBytes chunk

let private writeEncodedChunkTiffSliceFromOffset<'T when 'T: equality> options fileName (chunk: Chunk<'T>) width height sliceOffsetBytes =
    match bitMiracleTiffFastPathLayout<'T> () with
    | None -> invalidArg "T" $"BitMiracle TIFF encoded write does not support chunk type {typeof<'T>.Name}."
    | Some(bitsPerSample, sampleFormat, bytesPerSample) ->
        let pageBytes = uint64 width * uint64 height * uint64 bytesPerSample
        writeBitMiracleEncodedTiffPageFrom options fileName width height bitsPerSample sampleFormat 1us pageBytes sliceOffsetBytes chunk

let private scalarTiffWriterForOptions<'T when 'T: equality>
    options
    : string -> Chunk<'T> -> uint64 -> uint64 -> int -> unit =

    if options = defaultTiffWriteOptions then
        writeChunkTiffSliceFromOffset<'T>
    else
        writeEncodedChunkTiffSliceFromOffset<'T> options

let private writeChunkTiffSlice<'T when 'T: equality> fileName (chunk: Chunk<'T>) =
    let width, height, depth = chunk.Size
    if depth <> 1UL then
        invalidArg "chunk" $"writeChunkSlices expects 2D slice chunks with depth 1, got {chunk.Size}."

    writeChunkTiffSliceFromOffset<'T> fileName chunk width height 0

let private writeChunkTiffFile<'T when 'T: equality> fileName (_chunk: Chunk<'T>) =
    invalidOp $"Writing a multi-page TIFF chunk file is currently unsupported in the BitMiracle TIFF path. Use split slice TIFF output for '{fileName}'."

type private ColorTiffReadPlan =
    { Width: uint
      Height: uint
      RowBytes: int
      Strategy: TiffReadStrategy }

let private inspectColorChunkTiffSlice fileName =
    let info = bitMiracleTiffInfoPageOrFail fileName 0
    let width = uint info.Width
    let height = uint info.Height
    if width = 0u || height = 0u then
        invalidOp $"RGB TIFF slice '{fileName}' has invalid page dimensions {width}x{height}."
    if info.BitsPerSample <> 8us ||
       info.SampleFormat <> TiffSampleFormatUInt ||
       info.SamplesPerPixel <> 3us ||
       info.PlanarConfig <> TiffConstants.PlanarConfigContig ||
       info.IsTiled <> 0 then
        invalidOp $"RGB TIFF slices must be contiguous 8-bit unsigned RGB, got bits={info.BitsPerSample}, format={info.SampleFormat}, samples={info.SamplesPerPixel}, planar={info.PlanarConfig} in '{fileName}'."

    let rowBytes = int width * 3
    let pageBytes = uint64 rowBytes * uint64 height
    if info.PageBytes <> pageBytes then
        invalidOp $"RGB TIFF slice '{fileName}' has page byte count {info.PageBytes}, expected {pageBytes}."
    let strategy =
        if info.Compression = TiffConstants.CompressionNone && info.RawPageBytes = pageBytes && info.IsByteSwapped = 0 then
            RawFastPath
        else
            GeneralBitMiraclePath
    { Width = width
      Height = height
      RowBytes = rowBytes
      Strategy = strategy }

let private readColorChunkTiffSliceByPlan (plan: ColorTiffReadPlan) fileName : VectorChunk<uint8> =
    let width = plan.Width
    let height = plan.Height
    let rowBytes = plan.RowBytes
    let chunk = Chunk.create<uint8> (uint64 width * 3UL, uint64 height, 1UL)
    let vector: VectorChunk<uint8> =
        { SpatialSize = (uint64 width, uint64 height, 1UL)
          Components = 3u
          Chunk = chunk }
    try
        match plan.Strategy with
        | RawFastPath ->
            readBitMiracleRawTiffPageIntoOffsetNoMetadata fileName 0 (uint64 rowBytes * uint64 height) 0 chunk
        | GeneralBitMiraclePath ->
            readBitMiracleDecodedTiffPageIntoOffsetNoMetadata fileName 0 (uint64 rowBytes * uint64 height) 0 chunk
        vector
    with
    | _ ->
        Chunk.decRef chunk
        reraise()

let private writeColorChunkTiffSlice fileName (vector: VectorChunk<uint8>) =
    if vector.Components <> 3u then
        invalidArg "vector" $"writeColorChunkSlices expects 3-component RGB vectors, got {vector.Components} components."
    let width, height, depth = vector.SpatialSize
    if depth <> 1UL then
        invalidArg "vector" $"writeColorChunkSlices expects 2D color slice chunks with depth 1, got {vector.SpatialSize}."
    if vector.Chunk.Size <> (width * 3UL, height, 1UL) then
        invalidArg "vector" $"RGB vector chunk storage size {vector.Chunk.Size} does not match spatial size {vector.SpatialSize}."

    let rowBytes = int width * 3
    if vector.Chunk.ByteLength < rowBytes * int height then
        invalidArg "vector" $"RGB vector chunk byte length {vector.Chunk.ByteLength} is smaller than the TIFF page payload {rowBytes * int height}."

    let pageBytes = uint64 rowBytes * uint64 height
    writeBitMiracleRawTiffPageFrom fileName width height 8us TiffSampleFormatUInt 3us pageBytes 0 vector.Chunk

let private runTask (task: Task<'T>) : 'T =
    task.GetAwaiter().GetResult()

let private runUnitTask (task: Task) : unit =
    task.GetAwaiter().GetResult()

let private collectZarrChunks (array: ZarrArray) =
    let chunks = ResizeArray<ZarrChunkRef>()
    let enumerator = array.EnumerateChunksAsync(CancellationToken.None).GetAsyncEnumerator()
    try
        let mutable more = true
        while more do
            more <- enumerator.MoveNextAsync().AsTask() |> runTask
            if more then
                chunks.Add(enumerator.Current)
    finally
        enumerator.DisposeAsync().AsTask() |> runUnitTask
    chunks.ToArray()

let private deleteZarrNetDebugLogs () =
    let candidates =
        [ Path.Combine(Directory.GetCurrentDirectory(), @"C:\Users\Public\biolog.txt")
          Path.Combine(AppContext.BaseDirectory, @"C:\Users\Public\biolog.txt")
          Path.Combine(Directory.GetCurrentDirectory(), "log.txt")
          Path.Combine(AppContext.BaseDirectory, "log.txt") ]
        |> List.distinct

    for path in candidates do
        try
            if File.Exists path then
                File.Delete path
        with _ ->
            ()

let private suppressZarrNetDebugLogging () =
    let flags = BindingFlags.Static ||| BindingFlags.NonPublic

    let setField (typeName: string) (fieldName: string) (value: obj) =
        try
            match Type.GetType(typeName, throwOnError = false) with
            | null -> ()
            | typ ->
                match typ.GetField(fieldName, flags) with
                | null -> ()
                | field -> field.SetValue(null, value)
        with _ ->
            ()

    deleteZarrNetDebugLogs ()
    setField "ZarrNET.Core.Zarr.ZarrArray, ZarrNET" "s_writeDebugCount" (box Int32.MaxValue)
    setField "ZarrNET.Core.Zarr.ZarrArray, ZarrNET" "s_readDebugCount" (box Int32.MaxValue)
    setField "ZarrNET.Core.Zarr.Store.LocalFileSystemStore, ZarrNET" "s_debugCount" (box Int32.MaxValue)

let private jsonOptions =
    JsonSerializerOptions(WriteIndented = true)

let private jsonObjectProperty (name: string) (parent: JsonObject) =
    match parent[name] with
    | null ->
        let child = JsonObject()
        parent[name] <- child
        child
    | node -> node.AsObject()

let private jsonInt64Array (values: int list) =
    let array = JsonArray()
    for value in values do
        array.Add(JsonValue.Create<int64>(int64 value))
    array

let private tagSpectralZarrMetadataFile fileName (realWidth: int) (height: int) (depth: int) (packedAxis: int) (packedComplexWidth: int) =
    if File.Exists fileName then
        let root = JsonNode.Parse(File.ReadAllText fileName).AsObject()
        let attributes = jsonObjectProperty "attributes" root
        let stackProcessing = JsonObject()
        stackProcessing["complex_storage"] <- JsonValue.Create<string>("complex64_interleaved")
        stackProcessing["spectral_layout"] <- JsonValue.Create<string>("hermitian_packed")
        stackProcessing["packed_axis"] <- JsonValue.Create<int>(packedAxis)
        stackProcessing["real_size"] <- jsonInt64Array [ realWidth; height; depth ]
        stackProcessing["stored_complex_size"] <- jsonInt64Array [ packedComplexWidth; height; depth ]
        attributes["stackprocessing"] <- stackProcessing
        File.WriteAllText(fileName, root.ToJsonString(jsonOptions))

let private tagSpectralZarrMetadata outputPath realWidth height depth packedAxis packedComplexWidth =
    tagSpectralZarrMetadataFile (Path.Combine(outputPath, "zarr.json")) realWidth height depth packedAxis packedComplexWidth
    tagSpectralZarrMetadataFile (Path.Combine(outputPath, "0", "zarr.json")) realWidth height depth packedAxis packedComplexWidth

let private tryReadSpectralZarrMetadata outputPath =
    let readFrom fileName =
        if not (File.Exists fileName) then
            None
        else
            try
                let root = JsonNode.Parse(File.ReadAllText fileName).AsObject()
                match root["attributes"] with
                | null -> None
                | attributes ->
                    match attributes.AsObject()["stackprocessing"] with
                    | null -> None
                    | metadata ->
                        let metadata = metadata.AsObject()
                        let layout = metadata["spectral_layout"].GetValue<string>()
                        if not (String.Equals(layout, "hermitian_packed", StringComparison.OrdinalIgnoreCase)) then
                            None
                        else
                            let packedAxis = metadata["packed_axis"].GetValue<int>()
                            let realSize = metadata["real_size"].AsArray()
                            if realSize.Count <> 3 then
                                None
                            else
                                Some(
                                    packedAxis,
                                    uint64 (realSize[0].GetValue<int64>()),
                                    uint64 (realSize[1].GetValue<int64>()),
                                    uint64 (realSize[2].GetValue<int64>()))
            with _ ->
                None

    readFrom (Path.Combine(outputPath, "zarr.json"))
    |> Option.orElseWith (fun () -> readFrom (Path.Combine(outputPath, "0", "zarr.json")))

let private zarrDataType<'T> () =
    if typeof<'T> = typeof<uint8> then
        "uint8"
    elif typeof<'T> = typeof<uint16> then
        "uint16"
    elif typeof<'T> = typeof<float32> then
        "float32"
    elif typeof<'T> = typeof<float> then
        "float64"
    elif typeof<'T> = typeof<System.Numerics.Complex> then
        "complex128"
    else
        failwith $"ZarrNET image IO currently supports UInt8, UInt16, Float32, Float64, Complex64, and Complex128 images, but was {typeof<'T>.Name}."

let private zarrScalarElementBytes<'T> () =
    let t = typeof<'T>
    if t = typeof<uint8> then 1
    elif t = typeof<uint16> then 2
    elif t = typeof<float32> then 4
    elif t = typeof<float> then 8
    else
        zarrDataType<'T> () |> ignore
        failwith "unreachable"

let private isSupportedZarrDataType (dataType: string) =
    [ "uint8"; "uint16"; "float32"; "float64"; "complex64"; "complex128" ]
    |> List.exists (fun supported -> String.Equals(dataType, supported, StringComparison.OrdinalIgnoreCase))

let private nullableParallelChunks maxParallelChunks =
    if maxParallelChunks > 0 then
        Nullable<int>(maxParallelChunks)
    else
        Nullable<int>()

let private openZarrResolutionLevel (path: string) multiscaleIndex datasetIndex : ResolutionLevelNode =
    suppressZarrNetDebugLogging ()

    let reader: OmeZarrReader =
        OmeZarrReader.OpenAsync(path, ct = CancellationToken.None)
        |> runTask

    let multiscale = reader.AsMultiscaleImage()
    let level =
        multiscale.OpenResolutionLevelAsync(multiscaleIndex, datasetIndex, CancellationToken.None)
        |> runTask

    deleteZarrNetDebugLogs ()
    level

let private zarrShapeTCZYX (shape: int64[]) =
    if shape.Length <> 5 then
        let shapeText = String.Join("x", shape)
        failwith $"Expected a 5D OME-Zarr array with t,c,z,y,x axes, but shape was {shapeText}."

    int shape[0], int shape[1], int shape[2], int shape[3], int shape[4]

let private validateZarrScalarChunkType<'T> (dataType: string) =
    let expected = zarrDataType<'T> ()
    if not (String.Equals(dataType, expected, StringComparison.OrdinalIgnoreCase)) then
        invalidOp $"Expected Zarr data type {expected} for Chunk<{typeof<'T>.Name}>, got {dataType}."

let private zarrPlaneChunkAs<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (dataType: string)
    width
    height
    (bytes: byte[])
    =

    validateZarrScalarChunkType<'T> dataType
    let elementBytes = zarrScalarElementBytes<'T> ()
    let expectedBytes = width * height * elementBytes
    if bytes.Length < expectedBytes then
        invalidArg "bytes" $"Zarr byte payload is too short for a {width}x{height} {typeof<'T>.Name} plane: expected {expectedBytes}, got {bytes.Length}."

    let chunk = Chunk.create<'T> (uint64 width, uint64 height, 1UL)
    Buffer.BlockCopy(bytes, 0, chunk.Bytes, 0, expectedBytes)
    chunk

let private zarrThickChunkAs<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (dataType: string)
    width
    height
    depth
    (bytes: byte[])
    =

    validateZarrScalarChunkType<'T> dataType
    let elementBytes = zarrScalarElementBytes<'T> ()
    let expectedBytes = width * height * depth * elementBytes
    if bytes.Length < expectedBytes then
        invalidArg "bytes" $"Zarr byte payload is too short for a {width}x{height}x{depth} {typeof<'T>.Name} thick chunk: expected {expectedBytes}, got {bytes.Length}."

    let chunk = Chunk.create<'T> (uint64 width, uint64 height, uint64 depth)
    Buffer.BlockCopy(bytes, 0, chunk.Bytes, 0, expectedBytes)
    chunk

let private hdfDataType<'T> () =
    let t = typeof<'T>

    if t = typeof<uint8>
       || t = typeof<int8>
       || t = typeof<uint16>
       || t = typeof<int16>
       || t = typeof<uint32>
       || t = typeof<int32>
       || t = typeof<float32>
       || t = typeof<float> then
        t.Name
    else
        failwith $"HDF5/NeXus image IO supports scalar numeric images, but was {t.Name}."

let private hdfDataset (path: string) (datasetPath: string) =
    let file = H5File.OpenRead(path)
    file, file.Dataset(datasetPath)

let private addHdfDatasetPath (file: H5File) (datasetPath: string) (dataset: obj) =
    let parts = datasetPath.Trim('/').Split('/', StringSplitOptions.RemoveEmptyEntries)

    if parts.Length = 0 then
        invalidArg "datasetPath" "HDF5 dataset path must name a dataset."

    let mutable group = file :> H5Group

    for part in parts[0 .. parts.Length - 2] do
        let next = H5Group()
        group.Add(part, next)
        group <- next

    group.Add(parts[parts.Length - 1], dataset)

let private hdfDatasetChunks (dataset: IH5Dataset) =
    try
        dataset.Layout.Chunks
        |> Array.map int
        |> Array.toList
    with _ ->
        dataset.Space.Dimensions
        |> Array.map int
        |> Array.toList


let private validateHdfAxes rank frameAxis yAxis xAxis =
    let axes = [ frameAxis; yAxis; xAxis ]

    if axes |> List.exists (fun axis -> axis < 0 || axis >= rank) then
        invalidArg "axis" $"HDF5/NeXus axes must be in 0..{rank - 1}."

    if (axes |> Set.ofList).Count <> 3 then
        invalidArg "axis" "HDF5/NeXus frame, y, and x axes must be distinct."

let private normalizeSuffix (suffix: string) =
    let trimmed = suffix.Trim()

    if trimmed.StartsWith(".", StringComparison.Ordinal) then
        trimmed
    else
        "." + trimmed

let private suffixAliases (suffix: string) =
    match normalizeSuffix suffix with
    | suffix when String.Equals(suffix, ".tif", StringComparison.OrdinalIgnoreCase)
               || String.Equals(suffix, ".tiff", StringComparison.OrdinalIgnoreCase) ->
        [ ".tif"; ".tiff" ]
    | suffix when String.Equals(suffix, ".jpg", StringComparison.OrdinalIgnoreCase)
               || String.Equals(suffix, ".jpeg", StringComparison.OrdinalIgnoreCase) ->
        [ ".jpg"; ".jpeg" ]
    | suffix ->
        [ suffix ]

let private suffixDescription suffix =
    suffixAliases suffix |> String.concat " or "

let private isFSharpInteractiveProcess () =
    let friendlyName =
        try AppDomain.CurrentDomain.FriendlyName
        with _ -> ""

    let commandLine =
        try Environment.GetCommandLineArgs() |> String.concat " "
        with _ -> ""

    friendlyName.Contains("fsi", StringComparison.OrdinalIgnoreCase)
    || commandLine.Contains("fsi", StringComparison.OrdinalIgnoreCase)

let private stopWithInputError (message: string) =
    if isFSharpInteractiveProcess() then
        invalidOp message
    else
        Console.Error.WriteLine(message)
        Environment.ExitCode <- 2
        Environment.Exit 2
        failwith message

let volumeFilePath (input: string) (suffix: string) =
    if Path.HasExtension input then
        input
    else
        let primary = input + normalizeSuffix suffix

        if File.Exists primary then
            primary
        else
            suffixAliases suffix
            |> List.map (fun alias -> input + alias)
            |> List.tryFind File.Exists
            |> Option.defaultValue primary

let private getStackFiles inputDir suffix =
    let aliases = suffixAliases suffix

    if String.IsNullOrWhiteSpace inputDir then
        stopWithInputError "Input stack directory was empty. Please provide a directory containing image slices."

    if File.Exists inputDir then
        stopWithInputError $"Input path is a file, not an image stack directory: {inputDir}. getStackInfo expects a directory containing one image file per slice. For a single image or volume file, use getFileInfo or readVolume with the file path."

    if not (Directory.Exists inputDir) then
        stopWithInputError $"Input stack directory does not exist: {inputDir}"

    Directory.EnumerateFiles(inputDir)
    |> Seq.filter (fun file ->
        aliases
        |> List.exists (fun alias -> file.EndsWith(alias, StringComparison.OrdinalIgnoreCase)))
    |> Seq.distinct
    |> Seq.sort
    |> Seq.toArray

let private getStackPagesForFiles (files: string[]) =
    if files.Length = 0 then
        [||]
    else
        let firstPageCount = tiffDirectoryCount files[0] |> int
        if firstPageCount = 1 then
            files |> Array.map (fun fileName -> fileName, 0)
        else
            files
            |> Array.collect (fun fileName ->
                let pageCount = tiffDirectoryCount fileName |> int
                Array.init pageCount (fun pageIndex -> fileName, pageIndex))

let private getStackPages inputDir suffix =
    getStackFiles inputDir suffix
    |> getStackPagesForFiles

let getStackDepth (inputDir: string) (suffix: string) : uint =
    let files = getStackFiles inputDir suffix
    files.Length |> uint

let getStackInfo (inputDir: string) (suffix: string) : FileInfo =
    let files = getStackFiles inputDir suffix
    let depth = files.Length |> uint64
    if depth = 0uL then
        stopWithInputError $"No {suffixDescription suffix} files found in input stack directory: {inputDir}"
    let fi = getFileInfo(files[0])
    {fi with dimensions = fi.dimensions+1u; size = fi.size @ [depth]}

let private imageInfoOfFileInfo format chunks (info: FileInfo) : ImageInfo =
    { format = format
      dimensions = info.dimensions
      size = info.size
      componentType = info.componentType
      numberOfComponents = info.numberOfComponents
      chunks = chunks }

let getImageInfo (inputDir: string) (suffix: string) : ImageInfo =
    let info = getStackInfo inputDir suffix
    let chunks =
        match info.size with
        | width :: height :: _ -> [ int width; int height; 1 ]
        | _ -> [ 1; 1; 1 ]
    imageInfoOfFileInfo "Image stack" chunks info

let getImageFileInfo (input: string) (suffix: string) : ImageInfo =
    let info = getFileInfo (volumeFilePath input suffix)
    let chunks =
        match info.size with
        | width :: height :: depth :: _ -> [ int width; int height; int depth ]
        | width :: height :: _ -> [ int width; int height; 1 ]
        | _ -> [ 1; 1; 1 ]
    imageInfoOfFileInfo "Volume file" chunks info

let getStackSize (inputDir: string) (suffix: string) : uint*uint*uint =
    let fi = getStackInfo inputDir suffix 
    (uint fi.size[0],uint fi.size[1],uint fi.size[2])

let getStackWidth (inputDir: string) (suffix: string) : uint64 =
    let fi = getStackInfo inputDir suffix
    fi.size[0]

let getStackHeight (inputDir: string) (suffix: string) : uint64 =
    let fi = getStackInfo inputDir suffix
    fi.size[1]

let _getFilenames (inputDir: string) (suffix: string) (filter: string[]->string[]) =
    getStackFiles inputDir suffix |> filter

let getFilenames (inputDir: string) (suffix: string) (filter: string[]->string[]) (pl: Plan<unit, unit>) : Plan<unit, string> =
    let name = "getFilenames"
    let filenames = _getFilenames inputDir suffix filter
    let depth = uint64 filenames.Length

    let mapper (i: int) : string = 
        if pl.debug && DebugLevel.current() >= 2u then printfn "[%s] Supplying filename %i: %s" name i filenames[i]
        filenames[i]

    let transition = ProfileTransition.create Unit Streaming
    let memPeak = 256UL // surrugate string length
    let memoryNeed = fun _ -> memPeak
    let elementTransformation = id
    let memoryModel = StageMemoryModel.fromSinglePeak Source memoryNeed
    let timeCostModel =
        StageTimeCostModel.zero Source
    let stage =
        Stage.init $"{name}" (uint depth) mapper transition memoryNeed elementTransformation
        |> withCostModel (StageCostModel.create memoryModel timeCostModel)
        |> Some

    let memPerElem = 256UL // surrugate string length
    let length = depth
    Plan.createWithOptimizer stage pl.memAvail memPeak memPerElem length pl.debug pl.optimize
    |> Plan.withRuntimeOptionsFrom pl

let private readChunkSlicesExact<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> inputDir suffix (pl: Plan<unit, unit>) : Plan<unit, Chunk<'T>> =
    let name = "readChunkSlices"
    ensureDirectTiffChunkRead<'T> suffix
    let files = getStackFiles inputDir suffix
    if files.Length = 0 then
        stopWithInputError $"No {suffixDescription suffix} files found in input stack directory: {inputDir}"

    let readPlan = inspectChunkTiffSliceForRead<'T> files[0]
    let width = readPlan.Width
    let height = readPlan.Height
    let readSliceIntoOffset = scalarTiffReaderForPlan<'T> readPlan
    let elementBytes = uint64 readPlan.RowBytes * uint64 height
    let sourcePeek =
        SourcePeek.create
            name
            elementBytes
            (Some(uint64 files.Length))
            (Map.ofList
                [ "kind", "chunk-tiff-slices"
                  "inputDir", inputDir
                  "suffix", suffix
                  "width", string width
                  "height", string height
                  "depth", string files.Length
                  "pixelType", typeof<'T>.Name
                  "tiffStrategy", tiffReadStrategyName readPlan.Strategy
                  "rowBytes", string readPlan.RowBytes ])

    let mapper (i: int) =
        let fileName = files[i]
        if pl.debug then
            printfn $"[{name}] Reading chunk slice {i}: {fileName} as {typeof<'T>.Name}"
        let chunk = Chunk.create<'T> (uint64 width, uint64 height, 1UL)
        try
            readSliceIntoOffset fileName 0 chunk 0
            chunk
        with
        | _ ->
            Chunk.decRef chunk
            reraise()

    let transition = ProfileTransition.create Unit Streaming
    let memoryNeed _ = elementBytes
    let elementTransformation _ = uint64 width * uint64 height
    let memoryModel = StageMemoryModel.fromSinglePeak Source memoryNeed
    let timeCostModel =
        StageTimeCostModel.ioRead Source (Some $"readChunkSlices.{typeof<'T>.Name}") (fun _ -> elementBytes) (fun _ -> 1UL)
    let stage =
        Stage.init name (uint files.Length) mapper transition memoryNeed elementTransformation
        |> withCostModel (StageCostModel.create memoryModel timeCostModel)
        |> Some

    Plan.createWithOptimizer stage pl.memAvail elementBytes elementBytes (uint64 files.Length) pl.debug pl.optimize
    |> Plan.withRuntimeOptionsFrom pl
    |> Plan.withSourcePeek sourcePeek

let private readChunkSlicesAs<'S, 'T when 'S: equality and 'S: (new: unit -> 'S) and 'S: struct and 'S :> ValueType
                                      and 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    inputDir
    suffix
    (pl: Plan<unit, unit>)
    : Plan<unit, Chunk<'T>> =

    let source = readChunkSlicesExact<'S> inputDir suffix pl
    if typeof<'S> = typeof<'T> then
        unbox (box source)
    else
        source >=> castChunkStage<'S, 'T>

let readChunkSlices<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> inputDir suffix (pl: Plan<unit, unit>) : Plan<unit, Chunk<'T>> =
    if not (isTiffStackSuffix suffix) then
        invalidArg "suffix" $"readChunkSlices currently supports scalar TIFF stacks for UInt8, Int8, UInt16, Int16, UInt32, Int32, Float32, and Float64; got suffix '{suffix}'."

    let files = getStackFiles inputDir suffix
    if files.Length = 0 then
        stopWithInputError $"No {suffixDescription suffix} files found in input stack directory: {inputDir}"

    match (getFileInfo files[0]).componentType with
    | "UInt8" -> readChunkSlicesAs<uint8, 'T> inputDir suffix pl
    | "Int8" -> readChunkSlicesAs<int8, 'T> inputDir suffix pl
    | "UInt16" -> readChunkSlicesAs<uint16, 'T> inputDir suffix pl
    | "Int16" -> readChunkSlicesAs<int16, 'T> inputDir suffix pl
    | "UInt32" -> readChunkSlicesAs<uint32, 'T> inputDir suffix pl
    | "Int32" -> readChunkSlicesAs<int32, 'T> inputDir suffix pl
    | "Float32" -> readChunkSlicesAs<float32, 'T> inputDir suffix pl
    | "Float64" -> readChunkSlicesAs<float, 'T> inputDir suffix pl
    | sourceType ->
        invalidOp $"readChunkSlices cannot read TIFF source component type '{sourceType}' as {friendlyScalarTypeName typeof<'T>}."

let readComplex64ChunkSlices inputDir suffix (pl: Plan<unit, unit>) : Plan<unit, Chunk<float32>> =
    let name = "readComplex64ChunkSlices"
    if not (isTiffStackSuffix suffix) then
        invalidArg "suffix" $"readComplex64 currently supports TIFF stacks only; got suffix '{suffix}'."

    let files = getStackFiles inputDir suffix
    if files.Length = 0 then
        stopWithInputError $"No {suffixDescription suffix} files found in complex64 input stack directory: {inputDir}"

    let readPlan = inspectComplex64TiffSlice files[0]
    let logicalWidth = readPlan.LogicalWidth
    let height = readPlan.Height
    let storageWidth = 2u * logicalWidth
    let elementBytes = uint64 readPlan.RowBytes * uint64 height
    let sourcePeek =
        SourcePeek.create
            name
            elementBytes
            (Some(uint64 files.Length))
            (Map.ofList
                [ "kind", "complex64-tiff-slices"
                  "inputDir", inputDir
                  "suffix", suffix
                  "logicalWidth", string logicalWidth
                  "storageWidth", string storageWidth
                  "height", string height
                  "depth", string files.Length
                  "pixelType", "Complex64"
                  "samplesPerPixel", "2"
                  "tiffStrategy", tiffReadStrategyName readPlan.Strategy
                  "rowBytes", string readPlan.RowBytes ])

    let mapper (i: int) =
        let fileName = files[i]
        if pl.debug then
            printfn $"[{name}] Reading complex64 chunk slice {i}: {fileName}"
        let chunk = Chunk.create<float32> (uint64 storageWidth, uint64 height, 1UL)
        try
            readComplex64TiffSliceIntoOffset readPlan.Strategy fileName 0 chunk logicalWidth height 0
            chunk
        with
        | _ ->
            Chunk.decRef chunk
            reraise()

    let transition = ProfileTransition.create Unit Streaming
    let memoryNeed _ = elementBytes
    let elementTransformation _ = uint64 logicalWidth * uint64 height
    let memoryModel = StageMemoryModel.fromSinglePeak Source memoryNeed
    let timeCostModel =
        StageTimeCostModel.ioRead Source (Some "readComplex64ChunkSlices") (fun _ -> elementBytes) (fun _ -> 1UL)
    let stage =
        Stage.init name (uint files.Length) mapper transition memoryNeed elementTransformation
        |> withCostModel (StageCostModel.create memoryModel timeCostModel)
        |> Some

    Plan.createWithOptimizer stage pl.memAvail elementBytes elementBytes (uint64 files.Length) pl.debug pl.optimize
    |> Plan.withRuntimeOptionsFrom pl
    |> Plan.withSourcePeek sourcePeek

let readComplex128ChunkSlices inputDir suffix (pl: Plan<unit, unit>) : Plan<unit, Chunk<float>> =
    let name = "readComplex128ChunkSlices"
    if not (isTiffStackSuffix suffix) then
        invalidArg "suffix" $"readComplex128 currently supports TIFF stacks only; got suffix '{suffix}'."

    let files = getStackFiles inputDir suffix
    if files.Length = 0 then
        stopWithInputError $"No {suffixDescription suffix} files found in complex128 input stack directory: {inputDir}"

    let readPlan = inspectComplex128TiffSlice files[0]
    let logicalWidth = readPlan.LogicalWidth
    let height = readPlan.Height
    let storageWidth = 2u * logicalWidth
    let elementBytes = uint64 readPlan.RowBytes * uint64 height
    let sourcePeek =
        SourcePeek.create
            name
            elementBytes
            (Some(uint64 files.Length))
            (Map.ofList
                [ "kind", "complex128-tiff-slices"
                  "inputDir", inputDir
                  "suffix", suffix
                  "logicalWidth", string logicalWidth
                  "storageWidth", string storageWidth
                  "height", string height
                  "depth", string files.Length
                  "pixelType", "ComplexFloat64"
                  "samplesPerPixel", "2"
                  "tiffStrategy", tiffReadStrategyName readPlan.Strategy
                  "rowBytes", string readPlan.RowBytes ])

    let mapper (i: int) =
        let fileName = files[i]
        if pl.debug then
            printfn $"[{name}] Reading complex128 chunk slice {i}: {fileName}"
        let chunk = Chunk.create<float> (uint64 storageWidth, uint64 height, 1UL)
        try
            readComplex128TiffSliceIntoOffset readPlan.Strategy fileName 0 chunk logicalWidth height 0
            chunk
        with
        | _ ->
            Chunk.decRef chunk
            reraise()

    let transition = ProfileTransition.create Unit Streaming
    let memoryNeed _ = elementBytes
    let elementTransformation _ = uint64 logicalWidth * uint64 height
    let memoryModel = StageMemoryModel.fromSinglePeak Source memoryNeed
    let timeCostModel =
        StageTimeCostModel.ioRead Source (Some "readComplex128ChunkSlices") (fun _ -> elementBytes) (fun _ -> 1UL)
    let stage =
        Stage.init name (uint files.Length) mapper transition memoryNeed elementTransformation
        |> withCostModel (StageCostModel.create memoryModel timeCostModel)
        |> Some

    Plan.createWithOptimizer stage pl.memAvail elementBytes elementBytes (uint64 files.Length) pl.debug pl.optimize
    |> Plan.withRuntimeOptionsFrom pl
    |> Plan.withSourcePeek sourcePeek

let readChunkThick<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (chunkDepth: uint)
    inputDir
    suffix
    (pl: Plan<unit, unit>)
    : Plan<unit, Chunk<'T>> =

    let name = "readChunkThick"
    ensureDirectTiffChunkRead<'T> suffix
    let chunkDepth = max 1u chunkDepth |> int
    let files = getStackFiles inputDir suffix
    if files.Length = 0 then
        stopWithInputError $"No {suffixDescription suffix} files found in input stack directory: {inputDir}"

    let pages = getStackPagesForFiles files
    let readPlan = inspectChunkTiffSliceForRead<'T> files[0]
    let width = readPlan.Width
    let height = readPlan.Height
    let readSliceIntoOffset = scalarTiffReaderForPlan<'T> readPlan
    let sliceBytes = readPlan.RowBytes * int height
    let groupCount = (pages.Length + chunkDepth - 1) / chunkDepth
    let elementBytes = uint64 sliceBytes * uint64 chunkDepth
    let sourcePeek =
        SourcePeek.create
            name
            elementBytes
            (Some(uint64 groupCount))
            (Map.ofList
                [ "kind", "chunk-tiff-slice-groups"
                  "inputDir", inputDir
                  "suffix", suffix
                  "width", string width
                  "height", string height
                  "sourceDepth", string pages.Length
                  "sourceFiles", string files.Length
                  "chunkDepth", string chunkDepth
                  "groups", string groupCount
                  "pixelType", typeof<'T>.Name
                  "tiffStrategy", tiffReadStrategyName readPlan.Strategy
                  "rowBytes", string readPlan.RowBytes ])

    let mapper (groupIndex: int) =
        let firstSlice = groupIndex * chunkDepth
        let zCount = min chunkDepth (pages.Length - firstSlice)
        if pl.debug then
            printfn $"[{name}] Reading slices {firstSlice}..{firstSlice + zCount - 1} from {inputDir} as {typeof<'T>.Name}"

        let chunk = Chunk.create<'T> (uint64 width, uint64 height, uint64 zCount)
        try
            for localZ in 0 .. zCount - 1 do
                let fileName, pageIndex = pages[firstSlice + localZ]
                let sliceOffset = localZ * sliceBytes
                readSliceIntoOffset fileName pageIndex chunk sliceOffset
            chunk
        with
        | _ ->
            Chunk.decRef chunk
            reraise()

    let transition = ProfileTransition.create Unit Streaming
    let memoryNeed _ = elementBytes
    let elementTransformation _ = uint64 width * uint64 height * uint64 chunkDepth
    let memoryModel = StageMemoryModel.fromSinglePeak Source memoryNeed
    let timeCostModel =
        StageTimeCostModel.ioRead Source (Some $"readChunkThick.{typeof<'T>.Name}") (fun _ -> elementBytes) (fun _ -> uint64 chunkDepth)
    let stage =
        Stage.init name (uint groupCount) mapper transition memoryNeed elementTransformation
        |> withCostModel (StageCostModel.create memoryModel timeCostModel)
        |> Some

    Plan.createWithOptimizer stage pl.memAvail elementBytes elementBytes (uint64 groupCount) pl.debug pl.optimize
    |> Plan.withRuntimeOptionsFrom pl
    |> Plan.withSourcePeek sourcePeek

let readComplex64ChunkThick
    (chunkDepth: uint)
    inputDir
    suffix
    (pl: Plan<unit, unit>)
    : Plan<unit, Chunk<float32>> =

    let name = "readComplex64ChunkThick"
    if not (isTiffStackSuffix suffix) then
        invalidArg "suffix" $"readComplex64Thick currently supports TIFF stacks only; got suffix '{suffix}'."

    let chunkDepth = max 1u chunkDepth |> int
    let files = getStackFiles inputDir suffix
    if files.Length = 0 then
        stopWithInputError $"No {suffixDescription suffix} files found in complex64 input stack directory: {inputDir}"

    let pages = getStackPagesForFiles files
    let readPlan = inspectComplex64TiffSlice files[0]
    let logicalWidth = readPlan.LogicalWidth
    let height = readPlan.Height
    let storageWidth = 2u * logicalWidth
    let sliceBytes = readPlan.RowBytes * int height
    let groupCount = (pages.Length + chunkDepth - 1) / chunkDepth
    let elementBytes = uint64 sliceBytes * uint64 chunkDepth
    let sourcePeek =
        SourcePeek.create
            name
            elementBytes
            (Some(uint64 groupCount))
            (Map.ofList
                [ "kind", "complex64-tiff-slice-groups"
                  "inputDir", inputDir
                  "suffix", suffix
                  "logicalWidth", string logicalWidth
                  "storageWidth", string storageWidth
                  "height", string height
                  "sourceDepth", string pages.Length
                  "sourceFiles", string files.Length
                  "chunkDepth", string chunkDepth
                  "groups", string groupCount
                  "pixelType", "Complex64"
                  "samplesPerPixel", "2"
                  "tiffStrategy", tiffReadStrategyName readPlan.Strategy
                  "rowBytes", string readPlan.RowBytes ])

    let mapper (groupIndex: int) =
        let firstSlice = groupIndex * chunkDepth
        let zCount = min chunkDepth (pages.Length - firstSlice)
        if pl.debug then
            printfn $"[{name}] Reading complex64 slices {firstSlice}..{firstSlice + zCount - 1} from {inputDir}"

        let chunk = Chunk.create<float32> (uint64 storageWidth, uint64 height, uint64 zCount)
        try
            for localZ in 0 .. zCount - 1 do
                let fileName, pageIndex = pages[firstSlice + localZ]
                readComplex64TiffSliceIntoOffset readPlan.Strategy fileName pageIndex chunk logicalWidth height (localZ * sliceBytes)
            chunk
        with
        | _ ->
            Chunk.decRef chunk
            reraise()

    let transition = ProfileTransition.create Unit Streaming
    let memoryNeed _ = elementBytes
    let elementTransformation _ = uint64 logicalWidth * uint64 height * uint64 chunkDepth
    let memoryModel = StageMemoryModel.fromSinglePeak Source memoryNeed
    let timeCostModel =
        StageTimeCostModel.ioRead Source (Some "readComplex64ChunkThick") (fun _ -> elementBytes) (fun _ -> uint64 chunkDepth)
    let stage =
        Stage.init name (uint groupCount) mapper transition memoryNeed elementTransformation
        |> withCostModel (StageCostModel.create memoryModel timeCostModel)
        |> Some

    Plan.createWithOptimizer stage pl.memAvail elementBytes elementBytes (uint64 groupCount) pl.debug pl.optimize
    |> Plan.withRuntimeOptionsFrom pl
    |> Plan.withSourcePeek sourcePeek

let readComplex128ChunkThick
    (chunkDepth: uint)
    inputDir
    suffix
    (pl: Plan<unit, unit>)
    : Plan<unit, Chunk<float>> =

    let name = "readComplex128ChunkThick"
    if not (isTiffStackSuffix suffix) then
        invalidArg "suffix" $"readComplex128Thick currently supports TIFF stacks only; got suffix '{suffix}'."

    let chunkDepth = max 1u chunkDepth |> int
    let files = getStackFiles inputDir suffix
    if files.Length = 0 then
        stopWithInputError $"No {suffixDescription suffix} files found in complex128 input stack directory: {inputDir}"

    let pages = getStackPagesForFiles files
    let readPlan = inspectComplex128TiffSlice files[0]
    let logicalWidth = readPlan.LogicalWidth
    let height = readPlan.Height
    let storageWidth = 2u * logicalWidth
    let sliceBytes = readPlan.RowBytes * int height
    let groupCount = (pages.Length + chunkDepth - 1) / chunkDepth
    let elementBytes = uint64 sliceBytes * uint64 chunkDepth
    let sourcePeek =
        SourcePeek.create
            name
            elementBytes
            (Some(uint64 groupCount))
            (Map.ofList
                [ "kind", "complex128-tiff-slice-groups"
                  "inputDir", inputDir
                  "suffix", suffix
                  "logicalWidth", string logicalWidth
                  "storageWidth", string storageWidth
                  "height", string height
                  "sourceDepth", string pages.Length
                  "sourceFiles", string files.Length
                  "chunkDepth", string chunkDepth
                  "groups", string groupCount
                  "pixelType", "ComplexFloat64"
                  "samplesPerPixel", "2"
                  "tiffStrategy", tiffReadStrategyName readPlan.Strategy
                  "rowBytes", string readPlan.RowBytes ])

    let mapper (groupIndex: int) =
        let firstSlice = groupIndex * chunkDepth
        let zCount = min chunkDepth (pages.Length - firstSlice)
        if pl.debug then
            printfn $"[{name}] Reading complex128 slices {firstSlice}..{firstSlice + zCount - 1} from {inputDir}"

        let chunk = Chunk.create<float> (uint64 storageWidth, uint64 height, uint64 zCount)
        try
            for localZ in 0 .. zCount - 1 do
                let fileName, pageIndex = pages[firstSlice + localZ]
                readComplex128TiffSliceIntoOffset readPlan.Strategy fileName pageIndex chunk logicalWidth height (localZ * sliceBytes)
            chunk
        with
        | _ ->
            Chunk.decRef chunk
            reraise()

    let transition = ProfileTransition.create Unit Streaming
    let memoryNeed _ = elementBytes
    let elementTransformation _ = uint64 logicalWidth * uint64 height * uint64 chunkDepth
    let memoryModel = StageMemoryModel.fromSinglePeak Source memoryNeed
    let timeCostModel =
        StageTimeCostModel.ioRead Source (Some "readComplex128ChunkThick") (fun _ -> elementBytes) (fun _ -> uint64 chunkDepth)
    let stage =
        Stage.init name (uint groupCount) mapper transition memoryNeed elementTransformation
        |> withCostModel (StageCostModel.create memoryModel timeCostModel)
        |> Some

    Plan.createWithOptimizer stage pl.memAvail elementBytes elementBytes (uint64 groupCount) pl.debug pl.optimize
    |> Plan.withRuntimeOptionsFrom pl
    |> Plan.withSourcePeek sourcePeek

let readChunkThickFiles<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> inputDir suffix (pl: Plan<unit, unit>) : Plan<unit, Chunk<'T>> =
    let name = "readChunkThickFiles"
    ensureDirectTiffChunkRead<'T> suffix
    let files = getStackFiles inputDir suffix
    if files.Length = 0 then
        stopWithInputError $"No {suffixDescription suffix} files found in input stack directory: {inputDir}"

    let pageCounts = files |> Array.map (tiffDirectoryCount >> int)
    let readPlan = inspectChunkTiffSliceForRead<'T> files[0]
    let width = readPlan.Width
    let height = readPlan.Height
    let readSliceIntoOffset = scalarTiffReaderForPlan<'T> readPlan
    let sliceBytes = readPlan.RowBytes * int height
    let maxPagesPerFile = pageCounts |> Array.max
    let elementBytes = uint64 sliceBytes * uint64 maxPagesPerFile
    let sourcePeek =
        SourcePeek.create
            name
            elementBytes
            (Some(uint64 files.Length))
            (Map.ofList
                [ "kind", "chunk-tiff-files"
                  "inputDir", inputDir
                  "suffix", suffix
                  "width", string width
                  "height", string height
                  "files", string files.Length
                  "sourceDepth", string (pageCounts |> Array.sum)
                  "maxPagesPerFile", string maxPagesPerFile
                  "pixelType", typeof<'T>.Name
                  "tiffStrategy", tiffReadStrategyName readPlan.Strategy
                  "rowBytes", string readPlan.RowBytes ])

    let mapper (i: int) =
        let fileName = files[i]
        let zCount = pageCounts[i]
        if pl.debug then
            printfn $"[{name}] Reading chunk file {i}: {fileName} with {zCount} page(s) as {typeof<'T>.Name}"

        let chunk = Chunk.create<'T> (uint64 width, uint64 height, uint64 zCount)
        try
            for localZ in 0 .. zCount - 1 do
                let sliceOffset = localZ * sliceBytes
                readSliceIntoOffset fileName localZ chunk sliceOffset
            chunk
        with
        | _ ->
            Chunk.decRef chunk
            reraise()

    let transition = ProfileTransition.create Unit Streaming
    let memoryNeed _ = elementBytes
    let elementTransformation _ = uint64 width * uint64 height * uint64 maxPagesPerFile
    let memoryModel = StageMemoryModel.fromSinglePeak Source memoryNeed
    let timeCostModel =
        StageTimeCostModel.ioRead Source (Some $"readChunkThickFiles.{typeof<'T>.Name}") (fun _ -> elementBytes) (fun _ -> uint64 maxPagesPerFile)
    let stage =
        Stage.init name (uint files.Length) mapper transition memoryNeed elementTransformation
        |> withCostModel (StageCostModel.create memoryModel timeCostModel)
        |> Some

    Plan.createWithOptimizer stage pl.memAvail elementBytes elementBytes (uint64 files.Length) pl.debug pl.optimize
    |> Plan.withRuntimeOptionsFrom pl
    |> Plan.withSourcePeek sourcePeek

let private readSelectedChunkSlices<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (name: string)
    (inputDir: string)
    (suffix: string)
    (selectPages: (string * int)[] -> (string * int)[])
    (sourcePeekFields: (string * string) list)
    (pl: Plan<unit, unit>)
    : Plan<unit, Chunk<'T>> =
    ensureDirectTiffChunkRead<'T> suffix
    let sourceFiles = getStackFiles inputDir suffix
    if sourceFiles.Length = 0 then
        stopWithInputError $"No {suffixDescription suffix} files found in input stack directory: {inputDir}"

    let sourcePages = getStackPagesForFiles sourceFiles
    let pages = selectPages sourcePages
    if pages.Length = 0 then
        stopWithInputError $"{name} selected no {suffixDescription suffix} files from input stack directory: {inputDir}"

    let readPlan = inspectChunkTiffSliceForRead<'T> sourceFiles[0]
    let width = readPlan.Width
    let height = readPlan.Height
    let readSliceIntoOffset = scalarTiffReaderForPlan<'T> readPlan
    let elementBytes = uint64 readPlan.RowBytes * uint64 height
    let sourcePeek =
        SourcePeek.create
            name
            elementBytes
            (Some(uint64 pages.Length))
            (Map.ofList
                ([ "kind", "chunk-tiff-slices"
                   "inputDir", inputDir
                   "suffix", suffix
                   "width", string width
                   "height", string height
                   "depth", string pages.Length
                   "sourceDepth", string sourcePages.Length
                   "sourceFiles", string sourceFiles.Length
                   "pixelType", typeof<'T>.Name
                   "tiffStrategy", tiffReadStrategyName readPlan.Strategy
                   "rowBytes", string readPlan.RowBytes ]
                 @ sourcePeekFields))

    let mapper (i: int) =
        let fileName, pageIndex = pages[i]
        if pl.debug then
            printfn $"[{name}] Reading chunk slice {i}: {fileName} page {pageIndex} as {typeof<'T>.Name}"
        let chunk = Chunk.create<'T> (uint64 width, uint64 height, 1UL)
        try
            readSliceIntoOffset fileName pageIndex chunk 0
            chunk
        with
        | _ ->
            Chunk.decRef chunk
            reraise()

    let transition = ProfileTransition.create Unit Streaming
    let memoryNeed _ = elementBytes
    let elementTransformation _ = uint64 width * uint64 height
    let memoryModel = StageMemoryModel.fromSinglePeak Source memoryNeed
    let timeCostModel =
        StageTimeCostModel.ioRead Source (Some $"{name}.{typeof<'T>.Name}") (fun _ -> elementBytes) (fun _ -> 1UL)
    let stage =
        Stage.init name (uint pages.Length) mapper transition memoryNeed elementTransformation
        |> withCostModel (StageCostModel.create memoryModel timeCostModel)
        |> Some

    Plan.createWithOptimizer stage pl.memAvail elementBytes elementBytes (uint64 pages.Length) pl.debug pl.optimize
    |> Plan.withRuntimeOptionsFrom pl
    |> Plan.withSourcePeek sourcePeek

let private readSelectedChunkSlicesAs<'S, 'T when 'S: equality and 'S: (new: unit -> 'S) and 'S: struct and 'S :> ValueType
                                              and 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    name
    inputDir
    suffix
    selectPages
    sourcePeekFields
    (pl: Plan<unit, unit>)
    : Plan<unit, Chunk<'T>> =

    let source = readSelectedChunkSlices<'S> name inputDir suffix selectPages sourcePeekFields pl
    if typeof<'S> = typeof<'T> then
        unbox (box source)
    else
        source >=> castChunkStage<'S, 'T>

let private readSelectedChunkSlicesCast<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    name
    inputDir
    suffix
    selectPages
    sourcePeekFields
    (pl: Plan<unit, unit>)
    : Plan<unit, Chunk<'T>> =

    if not (isTiffStackSuffix suffix) then
        invalidArg "suffix" $"{name} currently supports scalar TIFF stacks for UInt8, Int8, UInt16, Int16, UInt32, Int32, Float32, and Float64; got suffix '{suffix}'."

    let files = getStackFiles inputDir suffix
    if files.Length = 0 then
        stopWithInputError $"No {suffixDescription suffix} files found in input stack directory: {inputDir}"

    match (getFileInfo files[0]).componentType with
    | "UInt8" -> readSelectedChunkSlicesAs<uint8, 'T> name inputDir suffix selectPages sourcePeekFields pl
    | "Int8" -> readSelectedChunkSlicesAs<int8, 'T> name inputDir suffix selectPages sourcePeekFields pl
    | "UInt16" -> readSelectedChunkSlicesAs<uint16, 'T> name inputDir suffix selectPages sourcePeekFields pl
    | "Int16" -> readSelectedChunkSlicesAs<int16, 'T> name inputDir suffix selectPages sourcePeekFields pl
    | "UInt32" -> readSelectedChunkSlicesAs<uint32, 'T> name inputDir suffix selectPages sourcePeekFields pl
    | "Int32" -> readSelectedChunkSlicesAs<int32, 'T> name inputDir suffix selectPages sourcePeekFields pl
    | "Float32" -> readSelectedChunkSlicesAs<float32, 'T> name inputDir suffix selectPages sourcePeekFields pl
    | "Float64" -> readSelectedChunkSlicesAs<float, 'T> name inputDir suffix selectPages sourcePeekFields pl
    | sourceType ->
        invalidOp $"{name} cannot read TIFF source component type '{sourceType}' as {friendlyScalarTypeName typeof<'T>}."

let readChunkSlicesRandom<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (count: uint)
    inputDir
    suffix
    (pl: Plan<unit, unit>)
    : Plan<unit, Chunk<'T>> =
    readSelectedChunkSlicesCast<'T>
        "readChunkSlicesRandom"
        inputDir
        suffix
        (Array.randomChoices (int count))
        [ "count", string count ]
        pl

let private readSelectedColorChunkSlices
    (name: string)
    (inputDir: string)
    (suffix: string)
    (selectFiles: string[] -> string[])
    (sourcePeekFields: (string * string) list)
    (pl: Plan<unit, unit>)
    : Plan<unit, VectorChunk<uint8>> =
    if not (isTiffStackSuffix suffix) then
        invalidArg "suffix" $"Color chunk slice IO currently supports TIFF stacks only; got suffix '{suffix}'."

    let sourceFiles = getStackFiles inputDir suffix
    if sourceFiles.Length = 0 then
        stopWithInputError $"No {suffixDescription suffix} files found in RGB input stack directory: {inputDir}"

    let files = selectFiles sourceFiles
    if files.Length = 0 then
        stopWithInputError $"{name} selected no {suffixDescription suffix} files from RGB input stack directory: {inputDir}"

    let readPlan = inspectColorChunkTiffSlice files[0]
    let width = readPlan.Width
    let height = readPlan.Height
    let rowBytes = readPlan.RowBytes
    let elementBytes = uint64 rowBytes * uint64 height
    let sourcePeek =
        SourcePeek.create
            name
            elementBytes
            (Some(uint64 files.Length))
            (Map.ofList
                ([ "kind", "chunk-rgb-tiff-slices"
                   "inputDir", inputDir
                   "suffix", suffix
                   "width", string width
                   "height", string height
                   "depth", string files.Length
                   "sourceDepth", string sourceFiles.Length
                   "pixelType", "Color"
                   "components", "3"
                   "tiffStrategy", tiffReadStrategyName readPlan.Strategy
                   "rowBytes", string readPlan.RowBytes ]
                 @ sourcePeekFields))

    let mapper (i: int) =
        let fileName = files[i]
        if pl.debug then
            printfn $"[{name}] Reading RGB chunk slice {i}: {fileName}"
        let vector = readColorChunkTiffSliceByPlan readPlan fileName
        let chunkWidth, chunkHeight, _ = vector.SpatialSize
        if chunkWidth <> uint64 width || chunkHeight <> uint64 height then
            Chunk.decRef vector.Chunk
            invalidOp $"Input RGB slice '{fileName}' has shape {chunkWidth}x{chunkHeight}, expected {width}x{height}."
        vector

    let transition = ProfileTransition.create Unit Streaming
    let memoryNeed _ = elementBytes
    let elementTransformation _ = uint64 width * uint64 height
    let memoryModel = StageMemoryModel.fromSinglePeak Source memoryNeed
    let timeCostModel =
        StageTimeCostModel.ioRead Source (Some $"{name}.Color") (fun _ -> elementBytes) (fun _ -> 1UL)
    let stage =
        Stage.init name (uint files.Length) mapper transition memoryNeed elementTransformation
        |> withCostModel (StageCostModel.create memoryModel timeCostModel)
        |> Some

    Plan.createWithOptimizer stage pl.memAvail elementBytes elementBytes (uint64 files.Length) pl.debug pl.optimize
    |> Plan.withRuntimeOptionsFrom pl
    |> Plan.withSourcePeek sourcePeek

let readColorChunkSlices inputDir suffix (pl: Plan<unit, unit>) : Plan<unit, VectorChunk<uint8>> =
    readSelectedColorChunkSlices
        "readColorChunkSlices"
        inputDir
        suffix
        id
        []
        pl

let readColorChunkSlicesRandom (count: uint) inputDir suffix (pl: Plan<unit, unit>) : Plan<unit, VectorChunk<uint8>> =
    readSelectedColorChunkSlices
        "readColorChunkSlicesRandom"
        inputDir
        suffix
        (Array.randomChoices (int count))
        [ "count", string count ]
        pl

let private colorRangeIndices (first: uint) step (last: uint) depth =
    if step = 0 then invalidArg "step" "Range step must be non-zero."
    let depthU = uint depth
    if depthU = 0u then [||]
    else
        let last = min last (depthU - 1u)
        let first = min first (depthU - 1u)
        let indices = ResizeArray<int>()
        if step > 0 then
            let mutable i = int first
            while i <= int last do
                indices.Add i
                i <- i + step
        else
            let mutable i = int first
            while i >= int last do
                indices.Add i
                i <- i + step
        indices.ToArray()

let readColorChunkSlicesRange (first: uint) (step: int) (last: uint) inputDir suffix (pl: Plan<unit, unit>) : Plan<unit, VectorChunk<uint8>> =
    readSelectedColorChunkSlices
        "readColorChunkSlicesRange"
        inputDir
        suffix
        (fun files -> colorRangeIndices first step last files.Length |> Array.map (fun index -> files[index]))
        [ "first", string first
          "step", string step
          "last", string last ]
        pl

let private isTiffVolumePath (filename: string) =
    match Path.GetExtension(filename).ToLowerInvariant() with
    | ".tif"
    | ".tiff"
    | ".btf"
    | ".bigtiff" -> true
    | _ -> false


let private readChunkVolumeExact<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (filename: string)
    (pl: Plan<unit, unit>)
    : Plan<unit, Chunk<'T>> =
    if not (isTiffVolumePath filename) then
        invalidArg "filename" "readChunkVolume currently supports TIFF/BigTIFF volumes only."

    let readPlan = inspectChunkTiffSliceForRead<'T> filename
    let width = readPlan.Width
    let height = readPlan.Height
    let readSliceIntoOffset = scalarTiffReaderForPlan<'T> readPlan
    let depth =
        tryBitMiracleTiffDirectoryCount filename
        |> Option.defaultWith (fun () -> tiffDirectoryCount filename)
    let elementBytes = uint64 readPlan.RowBytes * uint64 height
    let sourcePeek =
        SourcePeek.create
            "readChunkVolume"
            elementBytes
            (Some (uint64 depth))
            (Map.ofList
                [ "kind", "chunk-tiff-volume"
                  "filename", filename
                  "width", string width
                  "height", string height
                  "depth", string depth
                  "pixelType", typeof<'T>.Name
                  "tiffStrategy", tiffReadStrategyName readPlan.Strategy ])

    let mapper (index: int) =
        if pl.debug then
            printfn $"[readChunkVolume] Reading TIFF page {index} from {filename} as {typeof<'T>.Name}"

        let chunk = Chunk.create<'T> (uint64 width, uint64 height, 1UL)
        try
            readSliceIntoOffset filename index chunk 0
            chunk
        with
        | _ ->
            Chunk.decRef chunk
            reraise()

    let transition = ProfileTransition.create Unit Streaming
    let memoryNeed _ = elementBytes
    let elementTransformation _ = uint64 width * uint64 height
    let memoryModel = StageMemoryModel.fromSinglePeak Source memoryNeed
    let timeCostModel =
        imageIoCost<'T>
            "readVolume"
            Source
            $"readChunkVolume.tiff.{tiffReadStrategyName readPlan.Strategy}.{typeof<'T>.Name}"
            (fun _ -> elementBytes)
            (fun _ -> 1UL)

    let stage =
        Stage.init "readChunkVolume" depth mapper transition memoryNeed elementTransformation
        |> withCostModel (StageCostModel.create memoryModel timeCostModel)
        |> Some

    Plan.createWithOptimizer stage pl.memAvail elementBytes elementBytes (uint64 depth) pl.debug pl.optimize
    |> Plan.withRuntimeOptionsFrom pl
    |> Plan.withSourcePeek sourcePeek

let private readChunkVolumeAs<'S, 'T when 'S: equality and 'S: (new: unit -> 'S) and 'S: struct and 'S :> ValueType
                                      and 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    filename
    (pl: Plan<unit, unit>)
    : Plan<unit, Chunk<'T>> =

    let source = readChunkVolumeExact<'S> filename pl
    if typeof<'S> = typeof<'T> then
        unbox (box source)
    else
        source >=> castChunkStage<'S, 'T>

let readChunkVolume<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (filename: string)
    (pl: Plan<unit, unit>)
    : Plan<unit, Chunk<'T>> =

    if not (isTiffVolumePath filename) then
        invalidArg "filename" "readChunkVolume currently supports TIFF/BigTIFF volumes only."

    match (getFileInfo filename).componentType with
    | "UInt8" -> readChunkVolumeAs<uint8, 'T> filename pl
    | "Int8" -> readChunkVolumeAs<int8, 'T> filename pl
    | "UInt16" -> readChunkVolumeAs<uint16, 'T> filename pl
    | "Int16" -> readChunkVolumeAs<int16, 'T> filename pl
    | "UInt32" -> readChunkVolumeAs<uint32, 'T> filename pl
    | "Int32" -> readChunkVolumeAs<int32, 'T> filename pl
    | "Float32" -> readChunkVolumeAs<float32, 'T> filename pl
    | "Float64" -> readChunkVolumeAs<float, 'T> filename pl
    | sourceType ->
        invalidOp $"readChunkVolume cannot read TIFF source component type '{sourceType}' as {friendlyScalarTypeName typeof<'T>}."

let private randomIndices count depth =
    if depth <= 0 then
        invalidArg "depth" "Cannot sample random slices from an empty image source."

    let rng = Random()
    Array.init (int count) (fun _ -> rng.Next(depth))

let private rangeIndices (first: uint) step (last: uint) depth =
    if step = 0 then
        invalidArg "step" "readRange step must be non-zero."

    if depth <= 0 then
        [||]
    else
        let maxIndex = depth - 1
        let clampIndex value =
            let value64 = int64 value
            if value64 > int64 maxIndex then maxIndex else int value64

        let startIndex = clampIndex first
        let lastIndex = clampIndex last

        if step > 0 && startIndex > lastIndex then
            [||]
        elif step < 0 && startIndex < lastIndex then
            [||]
        else
            [| for index in startIndex .. step .. lastIndex -> index |]


let private rangeFilter first step last files =
    let sorted = Array.sort files
    rangeIndices first step last sorted.Length
    |> Array.map (fun index -> sorted[index])

let readChunkSlicesRange<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    first
    step
    last
    inputDir
    suffix
    (pl: Plan<unit, unit>)
    : Plan<unit, Chunk<'T>> =
    readSelectedChunkSlicesCast<'T>
        "readChunkSlicesRange"
        inputDir
        suffix
        (rangeFilter first step last)
        [ "first", string first; "step", string step; "last", string last ]
        pl


let getChunkInfo (inputDir: string) (suffix: string) : ChunkInfo =
    let (|IJK|_|) (s: string) =
        let rx = Regex(@"chunk(\d+)_(\d+)_(\d+)(.*)$", RegexOptions.Compiled)
        let m = rx.Match s
        if m.Success then
            Some (
                //int m.Groups[0].Value,   // s
                int m.Groups[1].Value,   // i
                int m.Groups[2].Value, // j
                int m.Groups[3].Value // k
                //m.Groups[4].Value    // suffix
            )
        else None    
    let files = Directory.GetFiles(inputDir, "*"+suffix)
    let maxI, maxJ, maxK, topLeft, bottomRight = 
        Array.fold
            (fun (maxI: int, maxJ: int, maxK: int, tl: string, br: string) (str: string) -> 
                let res = 
                    match str with 
                        IJK (i, j, k) when i >= maxI && j >= maxJ && k >= maxK -> (i,j,k,tl,str)
                        | IJK (i, j, k) when i = 0 && j = 0 && k = 0 -> (maxI,maxJ,maxK,str,br)
                        | _ -> (maxI, maxJ, maxK, tl, br)
                res
            ) (System.Int32.MinValue, System.Int32.MinValue, System.Int32.MinValue, files[0], files[0]) files
    let topLeftFi = getFileInfo topLeft
    let bottomRightFi = getFileInfo bottomRight

    let stackSize = 
        [
            (uint64 maxI) * topLeftFi.size[0] + bottomRightFi.size[0];
            (uint64 maxJ) * topLeftFi.size[1] + bottomRightFi.size[1];
            (uint64 maxK) * topLeftFi.size[2] + bottomRightFi.size[2];
        ]
    { chunks = [maxI+1;maxJ+1;maxK+1]; topLeftInfo = topLeftFi; size = stackSize }

let getZarrInfo (path: string) (multiscaleIndex: int) (datasetIndex: int) : ImageInfo =
    suppressZarrNetDebugLogging ()

    let reader: OmeZarrReader =
        OmeZarrReader.OpenAsync(path, ct = CancellationToken.None)
        |> runTask

    let level =
        reader.AsMultiscaleImage().OpenResolutionLevelAsync(multiscaleIndex, datasetIndex, CancellationToken.None)
        |> runTask

    let _sizeT, _sizeC, sizeZ, sizeY, sizeX = zarrShapeTCZYX level.Shape
    let rawChunks =
        reader.RootGroup.OpenArrayAsync(level.Dataset.Path, CancellationToken.None)
        |> runTask
        |> fun zarrArray -> zarrArray.Metadata.ChunkShape
        |> Array.toList
    let chunks =
        match rawChunks with
        | _t :: _c :: z :: y :: x :: _ -> [ x; y; z ]
        | z :: y :: x :: _ -> [ x; y; z ]
        | _ -> rawChunks

    deleteZarrNetDebugLogs ()

    { format = "OME-Zarr"
      dimensions = 3u
      size = [ uint64 sizeX; uint64 sizeY; uint64 sizeZ ]
      componentType = level.DataType
      numberOfComponents = 1u
      chunks = chunks }

let private zarrChunkDepthFromInfo (chunkInfo: ImageInfo) =
    match chunkInfo.chunks with
    | _x :: _y :: z :: _ -> max 1u (uint z)
    | _ -> 1u

let getNexusInfo (path: string) (datasetPath: string) (frameAxis: int) (yAxis: int) (xAxis: int) : ImageInfo =
    use file = H5File.OpenRead(path)
    let dataset = file.Dataset(datasetPath)
    let rank = int dataset.Space.Rank
    validateHdfAxes rank frameAxis yAxis xAxis

    if rank <> 3 then
        failwith $"getNexusInfo currently expects a rank-3 detector stack dataset, but {datasetPath} has rank {rank}."

    let dimensions = dataset.Space.Dimensions
    let rawChunks = hdfDatasetChunks dataset
    let chunkAt axis =
        rawChunks
        |> List.tryItem axis
        |> Option.defaultValue 1

    { format = "NeXus/HDF5"
      dimensions = 3u
      size = [ uint64 dimensions[xAxis]; uint64 dimensions[yAxis]; uint64 dimensions[frameAxis] ]
      componentType = dataset.Type.ToString()
      numberOfComponents = 1u
      chunks = [ chunkAt xAxis; chunkAt yAxis; chunkAt frameAxis ] }

let private nexusFrameChunkDepth (info: ImageInfo) (_frameAxis: int) =
    info.chunks
    |> List.tryItem 2
    |> Option.map (fun z -> max 1u (uint z))
    |> Option.defaultValue 1u

let getChunkFilename (path: string) (suffix: string) (i: int) (j: int) (k: int) =
    Path.Combine(path, sprintf "chunk%d_%d_%d%s" i j k suffix)


let readZarrChunkSlicesRange<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (first: uint)
    (step: int)
    (last: uint)
    (path: string)
    (multiscaleIndex: int)
    (datasetIndex: int)
    (timepoint: int)
    (channel: int)
    (maxParallelChunks: int)
    (pl: Plan<unit, unit>)
    : Plan<unit, Chunk<'T>> =

    suppressZarrNetDebugLogging ()

    let name = "readZarrChunkSlicesRange"
    let level = openZarrResolutionLevel path multiscaleIndex datasetIndex
    let sizeT, sizeC, sizeZ, sizeY, sizeX = zarrShapeTCZYX level.Shape
    validateZarrScalarChunkType<'T> level.DataType

    if timepoint < 0 || timepoint >= sizeT then
        invalidArg "timepoint" $"Timepoint {timepoint} is outside the Zarr time range 0..{sizeT - 1}."
    if channel < 0 || channel >= sizeC then
        invalidArg "channel" $"Channel {channel} is outside the Zarr channel range 0..{sizeC - 1}."

    let selected = rangeIndices first step last sizeZ
    let elementBytes = uint64 sizeX * uint64 sizeY * uint64 (zarrScalarElementBytes<'T> ())
    let parallelChunks = nullableParallelChunks maxParallelChunks
    let sourcePeek =
        SourcePeek.create
            name
            elementBytes
            (Some (uint64 selected.Length))
            (Map.ofList
                [ "kind", "zarr-chunk-range"
                  "path", path
                  "width", string sizeX
                  "height", string sizeY
                  "depth", string selected.Length
                  "sourceDepth", string sizeZ
                  "pixelType", typeof<'T>.Name
                  "multiscaleIndex", string multiscaleIndex
                  "datasetIndex", string datasetIndex
                  "timepoint", string timepoint
                  "channel", string channel
                  "first", string first
                  "step", string step
                  "last", string last ])

    let mapper (outputIndex: int) : Chunk<'T> =
        let zIndex = selected[outputIndex]
        if pl.debug then
            printfn $"[readZarrChunkSlicesRange] Reading z {zIndex} from {path} as {typeof<'T>.Name}"

        let region =
            PixelRegion(
                [| int64 timepoint; int64 channel; int64 zIndex; 0L; 0L |],
                [| int64 (timepoint + 1); int64 (channel + 1); int64 (zIndex + 1); int64 sizeY; int64 sizeX |])
        let result =
            level.ReadPixelRegionAsync(region, parallelChunks, CancellationToken.None)
            |> runTask
        deleteZarrNetDebugLogs ()
        zarrPlaneChunkAs<'T> level.DataType sizeX sizeY result.Data

    let transition = ProfileTransition.create Unit Streaming
    let memPeak = 256UL
    let memoryNeed = fun _ -> memPeak
    let elementTransformation = id
    let memoryModel = StageMemoryModel.fromSinglePeak Source memoryNeed
    let timeCostModel =
        StageTimeCostModel.ioRead Source (Some $"readZarrChunkSlicesRange.{typeof<'T>.Name}") (fun _ -> elementBytes) (fun _ -> 1UL)
    let stage =
        Stage.init name (uint selected.Length) mapper transition memoryNeed elementTransformation
        |> withCostModel (StageCostModel.create memoryModel timeCostModel)
        |> Some

    Plan.createWithOptimizer stage pl.memAvail memPeak memPeak (uint64 selected.Length) pl.debug pl.optimize
    |> Plan.withRuntimeOptionsFrom pl
    |> Plan.withSourcePeek sourcePeek

let readZarrChunkThickRange<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (first: uint)
    (last: uint)
    (thickDepth: uint)
    (path: string)
    (multiscaleIndex: int)
    (datasetIndex: int)
    (timepoint: int)
    (channel: int)
    (maxParallelChunks: int)
    (pl: Plan<unit, unit>)
    : Plan<unit, Chunk<'T>> =

    suppressZarrNetDebugLogging ()

    let name = "readZarrChunkThickRange"
    let level = openZarrResolutionLevel path multiscaleIndex datasetIndex
    let sizeT, sizeC, sizeZ, sizeY, sizeX = zarrShapeTCZYX level.Shape
    validateZarrScalarChunkType<'T> level.DataType

    if timepoint < 0 || timepoint >= sizeT then
        invalidArg "timepoint" $"Timepoint {timepoint} is outside the Zarr time range 0..{sizeT - 1}."
    if channel < 0 || channel >= sizeC then
        invalidArg "channel" $"Channel {channel} is outside the Zarr channel range 0..{sizeC - 1}."

    let first = int first
    let last =
        if last = UInt32.MaxValue then
            sizeZ - 1
        else
            min (int last) (sizeZ - 1)
    if first < 0 || first >= sizeZ || last < first then
        invalidArg "first" $"Invalid Zarr z range {first}..{last} for source depth {sizeZ}."

    let thickDepth = max 1u thickDepth |> int
    let groupCount = (last - first + 1 + thickDepth - 1) / thickDepth
    let elementBytes = uint64 sizeX * uint64 sizeY * uint64 thickDepth * uint64 (zarrScalarElementBytes<'T> ())
    let parallelChunks = nullableParallelChunks maxParallelChunks
    let sourcePeek =
        SourcePeek.create
            name
            elementBytes
            (Some (uint64 groupCount))
            (Map.ofList
                [ "kind", "zarr-chunk-thick"
                  "path", path
                  "width", string sizeX
                  "height", string sizeY
                  "depth", string (last - first + 1)
                  "sourceDepth", string sizeZ
                  "thickDepth", string thickDepth
                  "pixelType", typeof<'T>.Name
                  "multiscaleIndex", string multiscaleIndex
                  "datasetIndex", string datasetIndex
                  "timepoint", string timepoint
                  "channel", string channel
                  "first", string first
                  "last", string last ])

    let mapper (outputIndex: int) : Chunk<'T> =
        let zStart = first + outputIndex * thickDepth
        let zStop = min (last + 1) (zStart + thickDepth)
        let zCount = zStop - zStart
        if pl.debug then
            printfn $"[readZarrChunkThickRange] Reading z {zStart}..{zStop - 1} from {path} as {typeof<'T>.Name}"

        let region =
            PixelRegion(
                [| int64 timepoint; int64 channel; int64 zStart; 0L; 0L |],
                [| int64 (timepoint + 1); int64 (channel + 1); int64 zStop; int64 sizeY; int64 sizeX |])
        let result =
            level.ReadPixelRegionAsync(region, parallelChunks, CancellationToken.None)
            |> runTask
        deleteZarrNetDebugLogs ()
        zarrThickChunkAs<'T> level.DataType sizeX sizeY zCount result.Data

    let transition = ProfileTransition.create Unit Streaming
    let memPeak = 256UL
    let memoryNeed = fun _ -> memPeak
    let elementTransformation n = n * uint64 thickDepth
    let memoryModel = StageMemoryModel.fromSinglePeak Source memoryNeed
    let timeCostModel =
        StageTimeCostModel.ioRead Source (Some $"readZarrChunkThickRange.{typeof<'T>.Name}") (fun _ -> elementBytes) (fun _ -> uint64 thickDepth)
    let stage =
        Stage.init name (uint groupCount) mapper transition memoryNeed elementTransformation
        |> withCostModel (StageCostModel.create memoryModel timeCostModel)
        |> Some

    Plan.createWithOptimizer stage pl.memAvail memPeak memPeak (uint64 groupCount) pl.debug pl.optimize
    |> Plan.withRuntimeOptionsFrom pl
    |> Plan.withSourcePeek sourcePeek

let readZarrChunkSlicesAlignedRange<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (first: uint)
    (last: uint)
    (path: string)
    (multiscaleIndex: int)
    (datasetIndex: int)
    (timepoint: int)
    (channel: int)
    (pl: Plan<unit, unit>)
    : Plan<unit, Chunk<'T>> =

    suppressZarrNetDebugLogging ()

    let name = "readZarrChunkSlicesAlignedRange"
    let level = openZarrResolutionLevel path multiscaleIndex datasetIndex
    let sizeT, sizeC, sizeZ, sizeY, sizeX = zarrShapeTCZYX level.Shape
    validateZarrScalarChunkType<'T> level.DataType

    if timepoint < 0 || timepoint >= sizeT then
        invalidArg "timepoint" $"Timepoint {timepoint} is outside the Zarr time range 0..{sizeT - 1}."
    if channel < 0 || channel >= sizeC then
        invalidArg "channel" $"Channel {channel} is outside the Zarr channel range 0..{sizeC - 1}."

    let first = int first
    let last =
        if last = UInt32.MaxValue then
            sizeZ - 1
        else
            min (int last) (sizeZ - 1)
    if first < 0 || first >= sizeZ || last < first then
        invalidArg "first" $"Invalid Zarr z range {first}..{last} for source depth {sizeZ}."

    let store = new LocalFileSystemStore(path)
    let rootGroup =
        ZarrGroup.OpenRootAsync(store, CancellationToken.None)
        |> runTask
    let array =
        rootGroup.OpenArrayAsync(level.Dataset.Path, CancellationToken.None)
        |> runTask

    let rawChunkShape = array.Metadata.ChunkShape
    let chunkZ = rawChunkShape[2]
    let chunkY = rawChunkShape[3]
    let chunkX = rawChunkShape[4]
    let chunks =
        collectZarrChunks array
        |> Array.filter (fun chunkRef ->
            chunkRef.Origin[0] = int64 timepoint
            && chunkRef.Origin[1] = int64 channel)

    let chunksByZ =
        chunks
        |> Array.groupBy (fun chunkRef -> int chunkRef.ChunkCoord[2])
        |> Map.ofArray

    let elementBytes = zarrScalarElementBytes<'T> ()
    let sliceBytes = sizeX * sizeY * elementBytes
    let fullZarrChunkBytes = chunkX * chunkY * chunkZ * elementBytes
    let mutable cachedZChunk = -1
    let mutable cachedSlices: Chunk<'T>[] = Array.empty

    let loadZChunk zChunk =
        let zStart = zChunk * chunkZ
        let zStop = min (last + 1) (zStart + chunkZ)
        let zCount = zStop - zStart
        let outputSlices =
            Array.init zCount (fun _ -> Chunk.create<'T> (uint64 sizeX, uint64 sizeY, 1UL))

        try
            match chunksByZ.TryFind zChunk with
            | Some zChunks ->
                let parallelOptions = ParallelOptions(MaxDegreeOfParallelism = Environment.ProcessorCount)
                let readChunk =
                    Action<ZarrChunkRef>(fun (chunkRef: ZarrChunkRef) ->
                        let scratch = ArrayPool<byte>.Shared.Rent(fullZarrChunkBytes)
                        try
                            array.ReadChunkDecodedAsync(
                                chunkRef,
                                Memory<byte>(scratch, 0, fullZarrChunkBytes),
                                true,
                                CancellationToken.None)
                            |> runUnitTask

                            let originZ = int chunkRef.Origin[2]
                            let originY = int chunkRef.Origin[3]
                            let originX = int chunkRef.Origin[4]
                            let actualZ = int chunkRef.Shape[2]
                            let actualY = int chunkRef.Shape[3]
                            let actualX = int chunkRef.Shape[4]
                            let rowBytes = actualX * elementBytes

                            for localChunkZ in 0 .. actualZ - 1 do
                                let globalZ = originZ + localChunkZ
                                if globalZ >= first && globalZ <= last then
                                    let output = outputSlices[globalZ - zStart]
                                    for localY in 0 .. actualY - 1 do
                                        let sourceOffset =
                                            ((localChunkZ * chunkY + localY) * chunkX) * elementBytes
                                        let destinationOffset =
                                            ((originY + localY) * sizeX + originX) * elementBytes
                                        Buffer.BlockCopy(scratch, sourceOffset, output.Bytes, destinationOffset, rowBytes)
                        finally
                            ArrayPool<byte>.Shared.Return(scratch))
                Parallel.ForEach<ZarrChunkRef>(zChunks, parallelOptions, readChunk) |> ignore
            | None -> ()

            deleteZarrNetDebugLogs ()
            outputSlices
        with
        | ex ->
            for output in outputSlices do
                Chunk.decRef output
            raise ex

    let mapper (outputIndex: int) : Chunk<'T> =
        let globalZ = first + outputIndex
        let zChunk = globalZ / chunkZ
        if zChunk <> cachedZChunk then
            cachedSlices <- loadZChunk zChunk
            cachedZChunk <- zChunk

        let localZ = globalZ - zChunk * chunkZ
        if pl.debug then
            printfn $"[readZarrChunkSlicesAlignedRange] Reading z {globalZ} from {path} as {typeof<'T>.Name}"
        cachedSlices[localZ]

    let depth = last - first + 1
    let sourcePeek =
        SourcePeek.create
            name
            (uint64 sliceBytes)
            (Some (uint64 depth))
            (Map.ofList
                [ "kind", "zarr-chunk-slices-aligned"
                  "path", path
                  "width", string sizeX
                  "height", string sizeY
                  "depth", string depth
                  "sourceDepth", string sizeZ
                  "chunkX", string chunkX
                  "chunkY", string chunkY
                  "chunkZ", string chunkZ
                  "pixelType", typeof<'T>.Name ])

    let transition = ProfileTransition.create Unit Streaming
    let memPeak = uint64 sliceBytes
    let memoryNeed = fun _ -> memPeak
    let memoryModel = StageMemoryModel.fromSinglePeak Source memoryNeed
    let timeCostModel =
        StageTimeCostModel.ioRead Source (Some $"readZarrChunkSlicesAlignedRange.{typeof<'T>.Name}") (fun _ -> uint64 fullZarrChunkBytes) (fun _ -> 1UL)
    let stage =
        Stage.init name (uint depth) mapper transition memoryNeed id
        |> withCostModel (StageCostModel.create memoryModel timeCostModel)
        |> Some

    Plan.createWithOptimizer stage pl.memAvail memPeak memPeak (uint64 depth) pl.debug pl.optimize
    |> Plan.withRuntimeOptionsFrom pl
    |> Plan.withSourcePeek sourcePeek

let private spatialZarrLayout pixelType sizeX sizeY sizeZ chunkX chunkY chunkZ : ChunkLayout =
    let chunkCounts =
        ((sizeX + chunkX - 1) / chunkX,
         (sizeY + chunkY - 1) / chunkY,
         (sizeZ + chunkZ - 1) / chunkZ)

    { VolumeSize = (uint64 sizeX, uint64 sizeY, uint64 sizeZ)
      ChunkSize = (uint64 chunkX, uint64 chunkY, uint64 chunkZ)
      ChunkCounts = chunkCounts
      PixelType = pixelType
      Components = 1u }

let private copyDecodedZarrChunkToChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (decoded: byte[])
    fullX
    fullY
    actualX
    actualY
    actualZ
    =

    let output = Chunk.create<'T> (uint64 actualX, uint64 actualY, uint64 actualZ)
    let elementBytes = zarrScalarElementBytes<'T> ()
    let rowBytes = actualX * elementBytes

    for z in 0 .. actualZ - 1 do
        for y in 0 .. actualY - 1 do
            let sourceOffset = ((z * fullY + y) * fullX) * elementBytes
            let destinationOffset = ((z * actualY + y) * actualX) * elementBytes
            Buffer.BlockCopy(decoded, sourceOffset, output.Bytes, destinationOffset, rowBytes)

    output

let private readDecodedZarrChunkToLocatedChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (array: ZarrArray)
    (layout: ChunkLayout)
    fullX
    fullY
    fullZ
    (chunkRef: ZarrChunkRef)
    : LocatedChunk<'T> =

    let actualZ = int chunkRef.Shape[2]
    let actualY = int chunkRef.Shape[3]
    let actualX = int chunkRef.Shape[4]

    let chunk =
        if actualX = fullX && actualY = fullY && actualZ = fullZ then
            let output = Chunk.create<'T> (uint64 actualX, uint64 actualY, uint64 actualZ)
            try
                array.ReadChunkDecodedAsync(
                    chunkRef,
                    Memory<byte>(output.Bytes, 0, output.ByteLength),
                    true,
                    CancellationToken.None)
                |> runUnitTask
                output
            with
            | ex ->
                Chunk.decRef output
                raise ex
        else
            let decoded =
                array.ReadChunkDecodedAsync(chunkRef, CancellationToken.None)
                |> runTask
            copyDecodedZarrChunkToChunk<'T> decoded fullX fullY actualX actualY actualZ

    { Index = (int chunkRef.ChunkCoord[4], int chunkRef.ChunkCoord[3], int chunkRef.ChunkCoord[2])
      Layout = layout
      Chunk = chunk }

let private zarrChunkCoordKey (chunkRef: ZarrChunkRef) =
    String.Join("/", chunkRef.ChunkCoord)

let readZarrLocatedChunks<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (path: string)
    (multiscaleIndex: int)
    (datasetIndex: int)
    (timepoint: int)
    (channel: int)
    (pl: Plan<unit, unit>)
    : Plan<unit, LocatedChunk<'T>> =

    suppressZarrNetDebugLogging ()

    let name = "readZarrLocatedChunks"
    let level = openZarrResolutionLevel path multiscaleIndex datasetIndex
    let sizeT, sizeC, sizeZ, sizeY, sizeX = zarrShapeTCZYX level.Shape
    validateZarrScalarChunkType<'T> level.DataType

    if timepoint < 0 || timepoint >= sizeT then
        invalidArg "timepoint" $"Timepoint {timepoint} is outside the Zarr time range 0..{sizeT - 1}."
    if channel < 0 || channel >= sizeC then
        invalidArg "channel" $"Channel {channel} is outside the Zarr channel range 0..{sizeC - 1}."

    let store = new LocalFileSystemStore(path)
    let rootGroup =
        ZarrGroup.OpenRootAsync(store, CancellationToken.None)
        |> runTask
    let array =
        rootGroup.OpenArrayAsync(level.Dataset.Path, CancellationToken.None)
        |> runTask
    let chunks =
        collectZarrChunks array
        |> Array.filter (fun chunkRef ->
            chunkRef.Origin[0] = int64 timepoint
            && chunkRef.Origin[1] = int64 channel)

    let rawChunkShape = array.Metadata.ChunkShape
    let chunkZ = rawChunkShape[2]
    let chunkY = rawChunkShape[3]
    let chunkX = rawChunkShape[4]
    let dataType = zarrDataType<'T> ()
    let layout = spatialZarrLayout dataType sizeX sizeY sizeZ chunkX chunkY chunkZ
    let elementBytes = uint64 (zarrScalarElementBytes<'T> ())
    let elementCount = uint64 chunkX * uint64 chunkY * uint64 chunkZ
    let sourcePeek =
        SourcePeek.create
            name
            (elementCount * elementBytes)
            (Some (uint64 chunks.Length))
            (Map.ofList
                [ "kind", "zarr-located-chunks"
                  "path", path
                  "width", string sizeX
                  "height", string sizeY
                  "depth", string sizeZ
                  "chunkX", string chunkX
                  "chunkY", string chunkY
                  "chunkZ", string chunkZ
                  "pixelType", typeof<'T>.Name ])

    let mapper (outputIndex: int) : LocatedChunk<'T> =
        let chunkRef = chunks[outputIndex]
        if pl.debug then
            printfn $"[readZarrLocatedChunks] Reading chunk index ({chunkRef.ChunkCoord[4]}, {chunkRef.ChunkCoord[3]}, {chunkRef.ChunkCoord[2]}) from {path} as {typeof<'T>.Name}"

        let located = readDecodedZarrChunkToLocatedChunk<'T> array layout chunkX chunkY chunkZ chunkRef
        deleteZarrNetDebugLogs ()
        located

    let transition = ProfileTransition.create Unit Streaming
    let memPeak = elementCount * elementBytes
    let memoryNeed = fun _ -> memPeak
    let elementTransformation n = n * uint64 chunkZ
    let memoryModel = StageMemoryModel.fromSinglePeak Source memoryNeed
    let timeCostModel =
        StageTimeCostModel.ioRead Source (Some $"readZarrLocatedChunks.{typeof<'T>.Name}") (fun _ -> memPeak) (fun _ -> uint64 chunkZ)
    let stage =
        Stage.init name (uint chunks.Length) mapper transition memoryNeed elementTransformation
        |> withCostModel (StageCostModel.create memoryModel timeCostModel)
        |> Some

    Plan.createWithOptimizer stage pl.memAvail memPeak memPeak (uint64 chunks.Length) pl.debug pl.optimize
    |> Plan.withRuntimeOptionsFrom pl
    |> Plan.withSourcePeek sourcePeek

let readZarrEncodedChunks
    (path: string)
    (multiscaleIndex: int)
    (datasetIndex: int)
    (timepoint: int)
    (channel: int)
    (pl: Plan<unit, unit>)
    : Plan<unit, EncodedLocatedChunk> =

    suppressZarrNetDebugLogging ()

    let name = "readZarrEncodedChunks"
    let level = openZarrResolutionLevel path multiscaleIndex datasetIndex
    let sizeT, sizeC, sizeZ, sizeY, sizeX = zarrShapeTCZYX level.Shape

    if timepoint < 0 || timepoint >= sizeT then
        invalidArg "timepoint" $"Timepoint {timepoint} is outside the Zarr time range 0..{sizeT - 1}."
    if channel < 0 || channel >= sizeC then
        invalidArg "channel" $"Channel {channel} is outside the Zarr channel range 0..{sizeC - 1}."

    let store = new LocalFileSystemStore(path)
    let rootGroup =
        ZarrGroup.OpenRootAsync(store, CancellationToken.None)
        |> runTask
    let array =
        rootGroup.OpenArrayAsync(level.Dataset.Path, CancellationToken.None)
        |> runTask
    let chunks =
        collectZarrChunks array
        |> Array.filter (fun chunkRef ->
            chunkRef.Origin[0] = int64 timepoint
            && chunkRef.Origin[1] = int64 channel)

    let rawChunkShape = array.Metadata.ChunkShape
    let chunkZ = rawChunkShape[2]
    let chunkY = rawChunkShape[3]
    let chunkX = rawChunkShape[4]
    let layout = spatialZarrLayout level.DataType sizeX sizeY sizeZ chunkX chunkY chunkZ
    let elementBytes =
        match level.DataType.ToLowerInvariant() with
        | "uint8" -> 1UL
        | "uint16" -> 2UL
        | "float32" -> 4UL
        | "float64" -> 8UL
        | other -> failwith $"readZarrEncodedChunks currently supports scalar numeric chunks, got {other}."
    let chunkBytes = uint64 chunkX * uint64 chunkY * uint64 chunkZ * elementBytes
    let maxParallelReads = max 1 (min Environment.ProcessorCount 16)
    let mutable cachedBatchStart = -1
    let mutable cachedBatchResults: IReadOnlyList<ZarrEncodedChunk> = Array.Empty<ZarrEncodedChunk>() :> IReadOnlyList<ZarrEncodedChunk>
    let sourcePeek =
        SourcePeek.create
            name
            chunkBytes
            (Some (uint64 chunks.Length))
            (Map.ofList
                [ "kind", "zarr-encoded-chunks"
                  "path", path
                  "width", string sizeX
                  "height", string sizeY
                  "depth", string sizeZ
                  "chunkX", string chunkX
                  "chunkY", string chunkY
                  "chunkZ", string chunkZ
                  "pixelType", level.DataType ])

    let loadBatch outputIndex =
        let batchStart = (outputIndex / maxParallelReads) * maxParallelReads
        let batchStop = min chunks.Length (batchStart + maxParallelReads)
        let batchChunks = ArraySegment<ZarrChunkRef>(chunks, batchStart, batchStop - batchStart)
        let batchResults =
            array.ReadChunksEncodedAsync(batchChunks, maxParallelReads, CancellationToken.None)
            |> runTask
        deleteZarrNetDebugLogs ()
        cachedBatchStart <- batchStart
        cachedBatchResults <- batchResults

    let mapper (outputIndex: int) : EncodedLocatedChunk =
        if cachedBatchStart < 0
           || outputIndex < cachedBatchStart
           || outputIndex >= cachedBatchStart + cachedBatchResults.Count then
            loadBatch outputIndex

        let chunkRef = chunks[outputIndex]
        if pl.debug then
            printfn $"[readZarrEncodedChunks] Reading encoded chunk index ({chunkRef.ChunkCoord[4]}, {chunkRef.ChunkCoord[3]}, {chunkRef.ChunkCoord[2]}) from {path}"

        let payload =
            let encoded = cachedBatchResults[outputIndex - cachedBatchStart]
            if encoded.IsPresent then
                encoded.EncodedBytes |> Some
            else
                None

        { Index = (int chunkRef.ChunkCoord[4], int chunkRef.ChunkCoord[3], int chunkRef.ChunkCoord[2])
          Layout = layout
          Payload = payload }

    let transition = ProfileTransition.create Unit Streaming
    let memoryNeed = fun _ -> chunkBytes
    let memoryModel = StageMemoryModel.fromSinglePeak Source memoryNeed
    let timeCostModel =
        StageTimeCostModel.ioRead Source (Some "readZarrEncodedChunks") (fun _ -> chunkBytes) (fun _ -> uint64 chunkZ)
    let stage =
        Stage.init name (uint chunks.Length) mapper transition memoryNeed id
        |> withCostModel (StageCostModel.create memoryModel timeCostModel)
        |> Some

    Plan.createWithOptimizer stage pl.memAvail chunkBytes chunkBytes (uint64 chunks.Length) pl.debug pl.optimize
    |> Plan.withRuntimeOptionsFrom pl
    |> Plan.withSourcePeek sourcePeek

let writeZarrEncodedChunksWithCompression
    (compression: ZarrCompression)
    (outputPath: string)
    (name: string)
    (physicalSizeX: float)
    (physicalSizeY: float)
    (physicalSizeZ: float)
    (maxConcurrentWrites: int)
    : Stage<EncodedLocatedChunk, unit> =

    suppressZarrNetDebugLogging ()

    let mutable writer: OmeZarrWriter option = None
    let mutable zarrStore: LocalFileSystemStore option = None
    let mutable zarrArray: ZarrArray option = None
    let mutable outputChunkLookup: Map<string, ZarrChunkRef> = Map.empty
    let mutable layout: ChunkLayout option = None
    let mutable written = 0
    let writeParallelism =
        if maxConcurrentWrites > 0 then
            maxConcurrentWrites
        else
            max 1 (min Environment.ProcessorCount 16)
    let pendingEncoded = ResizeArray<ZarrEncodedChunk>()

    let createWriter (chunkLayout: ChunkLayout) =
        let sizeX64, sizeY64, sizeZ64 = chunkLayout.VolumeSize
        let chunkX64, chunkY64, chunkZ64 = chunkLayout.ChunkSize
        let descriptor =
            BioImageDescriptor(
                int sizeX64,
                int sizeY64,
                ZCT(int sizeZ64, 1, 1),
                Name = name,
                DataType = chunkLayout.PixelType,
                ChunkX = int chunkX64,
                ChunkY = int chunkY64,
                ChunkZ = int chunkZ64,
                ChunkC = 1,
                ChunkT = 1,
                PhysicalSizeX = physicalSizeX,
                PhysicalSizeY = physicalSizeY,
                PhysicalSizeZ = physicalSizeZ,
                Compression = compression)

        let created =
            OmeZarrWriter.CreateAsync(outputPath, descriptor, CancellationToken.None)
            |> runTask
        deleteZarrNetDebugLogs ()

        if maxConcurrentWrites > 0 then
            OmeZarrWriter.MaxConcurrentWrites <- maxConcurrentWrites

        let store = new LocalFileSystemStore(outputPath)
        let rootGroup =
            ZarrGroup.OpenRootAsync(store, CancellationToken.None)
            |> runTask
        let array =
            rootGroup.OpenArrayAsync("0", CancellationToken.None)
            |> runTask

        writer <- Some created
        zarrStore <- Some store
        zarrArray <- Some array
        outputChunkLookup <-
            collectZarrChunks array
            |> Array.map (fun chunkRef -> zarrChunkCoordKey chunkRef, chunkRef)
            |> Map.ofArray
        layout <- Some chunkLayout
        array

    let finishWriter () =
        match writer with
        | Some zarrWriter ->
            zarrWriter.DisposeAsync().AsTask()
            |> runUnitTask
        | None -> ()

        match zarrStore with
        | Some store ->
            store.DisposeAsync().AsTask()
            |> runUnitTask
        | None -> ()

        writer <- None
        zarrStore <- None
        zarrArray <- None
        outputChunkLookup <- Map.empty
        pendingEncoded.Clear()
        deleteZarrNetDebugLogs ()

    let flushPending (array: ZarrArray) =
        if pendingEncoded.Count > 0 then
            array.WriteChunksEncodedAsync(pendingEncoded, writeParallelism, false, CancellationToken.None)
            |> runUnitTask
            pendingEncoded.Clear()

    let mapper (debug: bool) (_idx: int64) (located: EncodedLocatedChunk) =
        let array =
            match layout, zarrArray with
            | Some expected, Some array ->
                if expected <> located.Layout then
                    failwith $"writeZarrEncodedChunks expected layout {expected}, got {located.Layout}."
                array
            | None, None ->
                createWriter located.Layout
            | _ ->
                failwith "writeZarrEncodedChunks has inconsistent writer state."

        try
            let ix, iy, iz = located.Index
            let key = String.Join("/", [| 0L; 0L; int64 iz; int64 iy; int64 ix |])
            match located.Payload with
            | Some payload ->
                let outputChunk =
                    match outputChunkLookup.TryFind key with
                    | Some chunk -> chunk
                    | None -> failwith $"writeZarrEncodedChunks could not find output chunk for coordinate {key}."
                if debug then
                    printfn $"[writeZarrEncodedChunks] Saved encoded chunk index {located.Index} to {outputPath}"
                pendingEncoded.Add(ZarrEncodedChunk.Present(outputChunk, payload))
                if pendingEncoded.Count >= writeParallelism then
                    flushPending array
            | None ->
                if debug then
                    printfn $"[writeZarrEncodedChunks] Skipped missing encoded chunk index {located.Index} in {outputPath}"

            written <- written + 1
            let countX, countY, countZ = located.Layout.ChunkCounts
            if written = countX * countY * countZ then
                flushPending array
                finishWriter ()
        with
        | ex ->
            finishWriter ()
            raise ex

    let memoryNeed (nBytes: uint64) = nBytes
    Stage.mapi $"writeZarrEncodedChunks \"{outputPath}\"" mapper memoryNeed id

let writeZarrEncodedChunks
    (outputPath: string)
    (name: string)
    (physicalSizeX: float)
    (physicalSizeY: float)
    (physicalSizeZ: float)
    (maxConcurrentWrites: int)
    : Stage<EncodedLocatedChunk, unit> =

    writeZarrEncodedChunksWithCompression
        ZarrCompression.BloscLz4
        outputPath
        name
        physicalSizeX
        physicalSizeY
        physicalSizeZ
        maxConcurrentWrites

let writeZarrLocatedChunksWithCompression<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (compression: ZarrCompression)
    (outputPath: string)
    (name: string)
    (physicalSizeX: float)
    (physicalSizeY: float)
    (physicalSizeZ: float)
    (maxConcurrentWrites: int)
    : Stage<LocatedChunk<'T>, unit> =

    suppressZarrNetDebugLogging ()

    let dataType = zarrDataType<'T> ()
    let elementBytes = zarrScalarElementBytes<'T> ()
    let mutable writer: OmeZarrWriter option = None
    let mutable zarrStore: LocalFileSystemStore option = None
    let mutable zarrArray: ZarrArray option = None
    let mutable layout: ChunkLayout option = None
    let mutable written = 0

    let createWriter (chunkLayout: ChunkLayout) =
        let sizeX64, sizeY64, sizeZ64 = chunkLayout.VolumeSize
        let chunkX64, chunkY64, chunkZ64 = chunkLayout.ChunkSize
        let sizeX = int sizeX64
        let sizeY = int sizeY64
        let sizeZ = int sizeZ64
        let chunkX = int chunkX64
        let chunkY = int chunkY64
        let chunkZ = int chunkZ64

        if sizeX <= 0 || sizeY <= 0 || sizeZ <= 0 then
            invalidArg "layout" $"writeZarrLocatedChunks expects positive volume size, got {chunkLayout.VolumeSize}."
        if chunkX <= 0 || chunkY <= 0 || chunkZ <= 0 then
            invalidArg "layout" $"writeZarrLocatedChunks expects positive chunk size, got {chunkLayout.ChunkSize}."

        let descriptor =
            BioImageDescriptor(
                sizeX,
                sizeY,
                ZCT(sizeZ, 1, 1),
                Name = name,
                DataType = dataType,
                ChunkX = chunkX,
                ChunkY = chunkY,
                ChunkZ = chunkZ,
                ChunkC = 1,
                ChunkT = 1,
                PhysicalSizeX = physicalSizeX,
                PhysicalSizeY = physicalSizeY,
                PhysicalSizeZ = physicalSizeZ,
                Compression = compression)

        let created =
            OmeZarrWriter.CreateAsync(outputPath, descriptor, CancellationToken.None)
            |> runTask
        deleteZarrNetDebugLogs ()

        if maxConcurrentWrites > 0 then
            OmeZarrWriter.MaxConcurrentWrites <- maxConcurrentWrites

        let store = new LocalFileSystemStore(outputPath)
        let rootGroup =
            ZarrGroup.OpenRootAsync(store, CancellationToken.None)
            |> runTask
        let array =
            rootGroup.OpenArrayAsync("0", CancellationToken.None)
            |> runTask

        writer <- Some created
        zarrStore <- Some store
        zarrArray <- Some array
        layout <- Some chunkLayout
        array

    let finishWriter () =
        match writer with
        | Some zarrWriter ->
            zarrWriter.DisposeAsync().AsTask()
            |> runUnitTask
        | None -> ()

        match zarrStore with
        | Some store ->
            store.DisposeAsync().AsTask()
            |> runUnitTask
        | None -> ()

        writer <- None
        zarrStore <- None
        zarrArray <- None
        deleteZarrNetDebugLogs ()

    let writeLocatedChunk (array: ZarrArray) (located: LocatedChunk<'T>) =
        let chunkX64, chunkY64, chunkZ64 = located.Layout.ChunkSize
        let volumeX64, volumeY64, volumeZ64 = located.Layout.VolumeSize
        let chunkX = int chunkX64
        let chunkY = int chunkY64
        let chunkZ = int chunkZ64
        let ix, iy, iz = located.Index
        let actualX64, actualY64, actualZ64 = located.Chunk.Size
        let actualX = int actualX64
        let actualY = int actualY64
        let actualZ = int actualZ64
        let originX = uint64 ix * chunkX64
        let originY = uint64 iy * chunkY64
        let originZ = uint64 iz * chunkZ64

        if originX >= volumeX64 || originY >= volumeY64 || originZ >= volumeZ64 then
            invalidArg "located" $"writeZarrLocatedChunks chunk index {located.Index} is outside volume {located.Layout.VolumeSize}."
        if originX + actualX64 > volumeX64 || originY + actualY64 > volumeY64 || originZ + actualZ64 > volumeZ64 then
            invalidArg "located" $"writeZarrLocatedChunks chunk index {located.Index} with size {located.Chunk.Size} exceeds volume {located.Layout.VolumeSize}."

        let expectedBytes = actualX * actualY * actualZ * elementBytes
        if located.Chunk.ByteLength <> expectedBytes then
            failwith $"writeZarrLocatedChunks expected {expectedBytes} bytes, got {located.Chunk.ByteLength}."

        if actualX = chunkX && actualY = chunkY && actualZ = chunkZ then
            let coord = [| 0L; 0L; int64 iz; int64 iy; int64 ix |]
            array.WriteChunkDecodedAsync(
                coord,
                ReadOnlyMemory<byte>(located.Chunk.Bytes, 0, located.Chunk.ByteLength),
                true,
                CancellationToken.None)
            |> runUnitTask
        else
            let regionStart =
                [| 0L
                   0L
                   int64 originZ
                   int64 originY
                   int64 originX |]
            let regionEnd =
                [| 1L
                   1L
                   int64 (originZ + actualZ64)
                   int64 (originY + actualY64)
                   int64 (originX + actualX64) |]
            array.WriteRegionAsync(regionStart, regionEnd, located.Chunk.Bytes[0 .. located.Chunk.ByteLength - 1], CancellationToken.None)
            |> runUnitTask

    let mapper (debug: bool) (_idx: int64) (located: LocatedChunk<'T>) =
        try
            if located.Layout.PixelType <> dataType then
                invalidArg "located" $"writeZarrLocatedChunks expected {dataType} chunks, got {located.Layout.PixelType}."
            if located.Layout.Components <> 1u then
                invalidArg "located" $"writeZarrLocatedChunks supports scalar chunks only, got {located.Layout.Components} components."

            let array =
                match layout, zarrArray with
                | Some expected, Some array ->
                    if expected <> located.Layout then
                        failwith $"writeZarrLocatedChunks expected layout {expected}, got {located.Layout}."
                    array
                | None, None ->
                    createWriter located.Layout
                | _ ->
                    failwith "writeZarrLocatedChunks has inconsistent writer state."

            if debug then
                printfn $"[writeZarrLocatedChunks] Saved chunk index {located.Index} to {outputPath} as {dataType}"

            writeLocatedChunk array located
            written <- written + 1

            let countX, countY, countZ = located.Layout.ChunkCounts
            if written = countX * countY * countZ then
                finishWriter ()
        finally
            Chunk.decRef located.Chunk

    let memoryNeed nPixels =
        nPixels * uint64 elementBytes

    Stage.mapi $"writeZarrLocatedChunks \"{outputPath}\"" mapper memoryNeed id

let writeZarrLocatedChunks<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (outputPath: string)
    (name: string)
    (physicalSizeX: float)
    (physicalSizeY: float)
    (physicalSizeZ: float)
    (maxConcurrentWrites: int)
    : Stage<LocatedChunk<'T>, unit> =

    writeZarrLocatedChunksWithCompression
        ZarrCompression.BloscLz4
        outputPath
        name
        physicalSizeX
        physicalSizeY
        physicalSizeZ
        maxConcurrentWrites

let icompare s1 s2  = 
    System.String.Equals(s1, s2, System.StringComparison.CurrentCultureIgnoreCase)

let rnd = new System.Random()
let rec getUnusedDirectoryName dir =
    if not (Directory.Exists(dir)) then
        dir
    else
        getUnusedDirectoryName (dir + string (rnd.Next(9)))

let deleteIfExists dir =
    if System.IO.Directory.Exists(dir) then 
        System.IO.Directory.Delete(dir,true)

let private cleanImageSeriesFiles outputDir suffix =
    if Directory.Exists(outputDir) then
        let pattern = sprintf "image_*%s" suffix
        Directory.GetFiles(outputDir, pattern)
        |> Array.iter File.Delete

let private cleanChunkFiles outputDir suffix =
    if Directory.Exists(outputDir) then
        let pattern = sprintf "chunk*%s" suffix
        Directory.GetFiles(outputDir, pattern)
        |> Array.iter File.Delete


let private writeChunkThickCoreWithOptions<'T when 'T: equality> options split3DChunks (outputDir: string) (suffix: string) : Stage<Chunk<'T>, unit> =
    ensureDirectTiffChunkWrite<'T> suffix
    if not split3DChunks && options <> defaultTiffWriteOptions then
        invalidArg "options" "Compressed TIFF writeThickFiles output is not supported for multipage chunk files yet. Use split writeThick or default TIFF options."

    Directory.CreateDirectory(outputDir) |> ignore
    let cleaned = lazy (cleanImageSeriesFiles outputDir suffix)
    let indexLock = obj()
    let mutable nextSliceIndex = 0L
    let writeSliceFromOffset = scalarTiffWriterForOptions<'T> options

    let mapper (debug: bool) (idx: int64) (chunk: Chunk<'T>) =
        cleaned.Force()
        try
            let width, height, depth = chunk.Size
            if depth = 0UL then
                invalidArg "chunk" $"writeChunkThick cannot write an empty-depth chunk: {chunk.Size}."

            let _bitsPerSample, _sampleFormat, bytesPerSample = tiffPixelLayout<'T> ()
            let sliceBytes = int width * int height * bytesPerSample
            if chunk.ByteLength < sliceBytes * int depth then
                invalidArg "chunk" $"Chunk byte length {chunk.ByteLength} is smaller than {depth} TIFF slice payloads of {sliceBytes} bytes."

            if split3DChunks then
                let firstSliceIndex =
                    lock indexLock (fun () ->
                        let first = nextSliceIndex
                        nextSliceIndex <- nextSliceIndex + int64 depth
                        first)

                for localZ in 0 .. int depth - 1 do
                    let sliceIndex = firstSliceIndex + int64 localZ
                    let fileName = Path.Combine(outputDir, sprintf "image_%03d%s" sliceIndex suffix)
                    let sliceOffset = localZ * sliceBytes
                    if debug then
                        if depth = 1UL then
                            printfn $"[writeChunkThick] Saved chunk slice {sliceIndex} to {fileName} as {typeof<'T>.Name}"
                        else
                            printfn $"[writeChunkThick] Saved chunk {idx} local z {localZ} as slice {sliceIndex} to {fileName} as {typeof<'T>.Name}"
                    writeSliceFromOffset fileName chunk width height sliceOffset
            else
                let fileName = Path.Combine(outputDir, sprintf "image_%03d%s" idx suffix)
                if debug then
                    printfn $"[writeChunkThick] Saved chunk {idx} with depth {depth} to {fileName} as {typeof<'T>.Name}"
                writeChunkTiffFile<'T> fileName chunk
        finally
            Chunk.decRef chunk

    let memoryNeed = id
    let memoryModel = StageMemoryModel.fromSinglePeak Iter memoryNeed
    let timeCostModel =
        imageIoCost<'T>
            "write"
            Iter
            $"writeChunkThick.{suffixCostLabel suffix}.{typeof<'T>.Name}"
            (fun input -> inputValue input)
            (fun _ -> 1UL)

    Stage.mapi $"writeChunkThick \"{outputDir}/*{suffix}\"" mapper memoryNeed id
    |> withCostModel (StageCostModel.create memoryModel timeCostModel)

let private writeChunkThickCore<'T when 'T: equality> split3DChunks (outputDir: string) (suffix: string) : Stage<Chunk<'T>, unit> =
    writeChunkThickCoreWithOptions<'T> defaultTiffWriteOptions split3DChunks outputDir suffix

let writeChunkSlicesWithOptions<'T when 'T: equality> options (outputDir: string) (suffix: string) : Stage<Chunk<'T>, unit> =
    ensureDirectTiffChunkWrite<'T> suffix
    Directory.CreateDirectory(outputDir) |> ignore
    let cleaned = lazy (cleanImageSeriesFiles outputDir suffix)
    let writeSliceFromOffset = scalarTiffWriterForOptions<'T> options

    let mapper (debug: bool) (idx: int64) (chunk: Chunk<'T>) =
        cleaned.Force()
        try
            let width, height, depth = chunk.Size
            if depth <> 1UL then
                invalidArg "chunk" $"write expects slice chunks with depth 1. Use writeThick for depth {depth} chunks."

            let fileName = Path.Combine(outputDir, sprintf "image_%03d%s" idx suffix)
            if debug then
                printfn $"[writeChunkSlices] Saved chunk slice {idx} to {fileName} as {typeof<'T>.Name}"
            writeSliceFromOffset fileName chunk width height 0
        finally
            Chunk.decRef chunk

    let memoryNeed = id
    let memoryModel = StageMemoryModel.fromSinglePeak Iter memoryNeed
    let timeCostModel =
        imageIoCost<'T>
            "write"
            Iter
            $"writeChunkSlices.{suffixCostLabel suffix}.{typeof<'T>.Name}"
            (fun input -> inputValue input)
            (fun _ -> 1UL)

    Stage.mapi $"writeChunkSlices \"{outputDir}/*{suffix}\"" mapper memoryNeed id
    |> withCostModel (StageCostModel.create memoryModel timeCostModel)

let writeChunkSlices<'T when 'T: equality> (outputDir: string) (suffix: string) : Stage<Chunk<'T>, unit> =
    writeChunkSlicesWithOptions<'T> defaultTiffWriteOptions outputDir suffix

let private complexLogicalShape stageName componentLabel (chunk: Chunk<'T>) =
    let storageWidth, height, depth = chunk.Size
    if storageWidth = 0UL || storageWidth % 2UL <> 0UL then
        invalidArg "chunk" $"{stageName} expects {componentLabel}-interleaved chunks with even nonzero storage width, got {chunk.Size}."
    storageWidth / 2UL, height, depth

let private complex64LogicalShape stageName (chunk: Chunk<float32>) =
    complexLogicalShape stageName "complex64 Float32" chunk

let private complex128LogicalShape stageName (chunk: Chunk<float>) =
    complexLogicalShape stageName "complex128 Float64" chunk

let writeComplex64ChunkSlices (outputDir: string) (suffix: string) : Stage<Chunk<float32>, unit> =
    if not (isTiffStackSuffix suffix) then
        invalidArg "suffix" $"writeComplex64 currently supports TIFF stack output only; got suffix '{suffix}'."
    Directory.CreateDirectory(outputDir) |> ignore
    let cleaned = lazy (cleanImageSeriesFiles outputDir suffix)

    let mapper (debug: bool) (idx: int64) (chunk: Chunk<float32>) =
        cleaned.Force()
        try
            let logicalWidth, height, depth = complex64LogicalShape "writeComplex64" chunk
            if depth <> 1UL then
                invalidArg "chunk" $"writeComplex64 expects slice chunks with depth 1. Use writeComplex64Thick for depth {depth} chunks."

            let fileName = Path.Combine(outputDir, sprintf "image_%03d%s" idx suffix)
            if debug then
                printfn $"[writeComplex64] Saved complex64 chunk slice {idx} to {fileName}"
            writeComplex64TiffSliceFromOffset fileName chunk logicalWidth height 0
        finally
            Chunk.decRef chunk

    let memoryNeed = id
    let memoryModel = StageMemoryModel.fromSinglePeak Iter memoryNeed
    let timeCostModel =
        imageIoCost<float32>
            "write"
            Iter
            $"writeComplex64.{suffixCostLabel suffix}.Complex64"
            (fun input -> inputValue input)
            (fun _ -> 1UL)

    Stage.mapi $"writeComplex64 \"{outputDir}/*{suffix}\"" mapper memoryNeed id
    |> withCostModel (StageCostModel.create memoryModel timeCostModel)

let writeComplex128ChunkSlices (outputDir: string) (suffix: string) : Stage<Chunk<float>, unit> =
    if not (isTiffStackSuffix suffix) then
        invalidArg "suffix" $"writeComplex128 currently supports TIFF stack output only; got suffix '{suffix}'."
    Directory.CreateDirectory(outputDir) |> ignore
    let cleaned = lazy (cleanImageSeriesFiles outputDir suffix)

    let mapper (debug: bool) (idx: int64) (chunk: Chunk<float>) =
        cleaned.Force()
        try
            let logicalWidth, height, depth = complex128LogicalShape "writeComplex128" chunk
            if depth <> 1UL then
                invalidArg "chunk" $"writeComplex128 expects slice chunks with depth 1. Use writeComplex128Thick for depth {depth} chunks."

            let fileName = Path.Combine(outputDir, sprintf "image_%03d%s" idx suffix)
            if debug then
                printfn $"[writeComplex128] Saved complex128 chunk slice {idx} to {fileName}"
            writeComplex128TiffSliceFromOffset fileName chunk logicalWidth height 0
        finally
            Chunk.decRef chunk

    let memoryNeed = id
    let memoryModel = StageMemoryModel.fromSinglePeak Iter memoryNeed
    let timeCostModel =
        imageIoCost<float>
            "write"
            Iter
            $"writeComplex128.{suffixCostLabel suffix}.Complex128"
            (fun input -> inputValue input)
            (fun _ -> 1UL)

    Stage.mapi $"writeComplex128 \"{outputDir}/*{suffix}\"" mapper memoryNeed id
    |> withCostModel (StageCostModel.create memoryModel timeCostModel)

let writeChunkThick<'T when 'T: equality> (outputDir: string) (suffix: string) : Stage<Chunk<'T>, unit> =
    writeChunkThickCore<'T> true outputDir suffix

let writeChunkThickWithOptions<'T when 'T: equality> options (outputDir: string) (suffix: string) : Stage<Chunk<'T>, unit> =
    writeChunkThickCoreWithOptions<'T> options true outputDir suffix

let writeChunkThickFiles<'T when 'T: equality> (outputDir: string) (suffix: string) : Stage<Chunk<'T>, unit> =
    writeChunkThickCore<'T> false outputDir suffix

let writeChunkThickFilesWithOptions<'T when 'T: equality> options (outputDir: string) (suffix: string) : Stage<Chunk<'T>, unit> =
    writeChunkThickCoreWithOptions<'T> options false outputDir suffix

let writeComplex64ChunkThick (outputDir: string) (suffix: string) : Stage<Chunk<float32>, unit> =
    if not (isTiffStackSuffix suffix) then
        invalidArg "suffix" $"writeComplex64Thick currently supports TIFF stack output only; got suffix '{suffix}'."
    Directory.CreateDirectory(outputDir) |> ignore
    let cleaned = lazy (cleanImageSeriesFiles outputDir suffix)

    let mapper (debug: bool) (idx: int64) (chunk: Chunk<float32>) =
        cleaned.Force()
        try
            let logicalWidth, height, depth = complex64LogicalShape "writeComplex64Thick" chunk
            if depth = 0UL then
                invalidArg "chunk" $"writeComplex64Thick cannot write an empty-depth chunk: {chunk.Size}."

            let sliceBytes = int logicalWidth * int height * 2 * sizeof<float32>
            for localZ in 0 .. int depth - 1 do
                let sliceIndex = idx + int64 localZ
                let fileName = Path.Combine(outputDir, sprintf "image_%03d%s" sliceIndex suffix)
                let sliceOffset = localZ * sliceBytes
                if debug then
                    printfn $"[writeComplex64Thick] Saved complex64 chunk {idx} local z {localZ} as slice {sliceIndex} to {fileName}"
                writeComplex64TiffSliceFromOffset fileName chunk logicalWidth height sliceOffset
        finally
            Chunk.decRef chunk

    let memoryNeed = id
    let memoryModel = StageMemoryModel.fromSinglePeak Iter memoryNeed
    let timeCostModel =
        imageIoCost<float32>
            "write"
            Iter
            $"writeComplex64Thick.{suffixCostLabel suffix}.Complex64"
            (fun input -> inputValue input)
            (fun _ -> 1UL)

    Stage.mapi $"writeComplex64Thick \"{outputDir}/*{suffix}\"" mapper memoryNeed id
    |> withCostModel (StageCostModel.create memoryModel timeCostModel)

let writeComplex128ChunkThick (outputDir: string) (suffix: string) : Stage<Chunk<float>, unit> =
    if not (isTiffStackSuffix suffix) then
        invalidArg "suffix" $"writeComplex128Thick currently supports TIFF stack output only; got suffix '{suffix}'."
    Directory.CreateDirectory(outputDir) |> ignore
    let cleaned = lazy (cleanImageSeriesFiles outputDir suffix)

    let mapper (debug: bool) (idx: int64) (chunk: Chunk<float>) =
        cleaned.Force()
        try
            let logicalWidth, height, depth = complex128LogicalShape "writeComplex128Thick" chunk
            if depth = 0UL then
                invalidArg "chunk" $"writeComplex128Thick cannot write an empty-depth chunk: {chunk.Size}."

            let sliceBytes = int logicalWidth * int height * 2 * sizeof<float>
            for localZ in 0 .. int depth - 1 do
                let sliceIndex = idx + int64 localZ
                let fileName = Path.Combine(outputDir, sprintf "image_%03d%s" sliceIndex suffix)
                let sliceOffset = localZ * sliceBytes
                if debug then
                    printfn $"[writeComplex128Thick] Saved complex128 chunk {idx} local z {localZ} as slice {sliceIndex} to {fileName}"
                writeComplex128TiffSliceFromOffset fileName chunk logicalWidth height sliceOffset
        finally
            Chunk.decRef chunk

    let memoryNeed = id
    let memoryModel = StageMemoryModel.fromSinglePeak Iter memoryNeed
    let timeCostModel =
        imageIoCost<float>
            "write"
            Iter
            $"writeComplex128Thick.{suffixCostLabel suffix}.Complex128"
            (fun input -> inputValue input)
            (fun _ -> 1UL)

    Stage.mapi $"writeComplex128Thick \"{outputDir}/*{suffix}\"" mapper memoryNeed id
    |> withCostModel (StageCostModel.create memoryModel timeCostModel)

let writeColorChunkSlices (outputDir: string) (suffix: string) : Stage<VectorChunk<uint8>, unit> =
    if not (isTiffStackSuffix suffix) then
        invalidArg "suffix" $"writeColorChunkSlices currently supports TIFF stack output only; got suffix '{suffix}'."
    Directory.CreateDirectory(outputDir) |> ignore
    let cleaned = lazy (cleanImageSeriesFiles outputDir suffix)

    let mapper (debug: bool) (idx: int64) (vector: VectorChunk<uint8>) =
        cleaned.Force()
        let fileName = Path.Combine(outputDir, sprintf "image_%03d%s" idx suffix)
        try
            if debug then
                printfn $"[writeColorChunkSlices] Saved RGB chunk slice {idx} to {fileName}"
            writeColorChunkTiffSlice fileName vector
        finally
            Chunk.decRef vector.Chunk

    let memoryNeed = id
    let memoryModel = StageMemoryModel.fromSinglePeak Iter memoryNeed
    let timeCostModel =
        imageIoCost<uint8>
            "write"
            Iter
            $"writeColorChunkSlices.{suffixCostLabel suffix}.Color"
            (fun input -> inputValue input)
            (fun _ -> 1UL)

    Stage.mapi $"writeColorChunkSlices \"{outputDir}/*{suffix}\"" mapper memoryNeed id
    |> withCostModel (StageCostModel.create memoryModel timeCostModel)


let writeZarrChunkSlicesWithCompression<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (compression: ZarrCompression)
    (outputPath: string)
    (name: string)
    (depth: uint)
    (chunkX: uint)
    (chunkY: uint)
    (chunkZ: uint)
    (physicalSizeX: float)
    (physicalSizeY: float)
    (physicalSizeZ: float)
    (maxConcurrentWrites: int)
    : Stage<Chunk<'T>, unit> =

    suppressZarrNetDebugLogging ()

    let dataType = zarrDataType<'T> ()
    let elementBytes = zarrScalarElementBytes<'T> ()
    let depth = int depth
    let chunkX = max 1u chunkX |> int
    let chunkY = max 1u chunkY |> int
    let chunkZ = max 1u chunkZ |> int
    let mutable writer: OmeZarrWriter option = None
    let mutable width = 0
    let mutable height = 0

    let createWriter chunkWidth chunkHeight =
        let descriptor =
            BioImageDescriptor(
                chunkWidth,
                chunkHeight,
                ZCT(depth, 1, 1),
                Name = name,
                DataType = dataType,
                ChunkX = chunkX,
                ChunkY = chunkY,
                ChunkZ = chunkZ,
                ChunkC = 1,
                ChunkT = 1,
                PhysicalSizeX = physicalSizeX,
                PhysicalSizeY = physicalSizeY,
                PhysicalSizeZ = physicalSizeZ,
                Compression = compression)

        let created =
            OmeZarrWriter.CreateAsync(outputPath, descriptor, CancellationToken.None)
            |> runTask
        deleteZarrNetDebugLogs ()

        if maxConcurrentWrites > 0 then
            OmeZarrWriter.MaxConcurrentWrites <- maxConcurrentWrites

        width <- chunkWidth
        height <- chunkHeight
        writer <- Some created
        created

    let mapper (debug: bool) (idx: int64) (chunk: Chunk<'T>) =
        try
            if depth <= 0 then
                invalidArg "depth" "writeZarrChunkSlices expects positive depth."
            if idx >= int64 depth then
                failwith $"writeZarrChunkSlices received slice {idx}, but declared depth is {depth}."

            let chunkWidth, chunkHeight, chunkDepth = chunk.Size
            if chunkDepth <> 1UL then
                failwith $"writeZarrChunkSlices expects 2D chunks with depth 1, got {chunk.Size}."

            let chunkWidth = int chunkWidth
            let chunkHeight = int chunkHeight
            let zarrWriter =
                match writer with
                | Some writer ->
                    if chunkWidth <> width || chunkHeight <> height then
                        failwith $"writeZarrChunkSlices expected all slices to be {width}x{height}, got {chunkWidth}x{chunkHeight}."
                    writer
                | None -> createWriter chunkWidth chunkHeight

            let expectedBytes = chunkWidth * chunkHeight * elementBytes
            if chunk.ByteLength <> expectedBytes then
                failwith $"writeZarrChunkSlices expected {expectedBytes} bytes, got {chunk.ByteLength}."

            if debug then
                printfn $"[writeZarrChunkSlices] Saved plane {idx} to {outputPath} as {dataType}"

            zarrWriter.WritePlaneAsync(int idx, ReadOnlyMemory<byte>(chunk.Bytes, 0, expectedBytes), CancellationToken.None)
            |> runUnitTask
            deleteZarrNetDebugLogs ()

            if idx = int64 (depth - 1) then
                zarrWriter.DisposeAsync().AsTask()
                |> runUnitTask
                writer <- None
                deleteZarrNetDebugLogs ()
        finally
            Chunk.decRef chunk

    let memoryNeed nPixels =
        nPixels * uint64 (elementBytes * 2)

    Stage.mapi $"writeZarrChunkSlices \"{outputPath}\"" mapper memoryNeed id

let writeZarrChunkThickWithCompression<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (compression: ZarrCompression)
    (outputPath: string)
    (name: string)
    (depth: uint)
    (chunkX: uint)
    (chunkY: uint)
    (chunkZ: uint)
    (physicalSizeX: float)
    (physicalSizeY: float)
    (physicalSizeZ: float)
    (maxConcurrentWrites: int)
    : Stage<Chunk<'T>, unit> =

    suppressZarrNetDebugLogging ()

    let dataType = zarrDataType<'T> ()
    let elementBytes = zarrScalarElementBytes<'T> ()
    let depth = int depth
    let chunkX = max 1u chunkX |> int
    let chunkY = max 1u chunkY |> int
    let chunkZ = max 1u chunkZ |> int
    let mutable writer: OmeZarrWriter option = None
    let mutable zarrStore: LocalFileSystemStore option = None
    let mutable zarrArray: ZarrArray option = None
    let mutable outputChunkLookup: Map<string, ZarrChunkRef> = Map.empty
    let mutable width = 0
    let mutable height = 0
    let mutable nextZ = 0
    let writeParallelism =
        if maxConcurrentWrites > 0 then
            maxConcurrentWrites
        else
            max 1 (min Environment.ProcessorCount 16)
    let pendingDecoded = ResizeArray<ZarrDecodedChunkWrite>()
    let pendingBuffers = ResizeArray<byte[]>()

    let createWriter chunkWidth chunkHeight =
        let descriptor =
            BioImageDescriptor(
                chunkWidth,
                chunkHeight,
                ZCT(depth, 1, 1),
                Name = name,
                DataType = dataType,
                ChunkX = chunkX,
                ChunkY = chunkY,
                ChunkZ = chunkZ,
                ChunkC = 1,
                ChunkT = 1,
                PhysicalSizeX = physicalSizeX,
                PhysicalSizeY = physicalSizeY,
                PhysicalSizeZ = physicalSizeZ,
                Compression = compression)

        let created =
            OmeZarrWriter.CreateAsync(outputPath, descriptor, CancellationToken.None)
            |> runTask
        deleteZarrNetDebugLogs ()

        if maxConcurrentWrites > 0 then
            OmeZarrWriter.MaxConcurrentWrites <- maxConcurrentWrites

        let store = new LocalFileSystemStore(outputPath)
        let rootGroup =
            ZarrGroup.OpenRootAsync(store, CancellationToken.None)
            |> runTask
        let array =
            rootGroup.OpenArrayAsync("0", CancellationToken.None)
            |> runTask

        width <- chunkWidth
        height <- chunkHeight
        writer <- Some created
        zarrStore <- Some store
        zarrArray <- Some array
        outputChunkLookup <-
            collectZarrChunks array
            |> Array.map (fun chunkRef -> zarrChunkCoordKey chunkRef, chunkRef)
            |> Map.ofArray
        created, array

    let flushPending (array: ZarrArray) =
        if pendingDecoded.Count > 0 then
            try
                array.WriteChunksDecodedAsync(pendingDecoded, writeParallelism, true, CancellationToken.None)
                |> runUnitTask
            finally
                for buffer in pendingBuffers do
                    ArrayPool<byte>.Shared.Return(buffer)
                pendingDecoded.Clear()
                pendingBuffers.Clear()

    let clearPending () =
        for buffer in pendingBuffers do
            ArrayPool<byte>.Shared.Return(buffer)
        pendingDecoded.Clear()
        pendingBuffers.Clear()

    let finishWriter () =
        match zarrArray with
        | Some array -> flushPending array
        | None -> ()

        match writer with
        | Some zarrWriter ->
            zarrWriter.DisposeAsync().AsTask()
            |> runUnitTask
        | None -> ()

        match zarrStore with
        | Some store ->
            store.DisposeAsync().AsTask()
            |> runUnitTask
        | None -> ()

        writer <- None
        zarrStore <- None
        zarrArray <- None
        outputChunkLookup <- Map.empty
        clearPending ()
        deleteZarrNetDebugLogs ()

    let copyChunkRegionToBuffer
        (source: Chunk<'T>)
        sourceWidth
        sourceHeight
        localZStart
        yStart
        xStart
        zCount
        yCount
        xCount
        outputZ
        outputY
        outputX
        (buffer: byte[])
        =

        let rowBytes = xCount * elementBytes
        for z in 0 .. zCount - 1 do
            let sourceZ = localZStart + z
            for y in 0 .. yCount - 1 do
                let sourceOffset =
                    ((sourceZ * sourceHeight + (yStart + y)) * sourceWidth + xStart) * elementBytes
                let destinationOffset =
                    ((z * outputY + y) * outputX) * elementBytes
                Buffer.BlockCopy(source.Bytes, sourceOffset, buffer, destinationOffset, rowBytes)

    let writeChunkRegion
        (array: ZarrArray)
        (source: Chunk<'T>)
        sourceWidth
        sourceHeight
        localZStart
        globalZStart
        yStart
        xStart
        zCount
        yCount
        xCount
        =

        let isFullZarrChunk =
            zCount = chunkZ
            && yCount = chunkY
            && xCount = chunkX
            && globalZStart % chunkZ = 0
            && yStart % chunkY = 0
            && xStart % chunkX = 0

        let outputZ, outputY, outputX =
            if isFullZarrChunk then
                chunkZ, chunkY, chunkX
            else
                zCount, yCount, xCount

        let bufferBytes = outputZ * outputY * outputX * elementBytes
        if isFullZarrChunk && compression = ZarrCompression.None then
            let buffer = ArrayPool<byte>.Shared.Rent(bufferBytes)
            try
                copyChunkRegionToBuffer
                    source
                    sourceWidth
                    sourceHeight
                    localZStart
                    yStart
                    xStart
                    zCount
                    yCount
                    xCount
                    outputZ
                    outputY
                    outputX
                    buffer

                let coord =
                    [| 0L
                       0L
                       int64 (globalZStart / chunkZ)
                       int64 (yStart / chunkY)
                       int64 (xStart / chunkX) |]
                let key = String.Join("/", coord)
                let outputChunk =
                    match outputChunkLookup.TryFind key with
                    | Some chunk -> chunk
                    | None -> failwith $"writeZarrChunkThick could not find output chunk for coordinate {key}."
                pendingDecoded.Add(ZarrDecodedChunkWrite(outputChunk, ReadOnlyMemory<byte>(buffer, 0, bufferBytes)))
                pendingBuffers.Add buffer
                if pendingDecoded.Count >= writeParallelism then
                    flushPending array
            with
            | ex ->
                ArrayPool<byte>.Shared.Return(buffer)
                raise ex
        else
            let buffer = ArrayPool<byte>.Shared.Rent(bufferBytes)
            try
                copyChunkRegionToBuffer
                    source
                    sourceWidth
                    sourceHeight
                    localZStart
                    yStart
                    xStart
                    zCount
                    yCount
                    xCount
                    outputZ
                    outputY
                    outputX
                    buffer

                let regionStart =
                    [| 0L
                       0L
                       int64 globalZStart
                       int64 yStart
                       int64 xStart |]
                let regionEnd =
                    [| 1L
                       1L
                       int64 (globalZStart + zCount)
                       int64 (yStart + yCount)
                       int64 (xStart + xCount) |]
                array.WriteRegionAsync(regionStart, regionEnd, buffer[0 .. bufferBytes - 1], CancellationToken.None)
                |> runUnitTask
            finally
                ArrayPool<byte>.Shared.Return(buffer)

    let writeFullChunkRegionsParallel
        (array: ZarrArray)
        (source: Chunk<'T>)
        sourceWidth
        sourceHeight
        localZStart
        globalZStart
        =

        let regions =
            [| for yStart in 0 .. chunkY .. sourceHeight - 1 do
                   let yCount = min chunkY (sourceHeight - yStart)
                   for xStart in 0 .. chunkX .. sourceWidth - 1 do
                       let xCount = min chunkX (sourceWidth - xStart)
                       if yCount = chunkY && xCount = chunkX then
                           yStart, xStart |]

        let bufferBytes = chunkZ * chunkY * chunkX * elementBytes
        let options = ParallelOptions(MaxDegreeOfParallelism = writeParallelism)
        let mutable batchStart = 0

        let batchSize = min regions.Length (writeParallelism * 16)

        while batchStart < regions.Length do
            let batchCount = min batchSize (regions.Length - batchStart)
            let buffers = Array.zeroCreate<byte[]> batchCount
            let chunks = Array.zeroCreate<ZarrDecodedChunkWrite> batchCount
            try
                Parallel.For(
                    0,
                    batchCount,
                    options,
                    fun i ->
                        let yStart, xStart = regions[batchStart + i]
                        let buffer = ArrayPool<byte>.Shared.Rent(bufferBytes)
                        buffers[i] <- buffer
                        copyChunkRegionToBuffer
                            source
                            sourceWidth
                            sourceHeight
                            localZStart
                            yStart
                            xStart
                            chunkZ
                            chunkY
                            chunkX
                            chunkZ
                            chunkY
                            chunkX
                            buffer

                        let coord =
                            [| 0L
                               0L
                               int64 (globalZStart / chunkZ)
                               int64 (yStart / chunkY)
                               int64 (xStart / chunkX) |]
                        let key = String.Join("/", coord)
                        let outputChunk =
                            match outputChunkLookup.TryFind key with
                            | Some chunk -> chunk
                            | None -> failwith $"writeZarrChunkThick could not find output chunk for coordinate {key}."
                        chunks[i] <- ZarrDecodedChunkWrite(outputChunk, ReadOnlyMemory<byte>(buffer, 0, bufferBytes))
                    )
                |> ignore

                array.WriteChunksDecodedAsync(chunks, writeParallelism, true, CancellationToken.None)
                |> runUnitTask
            finally
                for buffer in buffers do
                    if not (isNull buffer) then
                        ArrayPool<byte>.Shared.Return(buffer)

            batchStart <- batchStart + batchCount

    let mapper (debug: bool) (_idx: int64) (chunk: Chunk<'T>) =
        try
            try
                if depth <= 0 then
                    invalidArg "depth" "writeZarrChunkThick expects positive depth."

                let chunkWidth64, chunkHeight64, chunkDepth64 = chunk.Size
                if chunkDepth64 = 0UL then
                    invalidArg "chunk" $"writeZarrChunkThick cannot write an empty-depth chunk: {chunk.Size}."

                let chunkWidth = int chunkWidth64
                let chunkHeight = int chunkHeight64
                let chunkDepth = int chunkDepth64
                let array =
                    match writer, zarrArray with
                    | Some _writer, Some array ->
                        if chunkWidth <> width || chunkHeight <> height then
                            failwith $"writeZarrChunkThick expected all thick chunks to be {width}x{height}, got {chunkWidth}x{chunkHeight}."
                        array
                    | None, None ->
                        createWriter chunkWidth chunkHeight |> snd
                    | _ ->
                        failwith "writeZarrChunkThick has inconsistent writer state."

                let planeBytes = chunkWidth * chunkHeight * elementBytes
                let expectedBytes = planeBytes * chunkDepth
                if chunk.ByteLength <> expectedBytes then
                    failwith $"writeZarrChunkThick expected {expectedBytes} bytes, got {chunk.ByteLength}."
                if nextZ + chunkDepth > depth then
                    failwith $"writeZarrChunkThick received chunk depth {chunkDepth} at z={nextZ}, exceeding declared depth {depth}."

                let mutable localZ = 0
                while localZ < chunkDepth do
                    let globalZ = nextZ + localZ
                    let zOffset = globalZ % chunkZ
                    let zCount = min (chunkDepth - localZ) (chunkZ - zOffset)

                    if
                        not debug
                        && compression = ZarrCompression.None
                        && zCount = chunkZ
                        && zOffset = 0
                        && chunkWidth % chunkX = 0
                        && chunkHeight % chunkY = 0
                    then
                        flushPending array
                        writeFullChunkRegionsParallel
                            array
                            chunk
                            chunkWidth
                            chunkHeight
                            localZ
                            globalZ
                    else
                        for yStart in 0 .. chunkY .. chunkHeight - 1 do
                            let yCount = min chunkY (chunkHeight - yStart)
                            for xStart in 0 .. chunkX .. chunkWidth - 1 do
                                let xCount = min chunkX (chunkWidth - xStart)

                                if debug then
                                    printfn $"[writeZarrChunkThick] Saved region z {globalZ}..{globalZ + zCount - 1}, y {yStart}..{yStart + yCount - 1}, x {xStart}..{xStart + xCount - 1} to {outputPath} as {dataType}"

                                writeChunkRegion
                                    array
                                    chunk
                                    chunkWidth
                                    chunkHeight
                                    localZ
                                    globalZ
                                    yStart
                                    xStart
                                    zCount
                                    yCount
                                    xCount

                    localZ <- localZ + zCount

                nextZ <- nextZ + chunkDepth
                if nextZ = depth then
                    finishWriter ()
            with
            | ex ->
                clearPending ()
                raise ex
        finally
            Chunk.decRef chunk

    let memoryNeed nPixels =
        nPixels * uint64 (elementBytes * 2)

    Stage.mapi $"writeZarrChunkThick \"{outputPath}\"" mapper memoryNeed id

let writeZarrChunkSlices<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (outputPath: string)
    (name: string)
    (depth: uint)
    (chunkX: uint)
    (chunkY: uint)
    (chunkZ: uint)
    (physicalSizeX: float)
    (physicalSizeY: float)
    (physicalSizeZ: float)
    (maxConcurrentWrites: int)
    : Stage<Chunk<'T>, unit> =

    writeZarrChunkSlicesWithCompression
        ZarrCompression.BloscLz4
        outputPath
        name
        depth
        chunkX
        chunkY
        chunkZ
        physicalSizeX
        physicalSizeY
        physicalSizeZ
        maxConcurrentWrites

let writeZarrChunkThick<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (outputPath: string)
    (name: string)
    (depth: uint)
    (chunkX: uint)
    (chunkY: uint)
    (chunkZ: uint)
    (physicalSizeX: float)
    (physicalSizeY: float)
    (physicalSizeZ: float)
    (maxConcurrentWrites: int)
    : Stage<Chunk<'T>, unit> =

    writeZarrChunkThickWithCompression
        ZarrCompression.BloscLz4
        outputPath
        name
        depth
        chunkX
        chunkY
        chunkZ
        physicalSizeX
        physicalSizeY
        physicalSizeZ
        maxConcurrentWrites

let writeZarrChunkSlicesAlignedWithCompression<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (compression: ZarrCompression)
    (outputPath: string)
    (name: string)
    (depth: uint)
    (chunkX: uint)
    (chunkY: uint)
    (chunkZ: uint)
    (physicalSizeX: float)
    (physicalSizeY: float)
    (physicalSizeZ: float)
    (maxConcurrentWrites: int)
    : Stage<Chunk<'T>, unit> =

    suppressZarrNetDebugLogging ()

    let dataType = zarrDataType<'T> ()
    let elementBytes = zarrScalarElementBytes<'T> ()
    let depth = int depth
    let chunkX = max 1u chunkX |> int
    let chunkY = max 1u chunkY |> int
    let chunkZ = max 1u chunkZ |> int
    let mutable writer: OmeZarrWriter option = None
    let mutable zarrStore: LocalFileSystemStore option = None
    let mutable zarrArray: ZarrArray option = None
    let mutable width = 0
    let mutable height = 0
    let mutable zIndex = 0
    let mutable currentZChunk = -1
    let mutable buffers: byte[][] = Array.empty
    let mutable xChunkCount = 0
    let mutable yChunkCount = 0

    let chunkBufferBytes = chunkX * chunkY * chunkZ * elementBytes

    let createWriter chunkWidth chunkHeight =
        let descriptor =
            BioImageDescriptor(
                chunkWidth,
                chunkHeight,
                ZCT(depth, 1, 1),
                Name = name,
                DataType = dataType,
                ChunkX = chunkX,
                ChunkY = chunkY,
                ChunkZ = chunkZ,
                ChunkC = 1,
                ChunkT = 1,
                PhysicalSizeX = physicalSizeX,
                PhysicalSizeY = physicalSizeY,
                PhysicalSizeZ = physicalSizeZ,
                Compression = compression)

        let created =
            OmeZarrWriter.CreateAsync(outputPath, descriptor, CancellationToken.None)
            |> runTask
        deleteZarrNetDebugLogs ()

        if maxConcurrentWrites > 0 then
            OmeZarrWriter.MaxConcurrentWrites <- maxConcurrentWrites

        let store = new LocalFileSystemStore(outputPath)
        let rootGroup =
            ZarrGroup.OpenRootAsync(store, CancellationToken.None)
            |> runTask
        let array =
            rootGroup.OpenArrayAsync("0", CancellationToken.None)
            |> runTask

        width <- chunkWidth
        height <- chunkHeight
        xChunkCount <- (chunkWidth + chunkX - 1) / chunkX
        yChunkCount <- (chunkHeight + chunkY - 1) / chunkY
        writer <- Some created
        zarrStore <- Some store
        zarrArray <- Some array
        array

    let allocateBuffers () =
        buffers <- Array.init (xChunkCount * yChunkCount) (fun _ -> ArrayPool<byte>.Shared.Rent(chunkBufferBytes))
        for buffer in buffers do
            buffer.AsSpan(0, chunkBufferBytes).Clear()

    let returnBuffers () =
        for buffer in buffers do
            ArrayPool<byte>.Shared.Return(buffer)
        buffers <- Array.empty

    let finishWriter () =
        match writer with
        | Some zarrWriter ->
            zarrWriter.DisposeAsync().AsTask()
            |> runUnitTask
        | None -> ()

        match zarrStore with
        | Some store ->
            store.DisposeAsync().AsTask()
            |> runUnitTask
        | None -> ()

        writer <- None
        zarrStore <- None
        zarrArray <- None
        deleteZarrNetDebugLogs ()

    let flushBuffers () =
        if currentZChunk >= 0 && buffers.Length > 0 then
            let array =
                match zarrArray with
                | Some array -> array
                | None -> failwith "writeZarrChunkSlicesAligned has no open Zarr array."

            for yChunk in 0 .. yChunkCount - 1 do
                for xChunk in 0 .. xChunkCount - 1 do
                    let buffer = buffers[yChunk * xChunkCount + xChunk]
                    let coord = [| 0L; 0L; int64 currentZChunk; int64 yChunk; int64 xChunk |]
                    array.WriteChunkDecodedAsync(
                        coord,
                        ReadOnlyMemory<byte>(buffer, 0, chunkBufferBytes),
                        true,
                        CancellationToken.None)
                    |> runUnitTask
            deleteZarrNetDebugLogs ()
            returnBuffers ()

    let copySliceIntoBuffers (slice: Chunk<'T>) localZ =
        let sliceWidth64, sliceHeight64, sliceDepth64 = slice.Size
        if sliceDepth64 <> 1UL then
            invalidArg "slice" $"writeZarrChunkSlicesAligned expects 2D chunks with depth 1, got {slice.Size}."
        let sliceWidth = int sliceWidth64
        let sliceHeight = int sliceHeight64

        let array =
            match writer, zarrArray with
            | Some _writer, Some array ->
                if sliceWidth <> width || sliceHeight <> height then
                    failwith $"writeZarrChunkSlicesAligned expected all slices to be {width}x{height}, got {sliceWidth}x{sliceHeight}."
                array
            | None, None ->
                createWriter sliceWidth sliceHeight
            | _ ->
                failwith "writeZarrChunkSlicesAligned has inconsistent writer state."

        let expectedBytes = sliceWidth * sliceHeight * elementBytes
        if slice.ByteLength <> expectedBytes then
            failwith $"writeZarrChunkSlicesAligned expected {expectedBytes} bytes, got {slice.ByteLength}."

        if buffers.Length = 0 then
            allocateBuffers ()

        for yChunk in 0 .. yChunkCount - 1 do
            let yStart = yChunk * chunkY
            let yCount = min chunkY (height - yStart)
            for xChunk in 0 .. xChunkCount - 1 do
                let xStart = xChunk * chunkX
                let xCount = min chunkX (width - xStart)
                let rowBytes = xCount * elementBytes
                let buffer = buffers[yChunk * xChunkCount + xChunk]

                for y in 0 .. yCount - 1 do
                    let sourceOffset =
                        ((yStart + y) * width + xStart) * elementBytes
                    let destinationOffset =
                        ((localZ * chunkY + y) * chunkX) * elementBytes
                    Buffer.BlockCopy(slice.Bytes, sourceOffset, buffer, destinationOffset, rowBytes)

        array |> ignore

    let mapper (debug: bool) (_idx: int64) (slice: Chunk<'T>) =
        try
            if depth <= 0 then
                invalidArg "depth" "writeZarrChunkSlicesAligned expects positive depth."
            if zIndex >= depth then
                failwith $"writeZarrChunkSlicesAligned received slice {zIndex}, but declared depth is {depth}."

            let zChunk = zIndex / chunkZ
            if currentZChunk = -1 then
                currentZChunk <- zChunk
            elif zChunk <> currentZChunk then
                flushBuffers ()
                currentZChunk <- zChunk

            if debug then
                printfn $"[writeZarrChunkSlicesAligned] Buffered slice {zIndex} to {outputPath} as {dataType}"

            copySliceIntoBuffers slice (zIndex - zChunk * chunkZ)
            zIndex <- zIndex + 1

            if zIndex = depth then
                flushBuffers ()
                finishWriter ()
        finally
            Chunk.decRef slice

    let memoryNeed nPixels =
        nPixels * uint64 elementBytes

    Stage.mapi $"writeZarrChunkSlicesAligned \"{outputPath}\"" mapper memoryNeed id

let writeZarrChunkSlicesAligned<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (outputPath: string)
    (name: string)
    (depth: uint)
    (chunkX: uint)
    (chunkY: uint)
    (chunkZ: uint)
    (physicalSizeX: float)
    (physicalSizeY: float)
    (physicalSizeZ: float)
    (maxConcurrentWrites: int)
    : Stage<Chunk<'T>, unit> =

    writeZarrChunkSlicesAlignedWithCompression
        ZarrCompression.BloscLz4
        outputPath
        name
        depth
        chunkX
        chunkY
        chunkZ
        physicalSizeX
        physicalSizeY
        physicalSizeZ
        maxConcurrentWrites

let private writeZarrComplex64InterleavedFloat32PlanesWithCompression
    (compression: ZarrCompression)
    (outputPath: string)
    (name: string)
    (logicalWidth: uint)
    (height: uint)
    (depth: uint)
    (chunkX: uint)
    (chunkY: uint)
    (chunkZ: uint)
    (physicalSizeX: float)
    (physicalSizeY: float)
    (physicalSizeZ: float)
    (maxConcurrentWrites: int)
    : Stage<Chunk<float32>, unit> =

    suppressZarrNetDebugLogging ()

    let logicalWidth = int logicalWidth
    let height = int height
    let depth = int depth
    let chunkX = max 1u chunkX |> int
    let chunkY = max 1u chunkY |> int
    let chunkZ = max 1u chunkZ |> int
    let mutable writer: OmeZarrWriter option = None

    let createWriter () =
        let descriptor =
            BioImageDescriptor(
                logicalWidth,
                height,
                ZCT(depth, 1, 1),
                Name = name,
                DataType = "complex64",
                ChunkX = chunkX,
                ChunkY = chunkY,
                ChunkZ = chunkZ,
                ChunkC = 1,
                ChunkT = 1,
                PhysicalSizeX = physicalSizeX,
                PhysicalSizeY = physicalSizeY,
                PhysicalSizeZ = physicalSizeZ,
                Compression = compression)

        let created =
            OmeZarrWriter.CreateAsync(outputPath, descriptor, CancellationToken.None)
            |> runTask
        deleteZarrNetDebugLogs ()

        if maxConcurrentWrites > 0 then
            OmeZarrWriter.MaxConcurrentWrites <- maxConcurrentWrites

        writer <- Some created
        created

    let mapper (debug: bool) (idx: int64) (chunk: Chunk<float32>) =
        try
            if logicalWidth <= 0 || height <= 0 || depth <= 0 then
                invalidArg "shape" "writeZarrComplex64InterleavedFloat32 expects positive logical width, height, and depth."
            if idx >= int64 depth then
                failwith $"writeZarrComplex64InterleavedFloat32 received slice {idx}, but declared depth is {depth}."

            let chunkWidth, chunkHeight, chunkDepth = chunk.Size
            if chunkDepth <> 1UL then
                failwith $"writeZarrComplex64InterleavedFloat32 expects 2D interleaved complex64 chunks with depth 1, got {chunk.Size}."
            if chunkWidth <> uint64 (2 * logicalWidth) || chunkHeight <> uint64 height then
                failwith $"writeZarrComplex64InterleavedFloat32 expected chunk size {2 * logicalWidth}x{height}x1, got {chunk.Size}."

            let zarrWriter =
                match writer with
                | Some writer -> writer
                | None -> createWriter ()

            let expectedBytes = logicalWidth * height * 2 * sizeof<float32>
            if chunk.ByteLength <> expectedBytes then
                failwith $"writeZarrComplex64InterleavedFloat32 expected {expectedBytes} bytes, got {chunk.ByteLength}."

            if debug then
                printfn "[writeZarrComplex64InterleavedFloat32] Saved plane %d to %s as complex64" idx outputPath

            zarrWriter.WritePlaneAsync(int idx, ReadOnlyMemory<byte>(chunk.Bytes, 0, expectedBytes), CancellationToken.None)
            |> runUnitTask
            deleteZarrNetDebugLogs ()

            if idx = int64 (depth - 1) then
                zarrWriter.DisposeAsync().AsTask()
                |> runUnitTask
                writer <- None
                deleteZarrNetDebugLogs ()
        finally
            Chunk.decRef chunk

    let memoryNeed nPixels =
        // Input chunk plus the exact byte payload required by the current Zarr.NET writer.
        nPixels * uint64 (4 + 8)

    Stage.mapi $"writeZarrComplex64InterleavedFloat32 \"{outputPath}\"" mapper memoryNeed id

let private writeZarrComplex64InterleavedFloat32SlabsWithCompression
    (compression: ZarrCompression)
    (outputPath: string)
    (name: string)
    (logicalWidth: uint)
    (height: uint)
    (depth: uint)
    (chunkX: uint)
    (chunkY: uint)
    (chunkZ: uint)
    (physicalSizeX: float)
    (physicalSizeY: float)
    (physicalSizeZ: float)
    (maxConcurrentWrites: int)
    : Stage<Chunk<float32>, unit> =

    suppressZarrNetDebugLogging ()

    let logicalWidth = int logicalWidth
    let height = int height
    let depth = int depth
    let chunkX = max 1u chunkX |> int
    let chunkY = max 1u chunkY |> int
    let chunkZ = max 1u chunkZ |> int
    let mutable writer: OmeZarrWriter option = None
    let mutable level: ResolutionLevelNode option = None

    let createWriter () =
        let descriptor =
            BioImageDescriptor(
                logicalWidth,
                height,
                ZCT(depth, 1, 1),
                Name = name,
                DataType = "complex64",
                ChunkX = chunkX,
                ChunkY = chunkY,
                ChunkZ = chunkZ,
                ChunkC = 1,
                ChunkT = 1,
                PhysicalSizeX = physicalSizeX,
                PhysicalSizeY = physicalSizeY,
                PhysicalSizeZ = physicalSizeZ,
                Compression = compression)

        let created =
            OmeZarrWriter.CreateAsync(outputPath, descriptor, CancellationToken.None)
            |> runTask
        deleteZarrNetDebugLogs ()

        if maxConcurrentWrites > 0 then
            OmeZarrWriter.MaxConcurrentWrites <- maxConcurrentWrites

        writer <- Some created
        level <- Some(openZarrResolutionLevel outputPath 0 0)
        created

    let expectedPlaneBytes = logicalWidth * height * 2 * sizeof<float32>

    let validateChunk (chunk: Chunk<float32>) =
        let chunkWidth, chunkHeight, chunkDepth = chunk.Size
        if chunkDepth <> 1UL then
            failwith $"writeZarrComplex64InterleavedFloat32 expects 2D interleaved complex64 chunks with depth 1, got {chunk.Size}."
        if chunkWidth <> uint64 (2 * logicalWidth) || chunkHeight <> uint64 height then
            failwith $"writeZarrComplex64InterleavedFloat32 expected chunk size {2 * logicalWidth}x{height}x1, got {chunk.Size}."
        if chunk.ByteLength <> expectedPlaneBytes then
            failwith $"writeZarrComplex64InterleavedFloat32 expected {expectedPlaneBytes} bytes, got {chunk.ByteLength}."

    let writeSlab debug slabIndex (chunks: ResizeArray<Chunk<float32>>) =
        if logicalWidth <= 0 || height <= 0 || depth <= 0 then
            invalidArg "shape" "writeZarrComplex64InterleavedFloat32 expects positive logical width, height, and depth."
        if chunks.Count = 0 then
            ()
        else
            let zStart = slabIndex * chunkZ
            let zCount = chunks.Count
            let zStop = zStart + zCount
            if zStop > depth then
                failwith $"writeZarrComplex64InterleavedFloat32 received slab {slabIndex} ending at z={zStop}, but declared depth is {depth}."

            let slabBytes = Array.zeroCreate<byte> (expectedPlaneBytes * zCount)

            for i = 0 to chunks.Count - 1 do
                let chunk = chunks[i]
                validateChunk chunk
                Buffer.BlockCopy(chunk.Bytes, 0, slabBytes, i * expectedPlaneBytes, expectedPlaneBytes)

            let zarrLevel =
                match level with
                | Some level -> level
                | None ->
                    createWriter () |> ignore
                    level.Value

            let region =
                PixelRegion(
                    [| 0L; 0L; int64 zStart; 0L; 0L |],
                    [| 1L; 1L; int64 zStop; int64 height; int64 logicalWidth |])

            if debug then
                printfn "[writeZarrComplex64InterleavedFloat32] Saved z %d..%d to %s as complex64" zStart (zStop - 1) outputPath

            zarrLevel.WriteRegionAsync(region, slabBytes, CancellationToken.None)
            |> runUnitTask
            deleteZarrNetDebugLogs ()

            if zStop = depth then
                match writer with
                | Some zarrWriter ->
                    zarrWriter.DisposeAsync().AsTask()
                    |> runUnitTask
                    writer <- None
                    level <- None
                    deleteZarrNetDebugLogs ()
                | None -> ()

    let apply debug (input: AsyncSeq<Chunk<float32>>) =
        asyncSeq {
            let slab = ResizeArray<Chunk<float32>>(chunkZ)
            let mutable slabIndex = 0

            let releaseSlab () =
                for chunk in slab do
                    Chunk.decRef chunk
                slab.Clear()

            try
                for chunk in input do
                    slab.Add chunk
                    if slab.Count = chunkZ then
                        try
                            writeSlab debug slabIndex slab
                        finally
                            releaseSlab ()
                        slabIndex <- slabIndex + 1

                if slab.Count > 0 then
                    try
                        writeSlab debug slabIndex slab
                    finally
                        releaseSlab ()

                yield ()
            with
            | ex ->
                releaseSlab ()
                raise ex
        }

    let memoryNeed nPixels =
        nPixels * uint64 chunkZ * uint64 (4 + 8)

    Stage.fromAsyncSeq
        $"writeZarrComplex64InterleavedFloat32Slabs \"{outputPath}\""
        apply
        (ProfileTransition.create Streaming Constant)
        (StageMemoryModel.fromSinglePeak Reduce memoryNeed)
        id

let writeZarrComplex64InterleavedFloat32WithCompression
    (compression: ZarrCompression)
    (outputPath: string)
    (name: string)
    (logicalWidth: uint)
    (height: uint)
    (depth: uint)
    (chunkX: uint)
    (chunkY: uint)
    (chunkZ: uint)
    (physicalSizeX: float)
    (physicalSizeY: float)
    (physicalSizeZ: float)
    (maxConcurrentWrites: int)
    : Stage<Chunk<float32>, unit> =

    let chunkZ = max 1u chunkZ
    if chunkZ = 1u then
        writeZarrComplex64InterleavedFloat32PlanesWithCompression
            compression
            outputPath
            name
            logicalWidth
            height
            depth
            chunkX
            chunkY
            chunkZ
            physicalSizeX
            physicalSizeY
            physicalSizeZ
            maxConcurrentWrites
    else
        writeZarrComplex64InterleavedFloat32SlabsWithCompression
            compression
            outputPath
            name
            logicalWidth
            height
            depth
            chunkX
            chunkY
            chunkZ
            physicalSizeX
            physicalSizeY
            physicalSizeZ
            maxConcurrentWrites

let writeZarrComplex64InterleavedFloat32
    (outputPath: string)
    (name: string)
    (logicalWidth: uint)
    (height: uint)
    (depth: uint)
    (chunkX: uint)
    (chunkY: uint)
    (chunkZ: uint)
    (physicalSizeX: float)
    (physicalSizeY: float)
    (physicalSizeZ: float)
    (maxConcurrentWrites: int)
    : Stage<Chunk<float32>, unit> =

    writeZarrComplex64InterleavedFloat32WithCompression
        ZarrCompression.None
        outputPath
        name
        logicalWidth
        height
        depth
        chunkX
        chunkY
        chunkZ
        physicalSizeX
        physicalSizeY
        physicalSizeZ
        maxConcurrentWrites

let private validateSpectralComplex64InterleavedFloat32
    name
    (realWidth: int)
    (height: int)
    (depth: int)
    (idx: int64)
    (spectral: SpectralChunk)
    =

    if realWidth <= 0 || height <= 0 || depth <= 0 then
        invalidArg "shape" $"{name} expects positive logical width, height, and depth."
    if idx >= int64 depth then
        failwith $"{name} received slice {idx}, but declared depth is {depth}."

    match spectral.Layout with
    | SpectralLayout.HermitianPackedComplex64Interleaved(packedAxis, originalWidth) ->
        if packedAxis <> 0 then
            failwith $"{name} currently supports Hermitian packing along X only; got packed axis {packedAxis}."
        if originalWidth <> uint64 realWidth then
            failwith $"{name} expected real width {realWidth}, got spectral real width {originalWidth}."
    | SpectralLayout.FullComplex64Interleaved ->
        failwith $"{name} expects compact Hermitian spectra, got full complex spectra."

    let logicalWidth, logicalHeight, logicalDepth = spectral.LogicalSize
    if logicalWidth <> uint64 realWidth || logicalHeight <> uint64 height || logicalDepth <> uint64 depth then
        failwith $"{name} expected logical size {realWidth}x{height}x{depth}, got {spectral.LogicalSize}."

    let packedComplexWidth = realWidth / 2 + 1
    let chunkWidth, chunkHeight, chunkDepth = spectral.Chunk.Size
    if chunkDepth <> 1UL then
        failwith $"{name} expects 2D interleaved compact complex64 chunks with depth 1, got {spectral.Chunk.Size}."
    if chunkWidth <> uint64 (2 * packedComplexWidth) || chunkHeight <> uint64 height then
        failwith $"{name} expected chunk size {2 * packedComplexWidth}x{height}x1, got {spectral.Chunk.Size}."

    let expectedBytes = packedComplexWidth * height * 2 * sizeof<float32>
    if spectral.Chunk.ByteLength <> expectedBytes then
        failwith $"{name} expected {expectedBytes} bytes, got {spectral.Chunk.ByteLength}."

    packedComplexWidth, expectedBytes

let writeZarrSpectralComplex64InterleavedFloat32WithCompression
    (compression: ZarrCompression)
    (outputPath: string)
    (name: string)
    (realWidth: uint)
    (height: uint)
    (depth: uint)
    (chunkX: uint)
    (chunkY: uint)
    (chunkZ: uint)
    (physicalSizeX: float)
    (physicalSizeY: float)
    (physicalSizeZ: float)
    (maxConcurrentWrites: int)
    : Stage<SpectralChunk, unit> =

    let realWidth = int realWidth
    let height = int height
    let depth = int depth
    let packedComplexWidth = realWidth / 2 + 1
    let _requestedChunkX = max 1u chunkX |> int
    let _requestedChunkY = max 1u chunkY |> int
    let chunkZ = max 1u chunkZ |> int
    let mutable writer: OmeZarrWriter option = None
    let mutable level: ResolutionLevelNode option = None
    let mutable array: ZarrNET.Core.Zarr.ZarrArray option = None
    let expectedPlaneBytes = packedComplexWidth * height * 2 * sizeof<float32>
    let expectedFullChunkBytes = expectedPlaneBytes * chunkZ

    suppressZarrNetDebugLogging ()

    let createWriter () =
        let descriptor =
            BioImageDescriptor(
                packedComplexWidth,
                height,
                ZCT(depth, 1, 1),
                Name = name,
                DataType = "complex64",
                ChunkX = packedComplexWidth,
                ChunkY = height,
                ChunkZ = chunkZ,
                ChunkC = 1,
                ChunkT = 1,
                PhysicalSizeX = physicalSizeX,
                PhysicalSizeY = physicalSizeY,
                PhysicalSizeZ = physicalSizeZ,
                Compression = compression)

        let created =
            OmeZarrWriter.CreateAsync(outputPath, descriptor, CancellationToken.None)
            |> runTask
        deleteZarrNetDebugLogs ()
        tagSpectralZarrMetadata outputPath realWidth height depth 0 packedComplexWidth

        if maxConcurrentWrites > 0 then
            OmeZarrWriter.MaxConcurrentWrites <- maxConcurrentWrites

        writer <- Some created
        level <- Some(openZarrResolutionLevel outputPath 0 0)
        let zarrArray =
            let reader =
                OmeZarrReader.OpenAsync(outputPath, ct = CancellationToken.None)
                |> runTask
            reader.RootGroup.OpenArrayAsync("0", CancellationToken.None)
            |> runTask
        array <- Some zarrArray
        created

    let writeSlab debug slabIndex (spectra: ResizeArray<SpectralChunk>) =
        if spectra.Count = 0 then
            ()
        else
            let zStart = slabIndex * chunkZ
            let zCount = spectra.Count
            let zStop = zStart + zCount
            if zStop > depth then
                failwith $"writeZarrSpectralComplex64InterleavedFloat32 received slab {slabIndex} ending at z={zStop}, but declared depth is {depth}."

            let slabBytes = Array.zeroCreate<byte> expectedFullChunkBytes
            for i = 0 to spectra.Count - 1 do
                let spectral = spectra[i]
                let _packedWidth, _expectedBytes =
                    validateSpectralComplex64InterleavedFloat32
                        "writeZarrSpectralComplex64InterleavedFloat32"
                        realWidth
                        height
                        depth
                        (int64 (zStart + i))
                        spectral
                Buffer.BlockCopy(spectral.Chunk.Bytes, 0, slabBytes, i * expectedPlaneBytes, expectedPlaneBytes)

            let zarrArray =
                match array with
                | Some array -> array
                | None ->
                    createWriter () |> ignore
                    array.Value

            if debug then
                printfn "[writeZarrSpectralComplex64InterleavedFloat32] Saved chunk z %d..%d to %s as compact complex64" zStart (zStop - 1) outputPath

            zarrArray.WriteChunkDecodedAsync([| 0L; 0L; int64 slabIndex; 0L; 0L |], slabBytes.AsMemory(), true, CancellationToken.None)
            |> runUnitTask
            deleteZarrNetDebugLogs ()

            if zStop = depth then
                match writer with
                | Some zarrWriter ->
                    zarrWriter.DisposeAsync().AsTask()
                    |> runUnitTask
                    writer <- None
                    level <- None
                    array <- None
                    tagSpectralZarrMetadata outputPath realWidth height depth 0 packedComplexWidth
                    deleteZarrNetDebugLogs ()
                | None -> ()

    let apply debug (input: AsyncSeq<SpectralChunk>) =
        asyncSeq {
            let slab = ResizeArray<SpectralChunk>(chunkZ)
            let mutable slabIndex = 0

            let releaseSlab () =
                for spectral in slab do
                    Chunk.decRef spectral.Chunk
                slab.Clear()

            try
                for spectral in input do
                    slab.Add spectral
                    if slab.Count = chunkZ then
                        try
                            writeSlab debug slabIndex slab
                        finally
                            releaseSlab ()
                        slabIndex <- slabIndex + 1

                if slab.Count > 0 then
                    try
                        writeSlab debug slabIndex slab
                    finally
                        releaseSlab ()

                yield ()
            with
            | ex ->
                releaseSlab ()
                raise ex
        }

    let memoryNeed nPixels =
        nPixels * uint64 chunkZ * uint64 (sizeof<float32> * 4)

    Stage.fromAsyncSeq
        $"writeZarrSpectralComplex64InterleavedFloat32 \"{outputPath}\""
        apply
        (ProfileTransition.create Streaming Constant)
        (StageMemoryModel.fromSinglePeak Reduce memoryNeed)
        id

let writeZarrSpectralComplex64InterleavedFloat32
    (outputPath: string)
    (name: string)
    (realWidth: uint)
    (height: uint)
    (depth: uint)
    (chunkX: uint)
    (chunkY: uint)
    (chunkZ: uint)
    (physicalSizeX: float)
    (physicalSizeY: float)
    (physicalSizeZ: float)
    (maxConcurrentWrites: int)
    : Stage<SpectralChunk, unit> =

    writeZarrSpectralComplex64InterleavedFloat32WithCompression
        ZarrCompression.None
        outputPath
        name
        realWidth
        height
        depth
        chunkX
        chunkY
        chunkZ
        physicalSizeX
        physicalSizeY
        physicalSizeZ
        maxConcurrentWrites

let writeZarrSpectralComplex64InterleavedFloat32Tiled
    (outputPath: string)
    (name: string)
    (realWidth: uint)
    (height: uint)
    (depth: uint)
    (chunkX: uint)
    (chunkY: uint)
    (chunkZ: uint)
    (physicalSizeX: float)
    (physicalSizeY: float)
    (physicalSizeZ: float)
    (maxConcurrentWrites: int)
    : Stage<SpectralChunk, unit> =

    suppressZarrNetDebugLogging ()

    let realWidth = int realWidth
    let height = int height
    let depth = int depth
    let packedComplexWidth = realWidth / 2 + 1
    let chunkX = max 1u chunkX |> int
    let chunkY = max 1u chunkY |> int
    let chunkZ = max 1u chunkZ |> int
    let chunkBytes = chunkX * chunkY * chunkZ * 2 * sizeof<float32>
    let planeBytes = packedComplexWidth * height * 2 * sizeof<float32>
    let xChunks = (packedComplexWidth + chunkX - 1) / chunkX
    let yChunks = (height + chunkY - 1) / chunkY

    let mutable writer: OmeZarrWriter option = None
    let mutable array: ZarrArray option = None

    let createWriter () =
        let descriptor =
            BioImageDescriptor(
                packedComplexWidth,
                height,
                ZCT(depth, 1, 1),
                Name = name,
                DataType = "complex64",
                ChunkX = chunkX,
                ChunkY = chunkY,
                ChunkZ = chunkZ,
                ChunkC = 1,
                ChunkT = 1,
                PhysicalSizeX = physicalSizeX,
                PhysicalSizeY = physicalSizeY,
                PhysicalSizeZ = physicalSizeZ,
                Compression = ZarrCompression.None)

        let created =
            OmeZarrWriter.CreateAsync(outputPath, descriptor, CancellationToken.None)
            |> runTask
        deleteZarrNetDebugLogs ()
        tagSpectralZarrMetadata outputPath realWidth height depth 0 packedComplexWidth

        if maxConcurrentWrites > 0 then
            OmeZarrWriter.MaxConcurrentWrites <- maxConcurrentWrites

        let zarrArray =
            let reader =
                OmeZarrReader.OpenAsync(outputPath, ct = CancellationToken.None)
                |> runTask
            reader.RootGroup.OpenArrayAsync("0", CancellationToken.None)
            |> runTask

        writer <- Some created
        array <- Some zarrArray
        created

    let validate idx (spectral: SpectralChunk) =
        validateSpectralComplex64InterleavedFloat32
            "writeZarrSpectralComplex64InterleavedFloat32Tiled"
            realWidth
            height
            depth
            idx
            spectral
        |> ignore

    let writeSlab debug slabIndex (spectra: ResizeArray<SpectralChunk>) =
        if spectra.Count > 0 then
            let zStart = slabIndex * chunkZ
            let zStop = zStart + spectra.Count
            if zStop > depth then
                failwith $"writeZarrSpectralComplex64InterleavedFloat32Tiled received slab {slabIndex} ending at z={zStop}, but declared depth is {depth}."

            let zarrArray =
                match array with
                | Some zarrArray -> zarrArray
                | None ->
                    createWriter () |> ignore
                    array.Value

            for localZ = 0 to spectra.Count - 1 do
                validate (int64 (zStart + localZ)) spectra[localZ]

            for yc = 0 to yChunks - 1 do
                let y0 = yc * chunkY
                let validY = min chunkY (height - y0)
                for xc = 0 to xChunks - 1 do
                    let x0 = xc * chunkX
                    let validX = min chunkX (packedComplexWidth - x0)
                    let payload = Array.zeroCreate<byte> chunkBytes

                    for localZ = 0 to spectra.Count - 1 do
                        let source = spectra[localZ].Chunk.Bytes
                        for yy = 0 to validY - 1 do
                            let sourceOffset = ((y0 + yy) * packedComplexWidth + x0) * 2 * sizeof<float32>
                            let targetOffset = ((localZ * chunkY + yy) * chunkX) * 2 * sizeof<float32>
                            Buffer.BlockCopy(source, sourceOffset, payload, targetOffset, validX * 2 * sizeof<float32>)

                    if debug then
                        printfn "[writeZarrSpectralComplex64InterleavedFloat32Tiled] Saved chunk z=%d y=%d x=%d to %s" slabIndex yc xc outputPath

                    zarrArray.WriteChunkDecodedAsync(
                        [| 0L; 0L; int64 slabIndex; int64 yc; int64 xc |],
                        payload.AsMemory(),
                        true,
                        CancellationToken.None)
                    |> runUnitTask
                    deleteZarrNetDebugLogs ()

            if zStop = depth then
                match writer with
                | Some zarrWriter ->
                    zarrWriter.DisposeAsync().AsTask()
                    |> runUnitTask
                    writer <- None
                    array <- None
                    tagSpectralZarrMetadata outputPath realWidth height depth 0 packedComplexWidth
                    deleteZarrNetDebugLogs ()
                | None -> ()

    let apply debug (input: AsyncSeq<SpectralChunk>) =
        asyncSeq {
            let slab = ResizeArray<SpectralChunk>(chunkZ)
            let mutable slabIndex = 0

            let releaseSlab () =
                for spectral in slab do
                    Chunk.decRef spectral.Chunk
                slab.Clear()

            try
                for spectral in input do
                    slab.Add spectral
                    if slab.Count = chunkZ then
                        try
                            writeSlab debug slabIndex slab
                        finally
                            releaseSlab ()
                        slabIndex <- slabIndex + 1

                if slab.Count > 0 then
                    try
                        writeSlab debug slabIndex slab
                    finally
                        releaseSlab ()

                yield ()
            with
            | ex ->
                releaseSlab ()
                raise ex
        }

    Stage.fromAsyncSeq
        $"writeZarrSpectralComplex64InterleavedFloat32Tiled \"{outputPath}\""
        apply
        (ProfileTransition.create Streaming Constant)
        (StageMemoryModel.fromSinglePeak Reduce (fun _ -> uint64 (chunkBytes + planeBytes * chunkZ)))
        id

let readZarrSpectralComplex64InterleavedFloat32Range
    (first: uint)
    (step: int)
    (last: uint)
    (path: string)
    (multiscaleIndex: int)
    (datasetIndex: int)
    (timepoint: int)
    (channel: int)
    (maxParallelChunks: int)
    (pl: Plan<unit, unit>)
    : Plan<unit, SpectralChunk> =

    suppressZarrNetDebugLogging ()

    let name = "readZarrSpectralComplex64InterleavedFloat32Range"
    let level = openZarrResolutionLevel path multiscaleIndex datasetIndex
    let sizeT, sizeC, sizeZ, sizeY, sizeX = zarrShapeTCZYX level.Shape
    if not (String.Equals(level.DataType, "complex64", StringComparison.OrdinalIgnoreCase)) then
        invalidOp $"{name} expected complex64 Zarr data, got {level.DataType}."

    if timepoint < 0 || timepoint >= sizeT then
        invalidArg "timepoint" $"Timepoint {timepoint} is outside the Zarr time range 0..{sizeT - 1}."
    if channel < 0 || channel >= sizeC then
        invalidArg "channel" $"Channel {channel} is outside the Zarr channel range 0..{sizeC - 1}."

    let packedAxis, realWidth, realHeight, realDepth =
        match tryReadSpectralZarrMetadata path with
        | Some metadata -> metadata
        | None -> 0, uint64 ((sizeX - 1) * 2), uint64 sizeY, uint64 sizeZ

    if packedAxis <> 0 then
        invalidOp $"{name} currently supports compact spectra packed along X only; got packed axis {packedAxis}."
    if realHeight <> uint64 sizeY || realDepth <> uint64 sizeZ then
        invalidOp $"{name} metadata logical size {realWidth}x{realHeight}x{realDepth} does not match stored Zarr shape {sizeX}x{sizeY}x{sizeZ}."

    let expectedPackedWidth = int (realWidth / 2UL + 1UL)
    if expectedPackedWidth <> sizeX then
        invalidOp $"{name} metadata real width {realWidth} implies packed complex width {expectedPackedWidth}, but Zarr x-size is {sizeX}."

    let reader =
        OmeZarrReader.OpenAsync(path, ct = CancellationToken.None)
        |> runTask
    let zarrArray =
        reader.RootGroup.OpenArrayAsync(level.Dataset.Path, CancellationToken.None)
        |> runTask
    let chunkShape = zarrArray.Metadata.ChunkShape
    let chunkShapeText = String.Join("x", chunkShape)
    if chunkShape.Length <> 5 then
        invalidOp $"{name} expected 5D OME-Zarr chunk shape, got {chunkShapeText}."
    if chunkShape[0] <> 1 || chunkShape[1] <> 1 then
        invalidOp $"{name} expected singleton t/c Zarr chunks, got chunk shape {chunkShapeText}."
    if chunkShape[3] <> sizeY || chunkShape[4] <> sizeX then
        invalidOp $"{name} requires full packed XY chunks for direct chunk IO; got chunk shape {chunkShapeText} for stored XY {sizeX}x{sizeY}."

    let chunkZ = max 1 chunkShape[2]
    let selected = rangeIndices first step last sizeZ
    let selectedByChunk =
        selected
        |> Array.groupBy (fun z -> z / chunkZ)

    let expectedBytes = sizeX * sizeY * 2 * sizeof<float32>
    let expectedFullChunkBytes = expectedBytes * chunkZ
    let sourcePeek =
        SourcePeek.create
            name
            (uint64 expectedBytes)
            (Some (uint64 selected.Length))
            (Map.ofList
                [ "kind", "zarr-spectral-complex64-range"
                  "path", path
                  "realWidth", string realWidth
                  "height", string realHeight
                  "depth", string selected.Length
                  "sourceDepth", string sizeZ
                  "storedComplexWidth", string sizeX
                  "multiscaleIndex", string multiscaleIndex
                  "datasetIndex", string datasetIndex
                  "timepoint", string timepoint
                  "channel", string channel
                  "first", string first
                  "step", string step
                  "last", string last ])

    let mapper (outputIndex: int) : SpectralChunk list =
        let zChunk, zIndices = selectedByChunk[outputIndex]
        if pl.debug then
            printfn $"[{name}] Reading Zarr chunk z-block {zChunk} from {path} as compact complex64"

        let scratch = ArrayPool<byte>.Shared.Rent(expectedFullChunkBytes)
        try
            zarrArray.ReadChunkDecodedAsync(
                [| int64 timepoint; int64 channel; int64 zChunk; 0L; 0L |],
                scratch.AsMemory(0, expectedFullChunkBytes),
                true,
                CancellationToken.None)
            |> runUnitTask
            deleteZarrNetDebugLogs ()

            [ for zIndex in zIndices do
                let localZ = zIndex - zChunk * chunkZ
                let chunk = Chunk.create<float32> (uint64 (2 * sizeX), uint64 sizeY, 1UL)
                try
                    Buffer.BlockCopy(scratch, localZ * expectedBytes, chunk.Bytes, 0, expectedBytes)
                    ({ LogicalSize = (realWidth, uint64 sizeY, uint64 sizeZ)
                       Layout = SpectralLayout.HermitianPackedComplex64Interleaved(0, realWidth)
                       Chunk = chunk }
                     : SpectralChunk)
                with
                | ex ->
                    Chunk.decRef chunk
                    raise ex ]
        finally
            ArrayPool<byte>.Shared.Return(scratch)

    let transition = ProfileTransition.create Unit Streaming
    let memPeak = uint64 (expectedFullChunkBytes + expectedBytes * chunkZ)
    let memoryNeed = fun _ -> memPeak
    let elementTransformation = id
    let memoryModel = StageMemoryModel.fromSinglePeak Source memoryNeed
    let timeCostModel =
        StageTimeCostModel.ioRead Source (Some name) (fun _ -> uint64 expectedBytes) (fun _ -> 1UL)
    let stage =
        Stage.init name (uint selectedByChunk.Length) mapper transition memoryNeed elementTransformation
        |> withCostModel (StageCostModel.create memoryModel timeCostModel)
        |> Some

    Plan.createWithOptimizer stage pl.memAvail memPeak memPeak (uint64 selectedByChunk.Length) pl.debug pl.optimize
    |> Plan.withRuntimeOptionsFrom pl
    |> Plan.withSourcePeek sourcePeek
    >=> flattenList ()

let readZarrSpectralComplex64InterleavedFloat32TiledRange
    (first: uint)
    (step: int)
    (last: uint)
    (path: string)
    (multiscaleIndex: int)
    (datasetIndex: int)
    (timepoint: int)
    (channel: int)
    (maxParallelChunks: int)
    (pl: Plan<unit, unit>)
    : Plan<unit, SpectralChunk> =

    suppressZarrNetDebugLogging ()

    let name = "readZarrSpectralComplex64InterleavedFloat32TiledRange"
    let level = openZarrResolutionLevel path multiscaleIndex datasetIndex
    let sizeT, sizeC, sizeZ, sizeY, sizeX = zarrShapeTCZYX level.Shape
    if not (String.Equals(level.DataType, "complex64", StringComparison.OrdinalIgnoreCase)) then
        invalidOp $"{name} expected complex64 Zarr data, got {level.DataType}."

    if timepoint < 0 || timepoint >= sizeT then
        invalidArg "timepoint" $"Timepoint {timepoint} is outside the Zarr time range 0..{sizeT - 1}."
    if channel < 0 || channel >= sizeC then
        invalidArg "channel" $"Channel {channel} is outside the Zarr channel range 0..{sizeC - 1}."

    let packedAxis, realWidth, realHeight, realDepth =
        match tryReadSpectralZarrMetadata path with
        | Some metadata -> metadata
        | None -> 0, uint64 ((sizeX - 1) * 2), uint64 sizeY, uint64 sizeZ

    if packedAxis <> 0 then
        invalidOp $"{name} currently supports compact spectra packed along X only; got packed axis {packedAxis}."
    if realHeight <> uint64 sizeY || realDepth <> uint64 sizeZ then
        invalidOp $"{name} metadata logical size {realWidth}x{realHeight}x{realDepth} does not match stored Zarr shape {sizeX}x{sizeY}x{sizeZ}."

    let expectedPackedWidth = int (realWidth / 2UL + 1UL)
    if expectedPackedWidth <> sizeX then
        invalidOp $"{name} metadata real width {realWidth} implies packed complex width {expectedPackedWidth}, but Zarr x-size is {sizeX}."

    let reader =
        OmeZarrReader.OpenAsync(path, ct = CancellationToken.None)
        |> runTask
    let zarrArray =
        reader.RootGroup.OpenArrayAsync(level.Dataset.Path, CancellationToken.None)
        |> runTask
    let chunkShape = zarrArray.Metadata.ChunkShape
    let chunkShapeText = String.Join("x", chunkShape)
    if chunkShape.Length <> 5 then
        invalidOp $"{name} expected 5D OME-Zarr chunk shape, got {chunkShapeText}."
    if chunkShape[0] <> 1 || chunkShape[1] <> 1 then
        invalidOp $"{name} expected singleton t/c Zarr chunks, got chunk shape {chunkShapeText}."

    let chunkZ = max 1 chunkShape[2]
    let chunkY = max 1 chunkShape[3]
    let chunkX = max 1 chunkShape[4]
    let xChunks = (sizeX + chunkX - 1) / chunkX
    let yChunks = (sizeY + chunkY - 1) / chunkY
    let selected = rangeIndices first step last sizeZ
    let selectedByChunk =
        selected
        |> Array.groupBy (fun z -> z / chunkZ)

    let planeBytes = sizeX * sizeY * 2 * sizeof<float32>
    let chunkBytes = chunkX * chunkY * chunkZ * 2 * sizeof<float32>
    let memPeak = uint64 (chunkBytes + planeBytes * chunkZ)

    let mapper (outputIndex: int) : SpectralChunk list =
        let zChunk, zIndices = selectedByChunk[outputIndex]
        let zStart = zChunk * chunkZ
        let validDepth = min chunkZ (sizeZ - zStart)
        let outputs: SpectralChunk[] =
            zIndices
            |> Array.map (fun zIndex ->
                let chunk = Chunk.create<float32> (uint64 (2 * sizeX), uint64 sizeY, 1UL)
                let logicalSize = (realWidth, uint64 sizeY, uint64 sizeZ)
                let layout = SpectralLayout.HermitianPackedComplex64Interleaved(0, realWidth)
                { LogicalSize = logicalSize
                  Layout = layout
                  Chunk = chunk })

        let outputByLocalZ =
            zIndices
            |> Array.mapi (fun i zIndex -> zIndex - zStart, outputs[i])
            |> Map.ofArray

        let scratch = ArrayPool<byte>.Shared.Rent(chunkBytes)
        try
            try
                for yc = 0 to yChunks - 1 do
                    let y0 = yc * chunkY
                    let validY = min chunkY (sizeY - y0)
                    for xc = 0 to xChunks - 1 do
                        let x0 = xc * chunkX
                        let validX = min chunkX (sizeX - x0)

                        zarrArray.ReadChunkDecodedAsync(
                            [| int64 timepoint; int64 channel; int64 zChunk; int64 yc; int64 xc |],
                            scratch.AsMemory(0, chunkBytes),
                            true,
                            CancellationToken.None)
                        |> runUnitTask
                        deleteZarrNetDebugLogs ()

                        for localZ = 0 to validDepth - 1 do
                            match outputByLocalZ.TryFind localZ with
                            | None -> ()
                            | Some spectral ->
                                for yy = 0 to validY - 1 do
                                    let sourceOffset = ((localZ * chunkY + yy) * chunkX) * 2 * sizeof<float32>
                                    let targetOffset = ((y0 + yy) * sizeX + x0) * 2 * sizeof<float32>
                                    Buffer.BlockCopy(scratch, sourceOffset, spectral.Chunk.Bytes, targetOffset, validX * 2 * sizeof<float32>)

                outputs |> Array.toList
            with
            | ex ->
                for spectral in outputs do
                    Chunk.decRef spectral.Chunk
                raise ex
        finally
            ArrayPool<byte>.Shared.Return(scratch)

    let stage =
        Stage.init name (uint selectedByChunk.Length) mapper (ProfileTransition.create Unit Streaming) (fun _ -> memPeak) id
        |> withCostModel (StageCostModel.create (StageMemoryModel.fromSinglePeak Source (fun _ -> memPeak)) (StageTimeCostModel.ioRead Source (Some name) (fun _ -> uint64 planeBytes) (fun _ -> 1UL)))
        |> Some

    Plan.createWithOptimizer stage pl.memAvail memPeak memPeak (uint64 selectedByChunk.Length) pl.debug pl.optimize
    |> Plan.withRuntimeOptionsFrom pl
    >=> flattenList ()

let private transformZComplex64InterleavedZarrTiles
    inverse
    operationName
    (inputPath: string)
    (outputPath: string)
    (name: string)
    (width: uint)
    (height: uint)
    (depth: uint)
    (tileX: uint)
    (tileY: uint)
    (chunkX: uint)
    (chunkY: uint)
    (chunkZ: uint)
    (physicalSizeX: float)
    (physicalSizeY: float)
    (physicalSizeZ: float)
    (maxConcurrentWrites: int)
    =

    suppressZarrNetDebugLogging ()
    NativeSp.ensureAvailable ()

    let width = int width
    let height = int height
    let depth = int depth
    let tileX = max 1u tileX |> int
    let tileY = max 1u tileY |> int
    let chunkX = max 1u chunkX |> int
    let chunkY = max 1u chunkY |> int
    let chunkZ = max 1u chunkZ |> int

    if width <= 0 || height <= 0 || depth <= 0 then
        invalidArg "shape" $"{operationName} expects positive width, height, and depth."

    let inputLevel = openZarrResolutionLevel inputPath 0 0
    if not (String.Equals(inputLevel.DataType, "complex64", StringComparison.OrdinalIgnoreCase)) then
        invalidOp $"{operationName} expected complex64 input Zarr, got {inputLevel.DataType}."
    if inputLevel.Shape.Length <> 5 then
        invalidOp $"{operationName} expected 5D OME-Zarr input, got rank {inputLevel.Shape.Length}."
    if inputLevel.Shape[0] <> 1L || inputLevel.Shape[1] <> 1L || inputLevel.Shape[2] <> int64 depth || inputLevel.Shape[3] <> int64 height || inputLevel.Shape[4] <> int64 width then
        let actualShape = String.Join(",", inputLevel.Shape)
        invalidOp $"{operationName} expected input shape [1,1,{depth},{height},{width}], got [{actualShape}]."

    let spectralMetadata = tryReadSpectralZarrMetadata inputPath

    let descriptor =
        BioImageDescriptor(
            width,
            height,
            ZCT(depth, 1, 1),
            Name = name,
            DataType = "complex64",
            ChunkX = chunkX,
            ChunkY = chunkY,
            ChunkZ = chunkZ,
            ChunkC = 1,
            ChunkT = 1,
            PhysicalSizeX = physicalSizeX,
            PhysicalSizeY = physicalSizeY,
            PhysicalSizeZ = physicalSizeZ,
            Compression = ZarrCompression.None)

    let writer =
        OmeZarrWriter.CreateAsync(outputPath, descriptor, CancellationToken.None)
        |> runTask
    deleteZarrNetDebugLogs ()

    if maxConcurrentWrites > 0 then
        OmeZarrWriter.MaxConcurrentWrites <- maxConcurrentWrites

    let outputLevel = openZarrResolutionLevel outputPath 0 0
    spectralMetadata
    |> Option.iter (fun (packedAxis, realWidth, realHeight, realDepth) ->
        if packedAxis = 0 && realHeight = uint64 height && realDepth = uint64 depth then
            tagSpectralZarrMetadata outputPath (int realWidth) height depth packedAxis width)

    try
        let mutable y0 = 0
        while y0 < height do
            let currentTileY = min tileY (height - y0)
            let mutable x0 = 0
            while x0 < width do
                let currentTileX = min tileX (width - x0)
                let region =
                    PixelRegion(
                        [| 0L; 0L; 0L; int64 y0; int64 x0 |],
                        [| 1L; 1L; int64 depth; int64 (y0 + currentTileY); int64 (x0 + currentTileX) |])

                let result =
                    inputLevel.ReadPixelRegionAsync(region, Nullable<int>(1), CancellationToken.None)
                    |> runTask

                let expectedBytes = currentTileX * currentTileY * depth * 2 * sizeof<float32>
                if result.Data.Length <> expectedBytes then
                    invalidOp $"{operationName} expected tile payload {expectedBytes} bytes, got {result.Data.Length}."

                let mutable dataHandle = Unchecked.defaultof<GCHandle>
                let mutable dataPinned = false
                try
                    dataHandle <- GCHandle.Alloc(result.Data, GCHandleType.Pinned)
                    dataPinned <- true
                    NativeSp.fftwfComplexZInplace(dataHandle.AddrOfPinnedObject(), currentTileX, currentTileY, depth, if inverse then 1 else 0)
                    |> NativeSp.checkStatus (if inverse then "inverse fftwf z complex tiled" else "fftwf z complex tiled")
                finally
                    if dataPinned then
                        dataHandle.Free()

                outputLevel.WriteRegionAsync(region, result.Data, CancellationToken.None)
                |> runUnitTask
                deleteZarrNetDebugLogs ()

                x0 <- x0 + tileX
            y0 <- y0 + tileY
    finally
        writer.DisposeAsync().AsTask()
        |> runUnitTask
        spectralMetadata
        |> Option.iter (fun (packedAxis, realWidth, realHeight, realDepth) ->
            if packedAxis = 0 && realHeight = uint64 height && realDepth = uint64 depth then
                tagSpectralZarrMetadata outputPath (int realWidth) height depth packedAxis width)
        deleteZarrNetDebugLogs ()

let private transformZComplex64InterleavedZarrRawChunks
    inverse
    operationName
    (inputPath: string)
    (outputPath: string)
    (name: string)
    (width: uint)
    (height: uint)
    (depth: uint)
    (chunkZ: uint)
    (physicalSizeX: float)
    (physicalSizeY: float)
    (physicalSizeZ: float)
    (maxConcurrentWrites: int)
    =

    suppressZarrNetDebugLogging ()
    NativeSp.ensureAvailable ()

    let width = int width
    let height = int height
    let depth = int depth
    let chunkZ = max 1u chunkZ |> int

    if width <= 0 || height <= 0 || depth <= 0 then
        invalidArg "shape" $"{operationName} expects positive width, height, and depth."

    let inputLevel = openZarrResolutionLevel inputPath 0 0
    if not (String.Equals(inputLevel.DataType, "complex64", StringComparison.OrdinalIgnoreCase)) then
        invalidOp $"{operationName} expected complex64 input Zarr, got {inputLevel.DataType}."
    if inputLevel.Shape.Length <> 5 then
        invalidOp $"{operationName} expected 5D OME-Zarr input, got rank {inputLevel.Shape.Length}."
    if inputLevel.Shape[0] <> 1L || inputLevel.Shape[1] <> 1L || inputLevel.Shape[2] <> int64 depth || inputLevel.Shape[3] <> int64 height || inputLevel.Shape[4] <> int64 width then
        let actualShape = String.Join(",", inputLevel.Shape)
        invalidOp $"{operationName} expected input shape [1,1,{depth},{height},{width}], got [{actualShape}]."

    let inputReader =
        OmeZarrReader.OpenAsync(inputPath, ct = CancellationToken.None)
        |> runTask
    let inputArray =
        inputReader.RootGroup.OpenArrayAsync(inputLevel.Dataset.Path, CancellationToken.None)
        |> runTask
    let inputChunkShape = inputArray.Metadata.ChunkShape
    let expectedInputChunkShape = [| 1; 1; chunkZ; height; width |]
    if inputChunkShape <> expectedInputChunkShape then
        let expectedText = String.Join(",", expectedInputChunkShape)
        let actualText = String.Join(",", inputChunkShape)
        invalidOp
            $"{operationName} raw path expects input chunk shape [{expectedText}], got [{actualText}]."

    let spectralMetadata = tryReadSpectralZarrMetadata inputPath

    let descriptor =
        BioImageDescriptor(
            width,
            height,
            ZCT(depth, 1, 1),
            Name = name,
            DataType = "complex64",
            ChunkX = width,
            ChunkY = height,
            ChunkZ = chunkZ,
            ChunkC = 1,
            ChunkT = 1,
            PhysicalSizeX = physicalSizeX,
            PhysicalSizeY = physicalSizeY,
            PhysicalSizeZ = physicalSizeZ,
            Compression = ZarrCompression.None)

    let writer =
        OmeZarrWriter.CreateAsync(outputPath, descriptor, CancellationToken.None)
        |> runTask
    deleteZarrNetDebugLogs ()

    if maxConcurrentWrites > 0 then
        OmeZarrWriter.MaxConcurrentWrites <- maxConcurrentWrites

    let outputLevel = openZarrResolutionLevel outputPath 0 0
    let outputReader =
        OmeZarrReader.OpenAsync(outputPath, ct = CancellationToken.None)
        |> runTask
    let outputArray =
        outputReader.RootGroup.OpenArrayAsync(outputLevel.Dataset.Path, CancellationToken.None)
        |> runTask
    let outputChunkShape = outputArray.Metadata.ChunkShape
    if outputChunkShape <> expectedInputChunkShape then
        let expectedText = String.Join(",", expectedInputChunkShape)
        let actualText = String.Join(",", outputChunkShape)
        invalidOp
            $"{operationName} raw path expected output chunk shape [{expectedText}], got [{actualText}]."

    spectralMetadata
    |> Option.iter (fun (packedAxis, realWidth, realHeight, realDepth) ->
        if packedAxis = 0 && realHeight = uint64 height && realDepth = uint64 depth then
            tagSpectralZarrMetadata outputPath (int realWidth) height depth packedAxis width)

    let planeBytes = width * height * 2 * sizeof<float32>
    let fullBytes = planeBytes * depth
    let chunkBytes = planeBytes * chunkZ
    let chunkCount = (depth + chunkZ - 1) / chunkZ
    let inputChunks = collectZarrChunks inputArray
    let outputChunks = collectZarrChunks outputArray
    if inputChunks.Length <> chunkCount then
        invalidOp $"{operationName} expected {chunkCount} input chunks, got {inputChunks.Length}."
    if outputChunks.Length <> chunkCount then
        invalidOp $"{operationName} expected {chunkCount} output chunks, got {outputChunks.Length}."
    let buffer = ArrayPool<byte>.Shared.Rent(fullBytes)

    try
        for zChunk = 0 to chunkCount - 1 do
            let zStart = zChunk * chunkZ
            let validDepth = min chunkZ (depth - zStart)
            let validBytes = planeBytes * validDepth
            let targetOffset = zStart * planeBytes
            let encoded =
                inputArray.ReadChunkEncodedAsync(inputChunks[zChunk], CancellationToken.None)
                |> runTask

            if isNull encoded then
                buffer.AsSpan(targetOffset, validBytes).Clear()
            else
                if encoded.Length < validBytes then
                    invalidOp $"{operationName} raw chunk {zChunk} has {encoded.Length} bytes, expected at least {validBytes}."
                Buffer.BlockCopy(encoded, 0, buffer, targetOffset, validBytes)

        let mutable handle = Unchecked.defaultof<GCHandle>
        try
            handle <- GCHandle.Alloc(buffer, GCHandleType.Pinned)
            NativeSp.fftwfComplexZInplace(handle.AddrOfPinnedObject(), width, height, depth, if inverse then 1 else 0)
            |> NativeSp.checkStatus (if inverse then "inverse fftwf z complex raw chunks" else "fftwf z complex raw chunks")
        finally
            if handle.IsAllocated then
                handle.Free()

        for zChunk = 0 to chunkCount - 1 do
            let zStart = zChunk * chunkZ
            let sourceOffset = zStart * planeBytes
            outputArray.WriteChunkEncodedAsync(
                outputChunks[zChunk],
                buffer.AsMemory(sourceOffset, chunkBytes),
                CancellationToken.None)
            |> runUnitTask
            deleteZarrNetDebugLogs ()
    finally
        ArrayPool<byte>.Shared.Return(buffer)
        writer.DisposeAsync().AsTask()
        |> runUnitTask
        spectralMetadata
        |> Option.iter (fun (packedAxis, realWidth, realHeight, realDepth) ->
            if packedAxis = 0 && realHeight = uint64 height && realDepth = uint64 depth then
                tagSpectralZarrMetadata outputPath (int realWidth) height depth packedAxis width)
        deleteZarrNetDebugLogs ()

let private transformZComplex64InterleavedZarrSubchunksWithSigns
    (signs: int list)
    operationName
    (inputPath: string)
    (outputPath: string)
    (name: string)
    (width: uint)
    (height: uint)
    (depth: uint)
    (chunkX: uint)
    (chunkY: uint)
    (chunkZ: uint)
    (physicalSizeX: float)
    (physicalSizeY: float)
    (physicalSizeZ: float)
    (maxConcurrentWrites: int)
    =

    suppressZarrNetDebugLogging ()
    NativeSp.ensureAvailable ()

    let width = int width
    let height = int height
    let depth = int depth
    let chunkX = max 1u chunkX |> int
    let chunkY = max 1u chunkY |> int
    let chunkZ = max 1u chunkZ |> int

    if width <= 0 || height <= 0 || depth <= 0 then
        invalidArg "shape" $"{operationName} expects positive width, height, and depth."

    let inputLevel = openZarrResolutionLevel inputPath 0 0
    if not (String.Equals(inputLevel.DataType, "complex64", StringComparison.OrdinalIgnoreCase)) then
        invalidOp $"{operationName} expected complex64 input Zarr, got {inputLevel.DataType}."
    if inputLevel.Shape.Length <> 5 then
        invalidOp $"{operationName} expected 5D OME-Zarr input, got rank {inputLevel.Shape.Length}."
    if inputLevel.Shape[0] <> 1L || inputLevel.Shape[1] <> 1L || inputLevel.Shape[2] <> int64 depth || inputLevel.Shape[3] <> int64 height || inputLevel.Shape[4] <> int64 width then
        let actualShape = String.Join(",", inputLevel.Shape)
        invalidOp $"{operationName} expected input shape [1,1,{depth},{height},{width}], got [{actualShape}]."

    let inputReader =
        OmeZarrReader.OpenAsync(inputPath, ct = CancellationToken.None)
        |> runTask
    let inputArray =
        inputReader.RootGroup.OpenArrayAsync(inputLevel.Dataset.Path, CancellationToken.None)
        |> runTask
    let expectedChunkShape = [| 1; 1; chunkZ; chunkY; chunkX |]
    let inputChunkShape = inputArray.Metadata.ChunkShape
    if inputChunkShape <> expectedChunkShape then
        let expectedText = String.Join(",", expectedChunkShape)
        let actualText = String.Join(",", inputChunkShape)
        invalidOp $"{operationName} expects input chunk shape [{expectedText}], got [{actualText}]."

    let spectralMetadata = tryReadSpectralZarrMetadata inputPath

    let descriptor =
        BioImageDescriptor(
            width,
            height,
            ZCT(depth, 1, 1),
            Name = name,
            DataType = "complex64",
            ChunkX = chunkX,
            ChunkY = chunkY,
            ChunkZ = chunkZ,
            ChunkC = 1,
            ChunkT = 1,
            PhysicalSizeX = physicalSizeX,
            PhysicalSizeY = physicalSizeY,
            PhysicalSizeZ = physicalSizeZ,
            Compression = ZarrCompression.None)

    let writer =
        OmeZarrWriter.CreateAsync(outputPath, descriptor, CancellationToken.None)
        |> runTask
    deleteZarrNetDebugLogs ()

    if maxConcurrentWrites > 0 then
        OmeZarrWriter.MaxConcurrentWrites <- maxConcurrentWrites

    let outputLevel = openZarrResolutionLevel outputPath 0 0
    let outputReader =
        OmeZarrReader.OpenAsync(outputPath, ct = CancellationToken.None)
        |> runTask
    let outputArray =
        outputReader.RootGroup.OpenArrayAsync(outputLevel.Dataset.Path, CancellationToken.None)
        |> runTask
    let outputChunkShape = outputArray.Metadata.ChunkShape
    if outputChunkShape <> expectedChunkShape then
        let expectedText = String.Join(",", expectedChunkShape)
        let actualText = String.Join(",", outputChunkShape)
        invalidOp $"{operationName} expects output chunk shape [{expectedText}], got [{actualText}]."

    spectralMetadata
    |> Option.iter (fun (packedAxis, realWidth, realHeight, realDepth) ->
        if packedAxis = 0 && realHeight = uint64 height && realDepth = uint64 depth then
            tagSpectralZarrMetadata outputPath (int realWidth) height depth packedAxis width)

    let xChunks = (width + chunkX - 1) / chunkX
    let yChunks = (height + chunkY - 1) / chunkY
    let zChunks = (depth + chunkZ - 1) / chunkZ
    let fullChunkBytes = chunkX * chunkY * chunkZ * 2 * sizeof<float32>
    let fullTileValues = chunkX * chunkY * depth * 2
    let tile = ArrayPool<float32>.Shared.Rent(fullTileValues)
    let chunkBuffer = ArrayPool<byte>.Shared.Rent(fullChunkBytes)

    try
        for yc = 0 to yChunks - 1 do
            let y0 = yc * chunkY
            let validY = min chunkY (height - y0)
            for xc = 0 to xChunks - 1 do
                let x0 = xc * chunkX
                let validX = min chunkX (width - x0)
                let tileValues = validX * validY * depth * 2
                tile.AsSpan(0, tileValues).Clear()

                for zc = 0 to zChunks - 1 do
                    let zStart = zc * chunkZ
                    let validZ = min chunkZ (depth - zStart)

                    inputArray.ReadChunkDecodedAsync(
                        [| 0L; 0L; int64 zc; int64 yc; int64 xc |],
                        chunkBuffer.AsMemory(0, fullChunkBytes),
                        true,
                        CancellationToken.None)
                    |> runUnitTask
                    deleteZarrNetDebugLogs ()

                    for localZ = 0 to validZ - 1 do
                        for yy = 0 to validY - 1 do
                            let sourceOffset = ((localZ * chunkY + yy) * chunkX) * 2 * sizeof<float32>
                            let targetOffset = (((zStart + localZ) * validY + yy) * validX) * 2
                            Buffer.BlockCopy(chunkBuffer, sourceOffset, tile, targetOffset * sizeof<float32>, validX * 2 * sizeof<float32>)

                let mutable handle = Unchecked.defaultof<GCHandle>
                try
                    handle <- GCHandle.Alloc(tile, GCHandleType.Pinned)
                    for sign in signs do
                        NativeSp.fftwfComplexZInplace(handle.AddrOfPinnedObject(), validX, validY, depth, sign)
                        |> NativeSp.checkStatus (if sign = 0 then "fftwf z complex subchunks" else "inverse fftwf z complex subchunks")
                finally
                    if handle.IsAllocated then
                        handle.Free()

                for zc = 0 to zChunks - 1 do
                    let zStart = zc * chunkZ
                    let validZ = min chunkZ (depth - zStart)
                    chunkBuffer.AsSpan(0, fullChunkBytes).Clear()

                    for localZ = 0 to validZ - 1 do
                        for yy = 0 to validY - 1 do
                            let sourceOffset = (((zStart + localZ) * validY + yy) * validX) * 2 * sizeof<float32>
                            let targetOffset = ((localZ * chunkY + yy) * chunkX) * 2 * sizeof<float32>
                            Buffer.BlockCopy(tile, sourceOffset, chunkBuffer, targetOffset, validX * 2 * sizeof<float32>)

                    outputArray.WriteChunkDecodedAsync(
                        [| 0L; 0L; int64 zc; int64 yc; int64 xc |],
                        chunkBuffer.AsMemory(0, fullChunkBytes),
                        true,
                        CancellationToken.None)
                    |> runUnitTask
                    deleteZarrNetDebugLogs ()
    finally
        ArrayPool<byte>.Shared.Return(chunkBuffer)
        ArrayPool<float32>.Shared.Return(tile)
        writer.DisposeAsync().AsTask()
        |> runUnitTask
        spectralMetadata
        |> Option.iter (fun (packedAxis, realWidth, realHeight, realDepth) ->
            if packedAxis = 0 && realHeight = uint64 height && realDepth = uint64 depth then
                tagSpectralZarrMetadata outputPath (int realWidth) height depth packedAxis width)
        deleteZarrNetDebugLogs ()

let fftZComplex64InterleavedZarrTiles
    (inputPath: string)
    (outputPath: string)
    (name: string)
    (width: uint)
    (height: uint)
    (depth: uint)
    (tileX: uint)
    (tileY: uint)
    (chunkX: uint)
    (chunkY: uint)
    (chunkZ: uint)
    (physicalSizeX: float)
    (physicalSizeY: float)
    (physicalSizeZ: float)
    (maxConcurrentWrites: int)
    =
    transformZComplex64InterleavedZarrTiles
        false
        "fftZComplex64InterleavedZarrTiles"
        inputPath
        outputPath
        name
        width
        height
        depth
        tileX
        tileY
        chunkX
        chunkY
        chunkZ
        physicalSizeX
        physicalSizeY
        physicalSizeZ
        maxConcurrentWrites

let invFftZComplex64InterleavedZarrTiles
    (inputPath: string)
    (outputPath: string)
    (name: string)
    (width: uint)
    (height: uint)
    (depth: uint)
    (tileX: uint)
    (tileY: uint)
    (chunkX: uint)
    (chunkY: uint)
    (chunkZ: uint)
    (physicalSizeX: float)
    (physicalSizeY: float)
    (physicalSizeZ: float)
    (maxConcurrentWrites: int)
    =
    transformZComplex64InterleavedZarrTiles
        true
        "invFftZComplex64InterleavedZarrTiles"
        inputPath
        outputPath
        name
        width
        height
        depth
        tileX
        tileY
        chunkX
        chunkY
        chunkZ
        physicalSizeX
        physicalSizeY
        physicalSizeZ
        maxConcurrentWrites

let fftZComplex64InterleavedZarrRawChunks
    (inputPath: string)
    (outputPath: string)
    (name: string)
    (width: uint)
    (height: uint)
    (depth: uint)
    (chunkZ: uint)
    (physicalSizeX: float)
    (physicalSizeY: float)
    (physicalSizeZ: float)
    (maxConcurrentWrites: int)
    =
    transformZComplex64InterleavedZarrRawChunks
        false
        "fftZComplex64InterleavedZarrRawChunks"
        inputPath
        outputPath
        name
        width
        height
        depth
        chunkZ
        physicalSizeX
        physicalSizeY
        physicalSizeZ
        maxConcurrentWrites

let invFftZComplex64InterleavedZarrRawChunks
    (inputPath: string)
    (outputPath: string)
    (name: string)
    (width: uint)
    (height: uint)
    (depth: uint)
    (chunkZ: uint)
    (physicalSizeX: float)
    (physicalSizeY: float)
    (physicalSizeZ: float)
    (maxConcurrentWrites: int)
    =
    transformZComplex64InterleavedZarrRawChunks
        true
        "invFftZComplex64InterleavedZarrRawChunks"
        inputPath
        outputPath
        name
        width
        height
        depth
        chunkZ
        physicalSizeX
        physicalSizeY
        physicalSizeZ
        maxConcurrentWrites

let fftZComplex64InterleavedZarrSubchunks
    (inputPath: string)
    (outputPath: string)
    (name: string)
    (width: uint)
    (height: uint)
    (depth: uint)
    (chunkX: uint)
    (chunkY: uint)
    (chunkZ: uint)
    (physicalSizeX: float)
    (physicalSizeY: float)
    (physicalSizeZ: float)
    (maxConcurrentWrites: int)
    =
    transformZComplex64InterleavedZarrSubchunksWithSigns
        [ 0 ]
        "fftZComplex64InterleavedZarrSubchunks"
        inputPath
        outputPath
        name
        width
        height
        depth
        chunkX
        chunkY
        chunkZ
        physicalSizeX
        physicalSizeY
        physicalSizeZ
        maxConcurrentWrites

let invFftZComplex64InterleavedZarrSubchunks
    (inputPath: string)
    (outputPath: string)
    (name: string)
    (width: uint)
    (height: uint)
    (depth: uint)
    (chunkX: uint)
    (chunkY: uint)
    (chunkZ: uint)
    (physicalSizeX: float)
    (physicalSizeY: float)
    (physicalSizeZ: float)
    (maxConcurrentWrites: int)
    =
    transformZComplex64InterleavedZarrSubchunksWithSigns
        [ 1 ]
        "invFftZComplex64InterleavedZarrSubchunks"
        inputPath
        outputPath
        name
        width
        height
        depth
        chunkX
        chunkY
        chunkZ
        physicalSizeX
        physicalSizeY
        physicalSizeZ
        maxConcurrentWrites

let fftRoundtripZComplex64InterleavedZarrSubchunks
    (inputPath: string)
    (outputPath: string)
    (name: string)
    (width: uint)
    (height: uint)
    (depth: uint)
    (chunkX: uint)
    (chunkY: uint)
    (chunkZ: uint)
    (physicalSizeX: float)
    (physicalSizeY: float)
    (physicalSizeZ: float)
    (maxConcurrentWrites: int)
    =
    transformZComplex64InterleavedZarrSubchunksWithSigns
        [ 0; 1 ]
        "fftRoundtripZComplex64InterleavedZarrSubchunks"
        inputPath
        outputPath
        name
        width
        height
        depth
        chunkX
        chunkY
        chunkZ
        physicalSizeX
        physicalSizeY
        physicalSizeZ
        maxConcurrentWrites
