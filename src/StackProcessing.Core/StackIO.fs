module StackIO

open SlimPipeline // Core processing model
open System
open System.Buffers
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

let private fromVectorUInt64 (v: itk.simple.VectorUInt64) : uint64 list =
    v |> Seq.map uint64 |> Seq.toList

let private pixelIdToString (id: itk.simple.PixelIDValueEnum) : string =
    let text = id.ToString()
    if text.StartsWith("sitk", StringComparison.Ordinal) then
        text.Substring(4)
    else
        text

let getFileInfo (filename: string) : FileInfo =
    use reader = new itk.simple.ImageFileReader()
    reader.SetFileName(filename)
    reader.ReadImageInformation()
    { dimensions = reader.GetDimension()
      size = reader.GetSize() |> fromVectorUInt64
      componentType = reader.GetPixelID() |> pixelIdToString
      numberOfComponents = reader.GetNumberOfComponents() }

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
        || t = typeof<float32>)

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
    if t = typeof<uint8> then 8, SampleFormat.UINT, 1
    elif t = typeof<int8> then 8, SampleFormat.INT, 1
    elif t = typeof<uint16> then 16, SampleFormat.UINT, 2
    elif t = typeof<int16> then 16, SampleFormat.INT, 2
    elif t = typeof<uint32> then 32, SampleFormat.UINT, 4
    elif t = typeof<int32> then 32, SampleFormat.INT, 4
    elif t = typeof<float32> then 32, SampleFormat.IEEEFP, 4
    elif t = typeof<float> then 64, SampleFormat.IEEEFP, 8
    else
        invalidArg "T" $"TIFF scalar IO currently supports UInt8, Int8, UInt16, Int16, UInt32, Int32, Float32, and Float64 chunks; got {t.Name}."

let private tiffWriteMode (filename: string) =
    let ext = Path.GetExtension(filename).ToLowerInvariant()
    if ext = ".btf" || ext = ".bigtiff" then "w8" else "w"

let private tiffFieldInt (tiff: Tiff) tag fallback =
    let field = tiff.GetField(tag)
    if isNull field || field.Length = 0 then fallback else field[0].ToInt()

let private tiffFieldIntDefaulted (tiff: Tiff) tag fallback =
    let field = tiff.GetFieldDefaulted(tag)
    if isNull field || field.Length = 0 then fallback else field[0].ToInt()

[<Struct; StructLayout(LayoutKind.Sequential)>]
type private NativeUInt8TiffInfo =
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

module private NativeUInt8LibTiff =
    [<Literal>]
    let Ok = 0

    [<Literal>]
    let CompressionNone = 1us

    [<Literal>]
    let PlanarConfigContig = 1us

    [<Literal>]
    let SampleFormatUInt = 1us

    [<DllImport("sp_libtiff_shim", EntryPoint = "sp_tiff_read_info", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)>]
    extern int readInfo(string path, NativeUInt8TiffInfo& info)

    [<DllImport("sp_libtiff_shim", EntryPoint = "sp_tiff_read_raw_page", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)>]
    extern int readRawPage(string path, byte[] buffer, UIntPtr capacity, uint64& bytesRead)

    [<DllImport("sp_libtiff_shim", EntryPoint = "sp_tiff_read_raw_page_into", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)>]
    extern int readRawPageInto(string path, byte[] buffer, UIntPtr bufferOffset, UIntPtr capacity, uint64& bytesRead)

    [<DllImport("sp_libtiff_shim", EntryPoint = "sp_tiff_write_raw_page", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)>]
    extern int writeRawPage(string path, byte[] buffer, UIntPtr count, uint32 width, uint32 height, uint16 bitsPerSample, uint16 sampleFormat)

    [<DllImport("sp_libtiff_shim", EntryPoint = "sp_tiff_write_raw_page_from", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)>]
    extern int writeRawPageFrom(string path, byte[] buffer, UIntPtr bufferOffset, UIntPtr count, UIntPtr capacity, uint32 width, uint32 height, uint16 bitsPerSample, uint16 sampleFormat)

let private isNativeUInt8TiffPage expectedWidth expectedHeight (info: NativeUInt8TiffInfo) =
    info.Width = expectedWidth &&
    info.Height = expectedHeight &&
    info.BitsPerSample = 8us &&
    info.SampleFormat = NativeUInt8LibTiff.SampleFormatUInt &&
    info.SamplesPerPixel = 1us &&
    info.PlanarConfig = NativeUInt8LibTiff.PlanarConfigContig &&
    info.Compression = NativeUInt8LibTiff.CompressionNone &&
    info.IsTiled = 0 &&
    info.IsByteSwapped = 0 &&
    info.PageBytes = info.RawPageBytes &&
    info.PageBytes = uint64 expectedWidth * uint64 expectedHeight

let private tryReadNativeUInt8RawTiffSliceInto fileName expectedWidth expectedHeight sliceOffsetBytes (chunk: Chunk<uint8>) =
    try
        let mutable info = NativeUInt8TiffInfo()
        if NativeUInt8LibTiff.readInfo(fileName, &info) <> NativeUInt8LibTiff.Ok ||
           not (isNativeUInt8TiffPage expectedWidth expectedHeight info) ||
           info.PageBytes > uint64 Int32.MaxValue ||
           sliceOffsetBytes < 0 ||
           chunk.ByteLength < sliceOffsetBytes + int info.PageBytes then
            false
        else
            let mutable bytesRead = 0UL
            let status =
                NativeUInt8LibTiff.readRawPageInto(
                    fileName,
                    chunk.Bytes,
                    UIntPtr(uint64 sliceOffsetBytes),
                    UIntPtr(uint64 chunk.ByteLength),
                    &bytesRead)
            status = NativeUInt8LibTiff.Ok && bytesRead = info.PageBytes
    with
    | :? DllNotFoundException
    | :? EntryPointNotFoundException
    | :? BadImageFormatException ->
        false

let private tryReadNativeUInt8RawTiffSlice fileName expectedWidth expectedHeight (chunk: Chunk<uint8>) =
    tryReadNativeUInt8RawTiffSliceInto fileName expectedWidth expectedHeight 0 chunk

let private tryWriteNativeUInt8RawTiffSliceFrom fileName width height sliceOffsetBytes (chunk: Chunk<uint8>) =
    try
        let pageBytes = uint64 width * uint64 height
        if pageBytes > uint64 Int32.MaxValue ||
           sliceOffsetBytes < 0 ||
           chunk.ByteLength < sliceOffsetBytes + int pageBytes then
            false
        else
            let status =
                NativeUInt8LibTiff.writeRawPageFrom(
                    fileName,
                    chunk.Bytes,
                    UIntPtr(uint64 sliceOffsetBytes),
                    UIntPtr pageBytes,
                    UIntPtr(uint64 chunk.ByteLength),
                    uint32 width,
                    uint32 height,
                    8us,
                    NativeUInt8LibTiff.SampleFormatUInt)
            status = NativeUInt8LibTiff.Ok
    with
    | :? DllNotFoundException
    | :? EntryPointNotFoundException
    | :? BadImageFormatException ->
        false

let private tryWriteNativeUInt8RawTiffSlice fileName width height (chunk: Chunk<uint8>) =
    tryWriteNativeUInt8RawTiffSliceFrom fileName width height 0 chunk

let private tiffDirectoryCount (filename: string) =
    use tiff = Tiff.Open(filename, "r")
    if isNull tiff then
        invalidOp $"Could not open '{filename}' for TIFF volume reading."

    let mutable count = 0u
    let mutable keepGoing = true
    while keepGoing do
        count <- count + 1u
        keepGoing <- tiff.ReadDirectory()

    if count = 0u then
        invalidOp $"TIFF volume '{filename}' contains no readable pages."

    count

let private tiffBytesPerSample bitsPerSample sampleFormat =
    match sampleFormat, bitsPerSample with
    | SampleFormat.UINT, 8
    | SampleFormat.INT, 8 -> 1
    | SampleFormat.UINT, 16
    | SampleFormat.INT, 16 -> 2
    | SampleFormat.UINT, 32
    | SampleFormat.INT, 32
    | SampleFormat.IEEEFP, 32 -> 4
    | SampleFormat.IEEEFP, 64 -> 8
    | _ ->
        invalidOp $"TIFF scalar IO currently supports UInt8, Int8, UInt16, Int16, UInt32, Int32, Float32, and Float64 pages; got {bitsPerSample}-bit {sampleFormat}."

let private validateTiffSamples samplesPerPixel =
    if samplesPerPixel <> 1 then
        invalidOp $"TIFF scalar IO expects one sample per pixel; got {samplesPerPixel} samples per pixel."

let private ensureDirectTiffChunkRead<'T> suffix =
    if not (canReadDirectTiffStack<'T> suffix) then
        invalidArg "suffix" $"readChunkSlices currently supports direct scalar TIFF stacks for UInt8, Int8, UInt16, Int16, UInt32, Int32, Float32, and Float64; got suffix '{suffix}' and type {typeof<'T>.Name}."

let private ensureDirectTiffChunkWrite<'T> suffix =
    if not (canWriteDirectTiffStack<'T> suffix) then
        invalidArg "suffix" $"writeChunkSlices currently supports direct scalar TIFF stack output for UInt8, Int8, UInt16, Int16, UInt32, Int32, and Float32; got suffix '{suffix}' and type {typeof<'T>.Name}."

let private inspectChunkTiffSlice<'T> fileName =
    let expectedBits, expectedFormat, expectedBytesPerSample = tiffPixelLayout<'T> ()
    use tiff = Tiff.Open(fileName, "r")
    if isNull tiff then
        invalidOp $"Could not open '{fileName}' for TIFF reading."

    if not (tiff.SetDirectory(int16 0)) then
        invalidOp $"Could not select TIFF page 0 in '{fileName}'."

    let width = uint (tiffFieldInt tiff TiffTag.IMAGEWIDTH 0)
    let height = uint (tiffFieldInt tiff TiffTag.IMAGELENGTH 0)
    if width = 0u || height = 0u then
        invalidOp $"TIFF slice '{fileName}' has invalid page dimensions {width}x{height}."

    let bitsPerSample = tiffFieldIntDefaulted tiff TiffTag.BITSPERSAMPLE 1
    let samplesPerPixel = tiffFieldIntDefaulted tiff TiffTag.SAMPLESPERPIXEL 1
    validateTiffSamples samplesPerPixel

    let sampleFormat =
        tiffFieldIntDefaulted tiff TiffTag.SAMPLEFORMAT (int SampleFormat.UINT)
        |> enum<SampleFormat>

    if bitsPerSample <> expectedBits || sampleFormat <> expectedFormat then
        invalidOp $"Input slice '{fileName}' does not match {typeof<'T>.Name}: got {bitsPerSample}-bit {sampleFormat}."

    width, height, int width * expectedBytesPerSample, max (int width * expectedBytesPerSample) (tiff.ScanlineSize())

type private ChunkTiffReadPlan =
    { Width: uint
      Height: uint
      RowBytes: int
      ScanlineSize: int
      ConvertUInt8ToFloat32: bool }

let private inspectChunkTiffSliceForRead<'T> fileName =
    let expectedBits, expectedFormat, expectedBytesPerSample = tiffPixelLayout<'T> ()
    use tiff = Tiff.Open(fileName, "r")
    if isNull tiff then
        invalidOp $"Could not open '{fileName}' for TIFF reading."

    if not (tiff.SetDirectory(int16 0)) then
        invalidOp $"Could not select TIFF page 0 in '{fileName}'."

    let width = uint (tiffFieldInt tiff TiffTag.IMAGEWIDTH 0)
    let height = uint (tiffFieldInt tiff TiffTag.IMAGELENGTH 0)
    if width = 0u || height = 0u then
        invalidOp $"TIFF slice '{fileName}' has invalid page dimensions {width}x{height}."

    let bitsPerSample = tiffFieldIntDefaulted tiff TiffTag.BITSPERSAMPLE 1
    let samplesPerPixel = tiffFieldIntDefaulted tiff TiffTag.SAMPLESPERPIXEL 1
    validateTiffSamples samplesPerPixel

    let sampleFormat =
        tiffFieldIntDefaulted tiff TiffTag.SAMPLEFORMAT (int SampleFormat.UINT)
        |> enum<SampleFormat>

    if bitsPerSample = expectedBits && sampleFormat = expectedFormat then
        { Width = width
          Height = height
          RowBytes = int width * expectedBytesPerSample
          ScanlineSize = max (int width * expectedBytesPerSample) (tiff.ScanlineSize())
          ConvertUInt8ToFloat32 = false }
    elif typeof<'T> = typeof<float32> && bitsPerSample = 8 && sampleFormat = SampleFormat.UINT then
        let rowBytes = int width * sizeof<float32>
        { Width = width
          Height = height
          RowBytes = rowBytes
          ScanlineSize = max (int width) (tiff.ScanlineSize())
          ConvertUInt8ToFloat32 = true }
    else
        invalidOp $"Input slice '{fileName}' does not match {typeof<'T>.Name}: got {bitsPerSample}-bit {sampleFormat}."

let private tryReadTiffEncodedStripsInto (tiff: Tiff) (chunk: Chunk<'T>) sliceOffsetBytes sliceBytes =
    if tiff.IsTiled() then
        false
    else
        let planarConfig =
            tiffFieldIntDefaulted tiff TiffTag.PLANARCONFIG (int PlanarConfig.CONTIG)
            |> enum<PlanarConfig>

        if planarConfig <> PlanarConfig.CONTIG then
            false
        else
            let strips = tiff.NumberOfStrips()
            if strips < 1 then
                false
            else
                let stripSize = tiff.StripSize()

                let rec readStrip strip offset =
                    if strip = strips then
                        offset = sliceBytes
                    else
                        let remaining = sliceBytes - offset
                        if remaining <= 0 || stripSize <= 0 then
                            false
                        else
                            let requested = min remaining stripSize
                            let bytesRead = tiff.ReadEncodedStrip(strip, chunk.Bytes, sliceOffsetBytes + offset, requested)
                            if bytesRead < 0 || bytesRead > remaining then
                                false
                            else
                                readStrip (strip + 1) (offset + bytesRead)

                readStrip 0 0

let private tryReadTiffRawStripsInto expectedBytesPerSample (tiff: Tiff) (chunk: Chunk<'T>) sliceOffsetBytes sliceBytes =
    if tiff.IsTiled() then
        false
    else
        let compression =
            tiffFieldIntDefaulted tiff TiffTag.COMPRESSION (int Compression.NONE)
            |> enum<Compression>

        let planarConfig =
            tiffFieldIntDefaulted tiff TiffTag.PLANARCONFIG (int PlanarConfig.CONTIG)
            |> enum<PlanarConfig>

        if compression <> Compression.NONE ||
           planarConfig <> PlanarConfig.CONTIG ||
           (expectedBytesPerSample > 1 && tiff.IsByteSwapped()) then
            false
        else
            let strips = tiff.NumberOfStrips()
            if strips < 1 then
                false
            else
                let rawStripSizes =
                    Array.init strips (fun strip ->
                        let size = tiff.RawStripSize(strip)
                        if size <= 0L || size > int64 Int32.MaxValue then
                            -1
                        else
                            int size)

                if rawStripSizes |> Array.exists ((=) -1) then
                    false
                else
                    let rawBytes = rawStripSizes |> Array.sum
                    if rawBytes <> sliceBytes then
                        false
                    else
                        let rec readStrip strip offset =
                            if strip = strips then
                                offset = sliceBytes
                            else
                                let count = rawStripSizes[strip]
                                let bytesRead = tiff.ReadRawStrip(strip, chunk.Bytes, sliceOffsetBytes + offset, count)
                                if bytesRead <> count then
                                    false
                                else
                                    readStrip (strip + 1) (offset + count)

                        readStrip 0 0

let private readChunkTiffSliceIntoOffset<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    fileName
    pageIndex
    (chunk: Chunk<'T>)
    expectedWidth
    expectedHeight
    sliceOffsetBytes
    =

    use tiff = Tiff.Open(fileName, "r")
    if isNull tiff then
        invalidOp $"Could not open '{fileName}' for TIFF reading."
    if not (tiff.SetDirectory(int16 pageIndex)) then
        invalidOp $"Could not select TIFF page {pageIndex} in '{fileName}'."

    let expectedBits, expectedFormat, expectedBytesPerSample = tiffPixelLayout<'T> ()
    let width = uint (tiffFieldInt tiff TiffTag.IMAGEWIDTH 0)
    let height = uint (tiffFieldInt tiff TiffTag.IMAGELENGTH 0)
    if width = 0u || height = 0u then
        invalidOp $"TIFF slice '{fileName}' has invalid page dimensions {width}x{height}."

    let bitsPerSample = tiffFieldIntDefaulted tiff TiffTag.BITSPERSAMPLE 1
    let samplesPerPixel = tiffFieldIntDefaulted tiff TiffTag.SAMPLESPERPIXEL 1
    validateTiffSamples samplesPerPixel

    let sampleFormat =
        tiffFieldIntDefaulted tiff TiffTag.SAMPLEFORMAT (int SampleFormat.UINT)
        |> enum<SampleFormat>

    if bitsPerSample <> expectedBits || sampleFormat <> expectedFormat then
        invalidOp $"Input slice '{fileName}' does not match {typeof<'T>.Name}: got {bitsPerSample}-bit {sampleFormat}."

    if width <> expectedWidth || height <> expectedHeight then
        invalidOp $"Input slice '{fileName}' has shape {width}x{height}, expected {expectedWidth}x{expectedHeight}."

    let rowBytes = int width * expectedBytesPerSample
    let scanlineSize = max rowBytes (tiff.ScanlineSize())
    let sliceBytes = rowBytes * int height
    if sliceOffsetBytes < 0 || chunk.ByteLength < sliceOffsetBytes + sliceBytes then
        invalidArg "chunk" $"Chunk byte length {chunk.ByteLength} is too small to read slice '{fileName}' at offset {sliceOffsetBytes} with length {sliceBytes}."

    if tryReadTiffRawStripsInto expectedBytesPerSample tiff chunk sliceOffsetBytes sliceBytes then
        ()
    elif tryReadTiffEncodedStripsInto tiff chunk sliceOffsetBytes sliceBytes then
        ()
    elif scanlineSize <= rowBytes then
        for row in 0 .. int height - 1 do
            if not (tiff.ReadScanline(chunk.Bytes, sliceOffsetBytes + row * rowBytes, row, int16 0)) then
                invalidOp $"Failed to read TIFF scanline {row} from '{fileName}'."
    else
        let scratch = System.Buffers.ArrayPool<byte>.Shared.Rent(scanlineSize)
        try
            for row in 0 .. int height - 1 do
                if not (tiff.ReadScanline(scratch, row)) then
                    invalidOp $"Failed to read TIFF scanline {row} from '{fileName}'."
                Buffer.BlockCopy(scratch, 0, chunk.Bytes, sliceOffsetBytes + row * rowBytes, rowBytes)
        finally
            System.Buffers.ArrayPool<byte>.Shared.Return(scratch)

let private readChunkTiffUInt8SliceAsFloat32IntoOffset
    fileName
    pageIndex
    (chunk: Chunk<float32>)
    expectedWidth
    expectedHeight
    sliceOffsetBytes
    =

    use tiff = Tiff.Open(fileName, "r")
    if isNull tiff then
        invalidOp $"Could not open '{fileName}' for TIFF reading."
    if not (tiff.SetDirectory(int16 pageIndex)) then
        invalidOp $"Could not select TIFF page {pageIndex} in '{fileName}'."

    let width = uint (tiffFieldInt tiff TiffTag.IMAGEWIDTH 0)
    let height = uint (tiffFieldInt tiff TiffTag.IMAGELENGTH 0)
    if width <> expectedWidth || height <> expectedHeight then
        invalidOp $"Input slice '{fileName}' has shape {width}x{height}, expected {expectedWidth}x{expectedHeight}."

    let bitsPerSample = tiffFieldIntDefaulted tiff TiffTag.BITSPERSAMPLE 1
    let samplesPerPixel = tiffFieldIntDefaulted tiff TiffTag.SAMPLESPERPIXEL 1
    validateTiffSamples samplesPerPixel

    let sampleFormat =
        tiffFieldIntDefaulted tiff TiffTag.SAMPLEFORMAT (int SampleFormat.UINT)
        |> enum<SampleFormat>

    if bitsPerSample <> 8 || sampleFormat <> SampleFormat.UINT then
        invalidOp $"Input slice '{fileName}' cannot be converted to Single: got {bitsPerSample}-bit {sampleFormat}."

    let widthI = int width
    let heightI = int height
    let sliceBytes = widthI * heightI * sizeof<float32>
    if sliceOffsetBytes < 0 || chunk.ByteLength < sliceOffsetBytes + sliceBytes then
        invalidArg "chunk" $"Chunk byte length {chunk.ByteLength} is too small to read slice '{fileName}' at offset {sliceOffsetBytes} with length {sliceBytes}."

    let scanlineSize = max widthI (tiff.ScanlineSize())
    let scratch = ArrayPool<byte>.Shared.Rent(scanlineSize)
    try
        let outputBytes = chunk.Bytes.AsSpan(sliceOffsetBytes, sliceBytes)
        let output = MemoryMarshal.Cast<byte, float32>(outputBytes)
        for row in 0 .. heightI - 1 do
            if not (tiff.ReadScanline(scratch, row)) then
                invalidOp $"Failed to read TIFF scanline {row} from '{fileName}'."
            let rowOffset = row * widthI
            for x in 0 .. widthI - 1 do
                output[rowOffset + x] <- float32 scratch[x]
    finally
        ArrayPool<byte>.Shared.Return(scratch)

let private readChunkTiffSliceByPlanIntoOffset<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (plan: ChunkTiffReadPlan)
    fileName
    pageIndex
    (chunk: Chunk<'T>)
    sliceOffsetBytes
    =
    if plan.ConvertUInt8ToFloat32 then
        let floatChunk = box chunk :?> Chunk<float32>
        readChunkTiffUInt8SliceAsFloat32IntoOffset fileName pageIndex floatChunk plan.Width plan.Height sliceOffsetBytes
    else
        readChunkTiffSliceIntoOffset<'T> fileName pageIndex chunk plan.Width plan.Height sliceOffsetBytes

let private readChunkTiffSlice<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> fileName =
    let width, height, rowBytes, _scanlineSize = inspectChunkTiffSlice<'T> fileName
    let chunk = Chunk.create<'T> (uint64 width, uint64 height, 1UL)
    try
        readChunkTiffSliceIntoOffset<'T> fileName 0 chunk width height 0
        chunk
    with
    | ex ->
        Chunk.decRef chunk
        reraise()

let private setChunkTiffPageFields<'T> (tiff: Tiff) width height =
    let bitsPerSample, sampleFormat, _bytesPerSample = tiffPixelLayout<'T> ()
    tiff.SetField(TiffTag.IMAGEWIDTH, int width) |> ignore
    tiff.SetField(TiffTag.IMAGELENGTH, int height) |> ignore
    tiff.SetField(TiffTag.SAMPLESPERPIXEL, 1) |> ignore
    tiff.SetField(TiffTag.BITSPERSAMPLE, bitsPerSample) |> ignore
    tiff.SetField(TiffTag.SAMPLEFORMAT, sampleFormat) |> ignore
    tiff.SetField(TiffTag.PHOTOMETRIC, Photometric.MINISBLACK) |> ignore
    tiff.SetField(TiffTag.PLANARCONFIG, PlanarConfig.CONTIG) |> ignore
    tiff.SetField(TiffTag.ROWSPERSTRIP, int height) |> ignore
    tiff.SetField(TiffTag.COMPRESSION, Compression.NONE) |> ignore

let private writeChunkTiffSliceToOpenTiff<'T when 'T: equality> fileName (tiff: Tiff) (chunk: Chunk<'T>) width height sliceOffsetBytes =
    let _bitsPerSample, _sampleFormat, bytesPerSample = tiffPixelLayout<'T> ()
    let rowBytes = int width * bytesPerSample
    let sliceBytes = rowBytes * int height
    if sliceOffsetBytes < 0 || chunk.ByteLength < sliceOffsetBytes + sliceBytes then
        invalidArg "chunk" $"Chunk byte length {chunk.ByteLength} is smaller than the requested TIFF page payload at offset {sliceOffsetBytes} with length {sliceBytes}."

    setChunkTiffPageFields<'T> tiff width height
    let written = tiff.WriteRawStrip(0, chunk.Bytes, sliceOffsetBytes, sliceBytes)
    if written < 0 then
        invalidOp $"Failed to write TIFF strip to '{fileName}'."
    if written <> sliceBytes then
        invalidOp $"Wrote {written} bytes to TIFF strip in '{fileName}', expected {sliceBytes}."

    if not (tiff.WriteDirectory()) then
        invalidOp $"Failed to write TIFF directory to '{fileName}'."

let private writeChunkTiffSliceFromOffset<'T when 'T: equality> fileName (chunk: Chunk<'T>) width height sliceOffsetBytes =
    let writtenNative =
        if typeof<'T> = typeof<uint8> then
            let byteChunk = box chunk :?> Chunk<uint8>
            tryWriteNativeUInt8RawTiffSliceFrom fileName width height sliceOffsetBytes byteChunk
        else
            false

    if not writtenNative then
        use tiff = Tiff.Open(fileName, tiffWriteMode fileName)
        if isNull tiff then
            invalidOp $"Could not open '{fileName}' for TIFF writing."

        writeChunkTiffSliceToOpenTiff<'T> fileName tiff chunk width height sliceOffsetBytes

let private writeChunkTiffSlice<'T when 'T: equality> fileName (chunk: Chunk<'T>) =
    let width, height, depth = chunk.Size
    if depth <> 1UL then
        invalidArg "chunk" $"writeChunkSlices expects 2D slice chunks with depth 1, got {chunk.Size}."

    writeChunkTiffSliceFromOffset<'T> fileName chunk width height 0

let private writeChunkTiffFile<'T when 'T: equality> fileName (chunk: Chunk<'T>) =
    let width, height, depth = chunk.Size
    if depth = 0UL then
        invalidArg "chunk" $"Cannot write an empty-depth TIFF chunk: {chunk.Size}."

    let _bitsPerSample, _sampleFormat, bytesPerSample = tiffPixelLayout<'T> ()
    let sliceBytes = int width * int height * bytesPerSample
    if chunk.ByteLength < sliceBytes * int depth then
        invalidArg "chunk" $"Chunk byte length {chunk.ByteLength} is smaller than {depth} TIFF pages of {sliceBytes} bytes."

    use tiff = Tiff.Open(fileName, tiffWriteMode fileName)
    if isNull tiff then
        invalidOp $"Could not open '{fileName}' for TIFF writing."

    for localZ in 0 .. int depth - 1 do
        writeChunkTiffSliceToOpenTiff<'T> fileName tiff chunk width height (localZ * sliceBytes)

let private inspectColorChunkTiffSlice fileName =
    use tiff = Tiff.Open(fileName, "r")
    if isNull tiff then
        invalidOp $"Could not open '{fileName}' for RGB TIFF reading."

    let width = uint (tiffFieldInt tiff TiffTag.IMAGEWIDTH 0)
    let height = uint (tiffFieldInt tiff TiffTag.IMAGELENGTH 0)
    if width = 0u || height = 0u then
        invalidOp $"RGB TIFF slice '{fileName}' has invalid page dimensions {width}x{height}."

    let bitsPerSample = tiffFieldIntDefaulted tiff TiffTag.BITSPERSAMPLE 8
    let samplesPerPixel = tiffFieldIntDefaulted tiff TiffTag.SAMPLESPERPIXEL 3
    let sampleFormat =
        tiffFieldIntDefaulted tiff TiffTag.SAMPLEFORMAT (int SampleFormat.UINT)
        |> enum<SampleFormat>
    let planarConfig =
        tiffFieldIntDefaulted tiff TiffTag.PLANARCONFIG (int PlanarConfig.CONTIG)
        |> enum<PlanarConfig>

    if bitsPerSample <> 8 || sampleFormat <> SampleFormat.UINT || samplesPerPixel <> 3 || planarConfig <> PlanarConfig.CONTIG then
        invalidOp $"RGB TIFF slices must be contiguous 8-bit unsigned RGB, got bits={bitsPerSample}, format={sampleFormat}, samples={samplesPerPixel}, planar={planarConfig} in '{fileName}'."

    let rowBytes = int width * 3
    width, height, rowBytes, max rowBytes (tiff.ScanlineSize())

let private readColorChunkTiffSlice fileName : VectorChunk<uint8> =
    use tiff = Tiff.Open(fileName, "r")
    if isNull tiff then
        invalidOp $"Could not open '{fileName}' for RGB TIFF reading."

    let width, height, rowBytes, scanlineSize = inspectColorChunkTiffSlice fileName
    let chunk = Chunk.create<uint8> (uint64 width * 3UL, uint64 height, 1UL)
    let vector: VectorChunk<uint8> =
        { SpatialSize = (uint64 width, uint64 height, 1UL)
          Components = 3u
          Chunk = chunk }
    try
        if scanlineSize <= rowBytes then
            for row in 0 .. int height - 1 do
                if not (tiff.ReadScanline(chunk.Bytes, row * rowBytes, row, int16 0)) then
                    invalidOp $"Failed to read RGB TIFF scanline {row} from '{fileName}'."
        else
            let scratch = System.Buffers.ArrayPool<byte>.Shared.Rent(scanlineSize)
            try
                for row in 0 .. int height - 1 do
                    if not (tiff.ReadScanline(scratch, row)) then
                        invalidOp $"Failed to read RGB TIFF scanline {row} from '{fileName}'."
                    Buffer.BlockCopy(scratch, 0, chunk.Bytes, row * rowBytes, rowBytes)
            finally
                System.Buffers.ArrayPool<byte>.Shared.Return(scratch)
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

    use tiff = Tiff.Open(fileName, tiffWriteMode fileName)
    if isNull tiff then
        invalidOp $"Could not open '{fileName}' for RGB TIFF writing."

    tiff.SetField(TiffTag.IMAGEWIDTH, int width) |> ignore
    tiff.SetField(TiffTag.IMAGELENGTH, int height) |> ignore
    tiff.SetField(TiffTag.SAMPLESPERPIXEL, 3) |> ignore
    tiff.SetField(TiffTag.BITSPERSAMPLE, 8) |> ignore
    tiff.SetField(TiffTag.SAMPLEFORMAT, SampleFormat.UINT) |> ignore
    tiff.SetField(TiffTag.PHOTOMETRIC, Photometric.RGB) |> ignore
    tiff.SetField(TiffTag.PLANARCONFIG, PlanarConfig.CONTIG) |> ignore
    tiff.SetField(TiffTag.ROWSPERSTRIP, int height) |> ignore
    tiff.SetField(TiffTag.COMPRESSION, Compression.NONE) |> ignore

    for row in 0 .. int height - 1 do
        if not (tiff.WriteScanline(vector.Chunk.Bytes, row * rowBytes, row, int16 0)) then
            invalidOp $"Failed to write RGB TIFF scanline {row} to '{fileName}'."

    if not (tiff.WriteDirectory()) then
        invalidOp $"Failed to write RGB TIFF directory to '{fileName}'."

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

let private blockCopyZarrBytes<'T> elementCount expectedBytes (bytes: byte[]) =
    if bytes.Length < expectedBytes then
        invalidArg "bytes" $"Zarr byte payload is too short for {elementCount} {typeof<'T>.Name} values: expected {expectedBytes}, got {bytes.Length}."

    let pixels = Array.zeroCreate<'T> elementCount
    Buffer.BlockCopy(bytes, 0, pixels, 0, expectedBytes)
    pixels

let private flatArrayOfZarrBytes<'T> (width: int) (height: int) (depth: int) (bytes: byte[]) =
    let elementCount = width * height * depth

    if typeof<'T> = typeof<uint8> then
        blockCopyZarrBytes<uint8> elementCount elementCount bytes
        |> box
        |> unbox<'T[]>
    elif typeof<'T> = typeof<uint16> then
        blockCopyZarrBytes<uint16> elementCount (elementCount * 2) bytes
        |> box
        |> unbox<'T[]>
    elif typeof<'T> = typeof<float32> then
        blockCopyZarrBytes<float32> elementCount (elementCount * 4) bytes
        |> box
        |> unbox<'T[]>
    elif typeof<'T> = typeof<float> then
        blockCopyZarrBytes<float> elementCount (elementCount * 8) bytes
        |> box
        |> unbox<'T[]>
    else
        zarrDataType<'T> () |> ignore
        failwith "unreachable"

#if LEGACY_IMAGE
let private complexFloat32ImageOfZarrBytes width height depth (bytes: byte[]) name =
    let elementCount = width * height * depth
    let components = blockCopyZarrBytes<float32> (elementCount * 2) (elementCount * 8) bytes
    let real = Array.zeroCreate<float32> elementCount
    let imag = Array.zeroCreate<float32> elementCount
    for i in 0 .. elementCount - 1 do
        let j = 2 * i
        real[i] <- components[j]
        imag[i] <- components[j + 1]

    let realImage = Image<float32>.ofFlatArray([ uint width; uint height; uint depth ], real, $"{name}.Re")
    let imagImage = Image<float32>.ofFlatArray([ uint width; uint height; uint depth ], imag, $"{name}.Im")
    try
        Image<float32>.ofImagePairToComplexFloat32 realImage imagImage
    finally
        realImage.decRefCount()
        imagImage.decRefCount()

let private complexFloat64ImageOfZarrBytes width height depth (bytes: byte[]) name =
    let elementCount = width * height * depth
    let components = blockCopyZarrBytes<float> (elementCount * 2) (elementCount * 16) bytes
    let real = Array.zeroCreate<float> elementCount
    let imag = Array.zeroCreate<float> elementCount
    for i in 0 .. elementCount - 1 do
        let j = 2 * i
        real[i] <- components[j]
        imag[i] <- components[j + 1]

    let realImage = Image<float>.ofFlatArray([ uint width; uint height; uint depth ], real, $"{name}.Re")
    let imagImage = Image<float>.ofFlatArray([ uint width; uint height; uint depth ], imag, $"{name}.Im")
    try
        Image<float>.ofImagePairToComplex realImage imagImage
    finally
        realImage.decRefCount()
        imagImage.decRefCount()

let private zarrSlabImageAs<'T when 'T: equality> (dataType: string) width height depth (bytes: byte[]) name =
    let castNative (nativeImage: Image<'Native>) =
        let cast = nativeImage.castTo<'T>()
        nativeImage.decRefCount()
        cast

    if String.Equals(dataType, "uint8", StringComparison.OrdinalIgnoreCase) then
        flatArrayOfZarrBytes<uint8> width height depth bytes
        |> fun pixels -> Image<uint8>.ofFlatArray([ uint width; uint height; uint depth ], pixels, name)
        |> castNative
    elif String.Equals(dataType, "uint16", StringComparison.OrdinalIgnoreCase) then
        flatArrayOfZarrBytes<uint16> width height depth bytes
        |> fun pixels -> Image<uint16>.ofFlatArray([ uint width; uint height; uint depth ], pixels, name)
        |> castNative
    elif String.Equals(dataType, "float32", StringComparison.OrdinalIgnoreCase) then
        flatArrayOfZarrBytes<float32> width height depth bytes
        |> fun pixels -> Image<float32>.ofFlatArray([ uint width; uint height; uint depth ], pixels, name)
        |> castNative
    elif String.Equals(dataType, "float64", StringComparison.OrdinalIgnoreCase) then
        flatArrayOfZarrBytes<float> width height depth bytes
        |> fun pixels -> Image<float>.ofFlatArray([ uint width; uint height; uint depth ], pixels, name)
        |> castNative
    elif String.Equals(dataType, "complex64", StringComparison.OrdinalIgnoreCase) then
        complexFloat32ImageOfZarrBytes width height depth bytes name
        |> castNative
    elif String.Equals(dataType, "complex128", StringComparison.OrdinalIgnoreCase) then
        complexFloat64ImageOfZarrBytes width height depth bytes name
        |> castNative
    else
        failwith $"ZarrNET image IO currently supports UInt8, UInt16, Float32, Float64, Complex64, and Complex128 datasets, but dataset type was {dataType}."
#endif

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

#if LEGACY_IMAGE
let private flatArrayOfHdfSource<'Native> rank frameAxis yAxis xAxis sizeX sizeY zCount (source: 'Native[,,]) =
    let elementCount = sizeX * sizeY * zCount

    if rank = 3 && frameAxis = 0 && yAxis = 1 && xAxis = 2 then
        let pixels = Array.zeroCreate<'Native> elementCount
        let byteCount = elementCount * (typeof<'Native> |> Image.getBytesPerComponent |> int)
        Buffer.BlockCopy(source, 0, pixels, 0, byteCount)
        pixels
    else
        let sourceArray = source :> Array
        Array.init elementCount (fun i ->
            let x = i % sizeX
            let yz = i / sizeX
            let y = yz % sizeY
            let z = yz / sizeY
            let indices = Array.zeroCreate<int> rank
            indices[frameAxis] <- z
            indices[yAxis] <- y
            indices[xAxis] <- x
            sourceArray.GetValue(indices) |> unbox<'Native>)

let private castOrReuseNativeImage<'Native, 'T when 'Native: equality and 'T: equality> (nativeImage: Image<'Native>) =
    if typeof<'Native> = typeof<'T> then
        nativeImage |> box :?> Image<'T>
    else
        let cast = nativeImage.castTo<'T>()
        nativeImage.decRefCount()
        cast

let private readHdfSlabNative<'Native, 'T when 'Native: equality and 'T: equality>
    (dataset: IH5Dataset)
    rank
    starts
    blocks
    sizeX
    sizeY
    zCount
    frameAxis
    yAxis
    xAxis
    name =

    let selection = HyperslabSelection(rank, starts, blocks)
    let source = dataset.Read<'Native[,,]>(selection, AllSelection(), blocks)
    let pixels = flatArrayOfHdfSource<'Native> rank frameAxis yAxis xAxis sizeX sizeY zCount source
    let nativeImage = Image<'Native>.ofFlatArray([ uint sizeX; uint sizeY; uint zCount ], pixels, name)
    castOrReuseNativeImage<'Native, 'T> nativeImage

let private hdfSlabImageAs<'T when 'T: equality>
    (dataset: IH5Dataset)
    rank
    starts
    blocks
    sizeX
    sizeY
    zCount
    frameAxis
    yAxis
    xAxis
    name =

    match dataset.Type.Class, dataset.Type.Size with
    | H5DataTypeClass.FixedPoint, 1 when dataset.Type.FixedPoint.IsSigned ->
        readHdfSlabNative<int8, 'T> dataset rank starts blocks sizeX sizeY zCount frameAxis yAxis xAxis name
    | H5DataTypeClass.FixedPoint, 1 ->
        readHdfSlabNative<uint8, 'T> dataset rank starts blocks sizeX sizeY zCount frameAxis yAxis xAxis name
    | H5DataTypeClass.FixedPoint, 2 when dataset.Type.FixedPoint.IsSigned ->
        readHdfSlabNative<int16, 'T> dataset rank starts blocks sizeX sizeY zCount frameAxis yAxis xAxis name
    | H5DataTypeClass.FixedPoint, 2 ->
        readHdfSlabNative<uint16, 'T> dataset rank starts blocks sizeX sizeY zCount frameAxis yAxis xAxis name
    | H5DataTypeClass.FixedPoint, 4 when dataset.Type.FixedPoint.IsSigned ->
        readHdfSlabNative<int32, 'T> dataset rank starts blocks sizeX sizeY zCount frameAxis yAxis xAxis name
    | H5DataTypeClass.FixedPoint, 4 ->
        readHdfSlabNative<uint32, 'T> dataset rank starts blocks sizeX sizeY zCount frameAxis yAxis xAxis name
    | H5DataTypeClass.FloatingPoint, 4 ->
        readHdfSlabNative<float32, 'T> dataset rank starts blocks sizeX sizeY zCount frameAxis yAxis xAxis name
    | H5DataTypeClass.FloatingPoint, 8 ->
        readHdfSlabNative<float, 'T> dataset rank starts blocks sizeX sizeY zCount frameAxis yAxis xAxis name
    | dataClass, size ->
        failwith $"HDF5/NeXus image IO supports scalar numeric datasets UInt8, Int8, UInt16, Int16, UInt32, Int32, Float32, and Float64; dataset class was {dataClass} with size {size} bytes."
#endif

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

let readChunkSlices<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> inputDir suffix (pl: Plan<unit, unit>) : Plan<unit, Chunk<'T>> =
    let name = "readChunkSlices"
    ensureDirectTiffChunkRead<'T> suffix
    let files = getStackFiles inputDir suffix
    if files.Length = 0 then
        stopWithInputError $"No {suffixDescription suffix} files found in input stack directory: {inputDir}"

    let readPlan = inspectChunkTiffSliceForRead<'T> files[0]
    let width = readPlan.Width
    let height = readPlan.Height
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
                  "scanlineSize", string readPlan.ScanlineSize ])

    let mapper (i: int) =
        let fileName = files[i]
        if pl.debug then
            printfn $"[{name}] Reading chunk slice {i}: {fileName} as {typeof<'T>.Name}"
        let chunk = Chunk.create<'T> (uint64 width, uint64 height, 1UL)
        try
            let readNative =
                if typeof<'T> = typeof<uint8> then
                    let byteChunk = box chunk :?> Chunk<uint8>
                    tryReadNativeUInt8RawTiffSlice fileName width height byteChunk
                else
                    false

            if not readNative then
                readChunkTiffSliceByPlanIntoOffset readPlan fileName 0 chunk 0
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
    let width, height, rowBytes, scanlineSize = inspectChunkTiffSlice<'T> files[0]
    let sliceBytes = rowBytes * int height
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
                  "scanlineSize", string scanlineSize ])

    let mapper (groupIndex: int) =
        let firstSlice = groupIndex * chunkDepth
        let zCount = min chunkDepth (pages.Length - firstSlice)
        if pl.debug then
            printfn $"[{name}] Reading slices {firstSlice}..{firstSlice + zCount - 1} from {inputDir} as {typeof<'T>.Name}"

        let chunk = Chunk.create<'T> (uint64 width, uint64 height, uint64 zCount)
        try
            for localZ in 0 .. zCount - 1 do
                let fileName, pageIndex = pages[firstSlice + localZ]
                readChunkTiffSliceIntoOffset<'T> fileName pageIndex chunk width height (localZ * sliceBytes)
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

let readChunkThickFiles<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> inputDir suffix (pl: Plan<unit, unit>) : Plan<unit, Chunk<'T>> =
    let name = "readChunkThickFiles"
    ensureDirectTiffChunkRead<'T> suffix
    let files = getStackFiles inputDir suffix
    if files.Length = 0 then
        stopWithInputError $"No {suffixDescription suffix} files found in input stack directory: {inputDir}"

    let pageCounts = files |> Array.map (tiffDirectoryCount >> int)
    let width, height, rowBytes, scanlineSize = inspectChunkTiffSlice<'T> files[0]
    let sliceBytes = rowBytes * int height
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
                  "scanlineSize", string scanlineSize ])

    let mapper (i: int) =
        let fileName = files[i]
        let zCount = pageCounts[i]
        if pl.debug then
            printfn $"[{name}] Reading chunk file {i}: {fileName} with {zCount} page(s) as {typeof<'T>.Name}"

        let chunk = Chunk.create<'T> (uint64 width, uint64 height, uint64 zCount)
        try
            for localZ in 0 .. zCount - 1 do
                readChunkTiffSliceIntoOffset<'T> fileName localZ chunk width height (localZ * sliceBytes)
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
                   "scanlineSize", string readPlan.ScanlineSize ]
                 @ sourcePeekFields))

    let mapper (i: int) =
        let fileName, pageIndex = pages[i]
        if pl.debug then
            printfn $"[{name}] Reading chunk slice {i}: {fileName} page {pageIndex} as {typeof<'T>.Name}"
        let chunk = Chunk.create<'T> (uint64 width, uint64 height, 1UL)
        try
            readChunkTiffSliceByPlanIntoOffset readPlan fileName pageIndex chunk 0
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

let readChunkSlicesRandom<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (count: uint)
    inputDir
    suffix
    (pl: Plan<unit, unit>)
    : Plan<unit, Chunk<'T>> =
    readSelectedChunkSlices<'T>
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

    let width, height, rowBytes, scanlineSize = inspectColorChunkTiffSlice files[0]
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
                   "scanlineSize", string scanlineSize ]
                 @ sourcePeekFields))

    let mapper (i: int) =
        let fileName = files[i]
        if pl.debug then
            printfn $"[{name}] Reading RGB chunk slice {i}: {fileName}"
        let vector = readColorChunkTiffSlice fileName
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

#if LEGACY_IMAGE
let private readFilesWithShapeCore<'T when 'T: equality> suffix (debug: bool) (width: uint) (height: uint) : Stage<string, Image<'T>> =
    let name = "readFiles"
    if debug && DebugLevel.current() >= 2u then printfn $"[{name} cast to {typeof<'T>.Name}]"
    let mutable width = width
    let mutable height = height
    let useFastTiff =
        suffix
        |> Option.exists (canReadDirectTiffStack<'T>)

    let imageSeriesIndex fallbackIndex (fileName: string) =
        let name = Path.GetFileNameWithoutExtension(fileName)
        let m = Regex.Match(name, @"(\d+)$")
        if m.Success then
            int64 (Int32.Parse(m.Groups.[1].Value, System.Globalization.CultureInfo.InvariantCulture))
        else
            fallbackIndex

    let mapper (debug: bool) (sliceIndex: int64) (fileName: string) : Image<'T> =
        if debug then printfn "[%s] Reading image named %s as %s" name fileName (typeof<'T>.Name)
        let imageIndex = imageSeriesIndex sliceIndex fileName
        let image =
            if useFastTiff then
                ImageIO.readTiffSliceFile<'T> fileName imageIndex
            else
                let image = Image<'T>.ofFile fileName
                image.index <- int imageIndex
                image

        if width = 0u then
            width <- image.GetWidth()
            height <- image.GetHeight()
        image

    let memoryNeed = fun _ -> Image<'T>.memoryEstimate width height
    let elementTransformation _ = uint64 width * uint64 height

    let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed
    let suffixKey = suffix |> Option.map suffixCostLabel |> Option.defaultValue "stack"
    let fallbackTimeCostModel =
        imageIoCost<'T>
            "read"
            Map
            $"readFiles.{suffixKey}.{typeof<'T>.Name}"
            (fun _ -> Image<'T>.memoryEstimate width height)
            (fun _ -> 1UL)
    let timeCostModel =
        if width > 0u && height > 0u then
            match suffix with
            | Some suffix ->
                fixedImageStackOperatorTimeCost<'T>
                    "Read"
                    Map
                    suffix
                    (uint64 width * uint64 height)
                    fallbackTimeCostModel.Estimate
            | None ->
                fixedImageOperatorTimeCost<'T>
                    "Read"
                    Map
                    (uint64 width * uint64 height)
                    fallbackTimeCostModel.Estimate
        else
            fallbackTimeCostModel
    Stage.mapi name mapper memoryNeed elementTransformation
    |> withCostModel (StageCostModel.create memoryModel timeCostModel)

let readFilesWithShape<'T when 'T: equality> (debug: bool) (width: uint) (height: uint) : Stage<string, Image<'T>> =
    readFilesWithShapeCore<'T> None debug width height

let private readFilesWithShapeForSuffix<'T when 'T: equality> suffix (debug: bool) (width: uint) (height: uint) : Stage<string, Image<'T>> =
    readFilesWithShapeCore<'T> (Some suffix) debug width height

let readFiles<'T when 'T: equality> (debug: bool) : Stage<string, Image<'T>> =
    readFilesWithShape<'T> debug 0u 0u

let private readTiffVolume<'T when 'T: equality> (filename: string) (pl: Plan<unit, unit>) : Plan<unit, Image<'T>> =
    use header = Tiff.Open(filename, "r")
    if isNull header then
        invalidOp $"Could not open '{filename}' for TIFF volume reading."

    let width = uint (ImageIO.tiffFieldInt header TiffTag.IMAGEWIDTH 0)
    let height = uint (ImageIO.tiffFieldInt header TiffTag.IMAGELENGTH 0)
    if width = 0u || height = 0u then
        invalidOp $"TIFF volume '{filename}' has invalid page dimensions {width}x{height}."

    let bitsPerSample = ImageIO.tiffFieldIntDefaulted header TiffTag.BITSPERSAMPLE 1
    let samplesPerPixel = ImageIO.tiffFieldIntDefaulted header TiffTag.SAMPLESPERPIXEL 1
    let sampleFormat =
        let raw = ImageIO.tiffFieldIntDefaulted header TiffTag.SAMPLEFORMAT (int SampleFormat.UINT)
        enum<SampleFormat> raw

    ImageIO.validateTiffSamples samplesPerPixel
    ImageIO.tiffPixelLayout<'T> () |> ignore
    let bytesPerSample = ImageIO.tiffBytesPerSample bitsPerSample sampleFormat

    let depth = ImageIO.tiffDirectoryCount filename
    let pixelBytes = Image<'T>.memoryEstimate width height
    let sourcePeek =
        SourcePeek.create
            "readVolume"
            pixelBytes
            (Some (uint64 depth))
            (Map.ofList
                [ "kind", "tiff-volume-file"
                  "filename", filename
                  "width", string width
                  "height", string height
                  "depth", string depth
                  "pixelType", typeof<'T>.Name ])

    let transition = ProfileTransition.create Unit Streaming
    let memoryNeed _ = pixelBytes
    let elementTransformation _ = uint64 width * uint64 height
    let memoryModel = StageMemoryModel.fromSinglePeak Source memoryNeed
    let timeCostModel =
        imageIoCost<'T>
            "readVolume"
            Source
            $"readVolume.tiff.{typeof<'T>.Name}"
            (fun _ -> pixelBytes)
            (fun _ -> 1UL)

    let stage =
        let apply (_debug: bool) (_input: AsyncSeq<unit>) =
            asyncSeq {
                use reader = Tiff.Open(filename, "r")
                if isNull reader then
                    invalidOp $"Could not open '{filename}' for TIFF volume reading."

                for index in 0 .. int depth - 1 do
                    if _debug then
                        printfn $"[readVolume] Reading TIFF page {index} from {filename} as {typeof<'T>.Name}"

                    let pageWidth = uint (ImageIO.tiffFieldInt reader TiffTag.IMAGEWIDTH 0)
                    let pageHeight = uint (ImageIO.tiffFieldInt reader TiffTag.IMAGELENGTH 0)
                    if pageWidth <> width || pageHeight <> height then
                        invalidOp $"readVolume expected all TIFF pages to be {width}x{height}; page {index} is {pageWidth}x{pageHeight}."

                    yield ImageIO.readTiffPage<'T> reader width height bitsPerSample sampleFormat bytesPerSample index

                    if index < int depth - 1 then
                        if not (reader.ReadDirectory()) then
                            invalidOp $"TIFF volume '{filename}' ended after page {index}, but expected {depth} pages."
            }

        let pipe =
            { Name = "readVolume.tiff"
              Apply = apply
              Profile = transition.From }

        Stage.fromPipe "readVolume.tiff" transition memoryNeed elementTransformation pipe
        |> withCostModel (StageCostModel.create memoryModel timeCostModel)
        |> Some

    Plan.createWithOptimizer stage pl.memAvail pixelBytes (uint64 width * uint64 height) (uint64 depth) pl.debug pl.optimize
    |> Plan.withRuntimeOptionsFrom pl
    |> Plan.withSourcePeek sourcePeek

let private readSimpleItkVolume<'T when 'T: equality> (filename: string) (pl: Plan<unit, unit>) : Plan<unit, Image<'T>> =
    let info = ImageIO.imageFileInfo filename
    let dimension = info.Dimension
    if dimension < 2 || dimension > 3 then
        invalidArg "filename" $"readVolume expects a 2D or 3D image volume, got {dimension} dimensions in '{filename}'."

    let size = info.Size
    let width = size[0]
    let height = size[1]
    let depth = if dimension = 3 then size[2] else 1u
    let pixelBytes = Image<'T>.memoryEstimate width height
    let sourcePeek =
        SourcePeek.create
            "readVolume"
            pixelBytes
            (Some (uint64 depth))
            (Map.ofList
                [ "kind", "volume-file"
                  "filename", filename
                  "width", string width
                  "height", string height
                  "depth", string depth
                  "pixelType", typeof<'T>.Name ])

    let mapper (index: int) =
        if pl.debug then
            printfn $"[readVolume] Reading slice {index} from {filename} as {typeof<'T>.Name}"

        ImageIO.readSimpleItkSlice<'T> filename dimension width height index $"readVolume[{index}]" index

    let transition = ProfileTransition.create Unit Streaming
    let memoryNeed _ = pixelBytes
    let elementTransformation _ = uint64 width * uint64 height
    let memoryModel = StageMemoryModel.fromSinglePeak Source memoryNeed
    let timeCostModel =
        imageIoCost<'T>
            "readVolume"
            Source
            $"readVolume.{typeof<'T>.Name}"
            (fun _ -> pixelBytes)
            (fun _ -> 1UL)

    let stage =
        Stage.init "readVolume" depth mapper transition memoryNeed elementTransformation
        |> withCostModel (StageCostModel.create memoryModel timeCostModel)
        |> Some

    Plan.createWithOptimizer stage pl.memAvail pixelBytes (uint64 width * uint64 height) (uint64 depth) pl.debug pl.optimize
    |> Plan.withRuntimeOptionsFrom pl
    |> Plan.withSourcePeek sourcePeek

let readVolume<'T when 'T: equality> (filename: string) (pl: Plan<unit, unit>) : Plan<unit, Image<'T>> =
    if isTiffVolumePath filename then
        readTiffVolume<'T> filename pl
    else
        readSimpleItkVolume<'T> filename pl

let readChunkVolume<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (filename: string)
    (pl: Plan<unit, unit>)
    : Plan<unit, Chunk<'T>> =
    let toChunk =
        Stage.map
            $"readChunkVolume.toChunk.{typeof<'T>.Name}"
            (fun _ image ->
                try
                    Chunk.ofImage image
                finally
                    image.decRefCount())
            (fun n -> 2UL * imageBytes<'T> n)
            id

    readVolume<'T> filename pl >=> toChunk
#endif

let readChunkVolume<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    (filename: string)
    (pl: Plan<unit, unit>)
    : Plan<unit, Chunk<'T>> =
    if not (isTiffVolumePath filename) then
        invalidArg "filename" "readChunkVolume currently supports TIFF/BigTIFF volumes only."

    use header = Tiff.Open(filename, "r")
    if isNull header then
        invalidOp $"Could not open '{filename}' for TIFF volume reading."

    let expectedBits, expectedFormat, expectedBytesPerSample = tiffPixelLayout<'T> ()
    let width = uint (tiffFieldInt header TiffTag.IMAGEWIDTH 0)
    let height = uint (tiffFieldInt header TiffTag.IMAGELENGTH 0)
    if width = 0u || height = 0u then
        invalidOp $"TIFF volume '{filename}' has invalid page dimensions {width}x{height}."

    let bitsPerSample = tiffFieldIntDefaulted header TiffTag.BITSPERSAMPLE 1
    let samplesPerPixel = tiffFieldIntDefaulted header TiffTag.SAMPLESPERPIXEL 1
    validateTiffSamples samplesPerPixel

    let sampleFormat =
        tiffFieldIntDefaulted header TiffTag.SAMPLEFORMAT (int SampleFormat.UINT)
        |> enum<SampleFormat>

    if bitsPerSample <> expectedBits || sampleFormat <> expectedFormat then
        invalidOp $"TIFF volume '{filename}' does not match {typeof<'T>.Name}: got {bitsPerSample}-bit {sampleFormat}."

    let bytesPerSample = tiffBytesPerSample bitsPerSample sampleFormat
    if bytesPerSample <> expectedBytesPerSample then
        invalidOp $"TIFF volume '{filename}' has unexpected sample width {bytesPerSample}; expected {expectedBytesPerSample}."

    let depth = tiffDirectoryCount filename
    let rowBytes = int width * expectedBytesPerSample
    let elementBytes = uint64 rowBytes * uint64 height
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
                  "pixelType", typeof<'T>.Name ])

    let mapper (index: int) =
        use reader = Tiff.Open(filename, "r")
        if isNull reader then
            invalidOp $"Could not open '{filename}' for TIFF volume reading."
        if not (reader.SetDirectory(int16 index)) then
            invalidOp $"Could not seek to TIFF page {index} in '{filename}'."
        if pl.debug then
            printfn $"[readChunkVolume] Reading TIFF page {index} from {filename} as {typeof<'T>.Name}"

        let pageWidth = uint (tiffFieldInt reader TiffTag.IMAGEWIDTH 0)
        let pageHeight = uint (tiffFieldInt reader TiffTag.IMAGELENGTH 0)
        if pageWidth <> width || pageHeight <> height then
            invalidOp $"readChunkVolume expected all TIFF pages to be {width}x{height}; page {index} is {pageWidth}x{pageHeight}."

        let chunk = Chunk.create<'T> (uint64 width, uint64 height, 1UL)
        try
            let scanlineSize = max rowBytes (reader.ScanlineSize())
            if scanlineSize <= rowBytes then
                for row in 0 .. int height - 1 do
                    if not (reader.ReadScanline(chunk.Bytes, row * rowBytes, row, int16 0)) then
                        invalidOp $"Failed to read TIFF scanline {row} from page {index} in '{filename}'."
            else
                let scratch = System.Buffers.ArrayPool<byte>.Shared.Rent(scanlineSize)
                try
                    for row in 0 .. int height - 1 do
                        if not (reader.ReadScanline(scratch, row, int16 0)) then
                            invalidOp $"Failed to read TIFF scanline {row} from page {index} in '{filename}'."
                        Buffer.BlockCopy(scratch, 0, chunk.Bytes, row * rowBytes, rowBytes)
                finally
                    System.Buffers.ArrayPool<byte>.Shared.Return(scratch)
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
            $"readChunkVolume.tiff.{typeof<'T>.Name}"
            (fun _ -> elementBytes)
            (fun _ -> 1UL)

    let stage =
        Stage.init "readChunkVolume" depth mapper transition memoryNeed elementTransformation
        |> withCostModel (StageCostModel.create memoryModel timeCostModel)
        |> Some

    Plan.createWithOptimizer stage pl.memAvail elementBytes elementBytes (uint64 depth) pl.debug pl.optimize
    |> Plan.withRuntimeOptionsFrom pl
    |> Plan.withSourcePeek sourcePeek

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

#if LEGACY_IMAGE
let private readTiffVolumeRandom<'T when 'T: equality> (count: uint) (filename: string) (pl: Plan<unit, unit>) : Plan<unit, Image<'T>> =
    use header = Tiff.Open(filename, "r")
    if isNull header then
        invalidOp $"Could not open '{filename}' for TIFF volume reading."

    let width = uint (ImageIO.tiffFieldInt header TiffTag.IMAGEWIDTH 0)
    let height = uint (ImageIO.tiffFieldInt header TiffTag.IMAGELENGTH 0)
    if width = 0u || height = 0u then
        invalidOp $"TIFF volume '{filename}' has invalid page dimensions {width}x{height}."

    let bitsPerSample = ImageIO.tiffFieldIntDefaulted header TiffTag.BITSPERSAMPLE 1
    let samplesPerPixel = ImageIO.tiffFieldIntDefaulted header TiffTag.SAMPLESPERPIXEL 1
    let sampleFormat =
        let raw = ImageIO.tiffFieldIntDefaulted header TiffTag.SAMPLEFORMAT (int SampleFormat.UINT)
        enum<SampleFormat> raw

    ImageIO.validateTiffSamples samplesPerPixel
    ImageIO.tiffPixelLayout<'T> () |> ignore
    let bytesPerSample = ImageIO.tiffBytesPerSample bitsPerSample sampleFormat
    let sourceDepth = ImageIO.tiffDirectoryCount filename |> int
    let selected = randomIndices count sourceDepth
    let pixelBytes = Image<'T>.memoryEstimate width height
    let sourcePeek =
        SourcePeek.create
            "readVolumeRandom"
            pixelBytes
            (Some (uint64 selected.Length))
            (Map.ofList
                [ "kind", "tiff-volume-file-random"
                  "filename", filename
                  "width", string width
                  "height", string height
                  "depth", string selected.Length
                  "sourceDepth", string sourceDepth
                  "pixelType", typeof<'T>.Name ])

    let mapper (outputIndex: int) =
        let sourceIndex = selected[outputIndex]
        if pl.debug then
            printfn $"[readVolumeRandom] Reading TIFF page {sourceIndex} from {filename} as {typeof<'T>.Name}"

        use reader = Tiff.Open(filename, "r")
        if isNull reader then
            invalidOp $"Could not open '{filename}' for TIFF volume reading."
        if not (reader.SetDirectory(int16 sourceIndex)) then
            invalidOp $"Could not seek to TIFF page {sourceIndex} in '{filename}'."

        let pageWidth = uint (ImageIO.tiffFieldInt reader TiffTag.IMAGEWIDTH 0)
        let pageHeight = uint (ImageIO.tiffFieldInt reader TiffTag.IMAGELENGTH 0)
        if pageWidth <> width || pageHeight <> height then
            invalidOp $"readVolumeRandom expected all TIFF pages to be {width}x{height}; page {sourceIndex} is {pageWidth}x{pageHeight}."

        ImageIO.readTiffPage<'T> reader width height bitsPerSample sampleFormat bytesPerSample sourceIndex

    let transition = ProfileTransition.create Unit Streaming
    let memoryNeed _ = pixelBytes
    let elementTransformation _ = uint64 width * uint64 height
    let memoryModel = StageMemoryModel.fromSinglePeak Source memoryNeed
    let timeCostModel =
        imageIoCost<'T>
            "readVolumeRandom"
            Source
            $"readVolumeRandom.tiff.{typeof<'T>.Name}"
            (fun _ -> pixelBytes)
            (fun _ -> 1UL)
    let stage =
        Stage.init "readVolumeRandom" (uint selected.Length) mapper transition memoryNeed elementTransformation
        |> withCostModel (StageCostModel.create memoryModel timeCostModel)
        |> Some

    Plan.createWithOptimizer stage pl.memAvail pixelBytes (uint64 width * uint64 height) (uint64 selected.Length) pl.debug pl.optimize
    |> Plan.withRuntimeOptionsFrom pl
    |> Plan.withSourcePeek sourcePeek

let private readSimpleItkVolumeRandom<'T when 'T: equality> (count: uint) (filename: string) (pl: Plan<unit, unit>) : Plan<unit, Image<'T>> =
    let info = ImageIO.imageFileInfo filename
    let dimension = info.Dimension
    if dimension < 2 || dimension > 3 then
        invalidArg "filename" $"readVolumeRandom expects a 2D or 3D image volume, got {dimension} dimensions in '{filename}'."

    let size = info.Size
    let width = size[0]
    let height = size[1]
    let sourceDepth = if dimension = 3 then int size[2] else 1
    let selected = randomIndices count sourceDepth
    let pixelBytes = Image<'T>.memoryEstimate width height
    let sourcePeek =
        SourcePeek.create
            "readVolumeRandom"
            pixelBytes
            (Some (uint64 selected.Length))
            (Map.ofList
                [ "kind", "volume-file-random"
                  "filename", filename
                  "width", string width
                  "height", string height
                  "depth", string selected.Length
                  "sourceDepth", string sourceDepth
                  "pixelType", typeof<'T>.Name ])

    let mapper (outputIndex: int) =
        let sourceIndex = selected[outputIndex]
        if pl.debug then
            printfn $"[readVolumeRandom] Reading slice {sourceIndex} from {filename} as {typeof<'T>.Name}"

        ImageIO.readSimpleItkSlice<'T> filename dimension width height sourceIndex $"readVolumeRandom[{sourceIndex}]" sourceIndex

    let transition = ProfileTransition.create Unit Streaming
    let memoryNeed _ = pixelBytes
    let elementTransformation _ = uint64 width * uint64 height
    let memoryModel = StageMemoryModel.fromSinglePeak Source memoryNeed
    let timeCostModel =
        imageIoCost<'T>
            "readVolumeRandom"
            Source
            $"readVolumeRandom.{typeof<'T>.Name}"
            (fun _ -> pixelBytes)
            (fun _ -> 1UL)
    let stage =
        Stage.init "readVolumeRandom" (uint selected.Length) mapper transition memoryNeed elementTransformation
        |> withCostModel (StageCostModel.create memoryModel timeCostModel)
        |> Some

    Plan.createWithOptimizer stage pl.memAvail pixelBytes (uint64 width * uint64 height) (uint64 selected.Length) pl.debug pl.optimize
    |> Plan.withRuntimeOptionsFrom pl
    |> Plan.withSourcePeek sourcePeek

let readVolumeRandom<'T when 'T: equality> (count: uint) (filename: string) (pl: Plan<unit, unit>) : Plan<unit, Image<'T>> =
    if isTiffVolumePath filename then
        readTiffVolumeRandom<'T> count filename pl
    else
        readSimpleItkVolumeRandom<'T> count filename pl

let private readTiffVolumeRange<'T when 'T: equality> (first: uint) (step: int) (last: uint) (filename: string) (pl: Plan<unit, unit>) : Plan<unit, Image<'T>> =
    use header = Tiff.Open(filename, "r")
    if isNull header then
        invalidOp $"Could not open '{filename}' for TIFF volume reading."

    let width = uint (ImageIO.tiffFieldInt header TiffTag.IMAGEWIDTH 0)
    let height = uint (ImageIO.tiffFieldInt header TiffTag.IMAGELENGTH 0)
    if width = 0u || height = 0u then
        invalidOp $"TIFF volume '{filename}' has invalid page dimensions {width}x{height}."

    let bitsPerSample = ImageIO.tiffFieldIntDefaulted header TiffTag.BITSPERSAMPLE 1
    let samplesPerPixel = ImageIO.tiffFieldIntDefaulted header TiffTag.SAMPLESPERPIXEL 1
    let sampleFormat =
        let raw = ImageIO.tiffFieldIntDefaulted header TiffTag.SAMPLEFORMAT (int SampleFormat.UINT)
        enum<SampleFormat> raw

    ImageIO.validateTiffSamples samplesPerPixel
    ImageIO.tiffPixelLayout<'T> () |> ignore
    let bytesPerSample = ImageIO.tiffBytesPerSample bitsPerSample sampleFormat
    let sourceDepth = ImageIO.tiffDirectoryCount filename |> int
    let selected = rangeIndices first step last sourceDepth
    let pixelBytes = Image<'T>.memoryEstimate width height
    let sourcePeek =
        SourcePeek.create
            "readVolumeRange"
            pixelBytes
            (Some (uint64 selected.Length))
            (Map.ofList
                [ "kind", "tiff-volume-file-range"
                  "filename", filename
                  "width", string width
                  "height", string height
                  "depth", string selected.Length
                  "sourceDepth", string sourceDepth
                  "pixelType", typeof<'T>.Name
                  "first", string first
                  "step", string step
                  "last", string last ])

    let mapper (outputIndex: int) =
        let sourceIndex = selected[outputIndex]
        if pl.debug then
            printfn $"[readVolumeRange] Reading TIFF page {sourceIndex} from {filename} as {typeof<'T>.Name}"

        use reader = Tiff.Open(filename, "r")
        if isNull reader then
            invalidOp $"Could not open '{filename}' for TIFF volume reading."
        if not (reader.SetDirectory(int16 sourceIndex)) then
            invalidOp $"Could not seek to TIFF page {sourceIndex} in '{filename}'."

        let pageWidth = uint (ImageIO.tiffFieldInt reader TiffTag.IMAGEWIDTH 0)
        let pageHeight = uint (ImageIO.tiffFieldInt reader TiffTag.IMAGELENGTH 0)
        if pageWidth <> width || pageHeight <> height then
            invalidOp $"readVolumeRange expected all TIFF pages to be {width}x{height}; page {sourceIndex} is {pageWidth}x{pageHeight}."

        ImageIO.readTiffPage<'T> reader width height bitsPerSample sampleFormat bytesPerSample sourceIndex

    let transition = ProfileTransition.create Unit Streaming
    let memoryNeed _ = pixelBytes
    let elementTransformation _ = uint64 width * uint64 height
    let memoryModel = StageMemoryModel.fromSinglePeak Source memoryNeed
    let timeCostModel =
        imageIoCost<'T>
            "readVolumeRange"
            Source
            $"readVolumeRange.tiff.{typeof<'T>.Name}"
            (fun _ -> pixelBytes)
            (fun _ -> 1UL)
    let stage =
        Stage.init "readVolumeRange" (uint selected.Length) mapper transition memoryNeed elementTransformation
        |> withCostModel (StageCostModel.create memoryModel timeCostModel)
        |> Some

    Plan.createWithOptimizer stage pl.memAvail pixelBytes (uint64 width * uint64 height) (uint64 selected.Length) pl.debug pl.optimize
    |> Plan.withRuntimeOptionsFrom pl
    |> Plan.withSourcePeek sourcePeek

let private readSimpleItkVolumeRange<'T when 'T: equality> (first: uint) (step: int) (last: uint) (filename: string) (pl: Plan<unit, unit>) : Plan<unit, Image<'T>> =
    let info = ImageIO.imageFileInfo filename
    let dimension = info.Dimension
    if dimension < 2 || dimension > 3 then
        invalidArg "filename" $"readVolumeRange expects a 2D or 3D image volume, got {dimension} dimensions in '{filename}'."

    let size = info.Size
    let width = size[0]
    let height = size[1]
    let sourceDepth = if dimension = 3 then int size[2] else 1
    let selected = rangeIndices first step last sourceDepth
    let pixelBytes = Image<'T>.memoryEstimate width height
    let sourcePeek =
        SourcePeek.create
            "readVolumeRange"
            pixelBytes
            (Some (uint64 selected.Length))
            (Map.ofList
                [ "kind", "volume-file-range"
                  "filename", filename
                  "width", string width
                  "height", string height
                  "depth", string selected.Length
                  "sourceDepth", string sourceDepth
                  "pixelType", typeof<'T>.Name
                  "first", string first
                  "step", string step
                  "last", string last ])

    let mapper (outputIndex: int) =
        let sourceIndex = selected[outputIndex]
        if pl.debug then
            printfn $"[readVolumeRange] Reading slice {sourceIndex} from {filename} as {typeof<'T>.Name}"

        ImageIO.readSimpleItkSlice<'T> filename dimension width height sourceIndex $"readVolumeRange[{sourceIndex}]" sourceIndex

    let transition = ProfileTransition.create Unit Streaming
    let memoryNeed _ = pixelBytes
    let elementTransformation _ = uint64 width * uint64 height
    let memoryModel = StageMemoryModel.fromSinglePeak Source memoryNeed
    let timeCostModel =
        imageIoCost<'T>
            "readVolumeRange"
            Source
            $"readVolumeRange.{typeof<'T>.Name}"
            (fun _ -> pixelBytes)
            (fun _ -> 1UL)
    let stage =
        Stage.init "readVolumeRange" (uint selected.Length) mapper transition memoryNeed elementTransformation
        |> withCostModel (StageCostModel.create memoryModel timeCostModel)
        |> Some

    Plan.createWithOptimizer stage pl.memAvail pixelBytes (uint64 width * uint64 height) (uint64 selected.Length) pl.debug pl.optimize
    |> Plan.withRuntimeOptionsFrom pl
    |> Plan.withSourcePeek sourcePeek

let readVolumeRange<'T when 'T: equality> (first: uint) (step: int) (last: uint) (filename: string) (pl: Plan<unit, unit>) : Plan<unit, Image<'T>> =
    if isTiffVolumePath filename then
        readTiffVolumeRange<'T> first step last filename pl
    else
        readSimpleItkVolumeRange<'T> first step last filename pl

let readFilePairs<'T when 'T: equality> (debug: bool) : Stage<string*string, Image<'T>*Image<'T>> =
    let name = "readFilePairs"
    if debug && DebugLevel.current() >= 2u then printfn $"[{name} cast to {typeof<'T>.Name}]"
    let mutable width = 0u // We need to read the first image in order to find its size
    let mutable height = 0u

    let mapper (debug: bool) (fileName1: string, fileName2:string) : Image<'T>*Image<'T> = 
        if debug then printfn "[%s] Reading image named %s as %s" name fileName1 (typeof<'T>.Name)
        let image1 = Image<'T>.ofFile fileName1
        if debug then printfn "[%s] Reading image named %s as %s" name fileName2 (typeof<'T>.Name)
        let image2 = Image<'T>.ofFile fileName2
        if width = 0u then
            width <- image1.GetWidth()
            height <- image1.GetHeight()
        image1, image2

    let memoryNeed = fun _ -> 2UL*Image<'T>.memoryEstimate width height
    let elementTransformation = id

    let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed
    let timeCostModel =
        imageIoCost<'T>
            "read"
            Map
            $"readFilePairs.{typeof<'T>.Name}"
            (fun _ -> 2UL * Image<'T>.memoryEstimate width height)
            (fun _ -> 2UL)
    Stage.map name mapper memoryNeed elementTransformation
    |> withCostModel (StageCostModel.create memoryModel timeCostModel)

let readFiltered<'T when 'T: equality> (inputDir: string) (suffix: string) (filter: string[]->string[]) (pl: Plan<unit, unit>) : Plan<unit, Image<'T>> =
    let info = getStackInfo inputDir suffix
    let width = uint info.size[0]
    let height = uint info.size[1]
    let elementBytes = Image<'T>.memoryEstimate width height
    let sourcePeek =
        SourcePeek.create
            "read"
            elementBytes
            (Some (uint64 info.size[2]))
            (Map.ofList
                [ "kind", "image-stack"
                  "inputDir", inputDir
                  "suffix", suffix
                  "width", string width
                  "height", string height
                  "depth", string info.size[2]
                  "pixelType", typeof<'T>.Name ])

    pl
    |> getFilenames inputDir suffix filter
    >=> readFilesWithShapeForSuffix<'T> suffix pl.debug width height
    |> Plan.withSourcePeek sourcePeek

let read<'T when 'T: equality> (inputDir: string) (suffix: string) (pl: Plan<unit, unit>) : Plan<unit, Image<'T>> =
    readFiltered<'T> inputDir suffix Array.sort pl

let readRandom<'T when 'T: equality> (count: uint) (inputDir: string) (suffix: string) (pl: Plan<unit, unit>) : Plan<unit, Image<'T>> =
    readFiltered<'T> inputDir suffix (Array.randomChoices (int count)) pl
#endif

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
    readSelectedChunkSlices<'T>
        "readChunkSlicesRange"
        inputDir
        suffix
        (rangeFilter first step last)
        [ "first", string first; "step", string step; "last", string last ]
        pl

#if LEGACY_IMAGE
let readRange<'T when 'T: equality> (first: uint) (step: int) (last: uint) (inputDir: string) (suffix: string) (pl: Plan<unit, unit>) : Plan<unit, Image<'T>> =
    let info = getStackInfo inputDir suffix
    let width = uint info.size[0]
    let height = uint info.size[1]
    let selectedDepth =
        getStackFiles inputDir suffix
        |> rangeFilter first step last
        |> Array.length
        |> uint64
    let elementBytes = Image<'T>.memoryEstimate width height
    let sourcePeek =
        SourcePeek.create
            "readRange"
            elementBytes
            (Some selectedDepth)
            (Map.ofList
                [ "kind", "image-stack"
                  "inputDir", inputDir
                  "suffix", suffix
                  "width", string width
                  "height", string height
                  "depth", string selectedDepth
                  "sourceDepth", string info.size[2]
                  "pixelType", typeof<'T>.Name
                  "first", string first
                  "step", string step
                  "last", string last ])

    pl
    |> getFilenames inputDir suffix (rangeFilter first step last)
    >=> readFilesWithShapeForSuffix<'T> suffix pl.debug width height
    |> Plan.withSourcePeek sourcePeek
#endif

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

#if LEGACY_IMAGE
let _readChunk<'T when 'T: equality>  (inputDir: string) (suffix: string) i j k = 
    let filename = getChunkFilename inputDir suffix i j k
    if typeof<'T> = typeof<Image.ComplexFloat32> then
        Image<Image.ComplexFloat32>.ofFileComplexFloat32 filename
        |> box
        |> unbox<Image<'T>>
    elif typeof<'T> = typeof<System.Numerics.Complex> then
        Image<System.Numerics.Complex>.ofFileComplex filename
        |> box
        |> unbox<Image<'T>>
    else
        Image<'T>.ofFile filename

let _readSlabStacked<'T when 'T: equality>  (inputDir: string) (suffix: string) (chunkInfo: ChunkInfo) (udir: uint) (idx: int) =
    let dir = int udir
    let chunkWidth = int chunkInfo.topLeftInfo.size[0]
    let chunkHeight = int chunkInfo.topLeftInfo.size[1]
    let chunkDepth = int chunkInfo.topLeftInfo.size[2]

    let lastSz = 
        let lastIdx =  chunkInfo.size[dir]/chunkInfo.topLeftInfo.size[dir] |> int
        let slicesLeft = chunkInfo.size[dir] % chunkInfo.topLeftInfo.size[dir]
        if idx = lastIdx then slicesLeft else chunkInfo.topLeftInfo.size[dir]

    let sz, nChunks = 
        if dir = 0 then
            [lastSz; chunkInfo.size[1]; chunkInfo.size[2]], [1; chunkInfo.chunks[1]; chunkInfo.chunks[2]]
        elif dir = 1 then
            [chunkInfo.size[0]; lastSz; chunkInfo.size[2]], [chunkInfo.chunks[0]; 1; chunkInfo.chunks[2]]
        else
            [chunkInfo.size[0]; chunkInfo.size[1]; lastSz], [chunkInfo.chunks[0]; chunkInfo.chunks[1]; 1]

    let numberOfComponents =
        if typeof<'T> = typeof<System.Numerics.Complex> || typeof<'T> = typeof<Image.ComplexFloat32> then 1u
        else chunkInfo.topLeftInfo.numberOfComponents

    let slab = Image<'T>(sz |> List.map uint, numberOfComponents)
    for i in [0 .. nChunks[0]-1] do
        for j in [0 .. nChunks[1]-1] do
            for k in [0 .. nChunks[2]-1] do
                let img = 
                    if dir = 0 then   _readChunk<'T> inputDir suffix idx j k
                    elif dir = 1 then _readChunk<'T> inputDir suffix i idx k
                    else              _readChunk<'T> inputDir suffix i j idx
                let start0 = i*chunkWidth|>Some
                let stop0 = i*chunkWidth+(img.GetWidth()|>int)-1|>Some
                let start1 = j*chunkHeight|>Some
                let stop1 = j*chunkHeight+(img.GetHeight()|>int)-1|>Some
                let start2 = k*chunkDepth|>Some
                let stop2 = k*chunkDepth+(img.GetDepth()|>int)-1|>Some
                slab.SetSlice (start0, stop0, start1, stop1, start2, stop2) img |> ignore
                img.decRefCount()
    slab

let readSlabStacked<'T when 'T: equality> (inputDir: string) (suffix: string) (pl: Plan<unit, unit>) : Plan<unit, Image<'T>> =
    let name = "readSlabStacked"
    let chunkInfo = getChunkInfo inputDir suffix
    let depth = uint64 chunkInfo.chunks[2] // read each chunk_*_*_k layer as one full x-y slab
    let elementBytes =
        [ chunkInfo.size[0]; chunkInfo.size[1]; chunkInfo.topLeftInfo.size[2] ]
        |> List.fold (*) 1UL
        |> fun voxels -> voxels * (typeof<'T> |> Image.getBytesPerComponent |> uint64)
    let sourcePeek =
        SourcePeek.create
            name
            elementBytes
            (Some depth)
            (Map.ofList
                [ "kind", "image-slabs"
                  "inputDir", inputDir
                  "suffix", suffix
                  "chunkWidth", string chunkInfo.topLeftInfo.size[0]
                  "chunkHeight", string chunkInfo.topLeftInfo.size[1]
                  "chunkDepth", string chunkInfo.topLeftInfo.size[2]
                  "slabWidth", string chunkInfo.size[0]
                  "slabHeight", string chunkInfo.size[1]
                  "chunks", chunkInfo.chunks |> List.map string |> String.concat "x"
                  "pixelType", typeof<'T>.Name ])

    let mapper (k: int) : Image<'T> =
        if pl.debug then
            printfn $"[readSlabStacked] Reading slab {k} from {inputDir}/*{suffix} as {typeof<'T>.Name}"

        _readSlabStacked<'T> inputDir suffix chunkInfo 2u k

    let transition = ProfileTransition.create Unit Streaming
    let memPeak = 256UL // surrugate string length
    let memoryNeed = fun _ -> memPeak
    let elementTransformation = id
    let memoryModel = StageMemoryModel.fromSinglePeak Source memoryNeed
    let timeCostModel =
        StageTimeCostModel.ioRead Source (Some $"readSlabStacked.{typeof<'T>.Name}") (fun _ -> elementBytes) (fun _ -> 1UL)
    let stage =
        Stage.init $"{name}" (uint depth) mapper transition memoryNeed elementTransformation
        |> withCostModel (StageCostModel.create memoryModel timeCostModel)
        |> Some

    let memPerElem = 256UL // surrugate string length
    let length = depth
    Plan.createWithOptimizer stage pl.memAvail memPeak memPerElem depth pl.debug pl.optimize
    |> Plan.withRuntimeOptionsFrom pl
    |> Plan.withSourcePeek sourcePeek

let readSlabAsWindows<'T when 'T: equality> (inputDir: string) (suffix: string) (pl: Plan<unit, unit>) : Plan<unit, Image<'T> list> =
    pl
    |> readSlabStacked<'T> inputDir suffix
    >=> (slabToWindow<'T> --> windowItems ())

let readSlab<'T when 'T: equality> (inputDir: string) (suffix: string) (pl: Plan<unit, unit>) : Plan<unit, Image<'T>> =
    pl |> readSlabAsWindows<'T> inputDir suffix >=> flattenList ()

let private readZarrSlabStackedWithDepth<'T when 'T: equality>
    (path: string)
    (thickDepth: uint)
    (multiscaleIndex: int)
    (datasetIndex: int)
    (timepoint: int)
    (channel: int)
    (maxParallelChunks: int)
    (pl: Plan<unit, unit>)
    : Plan<unit, Image<'T>> =

    suppressZarrNetDebugLogging ()

    let name = "readZarrSlabStacked"
    ImageIO.validatePixelType<'T> ()
    let level = openZarrResolutionLevel path multiscaleIndex datasetIndex
    let sizeT, sizeC, sizeZ, sizeY, sizeX = zarrShapeTCZYX level.Shape

    if timepoint < 0 || timepoint >= sizeT then
        invalidArg "timepoint" $"Timepoint {timepoint} is outside the Zarr time range 0..{sizeT - 1}."
    if channel < 0 || channel >= sizeC then
        invalidArg "channel" $"Channel {channel} is outside the Zarr channel range 0..{sizeC - 1}."
    if not (isSupportedZarrDataType level.DataType) then
        failwith $"ZarrNET image IO currently supports UInt8, UInt16, Float32, Float64, Complex64, and Complex128 datasets, but dataset type was {level.DataType}."

    let slabDepth = max 1u slabDepth |> int
    let depth = (sizeZ + slabDepth - 1) / slabDepth |> uint64
    let elementBytes =
        uint64 sizeX * uint64 sizeY * uint64 slabDepth * (typeof<'T> |> Image.getBytesPerComponent |> uint64)
    let parallelChunks = nullableParallelChunks maxParallelChunks
    let sourcePeek =
        SourcePeek.create
            name
            elementBytes
            (Some depth)
            (Map.ofList
                [ "kind", "zarr-slabs"
                  "path", path
                  "slabDepth", string slabDepth
                  "width", string sizeX
                  "height", string sizeY
                  "depth", string sizeZ
                  "pixelType", typeof<'T>.Name
                  "multiscaleIndex", string multiscaleIndex
                  "datasetIndex", string datasetIndex
                  "timepoint", string timepoint
                  "channel", string channel ])

    let mapper (idx: int) : Image<'T> =
        let zStart = idx * slabDepth
        let zStop = min sizeZ (zStart + slabDepth)
        let zCount = zStop - zStart
        if pl.debug then
            printfn $"[readZarrSlabStacked] Reading z {zStart}..{zStop - 1} from {path} as {typeof<'T>.Name}"

        let region =
            PixelRegion(
                [| int64 timepoint; int64 channel; int64 zStart; 0L; 0L |],
                [| int64 (timepoint + 1); int64 (channel + 1); int64 zStop; int64 sizeY; int64 sizeX |])
        let result =
            level.ReadPixelRegionAsync(region, parallelChunks, CancellationToken.None)
            |> runTask
        deleteZarrNetDebugLogs ()
        zarrSlabImageAs<'T> level.DataType sizeX sizeY zCount result.Data $"readZarrSlabStacked.{idx}"

    let transition = ProfileTransition.create Unit Streaming
    let memPeak = 256UL
    let memoryNeed = fun _ -> memPeak
    let elementTransformation = id
    let memoryModel = StageMemoryModel.fromSinglePeak Source memoryNeed
    let timeCostModel =
        StageTimeCostModel.ioRead Source (Some $"readZarrSlabStacked.{typeof<'T>.Name}") (fun _ -> elementBytes) (fun _ -> 1UL)
    let stage =
        Stage.init name (uint depth) mapper transition memoryNeed elementTransformation
        |> withCostModel (StageCostModel.create memoryModel timeCostModel)
        |> Some

    Plan.createWithOptimizer stage pl.memAvail memPeak memPeak depth pl.debug pl.optimize
    |> Plan.withRuntimeOptionsFrom pl
    |> Plan.withSourcePeek sourcePeek

let readZarrSlabStacked<'T when 'T: equality>
    (path: string)
    (multiscaleIndex: int)
    (datasetIndex: int)
    (timepoint: int)
    (channel: int)
    (maxParallelChunks: int)
    (pl: Plan<unit, unit>)
    : Plan<unit, Image<'T>> =

    let slabDepth = getZarrInfo path multiscaleIndex datasetIndex |> zarrChunkDepthFromInfo
    pl
    |> readZarrSlabStackedWithDepth<'T> path slabDepth multiscaleIndex datasetIndex timepoint channel maxParallelChunks

let private readZarrSlabWithDepth<'T when 'T: equality>
    (path: string)
    (slabDepth: uint)
    (multiscaleIndex: int)
    (datasetIndex: int)
    (timepoint: int)
    (channel: int)
    (maxParallelChunks: int)
    (pl: Plan<unit, unit>)
    : Plan<unit, Image<'T>> =

    pl
    |> readZarrSlabStackedWithDepth<'T> path slabDepth multiscaleIndex datasetIndex timepoint channel maxParallelChunks
    >=> (slabToWindow<'T> --> windowItems ())
    >=> flattenList ()

let readZarrSlab<'T when 'T: equality>
    (path: string)
    (multiscaleIndex: int)
    (datasetIndex: int)
    (timepoint: int)
    (channel: int)
    (maxParallelChunks: int)
    (pl: Plan<unit, unit>)
    : Plan<unit, Image<'T>> =

    pl
    |> readZarrSlabStacked<'T> path multiscaleIndex datasetIndex timepoint channel maxParallelChunks
    >=> (slabToWindow<'T> --> windowItems ())
    >=> flattenList ()

let readZarrRandom<'T when 'T: equality>
    (count: uint)
    (path: string)
    (multiscaleIndex: int)
    (datasetIndex: int)
    (timepoint: int)
    (channel: int)
    (maxParallelChunks: int)
    (pl: Plan<unit, unit>)
    : Plan<unit, Image<'T>> =

    suppressZarrNetDebugLogging ()

    let name = "readZarrRandom"
    ImageIO.validatePixelType<'T> ()
    let level = openZarrResolutionLevel path multiscaleIndex datasetIndex
    let sizeT, sizeC, sizeZ, sizeY, sizeX = zarrShapeTCZYX level.Shape

    if timepoint < 0 || timepoint >= sizeT then
        invalidArg "timepoint" $"Timepoint {timepoint} is outside the Zarr time range 0..{sizeT - 1}."
    if channel < 0 || channel >= sizeC then
        invalidArg "channel" $"Channel {channel} is outside the Zarr channel range 0..{sizeC - 1}."
    if not (isSupportedZarrDataType level.DataType) then
        failwith $"ZarrNET image IO currently supports UInt8, UInt16, Float32, Float64, Complex64, and Complex128 datasets, but dataset type was {level.DataType}."

    let selected = randomIndices count sizeZ
    let elementBytes =
        uint64 sizeX * uint64 sizeY * (typeof<'T> |> Image.getBytesPerComponent |> uint64)
    let parallelChunks = nullableParallelChunks maxParallelChunks
    let sourcePeek =
        SourcePeek.create
            name
            elementBytes
            (Some (uint64 selected.Length))
            (Map.ofList
                [ "kind", "zarr-random"
                  "path", path
                  "width", string sizeX
                  "height", string sizeY
                  "depth", string selected.Length
                  "sourceDepth", string sizeZ
                  "pixelType", typeof<'T>.Name
                  "multiscaleIndex", string multiscaleIndex
                  "datasetIndex", string datasetIndex
                  "timepoint", string timepoint
                  "channel", string channel ])

    let mapper (outputIndex: int) : Image<'T> =
        let zIndex = selected[outputIndex]
        if pl.debug then
            printfn $"[readZarrRandom] Reading z {zIndex} from {path} as {typeof<'T>.Name}"

        let region =
            PixelRegion(
                [| int64 timepoint; int64 channel; int64 zIndex; 0L; 0L |],
                [| int64 (timepoint + 1); int64 (channel + 1); int64 (zIndex + 1); int64 sizeY; int64 sizeX |])
        let result =
            level.ReadPixelRegionAsync(region, parallelChunks, CancellationToken.None)
            |> runTask
        deleteZarrNetDebugLogs ()
        let slab = zarrSlabImageAs<'T> level.DataType sizeX sizeY 1 result.Data $"readZarrRandom.{outputIndex}"
        try
            ImageFunctions.extractSlice 2u 0 slab
        finally
            slab.decRefCount()

    let transition = ProfileTransition.create Unit Streaming
    let memPeak = 256UL
    let memoryNeed = fun _ -> memPeak
    let elementTransformation = id
    let memoryModel = StageMemoryModel.fromSinglePeak Source memoryNeed
    let timeCostModel =
        StageTimeCostModel.ioRead Source (Some $"readZarrRandom.{typeof<'T>.Name}") (fun _ -> elementBytes) (fun _ -> 1UL)
    let stage =
        Stage.init name (uint selected.Length) mapper transition memoryNeed elementTransformation
        |> withCostModel (StageCostModel.create memoryModel timeCostModel)
        |> Some

    Plan.createWithOptimizer stage pl.memAvail memPeak memPeak (uint64 selected.Length) pl.debug pl.optimize
    |> Plan.withRuntimeOptionsFrom pl
    |> Plan.withSourcePeek sourcePeek

let readZarrRange<'T when 'T: equality>
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
    : Plan<unit, Image<'T>> =

    suppressZarrNetDebugLogging ()

    let name = "readZarrRange"
    ImageIO.validatePixelType<'T> ()
    let level = openZarrResolutionLevel path multiscaleIndex datasetIndex
    let sizeT, sizeC, sizeZ, sizeY, sizeX = zarrShapeTCZYX level.Shape

    if timepoint < 0 || timepoint >= sizeT then
        invalidArg "timepoint" $"Timepoint {timepoint} is outside the Zarr time range 0..{sizeT - 1}."
    if channel < 0 || channel >= sizeC then
        invalidArg "channel" $"Channel {channel} is outside the Zarr channel range 0..{sizeC - 1}."
    if not (isSupportedZarrDataType level.DataType) then
        failwith $"ZarrNET image IO currently supports UInt8, UInt16, Float32, Float64, Complex64, and Complex128 datasets, but dataset type was {level.DataType}."

    let selected = rangeIndices first step last sizeZ
    let elementBytes =
        uint64 sizeX * uint64 sizeY * (typeof<'T> |> Image.getBytesPerComponent |> uint64)
    let parallelChunks = nullableParallelChunks maxParallelChunks
    let sourcePeek =
        SourcePeek.create
            name
            elementBytes
            (Some (uint64 selected.Length))
            (Map.ofList
                [ "kind", "zarr-range"
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

    let mapper (outputIndex: int) : Image<'T> =
        let zIndex = selected[outputIndex]
        if pl.debug then
            printfn $"[readZarrRange] Reading z {zIndex} from {path} as {typeof<'T>.Name}"

        let region =
            PixelRegion(
                [| int64 timepoint; int64 channel; int64 zIndex; 0L; 0L |],
                [| int64 (timepoint + 1); int64 (channel + 1); int64 (zIndex + 1); int64 sizeY; int64 sizeX |])
        let result =
            level.ReadPixelRegionAsync(region, parallelChunks, CancellationToken.None)
            |> runTask
        deleteZarrNetDebugLogs ()
        let slab = zarrSlabImageAs<'T> level.DataType sizeX sizeY 1 result.Data $"readZarrRange.{outputIndex}"
        try
            ImageFunctions.extractSlice 2u 0 slab
        finally
            slab.decRefCount()

    let transition = ProfileTransition.create Unit Streaming
    let memPeak = 256UL
    let memoryNeed = fun _ -> memPeak
    let elementTransformation = id
    let memoryModel = StageMemoryModel.fromSinglePeak Source memoryNeed
    let timeCostModel =
        StageTimeCostModel.ioRead Source (Some $"readZarrRange.{typeof<'T>.Name}") (fun _ -> elementBytes) (fun _ -> 1UL)
    let stage =
        Stage.init name (uint selected.Length) mapper transition memoryNeed elementTransformation
        |> withCostModel (StageCostModel.create memoryModel timeCostModel)
        |> Some

    Plan.createWithOptimizer stage pl.memAvail memPeak memPeak (uint64 selected.Length) pl.debug pl.optimize
    |> Plan.withRuntimeOptionsFrom pl
    |> Plan.withSourcePeek sourcePeek

let private readNexusSlabStackedWithDepth<'T when 'T: equality>
    (path: string)
    (datasetPath: string)
    (slabDepth: uint)
    (frameAxis: int)
    (yAxis: int)
    (xAxis: int)
    (pl: Plan<unit, unit>)
    : Plan<unit, Image<'T>> =

    let name = "readNexusSlabStacked"
    hdfDataType<'T> () |> ignore
    let file, dataset = hdfDataset path datasetPath
    let rank = int dataset.Space.Rank
    validateHdfAxes rank frameAxis yAxis xAxis

    if rank <> 3 then
        failwith $"readNexusSlabStacked currently expects a rank-3 detector stack dataset, but {datasetPath} has rank {rank}."

    let dimensions = dataset.Space.Dimensions
    let sizeZ = int dimensions[frameAxis]
    let sizeY = int dimensions[yAxis]
    let sizeX = int dimensions[xAxis]
    let slabDepth = max 1u slabDepth |> int
    let depth = (sizeZ + slabDepth - 1) / slabDepth |> uint64
    let elementBytes =
        uint64 sizeX * uint64 sizeY * uint64 slabDepth * (typeof<'T> |> Image.getBytesPerComponent |> uint64)
    let sourcePeek =
        SourcePeek.create
            name
            elementBytes
            (Some depth)
            (Map.ofList
                [ "kind", "nexus-slabs"
                  "path", path
                  "datasetPath", datasetPath
                  "slabDepth", string slabDepth
                  "width", string sizeX
                  "height", string sizeY
                  "depth", string sizeZ
                  "pixelType", typeof<'T>.Name
                  "frameAxis", string frameAxis
                  "yAxis", string yAxis
                  "xAxis", string xAxis ])

    let mapper (idx: int) : Image<'T> =
        let zStart = idx * slabDepth
        let zStop = min sizeZ (zStart + slabDepth)
        let zCount = zStop - zStart
        if pl.debug then
            printfn $"[readNexusSlabStacked] Reading z {zStart}..{zStop - 1} from {path}:{datasetPath} as {typeof<'T>.Name}"

        let starts = Array.zeroCreate<uint64> rank
        let blocks = Array.create rank 1UL
        starts[frameAxis] <- uint64 zStart
        blocks[frameAxis] <- uint64 zCount
        blocks[yAxis] <- uint64 sizeY
        blocks[xAxis] <- uint64 sizeX

        hdfSlabImageAs<'T>
            dataset
            rank
            starts
            blocks
            sizeX
            sizeY
            zCount
            frameAxis
            yAxis
            xAxis
            $"readNexusSlabStacked.{idx}"

    let transition = ProfileTransition.create Unit Streaming
    let memPeak = 256UL
    let memoryNeed = fun _ -> memPeak
    let elementTransformation = id
    let memoryModel = StageMemoryModel.fromSinglePeak Source memoryNeed
    let timeCostModel =
        StageTimeCostModel.ioRead Source (Some $"readNexusSlabStacked.{typeof<'T>.Name}") (fun _ -> elementBytes) (fun _ -> 1UL)
    let stage =
        Stage.init name (uint depth) mapper transition memoryNeed elementTransformation
        |> withCostModel (StageCostModel.create memoryModel timeCostModel)
        |> Some

    Plan.createWithOptimizer stage pl.memAvail memPeak memPeak depth pl.debug pl.optimize
    |> Plan.withRuntimeOptionsFrom pl
    |> Plan.withSourcePeek sourcePeek

let readNexusSlabStacked<'T when 'T: equality>
    (path: string)
    (datasetPath: string)
    (frameAxis: int)
    (yAxis: int)
    (xAxis: int)
    (pl: Plan<unit, unit>)
    : Plan<unit, Image<'T>> =

    let slabDepth =
        getNexusInfo path datasetPath frameAxis yAxis xAxis
        |> fun info -> nexusFrameChunkDepth info frameAxis

    pl
    |> readNexusSlabStackedWithDepth<'T> path datasetPath slabDepth frameAxis yAxis xAxis

let private readNexusSlabWithDepth<'T when 'T: equality>
    (path: string)
    (datasetPath: string)
    (slabDepth: uint)
    (frameAxis: int)
    (yAxis: int)
    (xAxis: int)
    (pl: Plan<unit, unit>)
    : Plan<unit, Image<'T>> =

    pl
    |> readNexusSlabStackedWithDepth<'T> path datasetPath slabDepth frameAxis yAxis xAxis
    >=> (slabToWindow<'T> --> windowItems ())
    >=> flattenList ()

let readNexusSlab<'T when 'T: equality>
    (path: string)
    (datasetPath: string)
    (frameAxis: int)
    (yAxis: int)
    (xAxis: int)
    (pl: Plan<unit, unit>)
    : Plan<unit, Image<'T>> =

    pl
    |> readNexusSlabStacked<'T> path datasetPath frameAxis yAxis xAxis
    >=> (slabToWindow<'T> --> windowItems ())
    >=> flattenList ()

let readNexusRandom<'T when 'T: equality>
    (count: uint)
    (path: string)
    (datasetPath: string)
    (frameAxis: int)
    (yAxis: int)
    (xAxis: int)
    (pl: Plan<unit, unit>)
    : Plan<unit, Image<'T>> =

    let name = "readNexusRandom"
    hdfDataType<'T> () |> ignore
    let file, dataset = hdfDataset path datasetPath
    let rank = int dataset.Space.Rank
    validateHdfAxes rank frameAxis yAxis xAxis

    if rank <> 3 then
        failwith $"readNexusRandom currently expects a rank-3 detector stack dataset, but {datasetPath} has rank {rank}."

    let dimensions = dataset.Space.Dimensions
    let sizeZ = int dimensions[frameAxis]
    let sizeY = int dimensions[yAxis]
    let sizeX = int dimensions[xAxis]
    let selected = randomIndices count sizeZ
    let elementBytes =
        uint64 sizeX * uint64 sizeY * (typeof<'T> |> Image.getBytesPerComponent |> uint64)
    let sourcePeek =
        SourcePeek.create
            name
            elementBytes
            (Some (uint64 selected.Length))
            (Map.ofList
                [ "kind", "nexus-random"
                  "path", path
                  "datasetPath", datasetPath
                  "width", string sizeX
                  "height", string sizeY
                  "depth", string selected.Length
                  "sourceDepth", string sizeZ
                  "pixelType", typeof<'T>.Name
                  "frameAxis", string frameAxis
                  "yAxis", string yAxis
                  "xAxis", string xAxis ])

    let mapper (outputIndex: int) : Image<'T> =
        let zIndex = selected[outputIndex]
        if pl.debug then
            printfn $"[readNexusRandom] Reading z {zIndex} from {path}:{datasetPath} as {typeof<'T>.Name}"

        let starts = Array.zeroCreate<uint64> rank
        let blocks = Array.create rank 1UL
        starts[frameAxis] <- uint64 zIndex
        blocks[frameAxis] <- 1UL
        blocks[yAxis] <- uint64 sizeY
        blocks[xAxis] <- uint64 sizeX

        let slab =
            hdfSlabImageAs<'T>
                dataset
                rank
                starts
                blocks
                sizeX
                sizeY
                1
                frameAxis
                yAxis
                xAxis
                $"readNexusRandom.{outputIndex}"
        try
            ImageFunctions.extractSlice 2u 0 slab
        finally
            slab.decRefCount()

    let transition = ProfileTransition.create Unit Streaming
    let memPeak = 256UL
    let memoryNeed = fun _ -> memPeak
    let elementTransformation = id
    let memoryModel = StageMemoryModel.fromSinglePeak Source memoryNeed
    let timeCostModel =
        StageTimeCostModel.ioRead Source (Some $"readNexusRandom.{typeof<'T>.Name}") (fun _ -> elementBytes) (fun _ -> 1UL)
    let stage =
        Stage.init name (uint selected.Length) mapper transition memoryNeed elementTransformation
        |> withCostModel (StageCostModel.create memoryModel timeCostModel)
        |> Some

    Plan.createWithOptimizer stage pl.memAvail memPeak memPeak (uint64 selected.Length) pl.debug pl.optimize
    |> Plan.withRuntimeOptionsFrom pl
    |> Plan.withSourcePeek sourcePeek

let readNexusRange<'T when 'T: equality>
    (first: uint)
    (step: int)
    (last: uint)
    (path: string)
    (datasetPath: string)
    (frameAxis: int)
    (yAxis: int)
    (xAxis: int)
    (pl: Plan<unit, unit>)
    : Plan<unit, Image<'T>> =

    let name = "readNexusRange"
    hdfDataType<'T> () |> ignore
    let file, dataset = hdfDataset path datasetPath
    let rank = int dataset.Space.Rank
    validateHdfAxes rank frameAxis yAxis xAxis

    if rank <> 3 then
        failwith $"readNexusRange currently expects a rank-3 detector stack dataset, but {datasetPath} has rank {rank}."

    let dimensions = dataset.Space.Dimensions
    let sizeZ = int dimensions[frameAxis]
    let sizeY = int dimensions[yAxis]
    let sizeX = int dimensions[xAxis]
    let selected = rangeIndices first step last sizeZ
    let elementBytes =
        uint64 sizeX * uint64 sizeY * (typeof<'T> |> Image.getBytesPerComponent |> uint64)
    let sourcePeek =
        SourcePeek.create
            name
            elementBytes
            (Some (uint64 selected.Length))
            (Map.ofList
                [ "kind", "nexus-range"
                  "path", path
                  "datasetPath", datasetPath
                  "width", string sizeX
                  "height", string sizeY
                  "depth", string selected.Length
                  "sourceDepth", string sizeZ
                  "pixelType", typeof<'T>.Name
                  "frameAxis", string frameAxis
                  "yAxis", string yAxis
                  "xAxis", string xAxis
                  "first", string first
                  "step", string step
                  "last", string last ])

    let mapper (outputIndex: int) : Image<'T> =
        let zIndex = selected[outputIndex]
        if pl.debug then
            printfn $"[readNexusRange] Reading z {zIndex} from {path}:{datasetPath} as {typeof<'T>.Name}"

        let starts = Array.zeroCreate<uint64> rank
        let blocks = Array.create rank 1UL
        starts[frameAxis] <- uint64 zIndex
        blocks[frameAxis] <- 1UL
        blocks[yAxis] <- uint64 sizeY
        blocks[xAxis] <- uint64 sizeX

        let slab =
            hdfSlabImageAs<'T>
                dataset
                rank
                starts
                blocks
                sizeX
                sizeY
                1
                frameAxis
                yAxis
                xAxis
                $"readNexusRange.{outputIndex}"
        try
            ImageFunctions.extractSlice 2u 0 slab
        finally
            slab.decRefCount()

    let transition = ProfileTransition.create Unit Streaming
    let memPeak = 256UL
    let memoryNeed = fun _ -> memPeak
    let elementTransformation = id
    let memoryModel = StageMemoryModel.fromSinglePeak Source memoryNeed
    let timeCostModel =
        StageTimeCostModel.ioRead Source (Some $"readNexusRange.{typeof<'T>.Name}") (fun _ -> elementBytes) (fun _ -> 1UL)
    let stage =
        Stage.init name (uint selected.Length) mapper transition memoryNeed elementTransformation
        |> withCostModel (StageCostModel.create memoryModel timeCostModel)
        |> Some

    Plan.createWithOptimizer stage pl.memAvail memPeak memPeak (uint64 selected.Length) pl.debug pl.optimize
    |> Plan.withRuntimeOptionsFrom pl
    |> Plan.withSourcePeek sourcePeek

#endif

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
    let mutable cachedBatchResults: ZarrEncodedChunk[] = Array.empty
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
        let batchChunks = chunks[batchStart .. batchStop - 1]
        let batchResults =
            array.ReadChunksEncodedAsync(batchChunks, maxParallelReads, CancellationToken.None)
            |> runTask
            |> Seq.toArray
        deleteZarrNetDebugLogs ()
        cachedBatchStart <- batchStart
        cachedBatchResults <- batchResults

    let mapper (outputIndex: int) : EncodedLocatedChunk =
        if cachedBatchStart < 0
           || outputIndex < cachedBatchStart
           || outputIndex >= cachedBatchStart + cachedBatchResults.Length then
            loadBatch outputIndex

        let chunkRef = chunks[outputIndex]
        if pl.debug then
            printfn $"[readZarrEncodedChunks] Reading encoded chunk index ({chunkRef.ChunkCoord[4]}, {chunkRef.ChunkCoord[3]}, {chunkRef.ChunkCoord[2]}) from {path}"

        let payload =
            let encoded = cachedBatchResults[outputIndex - cachedBatchStart]
            if encoded.IsPresent then
                encoded.EncodedBytes.ToArray() |> Some
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
            array.WriteChunksEncodedAsync(pendingEncoded.ToArray(), writeParallelism, false, CancellationToken.None)
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
                pendingEncoded.Add(ZarrEncodedChunk.Present(outputChunk, ReadOnlyMemory<byte>(payload)))
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

#if LEGACY_IMAGE
let write<'T when 'T: equality> (outputDir: string) (suffix: string) : Stage<Image<'T>, Image<'T>> =
    let t = typeof<'T>
    if (icompare suffix ".tif" || icompare suffix ".tiff") 
        && not (t = typeof<uint8> || t = typeof<uint8 list> || t = typeof<int8> || t = typeof<uint16> || t = typeof<int16> || t = typeof<int32> || t = typeof<float32>) then
        failwith $"[write] tiff images only supports (u)int8, (u)int16, int32 and float32 but was {t.Name}" 
    Directory.CreateDirectory(outputDir) |> ignore
    let cleaned = lazy (cleanImageSeriesFiles outputDir suffix)
    let mapper (debug: bool) (idx: int64) (image: Image<'T>) =
        cleaned.Force()
        let fileName = Path.Combine(outputDir, sprintf "image_%03d%s" idx suffix)
        if debug then printfn "[write] Saved image %d to %s as %s" idx fileName (friendlyImageTypeName image)
        if canWriteDirectTiffStack<'T> suffix then
            ImageIO.writeTiffSliceFile fileName image
        else
            image.toFile(fileName)
        image
    let memoryNeed = id
    let memoryModel = StageMemoryModel.fromSinglePeak Iter memoryNeed
    let fallbackTimeCostModel =
        imageIoCost<'T>
            "write"
            Iter
            $"write.{suffixCostLabel suffix}.{typeof<'T>.Name}"
            (fun input -> inputValue input |> imageBytes<'T>)
            (fun _ -> 1UL)
    let timeCostModel =
        imageStackOperatorTimeCost<'T>
            "Write"
            Iter
            suffix
            fallbackTimeCostModel.Estimate
    Stage.mapi $"write \"{outputDir}/*{suffix}\"" mapper memoryNeed id
    |> withCostModel (StageCostModel.create memoryModel timeCostModel)
#endif

let private writeChunkThickCore<'T when 'T: equality> split3DChunks (outputDir: string) (suffix: string) : Stage<Chunk<'T>, unit> =
    ensureDirectTiffChunkWrite<'T> suffix
    Directory.CreateDirectory(outputDir) |> ignore
    let cleaned = lazy (cleanImageSeriesFiles outputDir suffix)
    let indexLock = obj()
    let mutable nextSliceIndex = 0L

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
                    writeChunkTiffSliceFromOffset<'T> fileName chunk width height sliceOffset
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

let writeChunkSlices<'T when 'T: equality> (outputDir: string) (suffix: string) : Stage<Chunk<'T>, unit> =
    ensureDirectTiffChunkWrite<'T> suffix
    Directory.CreateDirectory(outputDir) |> ignore
    let cleaned = lazy (cleanImageSeriesFiles outputDir suffix)

    let mapper (debug: bool) (idx: int64) (chunk: Chunk<'T>) =
        cleaned.Force()
        try
            let width, height, depth = chunk.Size
            if depth <> 1UL then
                invalidArg "chunk" $"write expects slice chunks with depth 1. Use writeThick for depth {depth} chunks."

            let fileName = Path.Combine(outputDir, sprintf "image_%03d%s" idx suffix)
            if debug then
                printfn $"[writeChunkSlices] Saved chunk slice {idx} to {fileName} as {typeof<'T>.Name}"
            writeChunkTiffSliceFromOffset<'T> fileName chunk width height 0
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

let writeChunkThick<'T when 'T: equality> (outputDir: string) (suffix: string) : Stage<Chunk<'T>, unit> =
    writeChunkThickCore<'T> true outputDir suffix

let writeChunkThickFiles<'T when 'T: equality> (outputDir: string) (suffix: string) : Stage<Chunk<'T>, unit> =
    writeChunkThickCore<'T> false outputDir suffix

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

#if LEGACY_IMAGE
let writeSlabSlices<'T when 'T: equality> (outputDir: string) (suffix: string) (_winSz: uint) : Stage<Image<'T>, Image<'T>> =
    Directory.CreateDirectory(outputDir) |> ignore

    let mapper (debug: bool) (_idx: int64) (slab: Image<'T>) =
        Directory.CreateDirectory(outputDir) |> ignore

        let depth =
            if slab.GetDimensions() = 2u then 1
            else slab.GetDepth() |> int

        for localIndex in 0 .. depth - 1 do
            let slice =
                if slab.GetDimensions() = 2u then
                    slab.incRefCount()
                    slab
                else
                    ImageFunctions.extractSlice 2u localIndex slab

            let globalIndex = slab.index + localIndex
            let fileName = Path.Combine(outputDir, sprintf "image_%03d%s" globalIndex suffix)
            if debug then printfn "[writeSlabSlices] Saved image %d to %s as %s" globalIndex fileName (typeof<'T>.Name)
            slice.toFile(fileName)
            slice.decRefCount()

        slab

    Stage.mapi $"writeSlabSlices \"{outputDir}/*{suffix}\"" mapper id id

let writeVolume<'T when 'T: equality> (filename: string) : Stage<Image<'T>, unit> =
    let extension = Path.GetExtension(filename).ToLowerInvariant()
    if extension <> ".tif" && extension <> ".tiff" && extension <> ".btf" && extension <> ".bigtiff" then
        invalidArg "filename" "writeVolume currently supports streaming multipage TIFF/BigTIFF output only."

    ImageIO.tiffPixelLayout<'T> () |> ignore

    let reducer (debug: bool) (input: AsyncSeq<Image<'T>>) =
        async {
            let directory = Path.GetDirectoryName(Path.GetFullPath(filename))
            if not (String.IsNullOrWhiteSpace directory) then
                Directory.CreateDirectory(directory) |> ignore

            use tiff = Tiff.Open(filename, ImageIO.tiffWriteMode filename)
            if isNull tiff then
                invalidOp $"Could not open '{filename}' for TIFF volume writing."

            let mutable width = 0u
            let mutable height = 0u
            let mutable page = 0

            do!
                input
                |> AsyncSeq.iterAsync (fun image ->
                    async {
                        try
                            if image.GetDimensions() <> 2u then
                                invalidOp $"writeVolume expects 2D slices, got {image.GetDimensions()}D at slice {image.index}."

                            if page = 0 then
                                width <- image.GetWidth()
                                height <- image.GetHeight()
                            elif image.GetWidth() <> width || image.GetHeight() <> height then
                                invalidOp $"writeVolume expected all slices to be {width}x{height}; got {image.GetWidth()}x{image.GetHeight()} at slice {image.index}."

                            if debug then
                                printfn $"[writeVolume] Writing TIFF page {page} from slice {image.index} to {filename}"

                            ImageIO.writeTiffPage tiff image (Some page)
                            page <- page + 1
                        finally
                            image.decRefCount()
                    })
        }

    Stage.reduce "writeVolume" reducer Streaming id (fun _ -> 1UL)

let private bytesOfScalarImage<'T when 'T: equality> (image: Image<'T>) =
    if image.GetNumberOfComponentsPerPixel() <> 1u then
        invalidArg "image" $"Expected a scalar image, got {image.GetNumberOfComponentsPerPixel()} components per pixel."

    let pixels = image.toFlatArray()
    let byteCount = pixels.Length * int (Image.getBytesPerComponent typeof<'T>)
    let bytes = Array.zeroCreate<byte> byteCount
    Buffer.BlockCopy(pixels, 0, bytes, 0, byteCount)
    bytes

let private interleavedComplexBytes<'Component when 'Component: equality>
    (realImage: Image<'Component>)
    (imagImage: Image<'Component>) =

    let real = realImage.toFlatArray()
    let imag = imagImage.toFlatArray()
    if real.Length <> imag.Length then
        invalidOp $"Complex image real/imaginary component sizes differ: {real.Length} vs {imag.Length}."

    let components = Array.zeroCreate<'Component> (2 * real.Length)
    for i in 0 .. real.Length - 1 do
        let j = 2 * i
        components[j] <- real[i]
        components[j + 1] <- imag[i]

    let byteCount = components.Length * int (Image.getBytesPerComponent typeof<'Component>)
    let bytes = Array.zeroCreate<byte> byteCount
    Buffer.BlockCopy(components, 0, bytes, 0, byteCount)
    bytes

let private bytesOfZarrImage<'T when 'T: equality> (image: Image<'T>) =
    if typeof<'T> = typeof<Image.ComplexFloat32> then
        let realItk = extractComplexRealImage (image.toSimpleITK())
        let imagItk = extractComplexImagImage (image.toSimpleITK())
        let realImage = Image<float32>.ofSimpleITKNDispose(realItk, "writeZarr.Re", image.index)
        let imagImage = Image<float32>.ofSimpleITKNDispose(imagItk, "writeZarr.Im", image.index)
        try
            interleavedComplexBytes realImage imagImage
        finally
            realImage.decRefCount()
            imagImage.decRefCount()
    elif typeof<'T> = typeof<System.Numerics.Complex> then
        let realItk = extractComplexRealImage (image.toSimpleITK())
        let imagItk = extractComplexImagImage (image.toSimpleITK())
        let realImage = Image<float>.ofSimpleITKNDispose(realItk, "writeZarr.Re", image.index)
        let imagImage = Image<float>.ofSimpleITKNDispose(imagItk, "writeZarr.Im", image.index)
        try
            interleavedComplexBytes realImage imagImage
        finally
            realImage.decRefCount()
            imagImage.decRefCount()
    else
        bytesOfScalarImage image

let writeZarrWithCompression<'T when 'T: equality>
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
    : Stage<Image<'T>, Image<'T>> =

    suppressZarrNetDebugLogging ()

    let dataType = zarrDataType<'T> ()
    let mutable writer: OmeZarrWriter option = None
    let depth = int depth
    let chunkX = max 1u chunkX |> int
    let chunkY = max 1u chunkY |> int
    let chunkZ = max 1u chunkZ |> int

    let createWriter (image: Image<'T>) =
        let descriptor =
            BioImageDescriptor(
                int (image.GetWidth()),
                int (image.GetHeight()),
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

        writer <- Some created
        created

    let mapper (debug: bool) (idx: int64) (image: Image<'T>) =
        if depth <= 0 then
            invalidArg "depth" "writeZarr depth must be positive."
        if idx >= int64 depth then
            failwith $"writeZarr received slice {idx}, but the declared depth is {depth}."
        if image.GetDimensions() <> 2u then
            failwith $"writeZarr expects a stream of 2D slice images, but got {image.GetDimensions()}D."

        let zarrWriter =
            match writer with
            | Some writer -> writer
            | None -> createWriter image

        let planeBytes = bytesOfZarrImage image
        if debug then
            printfn "[writeZarr] Saved plane %d to %s as %s" idx outputPath (friendlyImageTypeName image)

        zarrWriter.WritePlaneAsync(int idx, planeBytes, CancellationToken.None)
        |> runUnitTask
        deleteZarrNetDebugLogs ()

        if idx = int64 (depth - 1) then
            zarrWriter.DisposeAsync().AsTask()
            |> runUnitTask
            writer <- None
            deleteZarrNetDebugLogs ()

        image

    let memoryNeed = id
    let memoryModel = StageMemoryModel.fromSinglePeak Iter memoryNeed
    let timeCostModel =
        imageIoCost<'T>
            "write"
            Iter
            $"writeZarr.{typeof<'T>.Name}"
            (fun input -> inputValue input |> imageBytes<'T>)
            (fun _ -> 1UL)

    Stage.mapi $"writeZarr \"{outputPath}\"" mapper memoryNeed id
    |> withCostModel (StageCostModel.create memoryModel timeCostModel)

let writeZarr<'T when 'T: equality>
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
    : Stage<Image<'T>, Image<'T>> =

    writeZarrWithCompression
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
#endif

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

            let planeBytes = Array.zeroCreate<byte> expectedBytes
            Buffer.BlockCopy(chunk.Bytes, 0, planeBytes, 0, expectedBytes)

            if debug then
                printfn $"[writeZarrChunkSlices] Saved plane {idx} to {outputPath} as {dataType}"

            zarrWriter.WritePlaneAsync(int idx, planeBytes, CancellationToken.None)
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
    let pendingEncoded = ResizeArray<ZarrEncodedChunk>()
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
        if pendingEncoded.Count > 0 then
            try
                array.WriteChunksEncodedAsync(pendingEncoded.ToArray(), writeParallelism, false, CancellationToken.None)
                |> runUnitTask
            finally
                for buffer in pendingBuffers do
                    ArrayPool<byte>.Shared.Return(buffer)
                pendingEncoded.Clear()
                pendingBuffers.Clear()

    let clearPending () =
        for buffer in pendingBuffers do
            ArrayPool<byte>.Shared.Return(buffer)
        pendingEncoded.Clear()
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
                pendingEncoded.Add(ZarrEncodedChunk.Present(outputChunk, ReadOnlyMemory<byte>(buffer, 0, bufferBytes)))
                pendingBuffers.Add buffer
                if pendingEncoded.Count >= writeParallelism then
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

            let planeBytes = Array.zeroCreate<byte> expectedBytes
            Buffer.BlockCopy(chunk.Bytes, 0, planeBytes, 0, expectedBytes)

            if debug then
                printfn "[writeZarrComplex64InterleavedFloat32] Saved plane %d to %s as complex64" idx outputPath

            zarrWriter.WritePlaneAsync(int idx, planeBytes, CancellationToken.None)
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

#if LEGACY_IMAGE
let private writeZarrSlabStage<'T when 'T: equality>
    (compression: ZarrCompression)
    (outputPath: string)
    (name: string)
    (depth: uint)
    (chunkX: uint)
    (chunkY: uint)
    (physicalSizeX: float)
    (physicalSizeY: float)
    (physicalSizeZ: float)
    (maxConcurrentWrites: int)
    : Stage<Image<'T>, Image<'T>> =

    suppressZarrNetDebugLogging ()

    let dataType = zarrDataType<'T> ()
    let mutable writer: OmeZarrWriter option = None
    let mutable level: ResolutionLevelNode option = None
    let depth = int depth
    let chunkX = max 1u chunkX |> int
    let chunkY = max 1u chunkY |> int
    let mutable slabChunkZ: int option = None

    let createWriter (image: Image<'T>) =
        let chunkZ = max 1u (image.GetDepth()) |> int
        slabChunkZ <- Some chunkZ
        let descriptor =
            BioImageDescriptor(
                int (image.GetWidth()),
                int (image.GetHeight()),
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

        writer <- Some created
        level <- Some(openZarrResolutionLevel outputPath 0 0)
        created

    let mapper (debug: bool) (idx: int64) (image: Image<'T>) =
        if depth <= 0 then
            invalidArg "depth" "writeZarrSlab depth must be positive."
        if image.GetDimensions() <> 3u then
            failwith $"writeZarrSlab expects a stream of 3D slab images, but got {image.GetDimensions()}D."

        let chunkZ =
            match slabChunkZ with
            | Some chunkZ -> chunkZ
            | None -> max 1u (image.GetDepth()) |> int
        let zStart = int idx * chunkZ
        let zCount = int (image.GetDepth())
        let zStop = zStart + zCount
        if zStop > depth then
            failwith $"writeZarrSlab received slab {idx} ending at z={zStop}, but the declared depth is {depth}."

        let zarrLevel =
            match level with
            | Some level -> level
            | None ->
                createWriter image |> ignore
                level.Value

        let slabBytes = bytesOfZarrImage image
        let region =
            PixelRegion(
                [| 0L; 0L; int64 zStart; 0L; 0L |],
                [| 1L; 1L; int64 zStop; int64 (image.GetHeight()); int64 (image.GetWidth()) |])

        if debug then
            printfn "[writeZarrSlab] Saved z %d..%d to %s as %s" zStart (zStop - 1) outputPath (friendlyImageTypeName image)

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

        image

    let memoryNeed = id
    let memoryModel = StageMemoryModel.fromSinglePeak Iter memoryNeed
    let timeCostModel =
        imageIoCost<'T>
            "write"
            Iter
            $"writeZarrSlab.{typeof<'T>.Name}"
            (fun input -> inputValue input |> imageBytes<'T>)
            (fun _ -> 1UL)

    Stage.mapi $"writeZarrSlab \"{outputPath}\"" mapper memoryNeed id
    |> withCostModel (StageCostModel.create memoryModel timeCostModel)

let private sourcePeekUInt<'S, 'T> (key: string) (pl: Plan<'S, 'T>) =
    match pl.sourcePeek with
    | Some sourcePeek ->
        match sourcePeek.Shape |> Map.tryFind key with
        | Some value ->
            match UInt32.TryParse(value) with
            | true, parsed -> parsed
            | false, _ -> failwith $"Source metadata field '{key}' has value '{value}', which is not a uint32."
        | None ->
            failwith $"Source metadata does not contain field '{key}'. Use the explicit writer when the output shape is not inherited from the input."
    | None ->
        failwith "Source metadata is unavailable. Use the explicit writer when the output shape is not inherited from a metadata-carrying source."

let writeZarrSlabNamedWithCompression<'S, 'T when 'T: equality>
    (compression: ZarrCompression)
    (outputPath: string)
    (name: string)
    (chunkX: uint)
    (chunkY: uint)
    (physicalSizeX: float)
    (physicalSizeY: float)
    (physicalSizeZ: float)
    (maxConcurrentWrites: int)
    (pl: Plan<'S, Image<'T>>)
    : Plan<'S, Image<'T>> =

    let depth = sourcePeekUInt "depth" pl
    pl
    >=> writeZarrSlabStage compression outputPath name depth chunkX chunkY physicalSizeX physicalSizeY physicalSizeZ maxConcurrentWrites

let writeZarrSlabNamed<'S, 'T when 'T: equality>
    (outputPath: string)
    (name: string)
    (chunkX: uint)
    (chunkY: uint)
    (physicalSizeX: float)
    (physicalSizeY: float)
    (physicalSizeZ: float)
    (maxConcurrentWrites: int)
    (pl: Plan<'S, Image<'T>>)
    : Plan<'S, Image<'T>> =

    pl
    |> writeZarrSlabNamedWithCompression ZarrCompression.BloscLz4 outputPath name chunkX chunkY physicalSizeX physicalSizeY physicalSizeZ maxConcurrentWrites

let writeZarrSlabWithCompression<'S, 'T when 'T: equality>
    (compression: ZarrCompression)
    (outputPath: string)
    (chunkX: uint)
    (chunkY: uint)
    (physicalSizeX: float)
    (physicalSizeY: float)
    (physicalSizeZ: float)
    (maxConcurrentWrites: int)
    (pl: Plan<'S, Image<'T>>)
    : Plan<'S, Image<'T>> =

    pl
    |> writeZarrSlabNamedWithCompression compression outputPath "image" chunkX chunkY physicalSizeX physicalSizeY physicalSizeZ maxConcurrentWrites

let writeZarrSlab<'S, 'T when 'T: equality>
    (outputPath: string)
    (chunkX: uint)
    (chunkY: uint)
    (physicalSizeX: float)
    (physicalSizeY: float)
    (physicalSizeZ: float)
    (maxConcurrentWrites: int)
    (pl: Plan<'S, Image<'T>>)
    : Plan<'S, Image<'T>> =

    pl
    |> writeZarrSlabWithCompression ZarrCompression.BloscLz4 outputPath chunkX chunkY physicalSizeX physicalSizeY physicalSizeZ maxConcurrentWrites

let writeNexus<'T when 'T: equality>
    (outputPath: string)
    (datasetPath: string)
    (depth: uint)
    (chunkX: uint)
    (chunkY: uint)
    (chunkZ: uint)
    (frameAxis: int)
    (yAxis: int)
    (xAxis: int)
    : Stage<Image<'T>, Image<'T>> =

    hdfDataType<'T> () |> ignore
    validateHdfAxes 3 frameAxis yAxis xAxis

    let depth = int depth
    let chunkX = max 1u chunkX |> uint32
    let chunkY = max 1u chunkY |> uint32
    let chunkZ = max 1u chunkZ |> uint32
    let mutable writer: H5NativeWriter option = None
    let mutable dataset: H5Dataset<'T[,,]> option = None

    let createWriter (image: Image<'T>) =
        if depth <= 0 then
            invalidArg "depth" "writeNexus depth must be positive."

        let fileDims = Array.zeroCreate<uint64> 3
        let chunks = Array.zeroCreate<uint32> 3
        fileDims[frameAxis] <- uint64 depth
        fileDims[yAxis] <- uint64 (image.GetHeight())
        fileDims[xAxis] <- uint64 (image.GetWidth())
        chunks[frameAxis] <- min chunkZ (uint32 depth)
        chunks[yAxis] <- min chunkY (uint32 (image.GetHeight()))
        chunks[xAxis] <- min chunkX (uint32 (image.GetWidth()))

        let file = H5File()
        let createdDataset = H5Dataset<'T[,,]>(fileDims, chunks = chunks)
        addHdfDatasetPath file datasetPath createdDataset
        let createdWriter = file.BeginWrite(filePath = outputPath)
        writer <- Some createdWriter
        dataset <- Some createdDataset
        createdWriter, createdDataset

    let mapper (debug: bool) (idx: int64) (image: Image<'T>) =
        if idx >= int64 depth then
            failwith $"writeNexus received slice {idx}, but the declared depth is {depth}."
        if image.GetDimensions() <> 2u then
            failwith $"writeNexus expects a stream of 2D slice images, but got {image.GetDimensions()}D."

        let hdfWriter, hdfDataset =
            match writer, dataset with
            | Some hdfWriter, Some hdfDataset -> hdfWriter, hdfDataset
            | _ -> createWriter image

        let width = int (image.GetWidth())
        let height = int (image.GetHeight())
        let plane = image.toFlatArray()
        let memDims = Array.zeroCreate<int> 3
        memDims[frameAxis] <- 1
        memDims[yAxis] <- height
        memDims[xAxis] <- width

        let data =
            if frameAxis = 0 && yAxis = 1 && xAxis = 2 then
                let block = Array3D.zeroCreate<'T> 1 height width
                let byteCount = width * height * (typeof<'T> |> Image.getBytesPerComponent |> int)
                Buffer.BlockCopy(plane, 0, block, 0, byteCount)
                block
            else
                Array3D.init memDims[0] memDims[1] memDims[2] (fun a b c ->
                    let indices = [| a; b; c |]
                    let x = indices[xAxis]
                    let y = indices[yAxis]
                    plane[y * width + x])

        let fileStarts = Array.zeroCreate<uint64> 3
        let blocks = Array.zeroCreate<uint64> 3
        fileStarts[frameAxis] <- uint64 idx
        blocks[frameAxis] <- 1UL
        blocks[yAxis] <- uint64 height
        blocks[xAxis] <- uint64 width
        let fileSelection = HyperslabSelection(3, fileStarts, blocks)

        if debug then
            printfn "[writeNexus] Saved frame %d to %s:%s as %s" idx outputPath datasetPath (friendlyImageTypeName image)

        hdfWriter.Write(hdfDataset, data, AllSelection(), fileSelection)

        if idx = int64 (depth - 1) then
            hdfWriter.Dispose()
            writer <- None
            dataset <- None

        image

    let memoryNeed = id
    let memoryModel = StageMemoryModel.fromSinglePeak Iter memoryNeed
    let timeCostModel =
        imageIoCost<'T>
            "write"
            Iter
            $"writeNexus.{typeof<'T>.Name}"
            (fun input -> inputValue input |> imageBytes<'T>)
            (fun _ -> 1UL)

    Stage.mapi $"writeNexus \"{outputPath}:{datasetPath}\"" mapper memoryNeed id
    |> withCostModel (StageCostModel.create memoryModel timeCostModel)

let private hdfBlockOfSlab<'T when 'T: equality> frameAxis yAxis xAxis (image: Image<'T>) =
    let width = int (image.GetWidth())
    let height = int (image.GetHeight())
    let depth = int (image.GetDepth())
    let pixels = image.toFlatArray()

    if frameAxis = 0 && yAxis = 1 && xAxis = 2 then
        let block = Array3D.zeroCreate<'T> depth height width
        let byteCount = width * height * depth * (typeof<'T> |> Image.getBytesPerComponent |> int)
        Buffer.BlockCopy(pixels, 0, block, 0, byteCount)
        block
    else
        let memDims = Array.zeroCreate<int> 3
        memDims[frameAxis] <- depth
        memDims[yAxis] <- height
        memDims[xAxis] <- width

        Array3D.init memDims[0] memDims[1] memDims[2] (fun a b c ->
            let indices = [| a; b; c |]
            let z = indices[frameAxis]
            let y = indices[yAxis]
            let x = indices[xAxis]
            pixels[(z * height + y) * width + x])

let private writeNexusSlabStage<'T when 'T: equality>
    (outputPath: string)
    (datasetPath: string)
    (depth: uint)
    (chunkX: uint)
    (chunkY: uint)
    (frameAxis: int)
    (yAxis: int)
    (xAxis: int)
    : Stage<Image<'T>, Image<'T>> =

    hdfDataType<'T> () |> ignore
    validateHdfAxes 3 frameAxis yAxis xAxis

    let depth = int depth
    let chunkX = max 1u chunkX |> uint32
    let chunkY = max 1u chunkY |> uint32
    let mutable chunkZ: int option = None
    let mutable writer: H5NativeWriter option = None
    let mutable dataset: H5Dataset<'T[,,]> option = None

    let createWriter (image: Image<'T>) =
        if depth <= 0 then
            invalidArg "depth" "writeNexusSlab depth must be positive."
        if image.GetDimensions() <> 3u then
            failwith $"writeNexusSlab expects a stream of 3D slab images, but got {image.GetDimensions()}D."

        let slabDepth = max 1u (image.GetDepth()) |> int
        chunkZ <- Some slabDepth

        let fileDims = Array.zeroCreate<uint64> 3
        let chunks = Array.zeroCreate<uint32> 3
        fileDims[frameAxis] <- uint64 depth
        fileDims[yAxis] <- uint64 (image.GetHeight())
        fileDims[xAxis] <- uint64 (image.GetWidth())
        chunks[frameAxis] <- min (uint32 slabDepth) (uint32 depth)
        chunks[yAxis] <- min chunkY (uint32 (image.GetHeight()))
        chunks[xAxis] <- min chunkX (uint32 (image.GetWidth()))

        let file = H5File()
        let createdDataset = H5Dataset<'T[,,]>(fileDims, chunks = chunks)
        addHdfDatasetPath file datasetPath createdDataset
        let createdWriter = file.BeginWrite(filePath = outputPath)
        writer <- Some createdWriter
        dataset <- Some createdDataset
        createdWriter, createdDataset

    let mapper (debug: bool) (idx: int64) (image: Image<'T>) =
        if image.GetDimensions() <> 3u then
            failwith $"writeNexusSlab expects a stream of 3D slab images, but got {image.GetDimensions()}D."

        let hdfWriter, hdfDataset =
            match writer, dataset with
            | Some hdfWriter, Some hdfDataset -> hdfWriter, hdfDataset
            | _ -> createWriter image

        let slabDepth =
            match chunkZ with
            | Some chunkZ -> chunkZ
            | None -> max 1u (image.GetDepth()) |> int

        let zStart = int idx * slabDepth
        let zCount = int (image.GetDepth())
        let zStop = zStart + zCount

        if zStop > depth then
            failwith $"writeNexusSlab received slab {idx} ending at z={zStop}, but the declared depth is {depth}."

        let data = hdfBlockOfSlab frameAxis yAxis xAxis image
        let fileStarts = Array.zeroCreate<uint64> 3
        let blocks = Array.zeroCreate<uint64> 3
        fileStarts[frameAxis] <- uint64 zStart
        blocks[frameAxis] <- uint64 zCount
        blocks[yAxis] <- uint64 (image.GetHeight())
        blocks[xAxis] <- uint64 (image.GetWidth())
        let fileSelection = HyperslabSelection(3, fileStarts, blocks)

        if debug then
            printfn "[writeNexusSlab] Saved frames %d..%d to %s:%s as %s" zStart (zStop - 1) outputPath datasetPath (friendlyImageTypeName image)

        hdfWriter.Write(hdfDataset, data, AllSelection(), fileSelection)

        if zStop = depth then
            hdfWriter.Dispose()
            writer <- None
            dataset <- None

        image

    let memoryNeed = id
    let memoryModel = StageMemoryModel.fromSinglePeak Iter memoryNeed
    let timeCostModel =
        imageIoCost<'T>
            "write"
            Iter
            $"writeNexusSlab.{typeof<'T>.Name}"
            (fun input -> inputValue input |> imageBytes<'T>)
            (fun _ -> 1UL)

    Stage.mapi $"writeNexusSlab \"{outputPath}:{datasetPath}\"" mapper memoryNeed id
    |> withCostModel (StageCostModel.create memoryModel timeCostModel)

let writeNexusSlab<'S, 'T when 'T: equality>
    (outputPath: string)
    (datasetPath: string)
    (chunkX: uint)
    (chunkY: uint)
    (frameAxis: int)
    (yAxis: int)
    (xAxis: int)
    (pl: Plan<'S, Image<'T>>)
    : Plan<'S, Image<'T>> =

    let depth = sourcePeekUInt "depth" pl
    pl
    >=> writeNexusSlabStage outputPath datasetPath depth chunkX chunkY frameAxis yAxis xAxis

let private writeSlabChunks (debug: bool) (outputDir: string) (suffix: string) (width: uint) (height: uint) (winSz: uint) (k: int) (stack: Image<'T>) =
    let chunksX = (stack.GetWidth() + width - 1u) / width
    let chunksY = (stack.GetHeight() + height - 1u) / height
    for i in 0u .. chunksX - 1u do
        for j in 0u .. chunksY - 1u do
            let fileName = getChunkFilename outputDir suffix (int i) (int j) (int k)
            let x00 = i*width |> int
            let x01 = ((i+1u)*width-1u |> int, stack.GetWidth()-1u |> int) ||> min
            let x10 = j*height |> int
            let x11 = ((j+1u)*height-1u |> int, stack.GetHeight()-1u |> int) ||> min
            let x20 = 0
            let x21 = winSz-1u |> int
            if x00<=x01 && x10<=x11 && x20<=x21 then
                let chunk = stack.[x00 .. x01, x10 .. x11 , x20 .. x21]
                if debug then printfn "[write] Saved chunk %d %d %d to %s as %s" i j k fileName (friendlyImageTypeName chunk)
                chunk.toFile(fileName)
                chunk.decRefCount()

let private writeChunksCore<'T when 'T: equality> (outputDir: string) (suffix: string) (width: uint) (height: uint) (winSz: uint) : Stage<Image<'T>, Image<'T>> =
    let t = typeof<'T>
    if (icompare suffix ".tif" || icompare suffix ".tiff") 
        && not (t = typeof<uint8> || t = typeof<int8> || t = typeof<uint16> || t = typeof<int16> || t = typeof<float32>) then
        failwith $"[write] tiff images only supports (u)int8, (u)int16 and float32 but was {t.Name}" 
    if not (Directory.Exists(outputDir)) then
        Directory.CreateDirectory(outputDir) |> ignore

    let cleaned = lazy (cleanChunkFiles outputDir suffix)
    let pad, stride = 0u, winSz
    let f (debug: bool) (k: int64) (stack: Image<'T>) = 
        cleaned.Force()
        if stack.GetDimensions() = 2u then
            let slab = ImageFunctions.stack [ stack ]
            writeSlabChunks debug outputDir suffix width height 1u (int k) slab
            slab.decRefCount()
        else
            writeSlabChunks debug outputDir suffix width height winSz (int k) stack
        stack.incRefCount() //to make sure volFctToLstFctReleaseAfter doesn't release it.
        stack
    let mapper (debug: bool) (idx: int64) (window: Window<Image<'T>>) =
        volFctToWindowFctReleaseAfterDebug debug (f debug idx) 1u pad stride window
    let btUint8 = typeof<uint8>|>Image.getBytesPerComponent |> uint64
    let btUint64 = typeof<uint64> |> Image.getBytesPerComponent |> uint64
    let memoryNeed nPixels = 
        let bt8 = typeof<uint8>|>Image.getBytesPerComponent |> uint64
        let bt64 = typeof<uint64> |> Image.getBytesPerComponent |> uint64
        let wsz = uint64 winSz
        let str = uint64 stride
        max (nPixels*(wsz*(2UL*bt8+bt64)-str*bt8)) (nPixels*(wsz*(bt8+bt64)+str*(bt64-bt8)))
    let elementTransformation = id
    let memoryModel = StageMemoryModel.fromSinglePeak Iter memoryNeed
    let timeCostModel =
        imageIoCost<'T>
            "write"
            Iter
            $"writeChunks.{typeof<'T>.Name}"
            (fun input -> inputValue input |> imageBytes<'T>)
            (fun _ -> 1UL)
    let stg =
        Stage.mapi "writeChunks" mapper memoryNeed elementTransformation
        |> withCostModel (StageCostModel.create memoryModel timeCostModel)
    (window winSz pad stride) --> stg --> flattenList ()

let writeChunks<'T when 'T: equality> outputDir suffix width height winSz =
    writeChunksCore<'T> outputDir suffix width height winSz

let writeVolumeAsChunks debug outputDir suffix chunkX chunkY chunkZ (volume: Image<'T>) =
    let depth = volume.GetDepth()
    let mutable k = 0u
    while k < depth do
        let last = min (depth - 1u) (k + chunkZ - 1u)
        let slab = ImageFunctions.extractSub [ 0u; 0u; k ] [ volume.GetWidth() - 1u; volume.GetHeight() - 1u; last ] volume
        writeSlabChunks debug outputDir suffix chunkX chunkY (last - k + 1u) (int (k / chunkZ)) slab
        slab.decRefCount()
        k <- k + chunkZ

let private readChunkDirectoryVolume<'T when 'T: equality> inputDir suffix =
    let chunkInfo = getChunkInfo inputDir suffix
    let slabs =
        [ for k in 0 .. chunkInfo.chunks[2] - 1 ->
            _readSlabStacked<'T> inputDir suffix chunkInfo 2u k ]
    let volume = ImageFunctions.stack slabs
    slabs |> List.iter (fun slab -> slab.decRefCount())
    volume, chunkInfo

let private readChunkColumn<'T when 'T: equality> inputDir suffix (chunkInfo: ChunkInfo) i j =
    let chunks =
        [ for k in 0 .. chunkInfo.chunks[2] - 1 ->
            _readChunk<'T> inputDir suffix i j k ]

    let column = ImageFunctions.stack chunks
    chunks |> List.iter (fun chunk -> chunk.decRefCount())
    column

let private writeChunkColumn debug outputDir suffix i j chunkZ (column: Image<'T>) =
    let depth = column.GetDepth()
    let mutable k = 0u

    while k < depth do
        let last = min (depth - 1u) (k + chunkZ - 1u)
        let chunk = ImageFunctions.extractSub [ 0u; 0u; k ] [ column.GetWidth() - 1u; column.GetHeight() - 1u; last ] column
        let fileName = getChunkFilename outputDir suffix i j (int (k / chunkZ))
        if debug then printfn "[write] Saved chunk %d %d %d to %s as %s" i j (int (k / chunkZ)) fileName (friendlyImageTypeName chunk)
        chunk.toFile(fileName)
        chunk.decRefCount()
        k <- k + chunkZ

let private chunkedFFTAlongZCore inverse debug inputDir outputDir suffix _chunkX _chunkY chunkZ =
    if Directory.Exists(outputDir) then Directory.Delete(outputDir, true)
    Directory.CreateDirectory(outputDir) |> ignore

    let chunkInfo = getChunkInfo inputDir suffix
    let chunkZ = max 1u chunkZ

    for i in 0 .. chunkInfo.chunks[0] - 1 do
        for j in 0 .. chunkInfo.chunks[1] - 1 do
            let column = readChunkColumn<System.Numerics.Complex> inputDir suffix chunkInfo i j
            let transformed = ImageFunctions.directionalFFTComplex 2u inverse column
            column.decRefCount()
            writeChunkColumn debug outputDir suffix i j chunkZ transformed
            transformed.decRefCount()

let chunkedFFTAlongZ debug inputDir outputDir suffix chunkX chunkY chunkZ =
    chunkedFFTAlongZCore false debug inputDir outputDir suffix chunkX chunkY chunkZ

let chunkedInvFFTAlongZ debug inputDir outputDir suffix chunkX chunkY chunkZ =
    chunkedFFTAlongZCore true debug inputDir outputDir suffix chunkX chunkY chunkZ

let private interleavedComplexOfComplex3D (values: Image.ComplexFloat32[,,]) =
    let width = values.GetLength 0
    let height = values.GetLength 1
    let depth = values.GetLength 2
    let plane = width * height
    let output = Array.zeroCreate<float32> (2 * plane * depth)

    for z in 0 .. depth - 1 do
        let zOffset = z * plane
        for y in 0 .. height - 1 do
            let row = zOffset + y * width
            for x in 0 .. width - 1 do
                let value = values[x, y, z]
                let i = 2 * (row + x)
                output[i] <- value.Real
                output[i + 1] <- value.Imaginary

    output

let private complex3DOfInterleaved width height depth (values: float32[]) =
    Array3D.init width height depth (fun x y z ->
        let i = 2 * (z * width * height + y * width + x)
        Image.ComplexFloat32(values[i], values[i + 1]))

let private fftwZComplexFloat32 inverse (image: Image<Image.ComplexFloat32>) : Image<Image.ComplexFloat32> =
    if image.GetDimensions() <> 3u then
        failwith $"fftwZComplexFloat32: image must be 3D, got {image.GetDimensions()}D"

    NativeSp.ensureAvailable ()
    let values = image.toComplexFloat32Array3D()
    let width = values.GetLength 0
    let height = values.GetLength 1
    let depth = values.GetLength 2
    let interleaved = interleavedComplexOfComplex3D values
    let mutable handle = Unchecked.defaultof<GCHandle>

    try
        handle <- GCHandle.Alloc(interleaved, GCHandleType.Pinned)
        NativeSp.fftwfComplexZInplace(handle.AddrOfPinnedObject(), width, height, depth, if inverse then 1 else 0)
        |> NativeSp.checkStatus "fftwf z complex"
    finally
        if handle.IsAllocated then
            handle.Free()

    complex3DOfInterleaved width height depth interleaved
    |> fun output -> Image<Image.ComplexFloat32>.ofComplexFloat32Array3D(output, "fftwZComplexFloat32", image.index)

let private chunkedFFTFloat32AlongZCore inverse debug inputDir outputDir suffix _chunkX _chunkY chunkZ =
    if Directory.Exists(outputDir) then Directory.Delete(outputDir, true)
    Directory.CreateDirectory(outputDir) |> ignore

    let chunkInfo = getChunkInfo inputDir suffix
    let chunkZ = max 1u chunkZ

    for i in 0 .. chunkInfo.chunks[0] - 1 do
        for j in 0 .. chunkInfo.chunks[1] - 1 do
            let column = readChunkColumn<Image.ComplexFloat32> inputDir suffix chunkInfo i j
            let transformed = fftwZComplexFloat32 inverse column
            column.decRefCount()
            writeChunkColumn debug outputDir suffix i j chunkZ transformed
            transformed.decRefCount()

let chunkedFFTFloat32AlongZ debug inputDir outputDir suffix chunkX chunkY chunkZ =
    chunkedFFTFloat32AlongZCore false debug inputDir outputDir suffix chunkX chunkY chunkZ

let chunkedInvFFTFloat32AlongZ debug inputDir outputDir suffix chunkX chunkY chunkZ =
    chunkedFFTFloat32AlongZCore true debug inputDir outputDir suffix chunkX chunkY chunkZ

let chunkedShiftFFT debug inputDir outputDir suffix chunkX chunkY chunkZ =
    if Directory.Exists(outputDir) then Directory.Delete(outputDir, true)
    Directory.CreateDirectory(outputDir) |> ignore
    let volume, _ = readChunkDirectoryVolume<System.Numerics.Complex> inputDir suffix
    let shifted = ImageFunctions.shiftFFT volume
    volume.decRefCount()
    writeVolumeAsChunks debug outputDir suffix chunkX chunkY chunkZ shifted
    shifted.decRefCount()

let readChunksAsSlices<'T when 'T: equality> name outputDir suffix =
    let mutable chunkInfo : ChunkInfo = { chunks = [0;0;0]; size = [0UL;0UL;0UL]; topLeftInfo = { dimensions = 0u; size = [0UL;0UL;0UL]; componentType = ""; numberOfComponents = 0u } }
    let memoryNeed = fun _ -> 256UL
    let elementTransformation = fun _ -> chunkInfo.chunks[2] |> uint64
    Stage.map name (fun _ _ -> chunkInfo <- getChunkInfo outputDir suffix) memoryNeed elementTransformation
    --> Stage.map name (fun _ _ -> [ 0 .. chunkInfo.chunks[2] - 1 ]) memoryNeed elementTransformation
    --> flattenList ()
    --> Stage.map name (fun _ idx ->
        let slab = _readSlabStacked<'T> outputDir suffix chunkInfo 2u idx
        let slices = ImageFunctions.unstack 2u slab
        slab.decRefCount()
        slices) memoryNeed elementTransformation
    --> flattenList ()

let chunkedVolumeOperation
    name
    (prepareInput: Stage<Image<'S>, Image<'I>>)
    (operation: bool -> string -> string -> string -> uint -> uint -> uint -> unit)
    (chunkX: uint)
    (chunkY: uint)
    (chunkZ: uint)
    : Stage<Image<'S>, Image<'T>> =
    let workspaceRoot =
        Path.Combine(Path.GetTempPath(), $"stackprocessing-{name}-{Guid.NewGuid():N}")
    let inputDir = Path.Combine(workspaceRoot, "input")
    let outputDir = Path.Combine(workspaceRoot, "output")
    let suffix = ".mha"
    let chunkX = max 1u chunkX
    let chunkY = max 1u chunkY
    let chunkZ = max 1u chunkZ
    let memoryNeed = fun _ -> 256UL
    let elementTransformation = fun _ -> 1UL

    prepareInput
    --> writeChunks inputDir suffix chunkX chunkY chunkZ
    --> cleanStage name (fun () -> deleteIfExists workspaceRoot)
    --> ignoreSingles ()
    --> Stage.map name (fun debug _ -> operation debug inputDir outputDir suffix chunkX chunkY chunkZ) memoryNeed elementTransformation
    --> readChunksAsSlices<'T> name outputDir suffix
#endif
