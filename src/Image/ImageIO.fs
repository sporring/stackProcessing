module ImageIO

open System
open System.IO
open System.Runtime.InteropServices
open BitMiracle.LibTiff.Classic
open Image
open Image.InternalHelpers

type ImageFileInfo =
    { Dimension: int
      Size: uint list }

let validatePixelType<'T> () =
    fromType<'T> |> ignore

let imageFileInfo (filename: string) =
    use reader = new itk.simple.ImageFileReader()
    reader.SetFileName(filename)
    reader.ReadImageInformation()
    { Dimension = int (reader.GetDimension())
      Size = reader.GetSize() |> fromVectorUInt64 |> List.map uint }

let readSimpleItkSlice<'T when 'T: equality>
    (filename: string)
    (dimension: int)
    (width: uint)
    (height: uint)
    (sourceIndex: int)
    (name: string)
    (index: int)
    =

    use reader = new itk.simple.ImageFileReader()
    reader.SetFileName(filename)
    if dimension = 3 then
        reader.SetExtractIndex([ 0; 0; sourceIndex ] |> toVectorInt32)
        reader.SetExtractSize([ width; height; 0u ] |> toVectorUInt32)

    let itkImage = reader.Execute()
    Image<'T>.ofSimpleITKNDispose(itkImage, name, index)

let tiffPixelLayout<'T> () =
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
        invalidArg "T" $"TIFF scalar IO currently supports UInt8, Int8, UInt16, Int16, UInt32, Int32, Float32, and Float64 images; got {t.Name}."

let supportsDirectTiffRead<'T> =
    let t = typeof<'T>
    t = typeof<uint8>
    || t = typeof<int8>
    || t = typeof<uint16>
    || t = typeof<int16>
    || t = typeof<uint32>
    || t = typeof<int32>
    || t = typeof<float32>

let supportsDirectTiffWrite<'T> =
    let t = typeof<'T>
    t = typeof<uint8>
    || t = typeof<int8>
    || t = typeof<uint16>
    || t = typeof<int16>
    || t = typeof<uint32>
    || t = typeof<int32>
    || t = typeof<float32>
    || t = typeof<float>

let tiffWriteMode (filename: string) =
    let ext = Path.GetExtension(filename).ToLowerInvariant()
    if ext = ".btf" || ext = ".bigtiff" then "w8" else "w"

let tiffFieldInt (tiff: Tiff) tag fallback =
    let field = tiff.GetField(tag)
    if isNull field || field.Length = 0 then fallback else field[0].ToInt()

let tiffFieldIntDefaulted (tiff: Tiff) tag fallback =
    let field = tiff.GetFieldDefaulted(tag)
    if isNull field || field.Length = 0 then fallback else field[0].ToInt()

let tiffDirectoryCount (filename: string) =
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

let tiffBytesPerSample bitsPerSample sampleFormat =
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

let validateTiffSamples samplesPerPixel =
    if samplesPerPixel <> 1 then
        invalidOp $"TIFF scalar IO expects one sample per pixel; got {samplesPerPixel} samples per pixel."

let private setImportImageBufferFromTiffLayout (importer: itk.simple.ImportImageFilter) bitsPerSample sampleFormat (buffer: IntPtr) =
    match sampleFormat, bitsPerSample with
    | SampleFormat.UINT, 8 -> importer.SetBufferAsUInt8(buffer)
    | SampleFormat.INT, 8 -> importer.SetBufferAsInt8(buffer)
    | SampleFormat.UINT, 16 -> importer.SetBufferAsUInt16(buffer)
    | SampleFormat.INT, 16 -> importer.SetBufferAsInt16(buffer)
    | SampleFormat.UINT, 32 -> importer.SetBufferAsUInt32(buffer)
    | SampleFormat.INT, 32 -> importer.SetBufferAsInt32(buffer)
    | SampleFormat.IEEEFP, 32 -> importer.SetBufferAsFloat(buffer)
    | SampleFormat.IEEEFP, 64 -> importer.SetBufferAsDouble(buffer)
    | _ ->
        invalidOp $"TIFF scalar IO currently supports UInt8, Int8, UInt16, Int16, UInt32, Int32, Float32, and Float64 pages; got {bitsPerSample}-bit {sampleFormat}."

let bytesOfScalarImage2D<'T when 'T: equality> (image: Image<'T>) =
    if image.GetDimensions() <> 2u then
        invalidArg "image" $"Expected a 2D image, got {image.GetDimensions()}D."

    let width = int (image.GetWidth())
    let height = int (image.GetHeight())
    let byteCount = width * height * scalarComponentByteSize<'T>
    let pixels = copyScalarPixels<'T> image.Image (width * height)
    let bytes = Array.zeroCreate<byte> byteCount
    Buffer.BlockCopy(pixels, 0, bytes, 0, byteCount)
    bytes

let readTiffPage<'T when 'T: equality> (tiff: Tiff) width height bitsPerSample sampleFormat bytesPerSample index =
    let rowBytes = int width * bytesPerSample
    let scanlineSize = max rowBytes (tiff.ScanlineSize())
    let buffer = Array.zeroCreate<byte> scanlineSize
    let pageBuffer = Array.zeroCreate<byte> (rowBytes * int height)

    for row in 0 .. int height - 1 do
        if not (tiff.ReadScanline(buffer, row)) then
            invalidOp $"Failed to read TIFF scanline {row} from page {index}."

        Buffer.BlockCopy(buffer, 0, pageBuffer, row * rowBytes, rowBytes)

    use importer = new itk.simple.ImportImageFilter()
    importer.SetSize([ width; height ] |> toVectorUInt32)

    let handle = GCHandle.Alloc(pageBuffer, GCHandleType.Pinned)
    try
        setImportImageBufferFromTiffLayout importer bitsPerSample sampleFormat (handle.AddrOfPinnedObject())
        use imported = importer.Execute()
        Image<'T>.ofSimpleITK(imported, $"readVolume[{index}]", index)
    finally
        handle.Free()

let readTiffSliceFile<'T when 'T: equality> (fileName: string) (sliceIndex: int64) =
    use tiff = Tiff.Open(fileName, "r")
    if isNull tiff then
        invalidOp $"Could not open '{fileName}' for TIFF slice reading."

    let width = uint (tiffFieldInt tiff TiffTag.IMAGEWIDTH 0)
    let height = uint (tiffFieldInt tiff TiffTag.IMAGELENGTH 0)
    if width = 0u || height = 0u then
        invalidOp $"TIFF slice '{fileName}' has invalid page dimensions {width}x{height}."

    let bitsPerSample = tiffFieldIntDefaulted tiff TiffTag.BITSPERSAMPLE 1
    let samplesPerPixel = tiffFieldIntDefaulted tiff TiffTag.SAMPLESPERPIXEL 1
    let sampleFormat =
        let raw = tiffFieldIntDefaulted tiff TiffTag.SAMPLEFORMAT (int SampleFormat.UINT)
        enum<SampleFormat> raw

    validateTiffSamples samplesPerPixel
    let bytesPerSample = tiffBytesPerSample bitsPerSample sampleFormat
    let image = readTiffPage<'T> tiff width height bitsPerSample sampleFormat bytesPerSample (int sliceIndex)
    image.index <- int sliceIndex
    image

let writeTiffPage<'T when 'T: equality> (tiff: Tiff) (image: Image<'T>) (page: int option) =
    if image.GetDimensions() <> 2u then
        invalidOp $"write TIFF expects 2D slices, got {image.GetDimensions()}D at slice {image.index}."

    let bitsPerSample, sampleFormat, bytesPerSample = tiffPixelLayout<'T> ()
    let width = image.GetWidth()
    let height = image.GetHeight()

    tiff.SetField(TiffTag.IMAGEWIDTH, int width) |> ignore
    tiff.SetField(TiffTag.IMAGELENGTH, int height) |> ignore
    tiff.SetField(TiffTag.SAMPLESPERPIXEL, 1) |> ignore
    tiff.SetField(TiffTag.BITSPERSAMPLE, bitsPerSample) |> ignore
    tiff.SetField(TiffTag.SAMPLEFORMAT, sampleFormat) |> ignore
    tiff.SetField(TiffTag.PHOTOMETRIC, Photometric.MINISBLACK) |> ignore
    tiff.SetField(TiffTag.PLANARCONFIG, PlanarConfig.CONTIG) |> ignore
    tiff.SetField(TiffTag.ROWSPERSTRIP, int height) |> ignore
    tiff.SetField(TiffTag.COMPRESSION, Compression.NONE) |> ignore
    match page with
    | Some pageNumber ->
        tiff.SetField(TiffTag.SUBFILETYPE, FileType.PAGE) |> ignore
        tiff.SetField(TiffTag.PAGENUMBER, pageNumber, 0) |> ignore
    | None -> ()

    let pageBytes = bytesOfScalarImage2D image
    let rowBytes = int width * bytesPerSample

    for row in 0 .. int height - 1 do
        let buffer = Array.zeroCreate<byte> rowBytes
        Buffer.BlockCopy(pageBytes, row * rowBytes, buffer, 0, rowBytes)
        if not (tiff.WriteScanline(buffer, int row)) then
            invalidOp $"Failed to write TIFF scanline {row}."

    if not (tiff.WriteDirectory()) then
        invalidOp "Failed to write TIFF directory."

let writeTiffSliceFile<'T when 'T: equality> (fileName: string) (image: Image<'T>) =
    use tiff = Tiff.Open(fileName, tiffWriteMode fileName)
    if isNull tiff then
        invalidOp $"Could not open '{fileName}' for TIFF slice writing."

    writeTiffPage tiff image None
