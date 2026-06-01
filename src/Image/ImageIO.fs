module ImageIO

open System
open System.Buffers
open System.IO
open System.Runtime.InteropServices
open BitMiracle.LibTiff.Classic
open Image
open Image.InternalHelpers
open SixLabors.ImageSharp.PixelFormats

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

let usePooledImageBackend () =
    match Environment.GetEnvironmentVariable("STACKPROCESSING_IMAGE_BACKEND") with
    | null -> false
    | value -> value.Equals("arraypool", StringComparison.OrdinalIgnoreCase)

let private isTiffFileName (filename: string) =
    let ext = Path.GetExtension(filename).ToLowerInvariant()
    ext = ".tif" || ext = ".tiff" || ext = ".btf" || ext = ".bigtiff"

let private isImageSharpFileName (filename: string) =
    match Path.GetExtension(filename).ToLowerInvariant() with
    | ".png"
    | ".jpg"
    | ".jpeg"
    | ".bmp"
    | ".gif"
    | ".tga"
    | ".webp" -> true
    | _ -> false

let private tryReadImageSharpPooled<'T when 'T: equality> (filename: string) (name: string) (index: int) : Image<'T> option =
    if not (usePooledImageBackend()) || isTiffFileName filename || not (isImageSharpFileName filename) then
        None
    elif typeof<'T> = typeof<uint8> then
        use image = SixLabors.ImageSharp.Image.Load<L8>(filename)
        let width = image.Width
        let height = image.Height
        let pixelCount = width * height
        let buffer = ArrayPool<uint8>.Shared.Rent(pixelCount)
        try
            image.ProcessPixelRows(fun accessor ->
                for y in 0 .. height - 1 do
                    let row = accessor.GetRowSpan(y)
                    for x in 0 .. width - 1 do
                        buffer[y * width + x] <- row[x].PackedValue)
            Image<uint8>.ofPooled1D(buffer, pixelCount, [ uint width; uint height ], name, index)
            |> box
            |> unbox<Image<'T>>
            |> Some
        with
        | _ ->
            ArrayPool<uint8>.Shared.Return(buffer)
            reraise()
    elif typeof<'T> = typeof<uint16> then
        use image = SixLabors.ImageSharp.Image.Load<L16>(filename)
        let width = image.Width
        let height = image.Height
        let pixelCount = width * height
        let buffer = ArrayPool<uint16>.Shared.Rent(pixelCount)
        try
            image.ProcessPixelRows(fun accessor ->
                for y in 0 .. height - 1 do
                    let row = accessor.GetRowSpan(y)
                    for x in 0 .. width - 1 do
                        buffer[y * width + x] <- row[x].PackedValue)
            Image<uint16>.ofPooled1D(buffer, pixelCount, [ uint width; uint height ], name, index)
            |> box
            |> unbox<Image<'T>>
            |> Some
        with
        | _ ->
            ArrayPool<uint16>.Shared.Return(buffer)
            reraise()
    else
        None

let readSimpleItkSlice<'T when 'T: equality>
    (filename: string)
    (dimension: int)
    (width: uint)
    (height: uint)
    (sourceIndex: int)
    (name: string)
    (index: int)
    =

    match if dimension = 2 then tryReadImageSharpPooled<'T> filename name index else None with
    | Some image -> image
    | None ->
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
    elif t = typeof<float32> then 32, SampleFormat.IEEEFP, 4
    elif t = typeof<float> then 64, SampleFormat.IEEEFP, 8
    else
        invalidArg "T" $"TIFF scalar IO currently supports UInt8, Int8, UInt16, Int16, Float32, and Float64 images; got {t.Name}."

let supportsDirectTiffRead<'T> =
    let t = typeof<'T>
    t = typeof<uint8>
    || t = typeof<int8>
    || t = typeof<uint16>
    || t = typeof<int16>
    || t = typeof<float32>
    || t = typeof<float>

let supportsDirectTiffWrite<'T> =
    let t = typeof<'T>
    t = typeof<uint8>
    || t = typeof<int8>
    || t = typeof<uint16>
    || t = typeof<int16>
    || t = typeof<float32>

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
    | SampleFormat.IEEEFP, 32 -> 4
    | SampleFormat.IEEEFP, 64 -> 8
    | _ ->
        invalidOp $"TIFF scalar IO currently supports UInt8, Int8, UInt16, Int16, Float32, and Float64 pages; got {bitsPerSample}-bit {sampleFormat}."

let validateTiffSamples samplesPerPixel =
    if samplesPerPixel <> 1 then
        invalidOp $"TIFF scalar IO expects one sample per pixel; got {samplesPerPixel} samples per pixel."

let private setImportImageBufferFromTiffLayout (importer: itk.simple.ImportImageFilter) bitsPerSample sampleFormat (buffer: IntPtr) =
    match sampleFormat, bitsPerSample with
    | SampleFormat.UINT, 8 -> importer.SetBufferAsUInt8(buffer)
    | SampleFormat.INT, 8 -> importer.SetBufferAsInt8(buffer)
    | SampleFormat.UINT, 16 -> importer.SetBufferAsUInt16(buffer)
    | SampleFormat.INT, 16 -> importer.SetBufferAsInt16(buffer)
    | SampleFormat.IEEEFP, 32 -> importer.SetBufferAsFloat(buffer)
    | SampleFormat.IEEEFP, 64 -> importer.SetBufferAsDouble(buffer)
    | _ ->
        invalidOp $"TIFF scalar IO currently supports UInt8, Int8, UInt16, Int16, Float32, and Float64 pages; got {bitsPerSample}-bit {sampleFormat}."

let bytesOfScalarImage2D<'T when 'T: equality> (image: Image<'T>) =
    if image.GetDimensions() <> 2u then
        invalidArg "image" $"Expected a 2D image, got {image.GetDimensions()}D."

    let width = int (image.GetWidth())
    let height = int (image.GetHeight())
    let byteCount = width * height * scalarComponentByteSize<'T>
    let bytes = Array.zeroCreate<byte> byteCount
    match image.TryGetPooled1D() with
    | Some(buffer, logicalLength, _, _) when logicalLength >= width * height ->
        Buffer.BlockCopy(buffer, 0, bytes, 0, byteCount)
    | _ ->
        let pixels = copyScalarPixels<'T> (image.toSimpleITK()) (width * height)
        Buffer.BlockCopy(pixels, 0, bytes, 0, byteCount)
    bytes

let readTiffPage<'T when 'T: equality> (tiff: Tiff) width height bitsPerSample sampleFormat bytesPerSample index =
    let rowBytes = int width * bytesPerSample
    let scanlineSize = max rowBytes (tiff.ScanlineSize())
    let scanlineBuffer = ArrayPool<byte>.Shared.Rent(scanlineSize)
    let expectedBits, expectedFormat, _ = tiffPixelLayout<'T> ()
    try
        if usePooledImageBackend() && bitsPerSample = expectedBits && sampleFormat = expectedFormat then
            let pixelCount = int width * int height
            let typedBuffer = ArrayPool<'T>.Shared.Rent(pixelCount)
            try
                for row in 0 .. int height - 1 do
                    if not (tiff.ReadScanline(scanlineBuffer, row)) then
                        invalidOp $"Failed to read TIFF scanline {row} from page {index}."

                    Buffer.BlockCopy(scanlineBuffer, 0, typedBuffer, row * rowBytes, rowBytes)

                Image<'T>.ofPooled1D(typedBuffer, pixelCount, [ width; height ], $"readVolume[{index}]", index)
            with
            | _ ->
                ArrayPool<'T>.Shared.Return(typedBuffer)
                reraise()
        else
            let pageBuffer = Array.zeroCreate<byte> (rowBytes * int height)
            for row in 0 .. int height - 1 do
                if not (tiff.ReadScanline(scanlineBuffer, row)) then
                    invalidOp $"Failed to read TIFF scanline {row} from page {index}."

                Buffer.BlockCopy(scanlineBuffer, 0, pageBuffer, row * rowBytes, rowBytes)

            use importer = new itk.simple.ImportImageFilter()
            importer.SetSize([ width; height ] |> toVectorUInt32)

            let handle = GCHandle.Alloc(pageBuffer, GCHandleType.Pinned)
            try
                setImportImageBufferFromTiffLayout importer bitsPerSample sampleFormat (handle.AddrOfPinnedObject())
                use imported = importer.Execute()
                Image<'T>.ofSimpleITK(imported, $"readVolume[{index}]", index)
            finally
                handle.Free()
    finally
        ArrayPool<byte>.Shared.Return(scanlineBuffer)

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

    let rowBytes = int width * bytesPerSample
    let rowBuffer = ArrayPool<byte>.Shared.Rent(rowBytes)
    try
        match image.TryGetPooled1D() with
        | Some(buffer, logicalLength, _, _) when logicalLength >= int width * int height ->
            for row in 0 .. int height - 1 do
                Buffer.BlockCopy(buffer, row * rowBytes, rowBuffer, 0, rowBytes)
                if not (tiff.WriteScanline(rowBuffer, int row)) then
                    invalidOp $"Failed to write TIFF scanline {row}."
        | _ ->
            let pageBytes = bytesOfScalarImage2D image
            for row in 0 .. int height - 1 do
                Buffer.BlockCopy(pageBytes, row * rowBytes, rowBuffer, 0, rowBytes)
                if not (tiff.WriteScanline(rowBuffer, int row)) then
                    invalidOp $"Failed to write TIFF scanline {row}."
    finally
        ArrayPool<byte>.Shared.Return(rowBuffer)

    if not (tiff.WriteDirectory()) then
        invalidOp "Failed to write TIFF directory."

let writeTiffSliceFile<'T when 'T: equality> (fileName: string) (image: Image<'T>) =
    use tiff = Tiff.Open(fileName, tiffWriteMode fileName)
    if isNull tiff then
        invalidOp $"Could not open '{fileName}' for TIFF slice writing."

    writeTiffPage tiff image None
