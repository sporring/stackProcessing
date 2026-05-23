module StackIO

open SlimPipeline // Core processing model
open System
open System.IO
open System.Reflection
open System.Runtime.InteropServices
open System.Text.RegularExpressions
open System.Threading
open System.Threading.Tasks
open BitMiracle.LibTiff.Classic
open FSharp.Control
open StackCore
open Image.InternalHelpers
open PureHDF
open PureHDF.Selections
open ZarrNET.Core
open ZarrNET.Core.Nodes
open ZarrNET.Core.OmeZarr.Coordinates

type FileInfo = ImageFunctions.FileInfo
type ChunkInfo = { chunks: int list; size: uint64 list; topLeftInfo: FileInfo}

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
    elif t = typeof<System.Numerics.Complex> then "Complex"
    else t.Name

let private friendlyImageTypeName (image: Image<'T>) =
    let t = typeof<'T>
    if t.IsGenericType && t.GetGenericTypeDefinition() = typedefof<list<_>> then
        let elementType = t.GetGenericArguments()[0]
        let elementName = friendlyScalarTypeName elementType
        let components = image.GetNumberOfComponentsPerPixel()

        if elementType = typeof<uint8> && components = 3u then
            "Color"
        else
            $"{elementName} vector[{components}]"
    else
        friendlyScalarTypeName t

let private imageIoCost<'T> kind evaluation calibrationKey bytes ops : StageTimeCostModel =
    StackProcessingCost.imageIoCost<'T> kind evaluation calibrationKey bytes ops

let private fixedImageOperatorTimeCost<'T> operator evaluation voxels fallback =
    let context _ =
        StackProcessingCost.Fitting.OperatorEstimateContext.create
            operator
            (Some(StackProcessingCost.pixelTypeName<'T>))
            (Some voxels)
            (Some(StackProcessingCost.imageBytes<'T> voxels))
            None
            None

    StackProcessingCost.Fitting.OperatorCostRuntime.timeCostModel evaluation context fallback

let private withCostModel costModel stage =
    StackProcessingCost.withCostModel costModel stage

let private identityStage name =
    Stage.map name (fun _ value -> value) id id

let private cleanStage name cleanup =
    { identityStage name with Cleaning = [ cleanup ] }

let private tiffPixelLayout<'T> () =
    let t = typeof<'T>
    if t = typeof<uint8> then 8, SampleFormat.UINT, 1
    elif t = typeof<int8> then 8, SampleFormat.INT, 1
    elif t = typeof<uint16> then 16, SampleFormat.UINT, 2
    elif t = typeof<int16> then 16, SampleFormat.INT, 2
    elif t = typeof<float32> then 32, SampleFormat.IEEEFP, 4
    elif t = typeof<float> then 64, SampleFormat.IEEEFP, 8
    else
        invalidArg "T" $"TIFF volume streaming currently supports UInt8, Int8, UInt16, Int16, Float32, and Float64 images; got {t.Name}."

let private runTask (task: Task<'T>) : 'T =
    task.GetAwaiter().GetResult()

let private runUnitTask (task: Task) : unit =
    task.GetAwaiter().GetResult()

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

let private zarrDataType<'T> () =
    if typeof<'T> = typeof<uint8> then
        "uint8"
    elif typeof<'T> = typeof<uint16> then
        "uint16"
    else
        failwith $"ZarrNET image IO currently supports UInt8 and UInt16 scalar images, but was {typeof<'T>.Name}."

let private nullableParallelChunks maxParallelChunks =
    if maxParallelChunks > 0 then
        Nullable<int>(maxParallelChunks)
    else
        Nullable<int>()

let private bytesOfScalarImage2D<'T when 'T: equality> (image: Image<'T>) =
    if image.GetDimensions() <> 2u then
        invalidArg "image" $"Expected a 2D image, got {image.GetDimensions()}D."

    let width = int (image.GetWidth())
    let height = int (image.GetHeight())
    let byteCount = width * height * scalarComponentByteSize<'T>
    let pixels = copyScalarPixels<'T> image.Image (width * height)
    let bytes = Array.zeroCreate<byte> byteCount
    Buffer.BlockCopy(pixels, 0, bytes, 0, byteCount)
    bytes

let private array3DOfZarrBytes<'T> (width: int) (height: int) (depth: int) (bytes: byte[]) =
    if typeof<'T> = typeof<uint8> then
        let arr =
            Array3D.init width height depth (fun x y z ->
                bytes[(z * height + y) * width + x] |> box |> unbox<'T>)

        arr
    elif typeof<'T> = typeof<uint16> then
        let arr =
            Array3D.init width height depth (fun x y z ->
                let offset = ((z * height + y) * width + x) * 2
                let value = uint16 bytes[offset] ||| (uint16 bytes[offset + 1] <<< 8)
                value |> box |> unbox<'T>)

        arr
    else
        zarrDataType<'T> () |> ignore
        failwith "unreachable"

let private zarrSlabImageAs<'T when 'T: equality> (dataType: string) width height depth (bytes: byte[]) name =
    let castNative (nativeImage: Image<'Native>) =
        let cast = nativeImage.castTo<'T>()
        nativeImage.decRefCount()
        cast

    if String.Equals(dataType, "uint8", StringComparison.OrdinalIgnoreCase) then
        array3DOfZarrBytes<uint8> width height depth bytes
        |> fun arr -> Image<uint8>.ofArray3D(arr, name)
        |> castNative
    elif String.Equals(dataType, "uint16", StringComparison.OrdinalIgnoreCase) then
        array3DOfZarrBytes<uint16> width height depth bytes
        |> fun arr -> Image<uint16>.ofArray3D(arr, name)
        |> castNative
    else
        failwith $"ZarrNET image IO currently supports UInt8 and UInt16 scalar datasets, but dataset type was {dataType}."

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
    let source = dataset.Read<'Native[,,]>(selection, AllSelection(), blocks) :> Array
    let arr =
        Array3D.init sizeX sizeY zCount (fun x y z ->
            let indices = Array.zeroCreate<int> rank
            indices[frameAxis] <- z
            indices[yAxis] <- y
            indices[xAxis] <- x
            source.GetValue(indices) |> unbox<'Native>)
    let nativeImage = Image<'Native>.ofArray3D(arr, name)
    let cast = nativeImage.castTo<'T>()
    nativeImage.decRefCount()
    cast

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

let getStackDepth (inputDir: string) (suffix: string) : uint =
    let files = getStackFiles inputDir suffix
    files.Length |> uint

let getStackInfo (inputDir: string) (suffix: string) : FileInfo =
    let files = getStackFiles inputDir suffix
    let depth = files.Length |> uint64
    if depth = 0uL then
        stopWithInputError $"No {suffixDescription suffix} files found in input stack directory: {inputDir}"
    let fi = ImageFunctions.getFileInfo(files[0])
    {fi with dimensions = fi.dimensions+1u; size = fi.size @ [depth]}

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

let readFilesWithShape<'T when 'T: equality> (debug: bool) (width: uint) (height: uint) : Stage<string, Image<'T>> =
    let name = "readFiles"
    if debug && DebugLevel.current() >= 2u then printfn $"[{name} cast to {typeof<'T>.Name}]"
    let mutable width = width
    let mutable height = height

    let mapper (debug: bool) (sliceIndex: int64) (fileName: string) : Image<'T> = 
        if debug then printfn "[%s] Reading image named %s as %s" name fileName (typeof<'T>.Name)
        let image = Image<'T>.ofFile fileName
        image.index <- int sliceIndex
        if width = 0u then
            width <- image.GetWidth()
            height <- image.GetHeight()
        image

    let memoryNeed = fun _ -> Image<'T>.memoryEstimate width height
    let elementTransformation _ = uint64 width * uint64 height

    let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed
    let fallbackTimeCostModel =
        imageIoCost<'T>
            "read"
            Map
            $"readFiles.{typeof<'T>.Name}"
            (fun _ -> Image<'T>.memoryEstimate width height)
            (fun _ -> 1UL)
    let timeCostModel =
        if width > 0u && height > 0u then
            fixedImageOperatorTimeCost<'T>
                "Read"
                Map
                (uint64 width * uint64 height)
                fallbackTimeCostModel.Estimate
        else
            fallbackTimeCostModel
    Stage.mapi name mapper memoryNeed elementTransformation
    |> withCostModel (StageCostModel.create memoryModel timeCostModel)

let readFiles<'T when 'T: equality> (debug: bool) : Stage<string, Image<'T>> =
    readFilesWithShape<'T> debug 0u 0u

let private imageFileReaderInfo filename =
    let reader = new itk.simple.ImageFileReader()
    reader.SetFileName(filename)
    reader.ReadImageInformation()
    reader

let private isTiffVolumePath (filename: string) =
    match Path.GetExtension(filename).ToLowerInvariant() with
    | ".tif"
    | ".tiff"
    | ".btf"
    | ".bigtiff" -> true
    | _ -> false

let private tiffFieldInt (tiff: Tiff) tag fallback =
    let values = tiff.GetField(tag)
    if isNull values || values.Length = 0 then fallback else values[0].ToInt()

let private tiffFieldIntDefaulted (tiff: Tiff) tag fallback =
    let values = tiff.GetFieldDefaulted(tag)
    if isNull values || values.Length = 0 then fallback else values[0].ToInt()

let private tiffDirectoryCount filename =
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

let private setImportImageBuffer<'T> (importer: itk.simple.ImportImageFilter) (buffer: IntPtr) =
    let t = typeof<'T>
    if t = typeof<uint8> then
        importer.SetBufferAsUInt8(buffer)
    elif t = typeof<int8> then
        importer.SetBufferAsInt8(buffer)
    elif t = typeof<uint16> then
        importer.SetBufferAsUInt16(buffer)
    elif t = typeof<int16> then
        importer.SetBufferAsInt16(buffer)
    elif t = typeof<float32> then
        importer.SetBufferAsFloat(buffer)
    elif t = typeof<float> then
        importer.SetBufferAsDouble(buffer)
    else
        invalidArg "T" $"readVolume TIFF streaming currently supports UInt8, Int8, UInt16, Int16, Float32, and Float64 images; got {t.Name}."

let private setImportImageBufferFromTiffLayout (importer: itk.simple.ImportImageFilter) bitsPerSample sampleFormat (buffer: IntPtr) =
    match sampleFormat, bitsPerSample with
    | SampleFormat.UINT, 8 -> importer.SetBufferAsUInt8(buffer)
    | SampleFormat.INT, 8 -> importer.SetBufferAsInt8(buffer)
    | SampleFormat.UINT, 16 -> importer.SetBufferAsUInt16(buffer)
    | SampleFormat.INT, 16 -> importer.SetBufferAsInt16(buffer)
    | SampleFormat.IEEEFP, 32 -> importer.SetBufferAsFloat(buffer)
    | SampleFormat.IEEEFP, 64 -> importer.SetBufferAsDouble(buffer)
    | _ ->
        invalidOp $"readVolume TIFF streaming currently supports UInt8, Int8, UInt16, Int16, Float32, and Float64 scalar pages; got {bitsPerSample}-bit {sampleFormat}."

let private tiffBytesPerSample bitsPerSample sampleFormat =
    match sampleFormat, bitsPerSample with
    | SampleFormat.UINT, 8
    | SampleFormat.INT, 8 -> 1
    | SampleFormat.UINT, 16
    | SampleFormat.INT, 16 -> 2
    | SampleFormat.IEEEFP, 32 -> 4
    | SampleFormat.IEEEFP, 64 -> 8
    | _ ->
        invalidOp $"readVolume TIFF streaming currently supports UInt8, Int8, UInt16, Int16, Float32, and Float64 scalar pages; got {bitsPerSample}-bit {sampleFormat}."

let private validateTiffSamples samplesPerPixel =
    if samplesPerPixel <> 1 then
        invalidOp $"readVolume TIFF streaming expects scalar pages with one sample per pixel; got {samplesPerPixel} samples per pixel."

let private readTiffPage<'T when 'T: equality> (tiff: Tiff) width height bitsPerSample sampleFormat bytesPerSample index =
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
        use cast = new itk.simple.CastImageFilter()
        cast.SetOutputPixelType(fromType<'T>)
        let itkImage = cast.Execute(imported)
        Image<'T>.ofSimpleITK(itkImage, $"readVolume[{index}]", index)
    finally
        handle.Free()

let private readTiffVolume<'T when 'T: equality> (filename: string) (pl: Plan<unit, unit>) : Plan<unit, Image<'T>> =
    use header = Tiff.Open(filename, "r")
    if isNull header then
        invalidOp $"Could not open '{filename}' for TIFF volume reading."

    let width = uint (tiffFieldInt header TiffTag.IMAGEWIDTH 0)
    let height = uint (tiffFieldInt header TiffTag.IMAGELENGTH 0)
    if width = 0u || height = 0u then
        invalidOp $"TIFF volume '{filename}' has invalid page dimensions {width}x{height}."

    let bitsPerSample = tiffFieldIntDefaulted header TiffTag.BITSPERSAMPLE 1
    let samplesPerPixel = tiffFieldIntDefaulted header TiffTag.SAMPLESPERPIXEL 1
    let sampleFormat =
        let raw = tiffFieldIntDefaulted header TiffTag.SAMPLEFORMAT (int SampleFormat.UINT)
        enum<SampleFormat> raw

    validateTiffSamples samplesPerPixel
    tiffPixelLayout<'T> () |> ignore
    let bytesPerSample = tiffBytesPerSample bitsPerSample sampleFormat

    let depth = tiffDirectoryCount filename
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

                    let pageWidth = uint (tiffFieldInt reader TiffTag.IMAGEWIDTH 0)
                    let pageHeight = uint (tiffFieldInt reader TiffTag.IMAGELENGTH 0)
                    if pageWidth <> width || pageHeight <> height then
                        invalidOp $"readVolume expected all TIFF pages to be {width}x{height}; page {index} is {pageWidth}x{pageHeight}."

                    yield readTiffPage<'T> reader width height bitsPerSample sampleFormat bytesPerSample index

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
    use infoReader = imageFileReaderInfo filename
    let dimension = int (infoReader.GetDimension())
    if dimension < 2 || dimension > 3 then
        invalidArg "filename" $"readVolume expects a 2D or 3D image volume, got {dimension} dimensions in '{filename}'."

    let size = infoReader.GetSize() |> fromVectorUInt64 |> List.map uint
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

        use reader = new itk.simple.ImageFileReader()
        reader.SetFileName(filename)
        if dimension = 3 then
            reader.SetExtractIndex([ 0; 0; index ] |> toVectorInt32)
            reader.SetExtractSize([ width; height; 0u ] |> toVectorUInt32)

        let itkImage = reader.Execute()
        Image<'T>.ofSimpleITK(itkImage, $"readVolume[{index}]", index)

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

let private readTiffVolumeRandom<'T when 'T: equality> (count: uint) (filename: string) (pl: Plan<unit, unit>) : Plan<unit, Image<'T>> =
    use header = Tiff.Open(filename, "r")
    if isNull header then
        invalidOp $"Could not open '{filename}' for TIFF volume reading."

    let width = uint (tiffFieldInt header TiffTag.IMAGEWIDTH 0)
    let height = uint (tiffFieldInt header TiffTag.IMAGELENGTH 0)
    if width = 0u || height = 0u then
        invalidOp $"TIFF volume '{filename}' has invalid page dimensions {width}x{height}."

    let bitsPerSample = tiffFieldIntDefaulted header TiffTag.BITSPERSAMPLE 1
    let samplesPerPixel = tiffFieldIntDefaulted header TiffTag.SAMPLESPERPIXEL 1
    let sampleFormat =
        let raw = tiffFieldIntDefaulted header TiffTag.SAMPLEFORMAT (int SampleFormat.UINT)
        enum<SampleFormat> raw

    validateTiffSamples samplesPerPixel
    tiffPixelLayout<'T> () |> ignore
    let bytesPerSample = tiffBytesPerSample bitsPerSample sampleFormat
    let sourceDepth = tiffDirectoryCount filename |> int
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

        let pageWidth = uint (tiffFieldInt reader TiffTag.IMAGEWIDTH 0)
        let pageHeight = uint (tiffFieldInt reader TiffTag.IMAGELENGTH 0)
        if pageWidth <> width || pageHeight <> height then
            invalidOp $"readVolumeRandom expected all TIFF pages to be {width}x{height}; page {sourceIndex} is {pageWidth}x{pageHeight}."

        readTiffPage<'T> reader width height bitsPerSample sampleFormat bytesPerSample sourceIndex

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
    use infoReader = imageFileReaderInfo filename
    let dimension = int (infoReader.GetDimension())
    if dimension < 2 || dimension > 3 then
        invalidArg "filename" $"readVolumeRandom expects a 2D or 3D image volume, got {dimension} dimensions in '{filename}'."

    let size = infoReader.GetSize() |> fromVectorUInt64 |> List.map uint
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

        use reader = new itk.simple.ImageFileReader()
        reader.SetFileName(filename)
        if dimension = 3 then
            reader.SetExtractIndex([ 0; 0; sourceIndex ] |> toVectorInt32)
            reader.SetExtractSize([ width; height; 0u ] |> toVectorUInt32)

        let itkImage = reader.Execute()
        Image<'T>.ofSimpleITK(itkImage, $"readVolumeRandom[{sourceIndex}]", sourceIndex)

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

    let width = uint (tiffFieldInt header TiffTag.IMAGEWIDTH 0)
    let height = uint (tiffFieldInt header TiffTag.IMAGELENGTH 0)
    if width = 0u || height = 0u then
        invalidOp $"TIFF volume '{filename}' has invalid page dimensions {width}x{height}."

    let bitsPerSample = tiffFieldIntDefaulted header TiffTag.BITSPERSAMPLE 1
    let samplesPerPixel = tiffFieldIntDefaulted header TiffTag.SAMPLESPERPIXEL 1
    let sampleFormat =
        let raw = tiffFieldIntDefaulted header TiffTag.SAMPLEFORMAT (int SampleFormat.UINT)
        enum<SampleFormat> raw

    validateTiffSamples samplesPerPixel
    tiffPixelLayout<'T> () |> ignore
    let bytesPerSample = tiffBytesPerSample bitsPerSample sampleFormat
    let sourceDepth = tiffDirectoryCount filename |> int
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

        let pageWidth = uint (tiffFieldInt reader TiffTag.IMAGEWIDTH 0)
        let pageHeight = uint (tiffFieldInt reader TiffTag.IMAGELENGTH 0)
        if pageWidth <> width || pageHeight <> height then
            invalidOp $"readVolumeRange expected all TIFF pages to be {width}x{height}; page {sourceIndex} is {pageWidth}x{pageHeight}."

        readTiffPage<'T> reader width height bitsPerSample sampleFormat bytesPerSample sourceIndex

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
    use infoReader = imageFileReaderInfo filename
    let dimension = int (infoReader.GetDimension())
    if dimension < 2 || dimension > 3 then
        invalidArg "filename" $"readVolumeRange expects a 2D or 3D image volume, got {dimension} dimensions in '{filename}'."

    let size = infoReader.GetSize() |> fromVectorUInt64 |> List.map uint
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

        use reader = new itk.simple.ImageFileReader()
        reader.SetFileName(filename)
        if dimension = 3 then
            reader.SetExtractIndex([ 0; 0; sourceIndex ] |> toVectorInt32)
            reader.SetExtractSize([ width; height; 0u ] |> toVectorUInt32)

        let itkImage = reader.Execute()
        Image<'T>.ofSimpleITK(itkImage, $"readVolumeRange[{sourceIndex}]", sourceIndex)

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
    >=> readFilesWithShape<'T> pl.debug width height
    |> Plan.withSourcePeek sourcePeek

let read<'T when 'T: equality> (inputDir: string) (suffix: string) (pl: Plan<unit, unit>) : Plan<unit, Image<'T>> =
    readFiltered<'T> inputDir suffix Array.sort pl

let readRandom<'T when 'T: equality> (count: uint) (inputDir: string) (suffix: string) (pl: Plan<unit, unit>) : Plan<unit, Image<'T>> =
    readFiltered<'T> inputDir suffix (Array.randomChoices (int count)) pl

let private rangeFilter first step last files =
    let sorted = Array.sort files
    rangeIndices first step last sorted.Length
    |> Array.map (fun index -> sorted[index])

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
    >=> readFilesWithShape<'T> pl.debug width height
    |> Plan.withSourcePeek sourcePeek

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
    let topLeftFi = ImageFunctions.getFileInfo topLeft
    let bottomRightFi = ImageFunctions.getFileInfo bottomRight

    let stackSize = 
        [
            (uint64 maxI) * topLeftFi.size[0] + bottomRightFi.size[0];
            (uint64 maxJ) * topLeftFi.size[1] + bottomRightFi.size[1];
            (uint64 maxK) * topLeftFi.size[2] + bottomRightFi.size[2];
        ]
    { chunks = [maxI+1;maxJ+1;maxK+1]; topLeftInfo = topLeftFi; size = stackSize }

let getZarrInfo (path: string) (multiscaleIndex: int) (datasetIndex: int) : ChunkInfo =
    suppressZarrNetDebugLogging ()

    let reader: OmeZarrReader =
        OmeZarrReader.OpenAsync(path, ct = CancellationToken.None)
        |> runTask

    let level =
        reader.AsMultiscaleImage().OpenResolutionLevelAsync(multiscaleIndex, datasetIndex, CancellationToken.None)
        |> runTask

    let _sizeT, _sizeC, sizeZ, sizeY, sizeX = zarrShapeTCZYX level.Shape
    let chunks =
        reader.RootGroup.OpenArrayAsync(level.Dataset.Path, CancellationToken.None)
        |> runTask
        |> fun zarrArray -> zarrArray.Metadata.ChunkShape
        |> Array.toList

    deleteZarrNetDebugLogs ()

    let fileInfo: FileInfo =
        { dimensions = 3u
          size = [ uint64 sizeX; uint64 sizeY; uint64 sizeZ ]
          componentType = level.DataType
          numberOfComponents = 1u }

    { chunks = chunks
      size = fileInfo.size
      topLeftInfo = fileInfo }

let getNexusInfo (path: string) (datasetPath: string) (frameAxis: int) (yAxis: int) (xAxis: int) : ChunkInfo =
    use file = H5File.OpenRead(path)
    let dataset = file.Dataset(datasetPath)
    let rank = int dataset.Space.Rank
    validateHdfAxes rank frameAxis yAxis xAxis

    if rank <> 3 then
        failwith $"getNexusInfo currently expects a rank-3 detector stack dataset, but {datasetPath} has rank {rank}."

    let dimensions = dataset.Space.Dimensions
    let chunks = hdfDatasetChunks dataset
    let fileInfo: FileInfo =
        { dimensions = 3u
          size = [ uint64 dimensions[xAxis]; uint64 dimensions[yAxis]; uint64 dimensions[frameAxis] ]
          componentType = dataset.Type.ToString()
          numberOfComponents = 1u }

    { chunks = chunks
      size = fileInfo.size
      topLeftInfo = fileInfo }

let getChunkFilename (path: string) (suffix: string) (i: int) (j: int) (k: int) =
    Path.Combine(path, sprintf "chunk%d_%d_%d%s" i j k suffix)

let _readChunk<'T when 'T: equality>  (inputDir: string) (suffix: string) i j k = 
    let filename = getChunkFilename inputDir suffix i j k
    if typeof<'T> = typeof<System.Numerics.Complex> then
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
        if typeof<'T> = typeof<System.Numerics.Complex> then 1u
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
    let memoryNeed = fun _ -> 256UL
    let elementTransformation = id
    let unstackSlab (slab: Image<'T>) =
        let result = ImageFunctions.unstack 2u slab
        slab.decRefCount()
        result

    pl
    |> readSlabStacked<'T> inputDir suffix
    >=> Stage.map $"readSlabAsWindows.{typeof<'T>.Name}" (fun _ slab -> unstackSlab slab) memoryNeed elementTransformation

let readSlab<'T when 'T: equality> (inputDir: string) (suffix: string) (pl: Plan<unit, unit>) : Plan<unit, Image<'T>> =
    pl |> readSlabAsWindows<'T> inputDir suffix >=> flattenList ()

let readZarrSlabStacked<'T when 'T: equality>
    (path: string)
    (slabDepth: uint)
    (multiscaleIndex: int)
    (datasetIndex: int)
    (timepoint: int)
    (channel: int)
    (maxParallelChunks: int)
    (pl: Plan<unit, unit>)
    : Plan<unit, Image<'T>> =

    suppressZarrNetDebugLogging ()

    let name = "readZarrSlabStacked"
    Image.InternalHelpers.fromType<'T> |> ignore
    let level = openZarrResolutionLevel path multiscaleIndex datasetIndex
    let sizeT, sizeC, sizeZ, sizeY, sizeX = zarrShapeTCZYX level.Shape

    if timepoint < 0 || timepoint >= sizeT then
        invalidArg "timepoint" $"Timepoint {timepoint} is outside the Zarr time range 0..{sizeT - 1}."
    if channel < 0 || channel >= sizeC then
        invalidArg "channel" $"Channel {channel} is outside the Zarr channel range 0..{sizeC - 1}."
    if not (String.Equals(level.DataType, "uint8", StringComparison.OrdinalIgnoreCase)
            || String.Equals(level.DataType, "uint16", StringComparison.OrdinalIgnoreCase)) then
        failwith $"ZarrNET image IO currently supports UInt8 and UInt16 scalar datasets, but dataset type was {level.DataType}."

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

let readZarrSlab<'T when 'T: equality>
    (path: string)
    (slabDepth: uint)
    (multiscaleIndex: int)
    (datasetIndex: int)
    (timepoint: int)
    (channel: int)
    (maxParallelChunks: int)
    (pl: Plan<unit, unit>)
    : Plan<unit, Image<'T>> =

    let memoryNeed = fun _ -> 256UL
    let elementTransformation = id
    let unstackSlab (slab: Image<'T>) =
        let result = ImageFunctions.unstack 2u slab
        slab.decRefCount()
        result

    pl
    |> readZarrSlabStacked<'T> path slabDepth multiscaleIndex datasetIndex timepoint channel maxParallelChunks
    >=> Stage.map $"readZarrSlab.{typeof<'T>.Name}" (fun _ slab -> unstackSlab slab) memoryNeed elementTransformation
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
    Image.InternalHelpers.fromType<'T> |> ignore
    let level = openZarrResolutionLevel path multiscaleIndex datasetIndex
    let sizeT, sizeC, sizeZ, sizeY, sizeX = zarrShapeTCZYX level.Shape

    if timepoint < 0 || timepoint >= sizeT then
        invalidArg "timepoint" $"Timepoint {timepoint} is outside the Zarr time range 0..{sizeT - 1}."
    if channel < 0 || channel >= sizeC then
        invalidArg "channel" $"Channel {channel} is outside the Zarr channel range 0..{sizeC - 1}."
    if not (String.Equals(level.DataType, "uint8", StringComparison.OrdinalIgnoreCase)
            || String.Equals(level.DataType, "uint16", StringComparison.OrdinalIgnoreCase)) then
        failwith $"ZarrNET image IO currently supports UInt8 and UInt16 scalar datasets, but dataset type was {level.DataType}."

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
    Image.InternalHelpers.fromType<'T> |> ignore
    let level = openZarrResolutionLevel path multiscaleIndex datasetIndex
    let sizeT, sizeC, sizeZ, sizeY, sizeX = zarrShapeTCZYX level.Shape

    if timepoint < 0 || timepoint >= sizeT then
        invalidArg "timepoint" $"Timepoint {timepoint} is outside the Zarr time range 0..{sizeT - 1}."
    if channel < 0 || channel >= sizeC then
        invalidArg "channel" $"Channel {channel} is outside the Zarr channel range 0..{sizeC - 1}."
    if not (String.Equals(level.DataType, "uint8", StringComparison.OrdinalIgnoreCase)
            || String.Equals(level.DataType, "uint16", StringComparison.OrdinalIgnoreCase)) then
        failwith $"ZarrNET image IO currently supports UInt8 and UInt16 scalar datasets, but dataset type was {level.DataType}."

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

let readNexusSlabStacked<'T when 'T: equality>
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

let readNexusSlab<'T when 'T: equality>
    (path: string)
    (datasetPath: string)
    (slabDepth: uint)
    (frameAxis: int)
    (yAxis: int)
    (xAxis: int)
    (pl: Plan<unit, unit>)
    : Plan<unit, Image<'T>> =

    let memoryNeed = fun _ -> 256UL
    let elementTransformation = id
    let unstackSlab (slab: Image<'T>) =
        let result = ImageFunctions.unstack 2u slab
        slab.decRefCount()
        result

    pl
    |> readNexusSlabStacked<'T> path datasetPath slabDepth frameAxis yAxis xAxis
    >=> Stage.map $"readNexusSlab.{typeof<'T>.Name}" (fun _ slab -> unstackSlab slab) memoryNeed elementTransformation
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

let write<'T when 'T: equality> (outputDir: string) (suffix: string) : Stage<Image<'T>, Image<'T>> =
    let t = typeof<'T>
    if (icompare suffix ".tif" || icompare suffix ".tiff") 
        && not (t = typeof<uint8> || t = typeof<uint8 list> || t = typeof<int8> || t = typeof<uint16> || t = typeof<int16> || t = typeof<float32>) then
        failwith $"[write] tiff images only supports (u)int8, (u)int16 and float32 but was {t.Name}" 
    Directory.CreateDirectory(outputDir) |> ignore
    let cleaned = lazy (cleanImageSeriesFiles outputDir suffix)
    let mapper (debug: bool) (idx: int64) (image: Image<'T>) =
        cleaned.Force()
        let fileName = Path.Combine(outputDir, sprintf "image_%03d%s" idx suffix)
        if debug then printfn "[write] Saved image %d to %s as %s" idx fileName (friendlyImageTypeName image)
        image.toFile(fileName)
        image
    let memoryNeed = id
    let memoryModel = StageMemoryModel.fromSinglePeak Iter memoryNeed
    let fallbackTimeCostModel =
        imageIoCost<'T>
            "write"
            Iter
            $"write.{typeof<'T>.Name}"
            (fun input -> inputValue input |> imageBytes<'T>)
            (fun _ -> 1UL)
    let timeCostModel =
        StackProcessingCost.operatorImageTimeCost<'T>
            "Write"
            Iter
            None
            None
            fallbackTimeCostModel.Estimate
    Stage.mapi $"write \"{outputDir}/*{suffix}\"" mapper memoryNeed id
    |> withCostModel (StageCostModel.create memoryModel timeCostModel)

let private tiffMode (filename: string) =
    let ext = Path.GetExtension(filename).ToLowerInvariant()
    if ext = ".btf" || ext = ".bigtiff" then "w8" else "w"

let writeVolume<'T when 'T: equality> (filename: string) : Stage<Image<'T>, unit> =
    let extension = Path.GetExtension(filename).ToLowerInvariant()
    if extension <> ".tif" && extension <> ".tiff" && extension <> ".btf" && extension <> ".bigtiff" then
        invalidArg "filename" "writeVolume currently supports streaming multipage TIFF/BigTIFF output only."

    let bitsPerSample, sampleFormat, bytesPerSample = tiffPixelLayout<'T> ()

    let reducer (debug: bool) (input: AsyncSeq<Image<'T>>) =
        async {
            let directory = Path.GetDirectoryName(Path.GetFullPath(filename))
            if not (String.IsNullOrWhiteSpace directory) then
                Directory.CreateDirectory(directory) |> ignore

            use tiff = Tiff.Open(filename, tiffMode filename)
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

                            tiff.SetField(TiffTag.IMAGEWIDTH, int width) |> ignore
                            tiff.SetField(TiffTag.IMAGELENGTH, int height) |> ignore
                            tiff.SetField(TiffTag.SAMPLESPERPIXEL, 1) |> ignore
                            tiff.SetField(TiffTag.BITSPERSAMPLE, bitsPerSample) |> ignore
                            tiff.SetField(TiffTag.SAMPLEFORMAT, sampleFormat) |> ignore
                            tiff.SetField(TiffTag.PHOTOMETRIC, Photometric.MINISBLACK) |> ignore
                            tiff.SetField(TiffTag.PLANARCONFIG, PlanarConfig.CONTIG) |> ignore
                            tiff.SetField(TiffTag.ROWSPERSTRIP, int height) |> ignore
                            tiff.SetField(TiffTag.COMPRESSION, Compression.NONE) |> ignore
                            tiff.SetField(TiffTag.SUBFILETYPE, FileType.PAGE) |> ignore
                            tiff.SetField(TiffTag.PAGENUMBER, page, 0) |> ignore

                            let pageBytes = bytesOfScalarImage2D image
                            let rowBytes = int width * bytesPerSample

                            for row in 0 .. int height - 1 do
                                let buffer = Array.zeroCreate<byte> rowBytes
                                Buffer.BlockCopy(pageBytes, row * rowBytes, buffer, 0, rowBytes)
                                if not (tiff.WriteScanline(buffer, int row)) then
                                    invalidOp $"Failed to write TIFF scanline {row} for page {page}."

                            if not (tiff.WriteDirectory()) then
                                invalidOp $"Failed to write TIFF directory for page {page}."

                            page <- page + 1
                        finally
                            image.decRefCount()
                    })
        }

    Stage.reduce "writeVolume" reducer Streaming id (fun _ -> 1UL)

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
                PhysicalSizeZ = physicalSizeZ)

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

        let planeBytes = bytesOfScalarImage2D image
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
        let plane = image.toArray2D()
        let memDims = Array.zeroCreate<int> 3
        memDims[frameAxis] <- 1
        memDims[yAxis] <- height
        memDims[xAxis] <- width

        let data =
            Array3D.init memDims[0] memDims[1] memDims[2] (fun a b c ->
                let indices = [| a; b; c |]
                plane[indices[xAxis], indices[yAxis]])

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

let _writeSlabChunks (debug: bool) (outputDir: string) (suffix: string) (width: uint) (height: uint) (winSz: uint) (k: int) (stack: Image<'T>) =
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
            _writeSlabChunks debug outputDir suffix width height 1u (int k) slab
            slab.decRefCount()
        else
            _writeSlabChunks debug outputDir suffix width height winSz (int k) stack
        stack.incRefCount() //to make sure volFctToLstFctReleaseAfter doesn't release it.
        stack
    let mapper (debug: bool) (idx: int64) (window: Window<Image<'T>>) =
        volFctToWindowFctReleaseAfterDebug debug (f debug idx) 1u pad stride window
    //let mapper (debug: bool) (idx: int64) = fun stack -> _writeSlabChunks debug outputDir suffix width height winSz (int idx) stack; stack
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
