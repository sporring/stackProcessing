module StackIO

open SlimPipeline // Core processing model
open System
open System.IO
open System.Text.RegularExpressions
open System.Threading
open System.Threading.Tasks
open StackCore
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

let private imageIoCost<'T> kind evaluation calibrationKey bytes ops : StageWorkModel =
    StackProcessingCost.imageIoCost<'T> kind evaluation calibrationKey bytes ops

let private withCostModel costModel stage =
    StackProcessingCost.withCostModel costModel stage

let private runTask (task: Task<'T>) : 'T =
    task.GetAwaiter().GetResult()

let private runUnitTask (task: Task) : unit =
    task.GetAwaiter().GetResult()

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

let private bytesOfArray2D<'T> (arr: 'T[,]) =
    let width = arr.GetLength(0)
    let height = arr.GetLength(1)

    if typeof<'T> = typeof<uint8> then
        let source = unbox<uint8[,]> (box arr)
        let bytes = Array.zeroCreate<byte> (width * height)
        let mutable offset = 0

        for y in 0 .. height - 1 do
            for x in 0 .. width - 1 do
                bytes[offset] <- source[x, y]
                offset <- offset + 1

        bytes
    elif typeof<'T> = typeof<uint16> then
        let source = unbox<uint16[,]> (box arr)
        let bytes = Array.zeroCreate<byte> (width * height * 2)
        let mutable offset = 0

        for y in 0 .. height - 1 do
            for x in 0 .. width - 1 do
                let value = source[x, y]
                bytes[offset] <- byte (value &&& 0x00FFus)
                bytes[offset + 1] <- byte (value >>> 8)
                offset <- offset + 2

        bytes
    else
        zarrDataType<'T> () |> ignore
        failwith "unreachable"

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

let private openZarrResolutionLevel (path: string) multiscaleIndex datasetIndex : ResolutionLevelNode =
    let reader: OmeZarrReader =
        OmeZarrReader.OpenAsync(path, ct = CancellationToken.None)
        |> runTask

    let multiscale = reader.AsMultiscaleImage()
    multiscale.OpenResolutionLevelAsync(multiscaleIndex, datasetIndex, CancellationToken.None)
    |> runTask

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

let private getStackFiles inputDir suffix =
    let aliases = suffixAliases suffix

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
        failwith $"No {suffixDescription suffix} files found in directory: {inputDir}"
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
        if pl.debug then printfn "[%s] Supplying filename %i: %s" name i filenames[i]
        filenames[i]

    let transition = ProfileTransition.create Unit Streaming
    let memPeak = 256UL // surrugate string length
    let memoryNeed = fun _ -> memPeak
    let elementTransformation = id
    let memoryModel = StageMemoryModel.fromSinglePeak Source memoryNeed
    let workModel =
        StageWorkModel.cpu Source (Some "getFilenames") (fun _ -> float depth)
    let stage =
        Stage.init $"{name}" (uint depth) mapper transition memoryNeed elementTransformation
        |> withCostModel (StageCostModel.create memoryModel workModel)
        |> Some

    let memPerElem = 256UL // surrugate string length
    let length = depth
    Plan.create stage pl.memAvail memPeak memPerElem length pl.debug

let readFilesWithShape<'T when 'T: equality> (debug: bool) (width: uint) (height: uint) : Stage<string, Image<'T>> =
    let name = "readFiles"
    if debug then printfn $"[{name} cast to {typeof<'T>.Name}]"
    let mutable width = width
    let mutable height = height

    let mapper (debug: bool) (fileName: string) : Image<'T> = 
        if debug then printfn "[%s] Reading image named %s as %s" name fileName (typeof<'T>.Name)
        let image = Image<'T>.ofFile fileName
        if width = 0u then
            width <- image.GetWidth()
            height <- image.GetHeight()
        image

    let memoryNeed = fun _ -> Image<'T>.memoryEstimate width height
    let elementTransformation _ = uint64 width * uint64 height

    let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed
    let workModel =
        imageIoCost<'T>
            "read"
            Map
            $"readFiles.{typeof<'T>.Name}"
            (fun _ -> Image<'T>.memoryEstimate width height)
            (fun _ -> 1UL)
    Stage.map name mapper memoryNeed elementTransformation
    |> withCostModel (StageCostModel.create memoryModel workModel)

let readFiles<'T when 'T: equality> (debug: bool) : Stage<string, Image<'T>> =
    readFilesWithShape<'T> debug 0u 0u

let readFilePairs<'T when 'T: equality> (debug: bool) : Stage<string*string, Image<'T>*Image<'T>> =
    let name = "readFilePairs"
    if debug then printfn $"[{name} cast to {typeof<'T>.Name}]"
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
    let workModel =
        imageIoCost<'T>
            "read"
            Map
            $"readFilePairs.{typeof<'T>.Name}"
            (fun _ -> 2UL * Image<'T>.memoryEstimate width height)
            (fun _ -> 2UL)
    Stage.map name mapper memoryNeed elementTransformation
    |> withCostModel (StageCostModel.create memoryModel workModel)

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
    let res = Image<'T>.ofFile filename
    res

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

    let slab = Image<'T>(sz |> List.map uint, chunkInfo.topLeftInfo.numberOfComponents)
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
        _readSlabStacked<'T> inputDir suffix chunkInfo 2u k

    let transition = ProfileTransition.create Unit Streaming
    let memPeak = 256UL // surrugate string length
    let memoryNeed = fun _ -> memPeak
    let elementTransformation = id
    let memoryModel = StageMemoryModel.fromSinglePeak Source memoryNeed
    let workModel =
        StageWorkModel.ioRead Source (Some $"readSlabStacked.{typeof<'T>.Name}") (fun _ -> elementBytes) (fun _ -> 1UL)
    let stage =
        Stage.init $"{name}" (uint depth) mapper transition memoryNeed elementTransformation
        |> withCostModel (StageCostModel.create memoryModel workModel)
        |> Some

    let memPerElem = 256UL // surrugate string length
    let length = depth
    Plan.create stage pl.memAvail memPeak memPerElem depth pl.debug
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

    let name = "readZarrSlabStacked"
    let dataType = zarrDataType<'T> ()
    let level = openZarrResolutionLevel path multiscaleIndex datasetIndex
    let sizeT, sizeC, sizeZ, sizeY, sizeX = zarrShapeTCZYX level.Shape

    if timepoint < 0 || timepoint >= sizeT then
        invalidArg "timepoint" $"Timepoint {timepoint} is outside the Zarr time range 0..{sizeT - 1}."
    if channel < 0 || channel >= sizeC then
        invalidArg "channel" $"Channel {channel} is outside the Zarr channel range 0..{sizeC - 1}."
    if not (String.Equals(level.DataType, dataType, StringComparison.OrdinalIgnoreCase)) then
        failwith $"Zarr dataset pixel type is {level.DataType}, but readZarrSlabStacked<{typeof<'T>.Name}> expects {dataType}."

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
        let region =
            PixelRegion(
                [| int64 timepoint; int64 channel; int64 zStart; 0L; 0L |],
                [| int64 (timepoint + 1); int64 (channel + 1); int64 zStop; int64 sizeY; int64 sizeX |])
        let result =
            level.ReadPixelRegionAsync(region, parallelChunks, CancellationToken.None)
            |> runTask
        let arr = array3DOfZarrBytes<'T> sizeX sizeY zCount result.Data
        Image<'T>.ofArray3D(arr, $"readZarrSlabStacked.{idx}")

    let transition = ProfileTransition.create Unit Streaming
    let memPeak = 256UL
    let memoryNeed = fun _ -> memPeak
    let elementTransformation = id
    let memoryModel = StageMemoryModel.fromSinglePeak Source memoryNeed
    let workModel =
        StageWorkModel.ioRead Source (Some $"readZarrSlabStacked.{typeof<'T>.Name}") (fun _ -> elementBytes) (fun _ -> 1UL)
    let stage =
        Stage.init name (uint depth) mapper transition memoryNeed elementTransformation
        |> withCostModel (StageCostModel.create memoryModel workModel)
        |> Some

    Plan.create stage pl.memAvail memPeak memPeak depth pl.debug
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
        let starts = Array.zeroCreate<uint64> rank
        let blocks = Array.create rank 1UL
        starts[frameAxis] <- uint64 zStart
        blocks[frameAxis] <- uint64 zCount
        blocks[yAxis] <- uint64 sizeY
        blocks[xAxis] <- uint64 sizeX

        let selection = HyperslabSelection(rank, starts, blocks)
        let source = dataset.Read<'T[,,]>(selection, AllSelection(), blocks) :> Array
        let arr =
            Array3D.init sizeX sizeY zCount (fun x y z ->
                let indices = Array.zeroCreate<int> rank
                indices[frameAxis] <- z
                indices[yAxis] <- y
                indices[xAxis] <- x
                source.GetValue(indices) |> unbox<'T>)

        Image<'T>.ofArray3D(arr, $"readNexusSlabStacked.{idx}")

    let transition = ProfileTransition.create Unit Streaming
    let memPeak = 256UL
    let memoryNeed = fun _ -> memPeak
    let elementTransformation = id
    let memoryModel = StageMemoryModel.fromSinglePeak Source memoryNeed
    let workModel =
        StageWorkModel.ioRead Source (Some $"readNexusSlabStacked.{typeof<'T>.Name}") (fun _ -> elementBytes) (fun _ -> 1UL)
    let stage =
        Stage.init name (uint depth) mapper transition memoryNeed elementTransformation
        |> withCostModel (StageCostModel.create memoryModel workModel)
        |> Some

    Plan.create stage pl.memAvail memPeak memPeak depth pl.debug
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

let write<'T when 'T: equality> (outputDir: string) (suffix: string) : Stage<Image<'T>, Image<'T>> =
    let t = typeof<'T>
    if (icompare suffix ".tif" || icompare suffix ".tiff") 
        && not (t = typeof<uint8> || t = typeof<int8> || t = typeof<uint16> || t = typeof<int16> || t = typeof<float32>) then
        failwith $"[write] tiff images only supports (u)int8, (u)int16 and float32 but was {t.Name}" 
    Directory.CreateDirectory(outputDir) |> ignore
    let mapper (debug: bool) (idx: int64) (image: Image<'T>) =
        let fileName = Path.Combine(outputDir, sprintf "image_%03d%s" idx suffix)
        if debug then printfn "[write] Saved image %d to %s as %s" idx fileName (typeof<'T>.Name) 
        image.toFile(fileName)
        image
    let memoryNeed = id
    let memoryModel = StageMemoryModel.fromSinglePeak Iter memoryNeed
    let workModel =
        imageIoCost<'T>
            "write"
            Iter
            $"write.{typeof<'T>.Name}"
            (fun input -> inputValue input |> imageBytes<'T>)
            (fun _ -> 1UL)
    Stage.mapi $"write \"{outputDir}/*{suffix}\"" mapper memoryNeed id
    |> withCostModel (StageCostModel.create memoryModel workModel)

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

        let planeBytes = image.toArray2D() |> bytesOfArray2D
        if debug then
            printfn "[writeZarr] Saved plane %d to %s as %s" idx outputPath (typeof<'T>.Name)

        zarrWriter.WritePlaneAsync(int idx, planeBytes, CancellationToken.None)
        |> runUnitTask

        if idx = int64 (depth - 1) then
            zarrWriter.DisposeAsync().AsTask()
            |> runUnitTask
            writer <- None

        image

    let memoryNeed = id
    let memoryModel = StageMemoryModel.fromSinglePeak Iter memoryNeed
    let workModel =
        imageIoCost<'T>
            "write"
            Iter
            $"writeZarr.{typeof<'T>.Name}"
            (fun input -> inputValue input |> imageBytes<'T>)
            (fun _ -> 1UL)

    Stage.mapi $"writeZarr \"{outputPath}\"" mapper memoryNeed id
    |> withCostModel (StageCostModel.create memoryModel workModel)

let _writeSlabChunks (debug: bool) (outputDir: string) (suffix: string) (width: uint) (height: uint) (winSz: uint) (k: int) (stack: Image<'T>) =
    for i in [0u..stack.GetWidth()/width] do
        for j in [0u..stack.GetHeight()/height] do
            let fileName = getChunkFilename outputDir suffix (int i) (int j) (int k)
            let x00 = i*width |> int
            let x01 = ((i+1u)*width-1u |> int, stack.GetWidth()-1u |> int) ||> min
            let x10 = j*height |> int
            let x11 = ((j+1u)*height-1u |> int, stack.GetHeight()-1u |> int) ||> min
            let x20 = 0
            let x21 = winSz-1u |> int
            if x00<=x01 && x10<=x11 && x20<=x21 then
                if debug then printfn "[write] Saved chunk %d %d %d to %s as %s" i j k fileName (typeof<'T>.Name) 
                let chunck = stack.[x00 .. x01, x10 .. x11 , x20 .. x21]
                chunck.toFile(fileName)
                chunck.decRefCount()

let _writeChunks (debug: bool) (outputDir: string) (suffix: string) (width: uint) (height: uint) (winSz: uint) (stack: Image<'T>) =
    ()

let private writeInSlabsCore<'T when 'T: equality> (outputDir: string) (suffix: string) (width: uint) (height: uint) (winSz: uint) : Stage<Image<'T>, Image<'T>> =
    let t = typeof<'T>
    if (icompare suffix ".tif" || icompare suffix ".tiff") 
        && not (t = typeof<uint8> || t = typeof<int8> || t = typeof<uint16> || t = typeof<int16> || t = typeof<float32>) then
        failwith $"[write] tiff images only supports (u)int8, (u)int16 and float32 but was {t.Name}" 
    if not (Directory.Exists(outputDir)) then
        Directory.CreateDirectory(outputDir) |> ignore

    let pad, stride = 0u, winSz
    let f (debug: bool) (k: int64) (stack: Image<'T>) = 
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
    let workModel =
        imageIoCost<'T>
            "write"
            Iter
            $"writeInSlabs.{typeof<'T>.Name}"
            (fun input -> inputValue input |> imageBytes<'T>)
            (fun _ -> 1UL)
    let stg =
        Stage.mapi "writeInSlabs" mapper memoryNeed elementTransformation
        |> withCostModel (StageCostModel.create memoryModel workModel)
    (window winSz pad stride) --> stg --> flattenList ()

let writeInSlabs<'T when 'T: equality> outputDir suffix width height winSz =
    writeInSlabsCore<'T> outputDir suffix width height winSz
