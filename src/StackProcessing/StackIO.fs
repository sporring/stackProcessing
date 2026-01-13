module StackIO

open SlimPipeline // Core processing model
open System.IO
open System.Text.RegularExpressions
open StackCore

type FileInfo = ImageFunctions.FileInfo
type ChunkInfo = { chunks: int list; size: uint64 list; topLeftInfo: FileInfo}

let getStackDepth (inputDir: string) (suffix: string) : uint =
    let files = Directory.GetFiles(inputDir, "*"+suffix) |> Array.sort
    files.Length |> uint

let getStackInfo (inputDir: string) (suffix: string) : FileInfo =
    let files = Directory.GetFiles(inputDir, "*"+suffix) |> Array.sort
    let depth = files.Length |> uint64
    if depth = 0uL then
        failwith $"No {suffix} files found in directory: {inputDir}"
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
    System.IO.Directory.GetFiles(inputDir, "*"+suffix) |> filter

let getFilenames (inputDir: string) (suffix: string) (filter: string[]->string[]) (pl: Plan<unit, unit>) : Plan<unit, string> =
    let name = "getFilenames"
    let filenames = _getFilenames inputDir ("*"+suffix) filter
    let depth = uint64 filenames.Length

    let mapper (i: int) : string = 
        if pl.debug then printfn "[%s] Supplying filename %i: %s" name i filenames[i]
        filenames[i]

    let transition = ProfileTransition.create Unit Streaming
    let memPeak = 256UL // surrugate string length
    let memoryNeed = fun _ -> memPeak
    let lengthTransformation = fun _ -> depth
    let stage = Stage.init $"{name}" (uint depth) mapper transition memoryNeed lengthTransformation |> Some

    let memPerElem = 256UL // surrugate string length
    let length = depth
    Plan.create stage pl.memAvail memPeak memPerElem length pl.debug

let readFiles<'T when 'T: equality> (debug: bool) : Stage<string, Image<'T>> =
    let name = "readFiles"
    if debug then printfn $"[{name} cast to {typeof<'T>.Name}]"
    let mutable width = 0u // We need to read the first image in order to find its size
    let mutable height = 0u

    let mapper (debug: bool) (fileName: string) : Image<'T> = 
        if debug then printfn "[%s] Reading image named %s as %s" name fileName (typeof<'T>.Name)
        let image = Image<'T>.ofFile fileName
        if width = 0u then
            width <- image.GetWidth()
            height <- image.GetHeight()
        image

    let memoryNeed = fun _ -> Image<'T>.memoryEstimate width height
    let lengthTransformation = id

    Stage.map name mapper memoryNeed lengthTransformation

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
    let lengthTransformation = id

    Stage.map name mapper memoryNeed lengthTransformation

let readFiltered<'T when 'T: equality> (inputDir: string) (suffix: string) (filter: string[]->string[]) (pl: Plan<unit, unit>) : Plan<unit, Image<'T>> =
    pl |> getFilenames inputDir suffix filter >=> readFiles pl.debug

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
            ) (System.Int32.MinValue, System.Int32.MinValue, System.Int32.MinValue, "", "") files
    let topLeftFi = ImageFunctions.getFileInfo topLeft
    let bottomRightFi = ImageFunctions.getFileInfo bottomRight

    let stackSize = 
        [
            (uint64 maxI) * topLeftFi.size[0] + bottomRightFi.size[0];
            (uint64 maxJ) * topLeftFi.size[1] + bottomRightFi.size[1];
            (uint64 maxK) * topLeftFi.size[2] + bottomRightFi.size[2];
        ]
    { chunks = [maxI+1;maxJ+1;maxK+1]; topLeftInfo = topLeftFi; size = stackSize }

let getChunkFilename (path: string) (suffix: string) (i: int) (j: int) (k: int) =
    Path.Combine(path, sprintf "chunk%d_%d_%d%s" i j k suffix)

let _readChunk (inputDir: string) (suffix: string) i j k = 
    let filename = getChunkFilename inputDir suffix i j k
    let res = Image<'T>.ofFile filename
    res

let _readChunkSlice (inputDir: string) (suffix: string) (chunkInfo: ChunkInfo) (dir: uint) (idx: int) =
    let depth = uint64 chunkInfo.chunks[2] // we will read chunks_*_*_i* as windows
    let chunkWidth = int chunkInfo.topLeftInfo.size[0]
    let chunkHeight = int chunkInfo.topLeftInfo.size[1]
    let chunkDepth = 
        if idx < chunkInfo.chunks[2]-1 then
           int chunkInfo.topLeftInfo.size[2]
        else
           chunkInfo.size[2] % chunkInfo.topLeftInfo.size[2] |> int
    let sz, nChunks = 
        if dir = 0u then
            [chunkWidth |> uint64; chunkInfo.size[1]; chunkInfo.size[2]], [1; chunkInfo.chunks[1]; chunkInfo.chunks[2]]
        elif dir = 1u then
            [chunkInfo.size[0]; chunkHeight |> uint64; chunkInfo.size[1]], [chunkInfo.chunks[0]; 1; chunkInfo.chunks[2]]
        else
            [chunkInfo.size[0]; chunkInfo.size[1]; chunkDepth |> uint64], [chunkInfo.chunks[0]; chunkInfo.chunks[1]; 1]
        
    let chunkSlice = Image<'T>(sz |> List.map uint, chunkInfo.topLeftInfo.numberOfComponents)
    for i in [0 .. nChunks[0]-1] do
        for j in [0 .. nChunks[1]-1] do
            for k in [0 .. nChunks[2]-1] do
                let img = 
                    if dir = 0u then   _readChunk inputDir suffix idx j k
                    elif dir = 1u then _readChunk inputDir suffix i idx k
                    else               _readChunk inputDir suffix i j idx
                let start0 = i*chunkWidth|>Some
                let stop0 = i*chunkWidth+(img.GetWidth()|>int)-1|>Some
                let start1 = j*chunkHeight|>Some
                let stop1 = j*chunkHeight+(img.GetHeight()|>int)-1|>Some
                let start2 = k*chunkDepth|>Some
                let stop2 = k*chunkDepth+(img.GetDepth()|>int)-1|>Some
                chunkSlice.SetSlice (start0, stop0, start1, stop1, start2, stop2) img |> ignore
                img.decRefCount()
    chunkSlice

let readChunksAsWindows<'T when 'T: equality> (inputDir: string) (suffix: string) (pl: Plan<unit, unit>) : Plan<unit, Image<'T> list> =
    let name = "readChunks"
    let chunkInfo = getChunkInfo inputDir suffix
    let depth = uint64 chunkInfo.chunks[2] // we will read chunks_*_*_i* as windows

    let mapper (k: int) : Image<'T> list = 
        let chunkSlice = _readChunkSlice inputDir suffix chunkInfo 2u k
        let res = chunkSlice |> ImageFunctions.unstack 2u
        chunkSlice.decRefCount()
        res

    let transition = ProfileTransition.create Unit Streaming
    let memPeak = 256UL // surrugate string length
    let memoryNeed = fun _ -> memPeak
    let lengthTransformation = fun _ -> depth
    let stage = Stage.init $"{name}" (uint depth) mapper transition memoryNeed lengthTransformation |> Some

    let memPerElem = 256UL // surrugate string length
    let length = depth
    Plan.create stage pl.memAvail memPeak memPerElem depth pl.debug

let readChunks<'T when 'T: equality> (inputDir: string) (suffix: string) (pl: Plan<unit, unit>) : Plan<unit, Image<'T>> =
    pl |> readChunksAsWindows inputDir suffix >=> flatten ()

let icompare s1 s2  = 
    System.String.Equals(s1, s2, System.StringComparison.CurrentCultureIgnoreCase)

let write (outputDir: string) (suffix: string) : Stage<Image<'T>, Image<'T>> =
    let t = typeof<'T>
    if (icompare suffix ".tif" || icompare suffix ".tiff") 
        && not (t = typeof<uint8> || t = typeof<int8> || t = typeof<uint16> || t = typeof<int16> || t = typeof<float32>) then
        failwith $"[write] tiff images only supports (u)int8, (u)int16 and float32 but was {t.Name}" 
    if not (Directory.Exists(outputDir)) then
        Directory.CreateDirectory(outputDir) |> ignore
    let mapper (debug: bool) (idx: int64) (image: Image<'T>) =
        let fileName = Path.Combine(outputDir, sprintf "image_%03d%s" idx suffix)
        if debug then printfn "[write] Saved image %d to %s as %s" idx fileName (typeof<'T>.Name) 
        image.toFile(fileName)
        image
    let memoryNeed = id
    Stage.mapi $"write \"{outputDir}/*{suffix}\"" mapper memoryNeed id

let _writeChunkSlice (debug: bool) (outputDir: string) (suffix: string) (width: uint) (height: uint) (winSz: uint) (k: int) (stack: Image<'T>) =
    for i in [0u..stack.GetWidth()/width] do
        for j in [0u..stack.GetHeight()/height] do
            let fileName = getChunkFilename outputDir suffix (int i) (int j) (int k)
            if debug then printfn "[write] Saved chunk %d %d %d to %s as %s" i j k fileName (typeof<'T>.Name) 
            let x00 = i*width |> int
            let x01 = ((i+1u)*width-1u |> int, stack.GetWidth()-1u |> int) ||> min
            let x10 = j*height |> int
            let x11 = ((j+1u)*height-1u |> int, stack.GetHeight()-1u |> int) ||> min
            let x20 = 0
            let x21 = winSz-1u |> int
            if x00<=x01 && x10<=x11 && x20<=x21 then
                let chunck = stack.[x00 .. x01, x10 .. x11 , x20 .. x21]
                chunck.toFile(fileName)
                chunck.decRefCount()

let _writeChunks (debug: bool) (outputDir: string) (suffix: string) (width: uint) (height: uint) (winSz: uint) (stack: Image<'T>) =
    ()

let writeInChunks (outputDir: string) (suffix: string) (width: uint) (height: uint) (winSz: uint) : Stage<Image<'T>, Image<'T>> =
    let t = typeof<'T>
    if (icompare suffix ".tif" || icompare suffix ".tiff") 
        && not (t = typeof<uint8> || t = typeof<int8> || t = typeof<uint16> || t = typeof<int16> || t = typeof<float32>) then
        failwith $"[write] tiff images only supports (u)int8, (u)int16 and float32 but was {t.Name}" 
    if not (Directory.Exists(outputDir)) then
        Directory.CreateDirectory(outputDir) |> ignore

    let pad, stride = 0u, winSz
    let f (debug: bool) (k: int64) (stack: Image<'T>) = 
        _writeChunkSlice debug outputDir suffix width height winSz (int k) stack
        stack.incRefCount()
        stack
    let mapper (debug: bool) (idx: int64) = volFctToLstFctReleaseAfter (f debug idx) pad stride
    //let mapper (debug: bool) (idx: int64) = fun stack -> _writeChunkSlice debug outputDir suffix width height winSz (int idx) stack; stack
    let btUint8 = typeof<uint8>|>Image.getBytesPerComponent |> uint64
    let btUint64 = typeof<uint64> |> Image.getBytesPerComponent |> uint64
    let memoryNeed nPixels = 
        let bt8 = typeof<uint8>|>Image.getBytesPerComponent |> uint64
        let bt64 = typeof<uint64> |> Image.getBytesPerComponent |> uint64
        let wsz = uint64 winSz
        let str = uint64 stride
        max (nPixels*(wsz*(2UL*bt8+bt64)-str*bt8)) (nPixels*(wsz*(bt8+bt64)+str*(bt64-bt8)))
    let lengthTransformation = id
    let stg = Stage.mapi "writeInChunks" mapper memoryNeed lengthTransformation
    (window winSz pad stride) --> stg --> flatten ()
