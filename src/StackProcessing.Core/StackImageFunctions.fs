module StackImageFunctions

open FSharp.Control
open SlimPipeline // Core processing model
open System
open System.IO
open System.Collections.Generic
open StackCore
open StackIO

let liftUnary name  = Stage.liftReleaseUnary name ignore
let liftUnaryReleaseAfter (name: string) (f: Image<'S> -> Image<'T>) (memoryNeed: MemoryNeed) (elementTransformation: ElementTransformation) = 
    Stage.liftResourceUnary name imageResourceOps f memoryNeed elementTransformation

let getBytesPerComponent<'T> = (typeof<'T> |> Image.getBytesPerComponent |> uint64)
let getImageFacts<'T when 'T: equality> (image: Image<'T>) = image.GetFacts()
let imageBytes<'T> nVoxels = StackProcessingCost.imageBytes<'T> nVoxels

let private inputValue input =
    input |> SingleOrPair.sum |> SingleOrPair.fst

let private withCostModel costModel stage =
    StackProcessingCost.withCostModel costModel stage

let private validSliceDomainForKernelDepth ksz =
    let before = ksz / 2u
    let after = (ksz - 1u) - before
    SlimPipeline.SliceDomain.trim before after

let private sameSliceDomainForKernelDepth ksz =
    let pad = ksz / 2u
    SlimPipeline.SliceDomain.compose
        (SlimPipeline.SliceDomain.expand pad pad)
        (validSliceDomainForKernelDepth ksz)

let private sliceCardinalityForConvolution ksz outputRegionMode =
    match outputRegionMode with
    | Some ImageFunctions.Valid ->
        SlimPipeline.Domain(validSliceDomainForKernelDepth ksz)
    | None
    | Some ImageFunctions.Same ->
        SlimPipeline.Domain(sameSliceDomainForKernelDepth ksz)

let private nativeImageStageCost name memoryModel costUnits =
    StageCostModel.create
        memoryModel
        (StageTimeCostModel.native Map (Some name) costUnits)

let identityStage name =
    Stage.map name (fun _ value -> value) id id

let private cleanStage name cleanup =
    { identityStage name with Cleaning = [ cleanup ] }

type ResampleInterpolation =
    | NearestNeighbor
    | Linear

module ResampleInterpolation =
    let parse (value: string) =
        match value.Trim().ToLowerInvariant().Replace("_", "").Replace("-", "").Replace(" ", "") with
        | "nearest"
        | "nearestneighbor"
        | "nn" -> NearestNeighbor
        | "linear" -> Linear
        | _ -> failwith $"Unknown resampling interpolation '{value}'. Use NearestNeighbor or Linear."

    let toItk = function
        | NearestNeighbor -> itk.simple.InterpolatorEnum.sitkNearestNeighbor
        | Linear -> itk.simple.InterpolatorEnum.sitkLinear

let private roundPositiveToUInt (value: float) =
    Math.Round(value, MidpointRounding.AwayFromZero)
    |> max 1.0
    |> uint

let private outputSpacingForSize inputSize outputSize =
    if inputSize <= 1u || outputSize <= 1u then
        1.0
    else
        float (inputSize - 1u) / float (outputSize - 1u)

let private convertFromFloat<'T> (value: float) : 'T =
    let t = typeof<'T>
    let inline clamp lo hi v = max lo (min hi v)

    if t = typeof<float> then
        box value :?> 'T
    elif t = typeof<float32> then
        box (float32 value) :?> 'T
    elif t = typeof<uint8> then
        box (uint8 (clamp 0.0 255.0 (Math.Round(value)))) :?> 'T
    elif t = typeof<int8> then
        box (int8 (clamp -128.0 127.0 (Math.Round(value)))) :?> 'T
    elif t = typeof<uint16> then
        box (uint16 (clamp 0.0 65535.0 (Math.Round(value)))) :?> 'T
    elif t = typeof<int16> then
        box (int16 (clamp -32768.0 32767.0 (Math.Round(value)))) :?> 'T
    elif t = typeof<uint32> || t = typeof<uint> then
        box (uint32 (clamp 0.0 (float UInt32.MaxValue) (Math.Round(value)))) :?> 'T
    elif t = typeof<int32> || t = typeof<int> then
        box (int32 (clamp (float Int32.MinValue) (float Int32.MaxValue) (Math.Round(value)))) :?> 'T
    elif t = typeof<uint64> then
        box (uint64 (clamp 0.0 (float UInt64.MaxValue) (Math.Round(value)))) :?> 'T
    elif t = typeof<int64> then
        box (int64 (clamp (float Int64.MinValue) (float Int64.MaxValue) (Math.Round(value)))) :?> 'T
    else
        failwith $"Linear z interpolation is not supported for pixel type {t.Name}."

let private pixelToFloat<'T> (value: 'T) =
    Convert.ToDouble(value)

let private blendImages<'T when 'T: equality> (t: float) (a: Image<'T>) (b: Image<'T>) =
    let aPixels = a.toArray2D()
    let bPixels = b.toArray2D()
    let width = aPixels.GetLength(0)
    let height = aPixels.GetLength(1)
    let arr =
        Array2D.init width height (fun x y ->
            let av = pixelToFloat aPixels[x, y]
            let bv = pixelToFloat bPixels[x, y]
            convertFromFloat<'T> ((1.0 - t) * av + t * bv))

    Image<'T>.ofArray2D(arr, "resampleZLinear")

let private resampleXYStage<'T when 'T: equality> outputWidth outputHeight spacingX spacingY interpolation =
    let itkInterpolator = ResampleInterpolation.toItk interpolation
    let mapper (image: Image<'T>) =
        ImageFunctions.resample2D itkInterpolator outputWidth outputHeight spacingX spacingY image

    let memoryNeed nPixels =
        3UL * nPixels * getBytesPerComponent<'T>

    liftUnaryReleaseAfter
        $"resampleXY.{typeof<'T>.Name}"
        mapper
        memoryNeed
        (fun _ -> uint64 outputWidth * uint64 outputHeight)

let private zPositions inputDepth outputDepth factor =
    [ for outputIndex in 0 .. int outputDepth - 1 ->
        let source =
            match factor with
            | Some scale -> float outputIndex / scale
            | None ->
                if outputDepth <= 1u || inputDepth <= 1UL then 0.0
                else float outputIndex * float (inputDepth - 1UL) / float (outputDepth - 1u)
        min (float (inputDepth - 1UL)) (max 0.0 source) ]

let private zResampleStage<'T when 'T: equality> inputDepth outputDepth factor interpolation =
    let positions = zPositions inputDepth outputDepth factor
    let memoryNeed nPixels = 3UL * nPixels * getBytesPerComponent<'T>

    let releaseWindow (window: Window<Image<'T>>) =
        let start, count = window.EmitRange
        window.Items
        |> List.skip (int start)
        |> List.take (min (int count) (max 0 (window.Items.Length - int start)))
        |> List.iter (fun image -> image.decRefCount())

    let outputForWindow (debug: bool) (windowStart: int64) (window: Window<Image<'T>>) =
        try
            match window.Items with
            | [] -> []
            | [ only ] ->
                positions
                |> List.mapi (fun outputIndex sourceZ -> outputIndex, sourceZ)
                |> List.filter (fun (_, sourceZ) -> int (Math.Floor(sourceZ)) = int windowStart)
                |> List.map (fun _ ->
                    only.incRefCount()
                    only)
            | first :: second :: _ ->
                let z0 = int windowStart
                positions
                |> List.mapi (fun outputIndex sourceZ -> outputIndex, sourceZ)
                |> List.filter (fun (_, sourceZ) ->
                    let low = int (Math.Floor(sourceZ))
                    low = z0)
                |> List.map (fun (_, sourceZ) ->
                    let fraction = sourceZ - float z0
                    match interpolation with
                    | NearestNeighbor ->
                        let chosen = if fraction < 0.5 then first else second
                        chosen.incRefCount()
                        chosen
                    | Linear ->
                        if fraction = 0.0 then
                            first.incRefCount()
                            first
                        else
                            blendImages fraction first second)
        finally
            releaseWindow window

    let stage =
        Stage.mapi
            $"resampleZ.{typeof<'T>.Name}"
            (fun debug idx window -> outputForWindow debug idx window)
            memoryNeed
            id
        |> Stage.withSliceCardinality (SliceCardinality.reduceTo (uint64 outputDepth))

    let windowSize =
        if inputDepth <= 1UL then 1u else 2u

    (window windowSize 0u 1u) --> stage --> flattenList ()

let private liftWindowedUnaryReleaseAfter
        (name: string)
        (winSz: uint)
        (f: Image<'S> -> Image<'T>)
        (memoryNeed: MemoryNeed)
        (elementTransformation: ElementTransformation)
        : Stage<Image<'S>, Image<'T>> =
    let win = max 1u winSz
    let mapper debug =
        volFctToWindowFctReleaseAfterDebug debug f 1u 0u win
    let stg = mapWindow name mapper memoryNeed elementTransformation
    (window win 0u win) --> stg --> flattenList ()

type System.String with // From https: //stackoverflow.com/questions/1936767/f-case-insensitive-string-compare
    member s1.icompare(s2: string) =
        System.String.Equals(s1, s2, System.StringComparison.CurrentCultureIgnoreCase)

(*
let liftImageSource (name: string) (img: Image<'T>) : Pipe<unit, Image<'T>> =
    {
        Name = name
        Profile = Streaming
        Apply = fun _ -> img |> Image.unstack |> AsyncSeq.ofSeq
    }

let axisSource
    (axis: int) 
    (size: int list)
    (pl: Plan<unit, unit>) 
   : Plan<unit, Image<uint>> =
    let img = Image.generateCoordinateAxis axis size
    let sz = Image.GetSize img
    let shapeUpdate = fun (s: Shape) -> s
    let op: Stage<unit, Image<uint>> =
        {
            Name = "axisSource"
            Pipe = img |> liftImageSource "axisSource"
            Transition = ProfileTransition.create Unit Streaming
            ShapeUpdate = shapeUpdate
        }
    let width, height, depth = sz[0], sz[1], sz[2]
    let context = ShapeContext.create (fun _ -> width*height |> uint64) (fun _ -> depth)
    {
        flow = Flow.returnM op
        mem = pl.memAvail
        shape = Slice (width,height) |> Some
        context = context
        debug = pl.debug
    }
*)

/// Pixel type casting
let cast<'S,'T when 'S: equality and 'T: equality> = 
    let name = sprintf "cast(%s->%s)" typeof<'S>.Name typeof<'T>.Name
    let f (I: Image<'S>) = 
        let result = I.castTo<'T> ()
        I.decRefCount()
        result
    Stage.cast<Image<'S>,Image<'T>> name f id

/// Basic arithmetic
let liftRelease2 f I J = releaseAfter2 (fun a b -> f a b) I J

let memNeeded<'T> nTimes nElems = 3UL*nElems*nTimes*getBytesPerComponent<'T> // We need input, output, and potentially a cast in between
let add (image: Image<'T>) = 
    liftUnaryReleaseAfter "add" ((+) image) id id
let addPair I J = liftRelease2 ( + ) I J
let inline scalarAddImage<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) = 
    liftUnaryReleaseAfter "scalarAddImage" (fun (s: Image<^T>)->ImageFunctions.scalarAddImage<^T> i s) id id
let inline imageAddScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    liftUnaryReleaseAfter "imageAddScalar" (fun (s: Image<^T>)->ImageFunctions.imageAddScalar<^T> s i) id id

let sub (image: Image<'T>) = 
    liftUnaryReleaseAfter "sub" ((-) image) id id

let subPair I J = liftRelease2 ( - ) I J
let inline scalarSubImage<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    liftUnaryReleaseAfter "scalarSubImage" (fun (s: Image<^T>)->ImageFunctions.scalarSubImage<^T> i s) id id
let inline imageSubScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    liftUnaryReleaseAfter "imageSubScalar" (fun (s: Image<^T>)->ImageFunctions.imageSubScalar<^T> s i) id id

let mul (image: Image<'T>) = liftUnaryReleaseAfter "mul" (( * ) image) id id
let mulPair I J = liftRelease2 ( * ) I J
let inline scalarMulImage<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    liftUnaryReleaseAfter "scalarMulImage" (fun (s: Image<^T>)->ImageFunctions.scalarMulImage<^T> i s) id id
let inline imageMulScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    liftUnaryReleaseAfter "imageMulScalar" (fun (s: Image<^T>)->ImageFunctions.imageMulScalar<^T> s i) id id

let div (image: Image<'T>) = liftUnaryReleaseAfter "div" ((/) image) id id

let divPair I J = liftRelease2 ( / ) I J
let inline scalarDivImage<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    liftUnaryReleaseAfter "scalarDivImage" (fun (s: Image<^T>)->ImageFunctions.scalarDivImage<^T> i s) id id
let inline imageDivScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    liftUnaryReleaseAfter "imageDivScalar" (fun (s: Image<^T>)->ImageFunctions.imageDivScalar<^T> s i) id id

let maxOfPair I J = liftRelease2 Image.maximumImage I J

let minOfPair I J = liftRelease2 Image.minimumImage I J
let getMinMax I = releaseAfter Image.getMinMax I;

let private releaseImagePair (f: Image<'S> -> Image<'T> -> 'U) (a: Image<'S>, b: Image<'T>) =
    try
        f a b
    finally
        imageResourceOps.Release a
        imageResourceOps.Release b

let private liftPairReleaseAfter name f : Stage<Image<'S> * Image<'T>, 'U> =
    Stage.map name (fun _ pair -> releaseImagePair f pair) id id

let failTypeMismatch<'T> name lst =
    let t = typeof<'T>
    if lst |> List.exists ((=) t) |> not then
        let names = List.map (fun (t: System.Type) -> t.Name) lst
        failwith $"[{name}] wrong type. Type {t} must be one of {names}"

/// Simple functions
let private floatNintTypes = [typeof<float>;typeof<float32>;typeof<int>]
let private floatTypes = [typeof<float>;typeof<float32>]
let private vectorElementFunction (name: string) =
    match name.Trim().ToLowerInvariant() with
    | "abs" -> abs
    | "acos" -> acos
    | "asin" -> asin
    | "atan" -> atan
    | "cos" -> cos
    | "sin" -> sin
    | "tan" -> tan
    | "exp" -> exp
    | "log10" -> log10
    | "log" -> log
    | "round" -> round
    | "sqrt" -> sqrt
    | "square" -> fun x -> x * x
    | "identity" -> id
    | other -> invalidArg "name" $"Unknown vector element function '{other}'."

let toVectorImage<'T when 'T: equality> : Stage<Image<'T> * Image<'T>, Image<'T list>> =
    liftPairReleaseAfter "toVectorImage" (fun a b -> ImageFunctions.toVectorImage [ a; b ])

let vectorElement<'T when 'T: equality> componentId : Stage<Image<'T list>, Image<'T>> =
    liftUnaryReleaseAfter "vectorElement" (ImageFunctions.vectorElement componentId) id id

let appendVectorElement : Stage<Image<float list> * Image<float>, Image<float list>> =
    liftPairReleaseAfter "appendVectorElement" ImageFunctions.appendVectorElement

let vectorMapElements functionName : Stage<Image<float list>, Image<float list>> =
    liftUnaryReleaseAfter "vectorMapElements" (ImageFunctions.mapVectorElements (vectorElementFunction functionName)) id id

let vectorDot : Stage<Image<float list> * Image<float list>, Image<float>> =
    liftPairReleaseAfter "vectorDot" ImageFunctions.vectorDot

let vectorCross3D : Stage<Image<float list> * Image<float list>, Image<float list>> =
    liftPairReleaseAfter "vectorCross3D" ImageFunctions.vectorCross3D

let vectorAngleTo (reference: float list) : Stage<Image<float list>, Image<float>> =
    liftUnaryReleaseAfter "vectorAngleTo" (ImageFunctions.vectorAngleTo reference) id id

let Re : Stage<Image<System.Numerics.Complex>, Image<float>> =
    liftUnaryReleaseAfter "Re" Image.Re id id

let Im : Stage<Image<System.Numerics.Complex>, Image<float>> =
    liftUnaryReleaseAfter "Im" Image.Im id id

let modulus : Stage<Image<System.Numerics.Complex>, Image<float>> =
    liftUnaryReleaseAfter "modulus" Image.modulus id id

let arg : Stage<Image<System.Numerics.Complex>, Image<float>> =
    liftUnaryReleaseAfter "arg" Image.arg id id

let conjugate : Stage<Image<System.Numerics.Complex>, Image<System.Numerics.Complex>> =
    liftUnaryReleaseAfter "conjugate" Image.conjugate id id

let toComplex : Stage<Image<float> * Image<float>, Image<System.Numerics.Complex>> =
    liftPairReleaseAfter "toComplex" Image.toComplex

let polarToComplex : Stage<Image<float> * Image<float>, Image<System.Numerics.Complex>> =
    liftPairReleaseAfter "polarToComplex" Image.polarToComplex

let private writeVolumeAsChunks debug outputDir suffix chunkX chunkY chunkZ (volume: Image<'T>) =
    let depth = volume.GetDepth()
    let mutable k = 0u
    while k < depth do
        let last = min (depth - 1u) (k + chunkZ - 1u)
        let slab = ImageFunctions.extractSub [ 0u; 0u; k ] [ volume.GetWidth() - 1u; volume.GetHeight() - 1u; last ] volume
        _writeSlabChunks debug outputDir suffix chunkX chunkY (last - k + 1u) (int (k / chunkZ)) slab
        slab.decRefCount()
        k <- k + chunkZ

let private readChunkVolume<'T when 'T: equality> inputDir suffix =
    let chunkInfo = getChunkInfo inputDir suffix
    let slabs =
        [ for k in 0 .. chunkInfo.chunks[2] - 1 ->
            _readSlabStacked<'T> inputDir suffix chunkInfo 2u k ]
    let volume = ImageFunctions.stack slabs
    slabs |> List.iter (fun slab -> slab.decRefCount())
    volume, chunkInfo

let private chunkedFFT3D debug inputDir outputDir suffix chunkX chunkY chunkZ =
    if Directory.Exists(outputDir) then Directory.Delete(outputDir, true)
    Directory.CreateDirectory(outputDir) |> ignore
    let volume, _ = readChunkVolume<System.Numerics.Complex> inputDir suffix
    let zTransformed = ImageFunctions.directionalFFTComplex 2u false volume
    volume.decRefCount()
    writeVolumeAsChunks debug outputDir suffix chunkX chunkY chunkZ zTransformed
    zTransformed.decRefCount()

let private chunkedInvFFT3D debug inputDir outputDir suffix chunkX chunkY chunkZ =
    if Directory.Exists(outputDir) then Directory.Delete(outputDir, true)
    Directory.CreateDirectory(outputDir) |> ignore
    let volume, _ = readChunkVolume<System.Numerics.Complex> inputDir suffix
    let zInverse = ImageFunctions.directionalFFTComplex 2u true volume
    volume.decRefCount()
    let slices = ImageFunctions.unstack 2u zInverse
    zInverse.decRefCount()
    let realSlices =
        slices
        |> List.map (fun slice ->
            let real = ImageFunctions.inverseFFTXY slice
            slice.decRefCount()
            real)
    let realVolume = ImageFunctions.stack realSlices
    realSlices |> List.iter (fun slice -> slice.decRefCount())
    writeVolumeAsChunks debug outputDir suffix chunkX chunkY chunkZ realVolume
    realVolume.decRefCount()

let private chunkedShiftFFT debug inputDir outputDir suffix chunkX chunkY chunkZ =
    if Directory.Exists(outputDir) then Directory.Delete(outputDir, true)
    Directory.CreateDirectory(outputDir) |> ignore
    let volume, _ = readChunkVolume<System.Numerics.Complex> inputDir suffix
    let shifted = ImageFunctions.shiftFFT volume
    volume.decRefCount()
    writeVolumeAsChunks debug outputDir suffix chunkX chunkY chunkZ shifted
    shifted.decRefCount()

let private readChunksAsSlices<'T when 'T: equality> name outputDir suffix =
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

let private chunkedVolumeOperation
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
    --> cleanStage name (fun () -> StackIO.deleteIfExists workspaceRoot)
    --> StackCore.ignoreSingles ()
    --> Stage.map name (fun debug _ -> operation debug inputDir outputDir suffix chunkX chunkY chunkZ) memoryNeed elementTransformation
    --> readChunksAsSlices<'T> name outputDir suffix

let FFT<'T when 'T: equality> chunkX chunkY chunkZ : Stage<Image<'T>, Image<System.Numerics.Complex>> =
    chunkedVolumeOperation
        "FFT"
        (liftUnaryReleaseAfter "FFTXY" ImageFunctions.FFTXY id id)
        chunkedFFT3D
        chunkX
        chunkY
        chunkZ

let invFFT chunkX chunkY chunkZ : Stage<Image<System.Numerics.Complex>, Image<float>> =
    chunkedVolumeOperation
        "invFFT"
        (identityStage "invFFT.input")
        chunkedInvFFT3D
        chunkX
        chunkY
        chunkZ

let shiftFFT chunkX chunkY chunkZ : Stage<Image<System.Numerics.Complex>, Image<System.Numerics.Complex>> =
    chunkedVolumeOperation
        "shiftFFT"
        (identityStage "shiftFFT.input")
        chunkedShiftFFT
        chunkX
        chunkY
        chunkZ

let abs<'T when 'T: equality>    : Stage<Image<'T>,Image<'T>> = 
    failTypeMismatch<'T> "abs" floatNintTypes
    liftUnaryReleaseAfter "abs"    ImageFunctions.absImage id id
let acos<'T when 'T: equality>   : Stage<Image<'T>,Image<'T>> =      
    failTypeMismatch<'T> "acos" floatTypes
    liftUnaryReleaseAfter "acos"   ImageFunctions.acosImage id id
let asin<'T when 'T: equality>   : Stage<Image<'T>,Image<'T>> =      
    failTypeMismatch<'T> "asin" floatTypes
    liftUnaryReleaseAfter "asin"   ImageFunctions.asinImage id id
let atan<'T when 'T: equality>   : Stage<Image<'T>,Image<'T>> =      
    failTypeMismatch<'T> "atan" floatTypes
    liftUnaryReleaseAfter "atan"   ImageFunctions.atanImage id id
let cos<'T when 'T: equality>    : Stage<Image<'T>,Image<'T>> =      
    failTypeMismatch<'T> "cos" floatTypes
    liftUnaryReleaseAfter "cos"    ImageFunctions.cosImage id id
let sin<'T when 'T: equality>    : Stage<Image<'T>,Image<'T>> =      
    failTypeMismatch<'T> "sin" floatTypes
    liftUnaryReleaseAfter "sin"    ImageFunctions.sinImage id id
let tan<'T when 'T: equality>    : Stage<Image<'T>,Image<'T>> =      
    failTypeMismatch<'T> "tan" floatTypes
    liftUnaryReleaseAfter "tan"    ImageFunctions.tanImage id id
let exp<'T when 'T: equality>    : Stage<Image<'T>,Image<'T>> =      
    failTypeMismatch<'T> "exp" floatTypes
    liftUnaryReleaseAfter "exp"    ImageFunctions.expImage id id
let log10<'T when 'T: equality>  : Stage<Image<'T>,Image<'T>> =      
    failTypeMismatch<'T> "log10" floatTypes
    liftUnaryReleaseAfter "log10"  ImageFunctions.log10Image id id
let log<'T when 'T: equality>    : Stage<Image<'T>,Image<'T>> =      
    failTypeMismatch<'T> "log" floatTypes
    liftUnaryReleaseAfter "log"    ImageFunctions.logImage id id
let round<'T when 'T: equality>  : Stage<Image<'T>,Image<'T>> =      
    failTypeMismatch<'T> "round" floatTypes
    liftUnaryReleaseAfter "round"  ImageFunctions.roundImage id id
let sqrt<'T when 'T: equality>   : Stage<Image<'T>,Image<'T>> =      
    failTypeMismatch<'T> "sqrt" floatNintTypes
    liftUnaryReleaseAfter "sqrt"   ImageFunctions.sqrtImage id id
let sqrtWindowed<'T when 'T: equality> (winSz: uint) : Stage<Image<'T>,Image<'T>> =
    failTypeMismatch<'T> "sqrtWindowed" floatNintTypes
    let win = max 1u winSz
    let memoryNeed nPixels = 2UL * nPixels * uint64 win * getBytesPerComponent<'T>
    let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed
    let costUnits input = float (inputValue input * uint64 win)
    liftWindowedUnaryReleaseAfter "sqrtWindowed" win ImageFunctions.sqrtImage memoryNeed id
    |> withCostModel (nativeImageStageCost $"sqrtWindowed.{typeof<'T>.Name}" memoryModel costUnits)
let square<'T when 'T: equality> : Stage<Image<'T>,Image<'T>> =      
    failTypeMismatch<'T> "square" floatNintTypes
    liftUnaryReleaseAfter "square" ImageFunctions.squareImage id id

let clamp<'T when 'T: equality> lower upper : Stage<Image<'T>, Image<'T>> =
    liftUnaryReleaseAfter "clamp" (ImageFunctions.clampImage lower upper) id id

let shiftScale<'T when 'T: equality> shift scale : Stage<Image<'T>, Image<'T>> =
    liftUnaryReleaseAfter "shiftScale" (ImageFunctions.shiftScale shift scale) id id

let intensityStretch<'T when 'T: equality> inputMinimum inputMaximum outputMinimum outputMaximum : Stage<Image<'T>, Image<'T>> =
    if inputMinimum = inputMaximum then
        invalidArg "inputMaximum" "Cannot stretch intensities from an input range with zero width."
    if outputMinimum = outputMaximum then
        invalidArg "outputMaximum" "Cannot stretch intensities to an output range with zero width using shiftScale."

    let scale = (outputMaximum - outputMinimum) / (inputMaximum - inputMinimum)
    let shift = outputMinimum / scale - inputMinimum
    shiftScale<'T> shift scale

let private defaultConvolutionWindowSize (kernelDepth: uint) =
    max kernelDepth (3u * kernelDepth)

let private makeWindowedLocalOp name ksz winSz core =
    let pad = ksz / 2u
    let win = max ksz winSz
    let stride = win - ksz + 1u
    let f debug = volFctToWindowFctReleaseAfterDebug debug core ksz pad stride
    let stg = mapWindow name f id id
    (window win pad stride) --> stg --> flattenList ()
    |> Stage.withSliceCardinality (SlimPipeline.Domain(sameSliceDomainForKernelDepth ksz))

let smoothWMedian<'T when 'T: equality> radius winSz : Stage<Image<'T>, Image<'T>> =
    let ksz = 2u * radius + 1u
    makeWindowedLocalOp "smoothWMedian" ksz winSz (ImageFunctions.median radius)

let smoothWBilateral<'T when 'T: equality> domainSigma rangeSigma winSz : Stage<Image<'T>, Image<'T>> =
    makeWindowedLocalOp "smoothWBilateral" (max 1u winSz) winSz (ImageFunctions.bilateral domainSigma rangeSigma)

let gradientMagnitude<'T when 'T: equality> winSz : Stage<Image<'T>, Image<'T>> =
    makeWindowedLocalOp "gradientMagnitude" 3u winSz ImageFunctions.gradientMagnitude

let private finiteDiffKernelDepth order =
    let kernel = ImageFunctions.finiteDiffFilter3D 2u order
    try
        max 1u (kernel.GetDepth())
    finally
        kernel.decRefCount()

let gradient (order: uint) (winSz: uint option) : Stage<Image<float>, Image<float list>> =
    let ksz = finiteDiffKernelDepth order
    let pad = ksz / 2u
    let win = Option.defaultValue ksz winSz |> max ksz
    let stride = win - ksz + 1u
    let f debug = volFctToWindowFctReleaseAfterDebug debug (ImageFunctions.gradientVector3D order) ksz pad stride
    let memoryNeed nPixels = (4UL * nPixels * uint64 win) * getBytesPerComponent<float>
    let stg = mapWindow "gradient" f memoryNeed id
    (window win pad stride) --> stg --> flattenList ()
    |> Stage.withSliceCardinality (SlimPipeline.Domain(sameSliceDomainForKernelDepth ksz))

let private gaussianVectorElements (sigma: float) : Stage<Image<float list>, Image<float list>> =
    if sigma <= 0.0 then
        identityStage "gaussianVectorElements.identity"
    else
        let roundFloatToUint v = uint (v + 0.5)
        let ksz = max 1u (4.0 * sigma + 1.0 |> roundFloatToUint)
        let pad = ksz / 2u
        let win = ksz
        let stride = win - ksz + 1u
        let f debug =
            volFctToWindowFctReleaseAfterDebug debug (ImageFunctions.smoothVectorElements3D sigma) ksz pad stride
        let memoryNeed nPixels = (8UL * nPixels * uint64 win) * getBytesPerComponent<float>
        let stg = mapWindow "gaussianVectorElements" f memoryNeed id
        (window win pad stride) --> stg --> flattenList ()
        |> Stage.withSliceCardinality (SlimPipeline.Domain(sameSliceDomainForKernelDepth ksz))

let private splitEigenImages : Stage<Image<float list>, Image<float list>> =
    Stage.map
        "structureTensorEigenImages"
        (fun _ tensor ->
            let result = ImageFunctions.structureTensorEigenImages tensor
            tensor.decRefCount()
            result)
        id
        (fun slices -> slices * 4UL)
    --> flattenList ()

type private PcaAccumulator =
    { Count: uint64
      Components: int
      Sums: float[]
      Products: float[,] }

let private zeroPcaAccumulator (components: int) : PcaAccumulator =
    { Count = 0UL
      Components = components
      Sums = Array.zeroCreate components
      Products = Array2D.zeroCreate components components }

let private addPcaVector (state: PcaAccumulator) (values: float list) : PcaAccumulator =
    let values = values |> List.toArray
    if values.Length <> state.Components then
        failwith $"PCA: expected {state.Components}-component vectors, got {values.Length}."

    let sums = Array.copy state.Sums
    let products = Array2D.copy state.Products
    values |> Array.iteri (fun i value -> sums[i] <- sums[i] + value)
    for i in 0 .. state.Components - 1 do
        for j in i .. state.Components - 1 do
            let product = values[i] * values[j]
            products[i, j] <- products[i, j] + product
            if i <> j then products[j, i] <- products[j, i] + product

    { state with
        Count = state.Count + 1UL
        Sums = sums
        Products = products }

let private vectorImageIndices (image: Image<float list>) =
    match image.GetSize() with
    | [ width; height ] ->
        seq {
            for y in 0u .. height - 1u do
                for x in 0u .. width - 1u do
                    yield [ x; y ]
        }
    | [ width; height; depth ] ->
        seq {
            for z in 0u .. depth - 1u do
                for y in 0u .. height - 1u do
                    for x in 0u .. width - 1u do
                        yield [ x; y; z ]
        }
    | size ->
        invalidArg "image" $"PCA: expected 2D or 3D vector images, got size {size}."

let private pcaOutputImage name components values =
    let image = new Image<float list>([ 1u; 1u ], uint components, name, 0)
    image.Set [ 0u; 0u ] values
    image

let private symmetricEigenN (matrix: float[,]) : (float * float list) list =
    let n = matrix.GetLength(0)
    if n <> matrix.GetLength(1) then invalidArg "matrix" "PCA eigen decomposition expects a square matrix."
    if n < 2 then invalidArg "matrix" "PCA eigen decomposition expects at least two components."

    let a = Array2D.copy matrix
    let v = Array2D.zeroCreate<float> n n
    for i in 0 .. n - 1 do
        v[i, i] <- 1.0

    let rotate p q =
        if System.Math.Abs a[p, q] > 1e-14 then
            let tau = (a[q, q] - a[p, p]) / (2.0 * a[p, q])
            let sign = if tau >= 0.0 then 1.0 else -1.0
            let t = sign / (System.Math.Abs tau + System.Math.Sqrt (1.0 + tau * tau))
            let c = 1.0 / System.Math.Sqrt (1.0 + t * t)
            let s = t * c
            let app = a[p, p]
            let aqq = a[q, q]
            let apq = a[p, q]

            a[p, p] <- app - t * apq
            a[q, q] <- aqq + t * apq
            a[p, q] <- 0.0
            a[q, p] <- 0.0

            for r in 0 .. n - 1 do
                if r <> p && r <> q then
                    let arp = a[r, p]
                    let arq = a[r, q]
                    a[r, p] <- c * arp - s * arq
                    a[p, r] <- a[r, p]
                    a[r, q] <- s * arp + c * arq
                    a[q, r] <- a[r, q]

            for r in 0 .. n - 1 do
                let vrp = v[r, p]
                let vrq = v[r, q]
                v[r, p] <- c * vrp - s * vrq
                v[r, q] <- s * vrp + c * vrq

    for _ in 1 .. System.Math.Max(32, 8 * n * n) do
        for p in 0 .. n - 2 do
            for q in p + 1 .. n - 1 do
                rotate p q

    [ for i in 0 .. n - 1 ->
        let vector = [ for r in 0 .. n - 1 -> v[r, i] ]
        let norm = vector |> List.sumBy (fun value -> value * value) |> System.Math.Sqrt
        let vector = if norm < 1e-18 then vector else vector |> List.map (fun value -> value / norm)
        a[i, i], vector ]
    |> List.sortByDescending fst

let private pcaImages (state: PcaAccumulator) : Image<float list> list =
    if state.Count = 0UL then
        invalidOp "PCA cannot reduce an empty vector image stream."

    let n = float state.Count
    let means = state.Sums |> Array.map (fun sum -> sum / n)
    let covariance = Array2D.zeroCreate<float> state.Components state.Components
    for i in 0 .. state.Components - 1 do
        for j in 0 .. state.Components - 1 do
            covariance[i, j] <- state.Products[i, j] / n - means[i] * means[j]

    let eigen = symmetricEigenN covariance
    let eigenvalues = eigen |> List.map fst
    [ yield pcaOutputImage "PCAEigenvalues" state.Components eigenvalues
      for index, (_, vector) in eigen |> List.indexed do
          yield pcaOutputImage $"PCAEigenvector{index}" state.Components vector ]

let PCA components : Stage<Image<float list>, Image<float list>> =
    if components < 2u then invalidArg "components" "PCA needs at least two vector components."
    let components = int components
    let reducer (_debug: bool) (input: AsyncSeq<Image<float list>>) =
        async {
            let! state =
                input
                |> AsyncSeq.foldAsync
                    (fun state image ->
                        async {
                            try
                                if image.GetNumberOfComponentsPerPixel() <> uint components then
                                    invalidArg "image" $"PCA: expected {components}-component vector images, got {image.GetNumberOfComponentsPerPixel()} components."

                                let state' =
                                    vectorImageIndices image
                                    |> Seq.fold (fun acc idx -> addPcaVector acc (image.Get idx)) state
                                return state'
                            finally
                                image.decRefCount()
                        })
                    (zeroPcaAccumulator components)

            return pcaImages state
        }

    Stage.reduce "PCA" reducer Streaming (fun _ -> uint64 ((components + 1) * components * sizeof<float>)) (fun _ -> uint64 (components + 1))
    --> flattenList ()

let selectGroupedOutput (groupSize: uint) (part: uint) : Stage<Image<'T>, Image<'T>> =
    if groupSize = 0u then
        invalidArg "groupSize" "selectGroupedOutput: groupSize must be positive."
    if part >= groupSize then
        invalidArg "part" $"selectGroupedOutput: part must be smaller than groupSize ({groupSize})."

    Stage.mapi
        "selectGroupedOutput"
        (fun _ index image ->
            if uint (index % int64 groupSize) = part then
                [ image ]
            else
                image.decRefCount()
                [])
        id
        (fun slices -> (slices + uint64 groupSize - 1UL) / uint64 groupSize)
    --> flattenList ()

let sobelEdge<'T when 'T: equality> winSz : Stage<Image<'T>, Image<'T>> =
    makeWindowedLocalOp "sobelEdge" 3u winSz ImageFunctions.sobelEdge

let laplacian<'T when 'T: equality> winSz : Stage<Image<'T>, Image<'T>> =
    makeWindowedLocalOp "laplacian" 3u winSz ImageFunctions.laplacian

let grayscaleErode<'T when 'T: equality> radius winSz : Stage<Image<'T>, Image<'T>> =
    let ksz = 2u * radius + 1u
    makeWindowedLocalOp "grayscaleErode" ksz winSz (ImageFunctions.grayscaleErode radius)

let grayscaleDilate<'T when 'T: equality> radius winSz : Stage<Image<'T>, Image<'T>> =
    let ksz = 2u * radius + 1u
    makeWindowedLocalOp "grayscaleDilate" ksz winSz (ImageFunctions.grayscaleDilate radius)

let grayscaleOpening<'T when 'T: equality> radius winSz : Stage<Image<'T>, Image<'T>> =
    let ksz = 2u * radius + 1u
    makeWindowedLocalOp "grayscaleOpening" ksz winSz (ImageFunctions.grayscaleOpening radius)

let grayscaleClosing<'T when 'T: equality> radius winSz : Stage<Image<'T>, Image<'T>> =
    let ksz = 2u * radius + 1u
    makeWindowedLocalOp "grayscaleClosing" ksz winSz (ImageFunctions.grayscaleClosing radius)

let whiteTopHat<'T when 'T: equality> radius winSz : Stage<Image<'T>, Image<'T>> =
    let ksz = 2u * radius + 1u
    makeWindowedLocalOp "whiteTopHat" ksz winSz (ImageFunctions.whiteTopHat radius)

let blackTopHat<'T when 'T: equality> radius winSz : Stage<Image<'T>, Image<'T>> =
    let ksz = 2u * radius + 1u
    makeWindowedLocalOp "blackTopHat" ksz winSz (ImageFunctions.blackTopHat radius)

let morphologicalGradient<'T when 'T: equality> radius winSz : Stage<Image<'T>, Image<'T>> =
    let ksz = 2u * radius + 1u
    makeWindowedLocalOp "morphologicalGradient" ksz winSz (ImageFunctions.morphologicalGradient radius)

let binaryContour (fullyConnected: bool) (winSz: uint) =
    makeWindowedLocalOp "binaryContour" 3u winSz (ImageFunctions.binaryContour fullyConnected)

let binaryMedian radius winSz =
    let ksz = 2u * radius + 1u
    makeWindowedLocalOp "binaryMedian" ksz winSz (ImageFunctions.binaryMedian radius)

let equal<'T when 'T: equality> : Stage<Image<'T> * Image<'T>, Image<uint8>> =
    liftPairReleaseAfter "equal" ImageFunctions.equalImage

let notEqual<'T when 'T: equality> : Stage<Image<'T> * Image<'T>, Image<uint8>> =
    liftPairReleaseAfter "notEqual" ImageFunctions.notEqualImage

let greater<'T when 'T: equality> : Stage<Image<'T> * Image<'T>, Image<uint8>> =
    liftPairReleaseAfter "greater" ImageFunctions.greaterImage

let greaterEqual<'T when 'T: equality> : Stage<Image<'T> * Image<'T>, Image<uint8>> =
    liftPairReleaseAfter "greaterEqual" ImageFunctions.greaterEqualImage

let less<'T when 'T: equality> : Stage<Image<'T> * Image<'T>, Image<uint8>> =
    liftPairReleaseAfter "less" ImageFunctions.lessImage

let lessEqual<'T when 'T: equality> : Stage<Image<'T> * Image<'T>, Image<uint8>> =
    liftPairReleaseAfter "lessEqual" ImageFunctions.lessEqualImage

let maskAnd : Stage<Image<uint8> * Image<uint8>, Image<uint8>> =
    liftPairReleaseAfter "maskAnd" ImageFunctions.andImage

let maskOr : Stage<Image<uint8> * Image<uint8>, Image<uint8>> =
    liftPairReleaseAfter "maskOr" ImageFunctions.orImage

let maskXor : Stage<Image<uint8> * Image<uint8>, Image<uint8>> =
    liftPairReleaseAfter "maskXor" ImageFunctions.xorImage

let maskNot : Stage<Image<uint8>, Image<uint8>> =
    liftUnaryReleaseAfter "maskNot" ImageFunctions.notImage id id

let labelContour<'T when 'T: equality> fullyConnected winSz : Stage<Image<'T>, Image<'T>> =
    makeWindowedLocalOp "labelContour" 3u winSz (ImageFunctions.labelContour fullyConnected)

let changeLabel<'T when 'T: equality> fromLabel toLabel : Stage<Image<'T>, Image<'T>> =
    liftUnaryReleaseAfter "changeLabel" (ImageFunctions.changeLabel fromLabel toLabel) id id

let resize<'T when 'T: equality>
    (outputWidth: uint)
    (outputHeight: uint)
    (outputDepth: uint)
    (interpolationName: string)
    (pl: Plan<unit, Image<'T>>)
    : Plan<unit, Image<'T>> =

    let outputWidth = max 1u outputWidth
    let outputHeight = max 1u outputHeight
    let outputDepth = max 1u outputDepth
    let interpolation = ResampleInterpolation.parse interpolationName

    let inputWidth =
        pl.sourcePeek
        |> Option.bind (fun peek -> peek.Shape |> Map.tryFind "width")
        |> Option.bind (fun text -> match UInt32.TryParse text with true, value -> Some value | _ -> None)
    let inputHeight =
        pl.sourcePeek
        |> Option.bind (fun peek -> peek.Shape |> Map.tryFind "height")
        |> Option.bind (fun text -> match UInt32.TryParse text with true, value -> Some value | _ -> None)

    let spacingX =
        inputWidth
        |> Option.map (fun width -> outputSpacingForSize width outputWidth)
        |> Option.defaultValue 1.0
    let spacingY =
        inputHeight
        |> Option.map (fun height -> outputSpacingForSize height outputHeight)
        |> Option.defaultValue 1.0

    let xy =
        resampleXYStage<'T> outputWidth outputHeight spacingX spacingY interpolation
    let z =
        zResampleStage<'T> pl.length outputDepth None interpolation

    pl >=> (xy --> z)

let resample<'T when 'T: equality>
    (factorX: float)
    (factorY: float)
    (factorZ: float)
    (interpolationName: string)
    (pl: Plan<unit, Image<'T>>)
    : Plan<unit, Image<'T>> =

    if factorX <= 0.0 || factorY <= 0.0 || factorZ <= 0.0 then
        invalidArg "factor" "resample factors must be positive."

    let interpolation = ResampleInterpolation.parse interpolationName
    let outputDepth = max 1u (uint (Math.Round(float pl.length * factorZ, MidpointRounding.AwayFromZero)))
    let inputWidth =
        pl.sourcePeek
        |> Option.bind (fun peek -> peek.Shape |> Map.tryFind "width")
        |> Option.bind (fun text -> match UInt32.TryParse text with true, value -> Some value | _ -> None)
    let inputHeight =
        pl.sourcePeek
        |> Option.bind (fun peek -> peek.Shape |> Map.tryFind "height")
        |> Option.bind (fun text -> match UInt32.TryParse text with true, value -> Some value | _ -> None)
    let outputWidth =
        inputWidth
        |> Option.map (fun width -> roundPositiveToUInt (float width * factorX))
        |> Option.defaultValue (roundPositiveToUInt (Math.Sqrt(float (SingleOrPair.fst pl.nElemsPerSlice)) * factorX))
    let outputHeight =
        inputHeight
        |> Option.map (fun height -> roundPositiveToUInt (float height * factorY))
        |> Option.defaultValue (roundPositiveToUInt (Math.Sqrt(float (SingleOrPair.fst pl.nElemsPerSlice)) * factorY))

    let xy =
        resampleXYStage<'T> outputWidth outputHeight (1.0 / factorX) (1.0 / factorY) interpolation
    let z =
        zResampleStage<'T> pl.length outputDepth (Some factorZ) interpolation

    pl >=> (xy --> z)

//let histogram<'T when 'T: comparison> = histogramOp<'T> "histogram"
let imageHistogram () =
    Stage.map<Image<'T>,Map<'T,uint64>> "histogram: map" (fun _ -> releaseAfter ImageFunctions.histogram) id id// Assumed max for uint8, can be done better

let imageHistogramFold () =
    Stage.fold<Map<'T,uint64>, Map<'T,uint64>> "histogram: fold" ImageFunctions.addHistogram (Map.empty<'T, uint64>) id (fun _ -> 1UL)

let imHistogram () =
    imageHistogram ()
    --> imageHistogramFold ()
    --> Stage.map "histogram: metadata" (fun _ -> Histogram.ofMap) id id

let imageHistogramFixedBins firstLeftEdge lastLeftEdge bins =
    Stage.map<Image<'T>,Map<float,uint64>> "histogramFixedBins: map" (fun _ -> releaseAfter (ImageFunctions.histogramFixedBins firstLeftEdge lastLeftEdge bins)) id id

let imHistogramFixedBins firstLeftEdge lastLeftEdge bins =
    imageHistogramFixedBins firstLeftEdge lastLeftEdge bins
    --> imageHistogramFold ()
    --> Stage.map "histogramFixedBins: metadata" (fun _ -> Histogram.withFixedEdges firstLeftEdge lastLeftEdge bins) id id

type HistogramEstimator =
    | DKW
    | Holdout
    | DKWAndHoldout

module HistogramEstimator =
    let parse (value: string) =
        match value.Trim().ToLowerInvariant().Replace("-", "").Replace("_", "").Replace(" ", "") with
        | "dkw" -> DKW
        | "holdout"
        | "halftest"
        | "halftesting" -> Holdout
        | ""
        | "both"
        | "dkwandholdout"
        | "holdoutanddkw" -> DKWAndHoldout
        | _ -> invalidArg "estimator" $"Unknown histogram estimator '{value}'. Use DKW, Holdout, or DKWAndHoldout."

type HistogramEstimate<'T when 'T: comparison> =
    { Histogram: Histogram<'T>
      Samples: uint64
      Confidence: float
      CdfHalfWidth: float
      HoldoutMaxCdfDelta: float }

let histogramEstimateMap<'T when 'T: comparison> =
    Stage.map<HistogramEstimate<'T>, Histogram<'T>> "histogramEstimateMap" (fun _ estimate -> estimate.Histogram) id id

let private addHistogramValue value histogram =
    histogram
    |> Map.change value (function
        | Some count -> Some(count + 1UL)
        | None -> Some 1UL)

let private histogramCdfValues<'T when 'T: comparison> (histogram: Map<'T,uint64>) =
    histogram
    |> Map.toArray
    |> Array.map (fun (key, count) -> Convert.ToDouble(key), count)
    |> Array.sortBy fst

let private histogramCdfAt (bins: (float * uint64) array) total value =
    if total = 0UL || bins.Length = 0 then
        0.0
    else
        let mutable cumulative = 0UL
        let mutable index = 0
        while index < bins.Length && fst bins[index] <= value do
            cumulative <- cumulative + snd bins[index]
            index <- index + 1
        float cumulative / float total

let private holdoutMaxCdfDelta<'T when 'T: comparison> (left: Map<'T,uint64>) (right: Map<'T,uint64>) =
    if Map.isEmpty left || Map.isEmpty right then
        nan
    else
        let leftBins = histogramCdfValues left
        let rightBins = histogramCdfValues right
        let leftTotal = leftBins |> Array.sumBy snd
        let rightTotal = rightBins |> Array.sumBy snd

        Array.concat [ leftBins |> Array.map fst; rightBins |> Array.map fst ]
        |> Array.distinct
        |> Array.sort
        |> Array.fold
            (fun maxDelta key ->
                let delta = Math.Abs(histogramCdfAt leftBins leftTotal key - histogramCdfAt rightBins rightTotal key)
                max maxDelta delta)
            0.0

let private dkwHalfWidth samples confidence =
    if samples = 0UL then
        nan
    else
        let confidence = min 0.999999999999 (max 0.0 confidence)
        let alpha = max 1e-12 (1.0 - confidence)
        Math.Sqrt(Math.Log(2.0 / alpha) / (2.0 * float samples))

let histogramEstimate<'T when 'T: equality and 'T: comparison> down estimatorName confidence =
    if down = 0u then
        invalidArg (nameof down) "histogramEstimate down must be positive."

    let estimator = HistogramEstimator.parse estimatorName
    let step = int down

    let reducer (_debug: bool) (input: AsyncSeq<Image<'T>>) =
        async {
            let mutable histogram = Map.empty<'T,uint64>
            let mutable holdoutLeft = Map.empty<'T,uint64>
            let mutable holdoutRight = Map.empty<'T,uint64>
            let mutable samples = 0UL

            do!
                input
                |> AsyncSeq.iterAsync (fun image ->
                    async {
                        try
                            let width = int (image.GetWidth())
                            let height = int (image.GetHeight())
                            let pixels = image.toArray2D()

                            for y in 0 .. step .. height - 1 do
                                for x in 0 .. step .. width - 1 do
                                    let value = pixels[x, y]
                                    histogram <- histogram |> addHistogramValue value

                                    if samples % 2UL = 0UL then
                                        holdoutLeft <- holdoutLeft |> addHistogramValue value
                                    else
                                        holdoutRight <- holdoutRight |> addHistogramValue value

                                    samples <- samples + 1UL
                        finally
                            image.decRefCount()
                    })

            let cdfHalfWidth =
                match estimator with
                | DKW
                | DKWAndHoldout -> dkwHalfWidth samples confidence
                | Holdout -> nan

            let holdoutDelta =
                match estimator with
                | Holdout
                | DKWAndHoldout -> holdoutMaxCdfDelta holdoutLeft holdoutRight
                | DKW -> nan

            return
                { Histogram = Histogram.ofMap histogram
                  Samples = samples
                  Confidence = confidence
                  CdfHalfWidth = cdfHalfWidth
                  HoldoutMaxCdfDelta = holdoutDelta }
        }

    Stage.reduce $"histogramEstimate.{typeof<'T>.Name}" reducer Streaming id id

let estimateHistogram<'T when 'T: equality and 'T: comparison>
    slices
    inputDir
    suffix
    down
    estimator
    confidence
    (pl: Plan<unit, unit>)
    : Plan<unit, HistogramEstimate<'T>> =

    if slices = 0u then
        invalidArg (nameof slices) "estimateHistogram slices must be positive."

    pl
    |> StackIO.readRandom<'T> slices inputDir suffix
    >=> histogramEstimate<'T> down estimator confidence

let private histogramKeyToFloat<'T> (value: 'T) =
    Convert.ToDouble(value)

let private histogramEqualizationLookup<'T when 'T: comparison> (histogram: Map<'T,uint64>) =
    if Map.isEmpty histogram then
        invalidArg (nameof histogram) "Cannot equalize from an empty histogram."

    let bins =
        histogram
        |> Map.toArray
        |> Array.map (fun (key, count) -> histogramKeyToFloat key, count)
        |> Array.sortBy fst

    let total = bins |> Array.sumBy snd
    if total = 0UL then
        invalidArg (nameof histogram) "Cannot equalize from a histogram whose total count is zero."

    let firstCount = snd bins[0]
    let denominator = float (total - firstCount)
    let keys = bins |> Array.map fst
    let equalized = Array.zeroCreate<float> bins.Length
    let mutable cumulative = 0UL

    for i in 0 .. bins.Length - 1 do
        cumulative <- cumulative + snd bins[i]
        equalized[i] <-
            if denominator <= 0.0 then
                0.0
            else
                (float (cumulative - firstCount)) / denominator

    fun value ->
        if value <= keys[0] then
            equalized[0]
        elif value >= keys[keys.Length - 1] then
            equalized[equalized.Length - 1]
        else
            let search = Array.BinarySearch(keys, value)
            if search >= 0 then
                equalized[search]
            else
                let upper = ~~~search
                let lower = upper - 1
                let span = keys[upper] - keys[lower]
                if span <= 0.0 then
                    equalized[lower]
                else
                    let t = (value - keys[lower]) / span
                    equalized[lower] + t * (equalized[upper] - equalized[lower])

let histogramEqualization<'T when 'T: equality and 'T: comparison> (histogram: Histogram<'T>) =
    let lookup = histogramEqualizationLookup histogram.Counts

    let equalize (_debug: bool) (image: Image<'T>) =
        try
            let width = int (image.GetWidth())
            let height = int (image.GetHeight())
            let pixels = image.toArray2D()
            let output =
                Array2D.init width height (fun x y ->
                    pixels[x, y]
                    |> histogramKeyToFloat
                    |> lookup)
                |> Image<float>.ofArray2D

            output.index <- image.index
            output
        finally
            image.decRefCount()

    Stage.map "histogramEqualization" equalize imageBytes<float> id

module ProjectionTransform =
    let values =
        [ "Identity"
          "Abs"
          "Square"
          "SqrtAbs"
          "Log1pAbs" ]

    let apply (name: string) : float -> float =
        match name.Trim().ToLowerInvariant().Replace("-", "").Replace("_", "").Replace(" ", "") with
        | ""
        | "identity"
        | "none" -> id
        | "abs"
        | "absolute" -> fun value -> Math.Abs value
        | "square"
        | "squared" -> fun value -> value * value
        | "sqrtabs"
        | "sqrt"
        | "squareroot" -> fun value -> Math.Sqrt(Math.Abs value)
        | "log1pabs"
        | "log"
        | "logabs" -> fun value -> Math.Log(1.0 + Math.Abs value)
        | _ ->
            let options = String.Join(", ", values)
            failwith $"Unknown projection transform '{name}'. Use one of: {options}."

let sumProjection<'T when 'T: equality> transformName : Stage<Image<'T>, Image<float>> =
    let transform = ProjectionTransform.apply transformName

    let reducer (_debug: bool) (input: AsyncSeq<Image<'T>>) =
        async {
            let! state =
                input
                |> AsyncSeq.foldAsync
                    (fun state image ->
                        async {
                            try
                                let width = int (image.GetWidth())
                                let height = int (image.GetHeight())
                                let pixels = image.toArray2D()
                                let accumulator =
                                    match state with
                                    | None ->
                                        Some(width, height, Array2D.zeroCreate<float> width height)
                                    | Some(expectedWidth, expectedHeight, values) ->
                                        if expectedWidth <> width || expectedHeight <> height then
                                            invalidOp $"sumProjection requires constant x-y slice size; got {width}x{height}, expected {expectedWidth}x{expectedHeight}."
                                        Some(expectedWidth, expectedHeight, values)

                                match accumulator with
                                | None -> return None
                                | Some(_, _, values) ->
                                    for y in 0 .. height - 1 do
                                        for x in 0 .. width - 1 do
                                            values[x, y] <- values[x, y] + transform (Convert.ToDouble pixels[x, y])
                                    return accumulator
                            finally
                                image.decRefCount()
                        })
                    None

            match state with
            | None ->
                return raise (InvalidOperationException "sumProjection cannot reduce an empty image stream.")
            | Some(_, _, values) ->
                let image = Image<float>.ofArray2D(values, "sumProjection")
                image.index <- 0
                return image
        }

    let memoryNeed nElems =
        imageBytes<float> nElems

    Stage.reduce $"sumProjection {transformName}" reducer Streaming memoryNeed id

let volume xUnit yUnit zUnit : Stage<Image<uint8>, float> =
    if xUnit <= 0.0 then invalidArg (nameof xUnit) "xUnit must be positive."
    if yUnit <= 0.0 then invalidArg (nameof yUnit) "yUnit must be positive."
    if zUnit <= 0.0 then invalidArg (nameof zUnit) "zUnit must be positive."

    let voxelVolume = xUnit * yUnit * zUnit

    let folder volume (image: Image<uint8>) =
        try
            let width = int (image.GetWidth())
            let height = int (image.GetHeight())
            let pixels = image.toArray2D()
            let mutable foreground = 0UL
            for y in 0 .. height - 1 do
                for x in 0 .. width - 1 do
                    match pixels[x, y] with
                    | 0uy -> ()
                    | 1uy -> foreground <- foreground + 1UL
                    | value ->
                        invalidOp $"volume expects a UInt8 0-1 mask stream; got pixel value {value} at ({x}, {y}) in slice {image.index}."

            volume + float foreground * voxelVolume
        finally
            image.decRefCount()

    Stage.fold "volume" folder 0.0 id (fun _ -> 1UL)

let quantiles (quantileValues: float list) (histogram: Histogram<'T>) =
    ImageFunctions.quantilesFromHistogram quantileValues histogram.Counts

let otsuThresholdFromHistogram histogram =
    ImageFunctions.otsuThresholdFromHistogram histogram.Counts

let momentsThresholdFromHistogram histogram =
    ImageFunctions.momentsThresholdFromHistogram histogram.Counts

let inline histogram2pairs< ^T when ^T: comparison and ^T: (static member op_Explicit: ^T -> float) > =
    let histogram2pairs (histogram: Histogram<'T>) : ('T * uint64) list =
        histogram.Counts |> Map.toList
    liftUnary "histogram2pairs" histogram2pairs id id

let histogramCounts<'T when 'T: comparison> =
    Stage.map<Histogram<'T>, Map<'T, uint64>> "histogramCounts" (fun _ histogram -> histogram.Counts) id id

let inline map2pairs< ^T, ^S when ^T: comparison and ^T: (static member op_Explicit: ^T -> float) and  ^S: (static member op_Explicit: ^S -> float) > = 
    let map2pairs (map: Map<'T, 'S>) : ('T * 'S) list =
        map |> Map.toList
    liftUnary "map2pairs" map2pairs id id
let inline pairs2floats< ^T, ^S when ^T: (static member op_Explicit: ^T -> float) and  ^S: (static member op_Explicit: ^S -> float) > = 
    let pairs2floats (pairs: (^T * ^S) list) : (float * float) list =
        pairs |> List.map (fun (k, v) -> (float k, float v)) 
    liftUnary "pairs2floats" pairs2floats id id
let inline pairs2ints< ^T, ^S when ^T: (static member op_Explicit: ^T -> int) and  ^S: (static member op_Explicit: ^S -> int) > = 
    let pairs2ints (pairs: (^T * ^S) list) : (int * int) list =
        pairs |> List.map (fun (k, v) -> (int k, int v)) 
    liftUnary "pairs2ints" pairs2ints id id

type ImageStats = ImageFunctions.ImageStats
let imageComputeStats () =
    Stage.map<Image<'T>,ImageStats> "computeStats: map" (fun _ -> releaseAfter ImageFunctions.computeStats) id id

let imageComputeStatsFold () =
    let zeroStats: ImageStats = { 
        NumPixels = 0u
        Mean = 0.0
        Std = 0.0
        Min = infinity
        Max = -infinity
        Sum = 0.0
        Var = 0.0
    }
    Stage.fold<ImageStats, ImageStats> "computeStats: fold" ImageFunctions.addComputeStats zeroStats id id

let computeStats () =
    imageComputeStats () --> imageComputeStatsFold ()

////////////////////////////////////////////////
/// Convolution like operators

// Chained type definitions do expose the originals
open type ImageFunctions.OutputRegionMode
open type ImageFunctions.BoundaryCondition

let stackFUnstack f (images: Image<'T> list) =
    let stck = images |> ImageFunctions.stack 
    stck |> releaseAfter (f >> ImageFunctions.unstack)

let skipNTakeM (n: uint) (m: uint) (lst: 'a list) : 'a list =
    let m = uint lst.Length - 2u*n;
    if m = 0u then []
    else lst |> List.skip (int n) |> List.take (int m) // This needs releaseAfter!!!

let stackFUnstackTrim trim (f: Image<'T>->Image<'S>) (images: Image<'T> list) =
    let stck = images |> ImageFunctions.stack 
    let result = 
        let volRes = f stck
        stck.decRefCount()
        let m = uint images.Length - 2u*trim // last stack may be smaller if stride > 1
        let imageLst = ImageFunctions.unstackSkipNTakeM trim m volRes
        volRes.decRefCount()
        imageLst
    result

let smoothWGauss (sigma: float) (outputRegionMode: ImageFunctions.OutputRegionMode option) (boundaryCondition: ImageFunctions.BoundaryCondition option) (winSz: uint option) : Stage<Image<float>, Image<float>> =
    let ksz = ImageFunctions.defaultGaussWindowSize sigma
    let kernel = ImageFunctions.gauss 3u sigma (Some ksz)
    let win = Option.defaultValue (defaultConvolutionWindowSize ksz) winSz |> max ksz // max should be found by memory availability
    let stride = win - ksz + 1u
    let pad = 
        match outputRegionMode with
            | Some Valid -> 0u
            | _ -> ksz/2u //floor
    let outputStart =
        match outputRegionMode with
            | Some Valid -> 0u
            | _ -> pad
    // length-2*pad = n*stride+windowSize
    // stride = windowSize-2*pad
    // => n = (windowSize-length-2*pad)/(2*pad-windowSize)
    // e.g., integer solutions for 
    // windowSize = 1, 6, 15, or 26, pad = 2, length = 22, => n = 21, 10, 1, or 0
    let f debug = 
        if debug && DebugLevel.current() >= 2u then printfn $"smoothWGauss: sigma {sigma}, ksz {ksz}, win {win}, stride {stride}, pad {pad}"
        volFctToWindowFctReleaseAfterDebug debug (fun image3D -> ImageFunctions.convolve outputRegionMode boundaryCondition image3D kernel) ksz outputStart stride
    let memoryNeed nPixels = (2UL*nPixels*(uint64 win) + (uint64 ksz))*getBytesPerComponent<float>
    let elementTransformation = id
    let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed
    let costUnits input =
        let kernelVoxels = uint64 ksz * uint64 ksz * uint64 ksz
        float (inputValue input * uint64 win * kernelVoxels)
    let stg =
        mapWindow "smoothWGauss" f memoryNeed elementTransformation
        |> withCostModel (nativeImageStageCost $"smoothWGauss.Float64" memoryModel costUnits)
    (window win pad stride) --> stg --> flattenList () --> cleanStage "smoothWGauss.cleanup" (fun () -> kernel.decRefCount())
    |> Stage.withSliceCardinality (sliceCardinalityForConvolution ksz outputRegionMode)

let structureTensor (sigma: float) (rho: float) : Stage<Image<float>, Image<float list>> =
    let preSmooth =
        if sigma <= 0.0 then identityStage "structureTensor.preSmooth.identity"
        else smoothWGauss sigma None None None

    preSmooth
    --> gradient 1u None
    --> liftUnaryReleaseAfter "structureTensorOuterProduct" ImageFunctions.structureTensorOuterProduct id id
    --> gaussianVectorElements rho
    --> splitEigenImages

// stride calculation example
// ker = 3, win = 7
// Image position: 2 1 0 1 2 3 4 5 6 7 8 9 
// First window         * * * * * * *
// Kern position1   * * *            
//                    * * *         
//                      * * * √        
//                        * * * √      
//                          * * * √   
//                            * * * √    
//                              * * * √   
//                                * * *
//                                  * * *
//                                    * * *
// Next window                    * * * * * * *
// Kern                       * * *
//                              * * *         
//                                * * * √  
//                                  * * * √   
//                                    * * * √
//.                                     * * * √
//                                        * * * √
//                                          * * *
//                                            * * *

let private constantImageLike<'T when 'T: equality> name index (size: uint list) components value =
    let image = Image<'T>(size, components, name, index)
    let filled = Image.map (fun _ -> value) image
    filled.index <- index
    image.decRefCount()
    filled

let private convertPaddingValue<'T> (value: double) : 'T =
    let t = typeof<'T>
    if t = typeof<float> then box value :?> 'T
    elif t = typeof<float32> then box (float32 value) :?> 'T
    elif t = typeof<uint8> then box (uint8 (max 0.0 (min 255.0 (Math.Round value)))) :?> 'T
    elif t = typeof<int8> then box (int8 (max -128.0 (min 127.0 (Math.Round value)))) :?> 'T
    elif t = typeof<uint16> then box (uint16 (max 0.0 (min 65535.0 (Math.Round value)))) :?> 'T
    elif t = typeof<int16> then box (int16 (max -32768.0 (min 32767.0 (Math.Round value)))) :?> 'T
    elif t = typeof<uint32> || t = typeof<uint> then box (uint32 (max 0.0 (min (float UInt32.MaxValue) (Math.Round value)))) :?> 'T
    elif t = typeof<int32> || t = typeof<int> then box (int32 (max (float Int32.MinValue) (min (float Int32.MaxValue) (Math.Round value)))) :?> 'T
    elif t = typeof<uint64> then box (uint64 (max 0.0 (min (float UInt64.MaxValue) (Math.Round value)))) :?> 'T
    elif t = typeof<int64> then box (int64 (max (float Int64.MinValue) (min (float Int64.MaxValue) (Math.Round value)))) :?> 'T
    else failwith $"Padding value conversion is not supported for pixel type {t.Name}."

let createPadding<'T when 'T: equality> beforeX afterX beforeY afterY beforeZ afterZ paddingValue : Stage<Image<'T>, Image<'T>> =
    let name = "createPadding"
    let padXY (image: Image<'T>) =
        try
            ImageFunctions.constantPad2D [ beforeX; beforeY ] [ afterX; afterY ] paddingValue image
        finally
            image.decRefCount()

    let apply (_debug: bool) (input: AsyncSeq<Image<'T>>) =
        asyncSeq {
            let! prefixAndRest = input |> AsyncSeq.splitAt 1
            let firstItems, rest = prefixAndRest
            match firstItems |> Array.tryHead with
            | None -> ()
            | Some first ->
                let paddingPixel = convertPaddingValue<'T> paddingValue
                let firstPadded = padXY first
                let mutable lastSize = firstPadded.GetSize()
                let mutable lastComponents = firstPadded.GetNumberOfComponentsPerPixel()
                let mutable nextIndex = firstPadded.index + 1

                for index in 0 .. int beforeZ - 1 do
                    yield constantImageLike<'T> "padding" (firstPadded.index - int beforeZ + index) lastSize lastComponents paddingPixel

                yield firstPadded

                for image in rest do
                    let padded = padXY image
                    lastSize <- padded.GetSize()
                    lastComponents <- padded.GetNumberOfComponentsPerPixel()
                    nextIndex <- padded.index + 1
                    yield padded

                for index in 0 .. int afterZ - 1 do
                    yield constantImageLike<'T> "padding" (nextIndex + index) lastSize lastComponents paddingPixel
        }

    let transition = ProfileTransition.create Streaming Streaming
    let memoryNeed nPixels = 2UL * nPixels * getBytesPerComponent<'T>
    let pipe = { Name = name; Apply = apply; Profile = Streaming }
    Stage.fromPipe name transition memoryNeed id pipe
    |> Stage.withSliceCardinality (SlimPipeline.Domain(SlimPipeline.SliceDomain.expand beforeZ afterZ))

let crop<'T when 'T: equality> beforeX afterX beforeY afterY beforeZ afterZ : Stage<Image<'T>, Image<'T>> =
    let name = "crop"
    let cropXY (image: Image<'T>) =
        try
            ImageFunctions.crop2D [ beforeX; beforeY ] [ afterX; afterY ] image
        finally
            image.decRefCount()

    let apply (_debug: bool) (input: AsyncSeq<Image<'T>>) =
        asyncSeq {
            let pending = Queue<Image<'T>>()
            let mutable skipped = 0u

            for image in input do
                if skipped < beforeZ then
                    skipped <- skipped + 1u
                    image.decRefCount()
                else
                    pending.Enqueue(cropXY image)
                    if pending.Count > int afterZ then
                        yield pending.Dequeue()

            while pending.Count > 0 do
                pending.Dequeue().decRefCount()
        }

    let transition = ProfileTransition.create Streaming Streaming
    let memoryNeed nPixels = nPixels * getBytesPerComponent<'T>
    let pipe = { Name = name; Apply = apply; Profile = Streaming }
    Stage.fromPipe name transition memoryNeed id pipe
    |> Stage.withSliceCardinality (SlimPipeline.Domain(SlimPipeline.SliceDomain.trim beforeZ afterZ))

let convolveOp (name: string) (kernel: Image<'T>) (outputRegionMode: ImageFunctions.OutputRegionMode option) (bc: ImageFunctions.BoundaryCondition option) (winSz: uint option) : Stage<Image<'T>, Image<'T>> =
    let kernel =
        if kernel.GetDimensions() = 3u && kernel.GetDepth() = 1u then
            ImageFunctions.extractSlice 2u 0 kernel
        else
            kernel

    if kernel.GetDimensions() < 3u then
        let memoryNeed nPixels =
            2UL * nPixels * getBytesPerComponent<'T>

        let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed
        let costUnits input =
            let kernelVoxels = uint64 (kernel.GetWidth()) * uint64 (kernel.GetHeight())
            float (inputValue input * kernelVoxels)

        liftUnaryReleaseAfter name (fun image -> ImageFunctions.convolve outputRegionMode bc image kernel) memoryNeed id
        |> withCostModel (nativeImageStageCost $"convolve.{typeof<'T>.Name}" memoryModel costUnits)
    else
        let windowFromKernel (k: Image<'T>) : uint =
            max 1u (k.GetDepth())
        let ksz = windowFromKernel kernel
        let win = Option.defaultValue (defaultConvolutionWindowSize ksz) winSz |> max ksz
        let stride = win-ksz+1u
        let pad =
            match outputRegionMode with
                | Some Valid -> 0u
                | _ -> ksz/2u //floor
        let outputStart =
            match outputRegionMode with
                | Some Valid -> 0u
                | _ -> pad
        let f debug =  volFctToWindowFctReleaseAfterDebug debug (fun image3D -> ImageFunctions.convolve outputRegionMode bc image3D kernel) ksz outputStart stride
        let memoryNeed nPixels = (2UL*nPixels*(uint64 win) + (uint64 ksz))*getBytesPerComponent<'T>
        let elementTransformation = id
        let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed
        let costUnits input =
            let kernelVoxels = uint64 (kernel.GetWidth()) * uint64 (kernel.GetHeight()) * uint64 (kernel.GetDepth())
            float (inputValue input * uint64 win * kernelVoxels)
        let stg =
            mapWindow name f memoryNeed elementTransformation
            |> withCostModel (nativeImageStageCost $"convolve.{typeof<'T>.Name}" memoryModel costUnits)
        (window win pad stride) --> stg --> flattenList ()
        |> Stage.withSliceCardinality (sliceCardinalityForConvolution ksz outputRegionMode)

let convolve kernel outputRegionMode boundaryCondition winSz = convolveOp "convolve" kernel outputRegionMode boundaryCondition winSz
let conv kernel = convolveOp "conv" kernel None None None

let finiteDiff (direction: uint) (order: uint) =
    let kernel = ImageFunctions.finiteDiffFilter3D direction order
    convolveOp "finiteDiff" kernel None None None

// these only works on uint8
let private makeMorphOp (name: string) (radius: uint) (winSz: uint option) (core: uint -> Image<'T> -> Image<'T>) : Stage<Image<'T>,Image<'T>> when 'T: equality =
    let ksz   = 2u * radius + 1u
    let pad = ksz/2u
    let win = Option.defaultValue (defaultConvolutionWindowSize ksz) winSz |> max ksz
    let stride = win - ksz + 1u

    let f debug = volFctToWindowFctReleaseAfterDebug debug (core radius) ksz pad stride
    let stg = mapWindow name f id id
    (window win pad stride) --> stg --> flattenList ()
    |> Stage.withSliceCardinality (SlimPipeline.Domain(sameSliceDomainForKernelDepth ksz))

let erode radius = makeMorphOp "binaryErode"  radius None ImageFunctions.binaryErode
let dilate radius = makeMorphOp "binaryDilate"  radius None ImageFunctions.binaryDilate
let opening radius = makeMorphOp "binaryOpening"  radius None ImageFunctions.binaryOpening
let closing radius = makeMorphOp "binaryClosing"  radius None ImageFunctions.binaryClosing

let connectedComponents (winSz: uint) =
    let pad, stride = 0u, winSz
    let btUint8 = typeof<uint8>|>Image.getBytesPerComponent |> uint64
    let btUint64 = typeof<uint64> |> Image.getBytesPerComponent |> uint64
    // Sliding window applying f cost assuming f has no extra costs: 
    //   window takes winSz Image<uint8>:           nPixels*winSz*1
    //   produces a stack of equal size in <uint8>: 2*nPixels*winSz*1
    //   releases stride Image<uint8>:              nPixels*(2*winSz*1-*stride*1)
    //   produces a stack of equal size in <uint64>: nPixels*(winSz*(2*1+8)-stride*1)
    //   releases <uint8> stack:                    nPixels*(winSz*(1+8)-stride*1)
    //   produces a stride list of Image<uint64>:   nPixels*(winSz*(1+8)+stride*(8-1))
    //   releases <uint64> stack:                   nPixels*(winSz*1+stride*(8-1))
    let memoryNeed nPixels = 
        let bt8 = typeof<uint8>|>Image.getBytesPerComponent |> uint64
        let bt64 = typeof<uint64> |> Image.getBytesPerComponent |> uint64
        let wsz = uint64 winSz
        let str = uint64 stride
        max (nPixels*(wsz*(2UL*bt8+bt64)-str*bt8)) (nPixels*(wsz*(bt8+bt64)+str*(bt64-bt8)))

    let mapper (_debug: bool) (chunkIndex: int64) (window: Window<Image<uint8>>) : Image<uint64> * uint64 =
        let images = window.Items
        let stack = ImageFunctions.stack images
        images |> List.take (min (int stride) images.Length) |> List.iter (fun image -> image.decRefCount())
        let result = ImageFunctions.connectedComponents stack
        stack.decRefCount()
        result.Labels.index <- int chunkIndex * int stride
        result.Labels, result.ObjectCount

    (window winSz pad stride) --> Stage.mapi "connectedComponents" mapper memoryNeed id

let relabelComponents a (winSz: uint) = 
    let pad, stride = 0u, winSz
    let f debug = volFctToWindowFctReleaseAfterDebug debug (ImageFunctions.relabelComponents a) 1u pad stride
    let stg = mapWindow "relabelComponents" f id id
    (window winSz pad stride) --> stg --> flattenList ()

let signedDistanceBand (bandRadius: uint) (stride: uint) =
    if bandRadius = 0u then
        invalidArg "bandRadius" "Band signed distance requires a positive band radius."
    if stride = 0u then
        invalidArg "stride" "Band signed distance requires a positive stride."

    let pad = bandRadius
    let winSz = stride + 2u * bandRadius
    let f debug = volFctToWindowFctReleaseAfterDebug debug (ImageFunctions.bandSignedDistanceMap bandRadius) 1u pad stride
    let stg = mapWindow "signedDistanceBand" f id id
    (window winSz pad stride) --> stg --> flattenList ()
let threshold a b = liftUnaryReleaseAfter "threshold" (ImageFunctions.threshold a b) id id

let addNormalNoise a b = liftUnaryReleaseAfter "addNormalNoise" (ImageFunctions.addNormalNoise a b) id id
let addSaltAndPepperNoise probability = liftUnaryReleaseAfter "addSaltAndPepperNoise" (ImageFunctions.addSaltAndPepperNoise probability) id id
let addShotNoise scale = liftUnaryReleaseAfter "addShotNoise" (ImageFunctions.addShotNoise scale) id id
let addSpeckleNoise stddev = liftUnaryReleaseAfter "addSpeckleNoise" (ImageFunctions.addSpeckleNoise stddev) id id

let show (plt: Image<'T> -> unit) : Stage<Image<'T>, unit> =
    let consumer (debug: bool) (idx: int) (image: Image<'T>) =
        if debug && DebugLevel.current() >= 2u then printfn "[show] Showing image %d" idx
        let width = image.GetWidth() |> int
        let height = image.GetHeight() |> int
        plt image
        image.decRefCount()
    let memoryNeed = id
    Stage.consumeWith "show" consumer memoryNeed

let plot (plt: (float list)->(float list)->unit) : Stage<(float * float) list, unit> = // better be (float*float) list
    let consumer (debug: bool) (idx: int) (points: (float*float) list) =
        if debug && DebugLevel.current() >= 2u then printfn $"[plot] Plotting {points.Length} 2D points"
        let x,y = points |> List.unzip
        plt x y
    let memoryNeed = id
    Stage.consumeWith "plot" consumer memoryNeed

let print () : Stage<'T, unit> =
    let consumer (debug: bool) (idx: int) (elm: 'T) =
        if debug && DebugLevel.current() >= 2u then printfn "[print]"
        printfn "%d -> %A" idx elm
    let memoryNeed = id
    Stage.consumeWith "print" consumer memoryNeed

// Not Pipes nor Operators
let srcStage (name: string) (width: uint) (height: uint) (depth: uint) (mapper: int->Image<'T>) =
    let transition = ProfileTransition.create Unit Streaming
    let memoryNeed = fun _ -> Image<'T>.memoryEstimate width height
    let elementTransformation = id
    Stage.init name depth mapper transition memoryNeed elementTransformation

let srcPlan (debug: bool) (memAvail: uint64) (width: uint) (height: uint) (depth: uint) (stage: Stage<unit,Image<'T>> option) =
    let nElems = (uint64 width) * (uint64 height)
    let memPeak = Image<'T>.memoryEstimate width height
    let sourcePeek =
        SourcePeek.create
            "synthetic-image-stack"
            memPeak
            (Some (uint64 depth))
            (Map.ofList
                [ "kind", "synthetic-image-stack"
                  "width", string width
                  "height", string height
                  "depth", string depth
                  "pixelType", typeof<'T>.Name ])
    Plan.create stage memAvail memPeak nElems (uint64 depth)  debug
    |> Plan.withSourcePeek sourcePeek

let zero<'T when 'T: equality> (width: uint) (height: uint) (depth: uint) (pl: Plan<unit, unit>) : Plan<unit, Image<'T>> =
    let mapper (i: int) : Image<'T> = 
        let image = new Image<'T>([width; height], 1u,$"zero[{i}]", i)
        if pl.debug && DebugLevel.current() >= 1u then printfn "[zero] Created slice %A" i
        image
    let stage = srcStage "zero" width height depth mapper |> Some
    srcPlan pl.debug pl.memAvail width height depth stage

let normalNoise<'T when 'T: equality> (width: uint) (height: uint) (depth: uint) (mean: float) (stddev: float) (pl: Plan<unit, unit>) : Plan<unit, Image<'T>> =
    let mapper (i: int) : Image<'T> =
        let zero = new Image<'T>([width; height], 1u, $"normalNoise.zero[{i}]", i)
        let image = ImageFunctions.addNormalNoise mean stddev zero
        zero.decRefCount()
        if pl.debug && DebugLevel.current() >= 1u then printfn "[normalNoise] Created slice %A" i
        image
    let stage = srcStage "normalNoise" width height depth mapper |> Some
    srcPlan pl.debug pl.memAvail width height depth stage

let private noiseSource<'T when 'T: equality> name width height depth addNoise (pl: Plan<unit, unit>) : Plan<unit, Image<'T>> =
    let mapper (i: int) : Image<'T> =
        let zero = new Image<'T>([width; height], 1u, $"{name}.zero[{i}]", i)
        let image = addNoise zero
        zero.decRefCount()
        if pl.debug && DebugLevel.current() >= 1u then printfn "[%s] Created slice %A" name i
        image
    let stage = srcStage name width height depth mapper |> Some
    srcPlan pl.debug pl.memAvail width height depth stage

let saltAndPepperNoise<'T when 'T: equality> (width: uint) (height: uint) (depth: uint) (probability: float) (pl: Plan<unit, unit>) : Plan<unit, Image<'T>> =
    noiseSource "saltAndPepperNoise" width height depth (ImageFunctions.addSaltAndPepperNoise probability) pl

let shotNoise<'T when 'T: equality> (width: uint) (height: uint) (depth: uint) (scale: float) (pl: Plan<unit, unit>) : Plan<unit, Image<'T>> =
    noiseSource "shotNoise" width height depth (ImageFunctions.addShotNoise scale) pl

let speckleNoise<'T when 'T: equality> (width: uint) (height: uint) (depth: uint) (stddev: float) (pl: Plan<unit, unit>) : Plan<unit, Image<'T>> =
    noiseSource "speckleNoise" width height depth (ImageFunctions.addSpeckleNoise stddev) pl

let createByEuler2DTransform<'T when 'T: equality> (img: Image<'T>) (depth: uint) (transform: uint -> (float*float*float) * (float*float)) (pl: Plan<unit, unit>) : Plan<unit, Image<'T>> =
    // width, heigth, depth should be replaced with shape and shapeUpdate, and mapper should be deferred to outside Core!!!
    let width= img.GetWidth()
    let height = img.GetHeight()
    let mapper (i: int) : Image<'T> =
        let rot, trans = transform (uint i)
        let image = ImageFunctions.euler2DTransform img rot trans
        if pl.debug && DebugLevel.current() >= 1u then printfn "[createByEuler2DTransform] Created slice %A" i
        image
    let stage = srcStage "createByEuler2DTransform" width height depth mapper |> Some
    srcPlan pl.debug pl.memAvail width height depth stage


let empty (pl: Plan<unit, unit>) : Plan<unit, unit> =
    let stage = "empty" |> Stage.empty |> Some
    Plan.create stage pl.memAvail 0UL 0UL 0UL  pl.debug

let writeSlabSlices (outputDir: string) (suffix: string) (winSz: uint) : Stage<Image<'T>, Image<'T>> =
    if (outputDir |> System.IO.Directory.Exists) |> not then
        System.IO.Directory.CreateDirectory(outputDir) |> ignore

    let mapper (debug: bool) (_idx: int64) (labelChunk: Image<'T>) =
        let slices = ImageFunctions.unstack 2u labelChunk
        slices
        |> List.iteri (fun localIndex slice ->
            let globalIndex = labelChunk.index + localIndex
            let fileName = System.IO.Path.Combine(outputDir, sprintf "image_%03d%s" globalIndex suffix)
            if debug then printfn "[writeSlabSlices] Saved image %d to %s as %s" globalIndex fileName (typeof<'T>.Name)
            slice.toFile(fileName)
            slice.decRefCount())
        labelChunk

    Stage.mapi $"writeSlabSlices \"{outputDir}/*{suffix}\"" mapper id id

type ComponentStatistics =
    { Label: uint64
      NumberOfPixels: uint64
      SumX: uint64
      SumY: uint64
      SumZ: uint64
      MinX: uint
      MaxX: uint
      MinY: uint
      MaxY: uint
      MinZ: uint
      MaxZ: uint }

type ConnectedComponentTranslationTable =
    { Labels: (uint * uint64 * uint64) list
      Statistics: ComponentStatistics list }

module ComponentStatistics =
    let create label x y z : ComponentStatistics =
        { Label = label
          NumberOfPixels = 1UL
          SumX = uint64 x
          SumY = uint64 y
          SumZ = uint64 z
          MinX = x
          MaxX = x
          MinY = y
          MaxY = y
          MinZ = z
          MaxZ = z }

    let add left right =
        if left.Label <> right.Label then
            invalidArg "right" "Component statistics can only be added for the same label."
        { Label = left.Label
          NumberOfPixels = left.NumberOfPixels + right.NumberOfPixels
          SumX = left.SumX + right.SumX
          SumY = left.SumY + right.SumY
          SumZ = left.SumZ + right.SumZ
          MinX = min left.MinX right.MinX
          MaxX = max left.MaxX right.MaxX
          MinY = min left.MinY right.MinY
          MaxY = max left.MaxY right.MaxY
          MinZ = min left.MinZ right.MinZ
          MaxZ = max left.MaxZ right.MaxZ }

    let addToMap key stats map =
        map
        |> Map.change key (function
            | Some existing -> Some(add existing stats)
            | None -> Some stats)

    let relabel newLabel (stats: ComponentStatistics) =
        { stats with Label = newLabel }

    let centroid stats =
        if stats.NumberOfPixels = 0UL then
            0.0, 0.0, 0.0
        else
            let n = float stats.NumberOfPixels
            float stats.SumX / n, float stats.SumY / n, float stats.SumZ / n

let private labelChunkStatistics chunk (labelChunk: Image<uint64>) =
    Image.foldi
        (fun index acc label ->
            if label = 0UL then
                acc
            else
                let x = index |> List.tryItem 0 |> Option.defaultValue 0u
                let y = index |> List.tryItem 1 |> Option.defaultValue 0u
                let localZ = index |> List.tryItem 2 |> Option.defaultValue 0u
                let z = uint (labelChunk.index + int localZ)
                let stats = ComponentStatistics.create label x y z
                acc |> ComponentStatistics.addToMap (chunk, label) stats)
        Map.empty
        labelChunk

let makeConnectedComponentTranslationTable (winSz: uint) : Stage<Image<uint64> * uint64, ConnectedComponentTranslationTable> =
    let name = "makeConnectedComponentTranslationTable"

    let addBoundaryEdges graph previousChunk (previous: Image<uint64>) (current: Image<uint64>) =
        Image.fold2
            (fun g p1 p2 ->
                if p1 <> 0UL && p2 <> 0UL then
                    simpleGraph.addEdge (previousChunk,p1) (previousChunk+1u,p2) g
                else
                    g)
            graph
            previous
            current

    let makeBaseMap chunkCounts =
        chunkCounts
        |> Map.toList
        |> List.sortBy fst
        |> List.scan
            (fun (_, nextLabel, _) (chunk, objectCount) ->
                let labels =
                    if objectCount = 0UL then
                        [ (chunk, 0UL), 0UL ]
                    else
                        [ yield (chunk, 0UL), 0UL
                          for label in 1UL .. objectCount do
                              yield (chunk, label), nextLabel + label - 1UL ]
                chunk, nextLabel + objectCount, labels)
            (0u, 1UL, [])
        |> List.collect (fun (_, _, labels) -> labels)
        |> Map.ofList

    let collapseTouchingComponents graph baseMap =
        simpleGraph.connectedComponents graph
        |> List.fold
            (fun translation componentNodes ->
                let target =
                    componentNodes
                    |> List.choose (fun key -> translation |> Map.tryFind key)
                    |> function
                        | [] -> 0UL
                        | values -> List.min values

                componentNodes
                |> List.fold (fun acc key -> acc |> Map.add key target) translation)
            baseMap

    let toTranslationList translation =
        translation
        |> Map.toList
        |> List.map (fun ((chunk, oldLabel), newLabel) -> chunk, oldLabel, newLabel)
        |> List.sort

    let mergeTranslatedStats translation localStatistics =
        localStatistics
        |> Map.toSeq
        |> Seq.fold
            (fun acc (key, stats) ->
                match translation |> Map.tryFind key with
                | Some newLabel when newLabel <> 0UL ->
                    acc
                    |> ComponentStatistics.addToMap newLabel (stats |> ComponentStatistics.relabel newLabel)
                | _ -> acc)
            Map.empty<uint64, ComponentStatistics>
        |> Map.toList
        |> List.map snd
        |> List.sortBy _.Label

    let reducer (debug: bool) (input: AsyncSeq<Image<uint64> * uint64>) =
        async {
            let mutable previousBoundary : (uint * Image<uint64>) option = None
            let mutable graph = simpleGraph.empty
            let mutable chunkCounts = Map.empty<uint,uint64>
            let mutable localStatistics = Map.empty<uint * uint64, ComponentStatistics>

            do!
                input
                |> AsyncSeq.iterAsync (fun (labelChunk, objectCount) ->
                    async {
                        let chunk = uint (labelChunk.index / int winSz)
                        chunkCounts <- chunkCounts |> Map.add chunk objectCount
                        localStatistics <-
                            labelChunk
                            |> labelChunkStatistics chunk
                            |> Map.fold (fun acc key stats -> acc |> ComponentStatistics.addToMap key stats) localStatistics

                        let firstSlice = ImageFunctions.extractSlice 2u 0 labelChunk

                        match previousBoundary with
                        | Some (previousChunk, previousSlice) when chunk = previousChunk + 1u ->
                            graph <- addBoundaryEdges graph previousChunk previousSlice firstSlice
                            previousSlice.decRefCount()
                        | Some (_, previousSlice) ->
                            previousSlice.decRefCount()
                        | None -> ()

                        firstSlice.decRefCount()

                        let depth = labelChunk.GetDepth() |> int
                        let lastSlice = ImageFunctions.extractSlice 2u (depth - 1) labelChunk
                        labelChunk.decRefCount()
                        previousBoundary <- Some (chunk, lastSlice)
                    })

            previousBoundary |> Option.iter (fun (_, image) -> image.decRefCount())

            let translation =
                chunkCounts
                |> makeBaseMap
                |> collapseTouchingComponents graph

            return
                { Labels = translation |> toTranslationList
                  Statistics = localStatistics |> mergeTranslatedStats translation }
        }

    let memoryNeed nPixels = 2UL * nPixels * uint64 sizeof<uint64>
    let elementTransformation = fun _ -> 1UL
    Stage.reduce name reducer Streaming memoryNeed elementTransformation

let trd (_,_,c) = c

let updateConnectedComponents (winSz: uint) (translationTable: ConnectedComponentTranslationTable) : Stage<Image<uint64>,Image<uint64>> =
    let name = "updateConnectedComponents"
    let translationTableChunked = List.groupBy (fun (c,_,_) -> c) translationTable.Labels
    let translationMap =
        translationTableChunked
        |> List.map (fun (chunk,lst) ->
            let chunkTranslation =
                (0u,0UL,0UL)::lst
                |> List.map (fun (_,i,j)->(i,j))
                |> Map.ofList
            chunk, chunkTranslation)
        |> Map.ofList

    let mapper (debug: bool) (sliceIndex: int64) (image: Image<uint64>) : Image<uint64> = 
        let chunk = int sliceIndex / int winSz
        //let _,trans = translationTableChunked[chunk]
        //let res = Image.map (fun v -> if v=0UL then 0UL else trans |> List.find (fun (_,w,_) -> v = w) |> trd) image
        let trans = translationMap |> Map.tryFind (uint chunk) |> Option.defaultValue (Map.ofList [(0UL,0UL)])
        let res = Image.map (fun v -> Map.find v trans) image
        image.decRefCount()
        res

    let memoryNeed = fun _ -> 2*sizeof<uint> |> uint64
    let elementTransformation = id
    Stage.mapi "updateConnectedComponents" mapper memoryNeed elementTransformation

let permuteAxes (i: uint, j: uint, k: uint) (winSz: uint): Stage<Image<'T>,Image<'T>> =
    let name = "permuteAxes"
    // There are the following 6 possible combinations: 0 1 2; 0 2 1; 1 0 2; 1 2 0; 2 1 0; 2 0 1
    if i = j || i = k || j = k then
        failwith "Order must be a permuation of [0u;1u;2u]"
    elif i = 0u && j = 1u then // k = 2u
        identityStage name
    elif i = 1u && j = 0u then // k = 2u
        // permute 0 1 does not require chunking
        let memoryNeed = fun _ -> 2*sizeof<uint> |> uint64
        let elementTransformation = id
        Stage.map name (fun _ -> ImageFunctions.permuteAxes [i;j;k]) memoryNeed elementTransformation
    else // k neq 2u
        // writechunks and reread in permuted order
        let tmpDir = getUnusedDirectoryName "tmp"
        let tmpSuffix = ".tiff"

        let mapper (chunkInfo: ChunkInfo) (debug: bool) (idx: int): Image<'T> list = 
            let slab = _readSlabStacked<'T> tmpDir tmpSuffix chunkInfo k (int idx)
            let sz = slab.GetSize ()
            let stack =  ImageFunctions.unstack k slab
            slab.decRefCount()
            let res = 
                if j < i then // since we use unstack, we need to transpose some
                    stack 
                    |> List.map (fun im -> 
                        let trnsp = ImageFunctions.permuteAxes [1u;0u;2u] im
                        im.decRefCount()
                        trnsp)
                else
                    stack
            res

        let mutable chunkInfo : ChunkInfo = {chunks = [0;0;0] ; size = [0UL;0UL;0UL]; topLeftInfo = {dimensions = 0u; size = [0UL;0UL;0UL]; componentType = ""; numberOfComponents = 0u}}
        let memPeak = 256UL // surrugate string length
        let memoryNeed = fun _ -> memPeak
        let elementTransformation = fun _ -> chunkInfo.chunks[int k] |> uint64

        (writeChunks tmpDir tmpSuffix winSz winSz winSz)
        --> cleanStage name (fun () -> StackIO.deleteIfExists tmpDir) 
        --> StackCore.ignoreSingles () // force calculation of full stream and decrease references
        --> Stage.map name (fun _ _ -> chunkInfo <- getChunkInfo tmpDir tmpSuffix) memoryNeed elementTransformation // insert side-effect
        --> Stage.map name (fun _ _ -> [0..(chunkInfo.chunks[int k]-1)]) memoryNeed elementTransformation
        --> flattenList () // expand to a new, non-empty stream
        --> Stage.map name (fun debug idx -> mapper chunkInfo debug idx) memoryNeed elementTransformation // mapper chunkInfo does not work, since argument is copied at compile time
        --> flattenList ()
