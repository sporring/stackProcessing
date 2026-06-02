module StackImageFunctions

open FSharp.Control
open SlimPipeline // Core processing model
open System
open System.Collections.Generic
open System.Globalization
open System.IO
open StackCore
open StackIO
open TinyLinAlg
open Image.InternalHelpers

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

let private imageOperatorCost<'T> operator evaluation memoryModel windowSize radius kernelSize sigma tags costUnits =
    StackProcessingCost.imageOperatorStageCost<'T>
        { Operator = operator
          Evaluation = evaluation
          Memory = memoryModel
          WindowSize = windowSize
          Radius = radius
          KernelSize = kernelSize
          Sigma = sigma
          Tags = tags
          FallbackCostUnits = costUnits }

let private operatorUnaryStageCost<'T> operator memoryNeed =
    StackProcessingCost.imageUnaryStageCost<'T> operator memoryNeed

let private castStageCost<'S, 'T> memoryNeed =
    let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed
    let pixelType = $"{StackProcessingCost.pixelTypeName<'S>}To{StackProcessingCost.pixelTypeName<'T>}"
    let fallback =
        StageTimeCostModel.native
            Map
            (Some $"Cast.{pixelType}")
            (fun input -> inputValue input |> float)
    let context input =
        let voxels = inputValue input
        StackProcessingCost.Fitting.OperatorEstimateContext.create
            "Cast"
            (Some pixelType)
            (Some voxels)
            (Some(StackProcessingCost.imageBytes<'T> voxels))
            None
            None
            None
            None

    StageCostModel.create
        memoryModel
        (StackProcessingCost.Fitting.OperatorCostRuntime.timeCostModel Map context fallback.Estimate)

let private liftOperatorUnaryReleaseAfter<'T when 'T: equality>
    name
    operator
    (f: Image<'T> -> Image<'T>)
    : Stage<Image<'T>, Image<'T>> =
    let memoryNeed nPixels = 2UL * nPixels * getBytesPerComponent<'T>

    liftUnaryReleaseAfter name f memoryNeed id
    |> withCostModel (operatorUnaryStageCost<'T> operator memoryNeed)

let private cleanStage name cleanup =
    { identityStage name with Cleaning = [ cleanup ] }

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
    let width = int (a.GetWidth())
    let height = int (a.GetHeight())
    if b.GetWidth() <> uint width || b.GetHeight() <> uint height then
        invalidArg "b" $"blendImages expects matching slice sizes, got {a.GetWidth()}x{a.GetHeight()} and {b.GetWidth()}x{b.GetHeight()}."

    let aPixels = a.toFlatArray()
    let bPixels = b.toFlatArray()
    let output = Array.zeroCreate<'T> aPixels.Length
    let mutable i = 0
    while i < output.Length do
        let av = pixelToFloat aPixels[i]
        let bv = pixelToFloat bPixels[i]
        output[i] <- convertFromFloat<'T> ((1.0 - t) * av + t * bv)
        i <- i + 1

    Image<'T>.ofFlatArray([ uint width; uint height ], output, "resampleZLinear")

let private resampleXYStage<'T when 'T: equality> outputWidth outputHeight spacingX spacingY interpolation =
    let mapper (image: Image<'T>) =
        ImageFunctions.resample2D interpolation outputWidth outputHeight spacingX spacingY image

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
                    | ImageFunctions.ResampleInterpolation.NearestNeighbor ->
                        let chosen = if fraction < 0.5 then first else second
                        chosen.incRefCount()
                        chosen
                    | ImageFunctions.ResampleInterpolation.Linear ->
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
    let memoryNeed nPixels = nPixels * (getBytesPerComponent<'S> + getBytesPerComponent<'T>)
    Stage.cast<Image<'S>,Image<'T>> name f id
    |> withCostModel (castStageCost<'S, 'T> memoryNeed)

/// Basic arithmetic
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
    let memoryNeed nPixels = 2UL * nPixels * getBytesPerComponent<'T>
    let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed
    let costUnits input = inputValue input |> float
    liftUnaryReleaseAfter "vectorElement" (ImageFunctions.vectorElement componentId) memoryNeed id
    |> withCostModel (imageOperatorCost<'T> "VectorElement" Map memoryModel None None None None [] costUnits)

let vectorRange<'T when 'T: equality> firstComponent componentCount : Stage<Image<'T list>, Image<'T list>> =
    liftUnaryReleaseAfter "vectorRange" (ImageFunctions.vectorRange firstComponent componentCount) id id

let vector3ToColor inputMinimum inputMaximum : Stage<Image<float list>, Image<uint8 list>> =
    let memoryNeed nPixels = 2UL * nPixels * getBytesPerComponent<float>
    liftUnaryReleaseAfter "vector3ToColor" (ImageFunctions.vector3ToColor inputMinimum inputMaximum) memoryNeed id
    |> withCostModel (operatorUnaryStageCost<float> "VectorElement" memoryNeed)

let colorToVector3 outputMinimum outputMaximum : Stage<Image<uint8 list>, Image<float list>> =
    liftUnaryReleaseAfter "colorToVector3" (ImageFunctions.colorToVector3 outputMinimum outputMaximum) id id

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
    let memoryNeed nPixels = 2UL * nPixels * getBytesPerComponent<float>
    let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed
    let costUnits input = inputValue input |> float
    liftUnaryReleaseAfter "modulus" Image.modulus memoryNeed id
    |> withCostModel (imageOperatorCost<float> "ComplexModulus" Map memoryModel None None None None [] costUnits)

let arg : Stage<Image<System.Numerics.Complex>, Image<float>> =
    liftUnaryReleaseAfter "arg" Image.arg id id

let conjugate : Stage<Image<System.Numerics.Complex>, Image<System.Numerics.Complex>> =
    liftUnaryReleaseAfter "conjugate" Image.conjugate id id

let toComplex : Stage<Image<float> * Image<float>, Image<System.Numerics.Complex>> =
    liftPairReleaseAfter "toComplex" Image.toComplex

let polarToComplex : Stage<Image<float> * Image<float>, Image<System.Numerics.Complex>> =
    liftPairReleaseAfter "polarToComplex" Image.polarToComplex

let FFT<'T when 'T: equality> chunkX chunkY chunkZ : Stage<Image<'T>, Image<System.Numerics.Complex>> =
    let stage =
        StackIO.chunkedVolumeOperation
            "FFT"
            (liftUnaryReleaseAfter "FFTXY" ImageFunctions.FFTXY id id)
            StackIO.chunkedFFTAlongZ
            chunkX
            chunkY
            chunkZ
    let memoryNeed nPixels = 2UL * nPixels * getBytesPerComponent<'T>
    let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed
    let costUnits input = inputValue input |> float
    stage
    |> withCostModel (imageOperatorCost<'T> "FFT" Map memoryModel None None None None [] costUnits)

let invFFT chunkX chunkY chunkZ : Stage<Image<System.Numerics.Complex>, Image<float>> =
    let stage =
        (StackIO.chunkedVolumeOperation
            "invFFT"
            (liftUnaryReleaseAfter "inverseFFTXY" ImageFunctions.inverseFFTXY id id)
            StackIO.chunkedInvFFTAlongZ
            chunkX
            chunkY
            chunkZ)
        --> liftUnaryReleaseAfter "invFFT.realPart" ImageFunctions.realPart id id
    let memoryNeed nPixels = 2UL * nPixels * getBytesPerComponent<float>
    let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed
    let costUnits input = inputValue input |> float
    stage
    |> withCostModel (imageOperatorCost<float> "InvFFT" Map memoryModel None None None None [] costUnits)

let shiftFFT chunkX chunkY chunkZ : Stage<Image<System.Numerics.Complex>, Image<System.Numerics.Complex>> =
    let stage =
        StackIO.chunkedVolumeOperation
            "shiftFFT"
            (identityStage "shiftFFT.input")
            StackIO.chunkedShiftFFT
            chunkX
            chunkY
            chunkZ
    let memoryNeed nPixels = 2UL * nPixels * getBytesPerComponent<float>
    let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed
    let costUnits input = inputValue input |> float
    stage
    |> withCostModel (imageOperatorCost<float> "ShiftFFT" Map memoryModel None None None None [] costUnits)

let abs<'T when 'T: equality>    : Stage<Image<'T>,Image<'T>> = 
    failTypeMismatch<'T> "abs" floatNintTypes
    liftOperatorUnaryReleaseAfter "abs" "UnaryImageFunction" ImageFunctions.absImage
let acos<'T when 'T: equality>   : Stage<Image<'T>,Image<'T>> =      
    failTypeMismatch<'T> "acos" floatTypes
    liftOperatorUnaryReleaseAfter "acos" "UnaryImageFunction" ImageFunctions.acosImage
let asin<'T when 'T: equality>   : Stage<Image<'T>,Image<'T>> =      
    failTypeMismatch<'T> "asin" floatTypes
    liftOperatorUnaryReleaseAfter "asin" "UnaryImageFunction" ImageFunctions.asinImage
let atan<'T when 'T: equality>   : Stage<Image<'T>,Image<'T>> =      
    failTypeMismatch<'T> "atan" floatTypes
    liftOperatorUnaryReleaseAfter "atan" "UnaryImageFunction" ImageFunctions.atanImage
let cos<'T when 'T: equality>    : Stage<Image<'T>,Image<'T>> =      
    failTypeMismatch<'T> "cos" floatTypes
    liftOperatorUnaryReleaseAfter "cos" "UnaryImageFunction" ImageFunctions.cosImage
let sin<'T when 'T: equality>    : Stage<Image<'T>,Image<'T>> =      
    failTypeMismatch<'T> "sin" floatTypes
    liftOperatorUnaryReleaseAfter "sin" "UnaryImageFunction" ImageFunctions.sinImage
let tan<'T when 'T: equality>    : Stage<Image<'T>,Image<'T>> =      
    failTypeMismatch<'T> "tan" floatTypes
    liftOperatorUnaryReleaseAfter "tan" "UnaryImageFunction" ImageFunctions.tanImage
let exp<'T when 'T: equality>    : Stage<Image<'T>,Image<'T>> =      
    failTypeMismatch<'T> "exp" floatTypes
    liftOperatorUnaryReleaseAfter "exp" "UnaryImageFunction" ImageFunctions.expImage
let log10<'T when 'T: equality>  : Stage<Image<'T>,Image<'T>> =      
    failTypeMismatch<'T> "log10" floatTypes
    liftOperatorUnaryReleaseAfter "log10" "UnaryImageFunction" ImageFunctions.log10Image
let log<'T when 'T: equality>    : Stage<Image<'T>,Image<'T>> =      
    failTypeMismatch<'T> "log" floatTypes
    liftOperatorUnaryReleaseAfter "log" "UnaryImageFunction" ImageFunctions.logImage
let round<'T when 'T: equality>  : Stage<Image<'T>,Image<'T>> =      
    failTypeMismatch<'T> "round" floatTypes
    liftOperatorUnaryReleaseAfter "round" "UnaryImageFunction" ImageFunctions.roundImage
let sqrt<'T when 'T: equality>   : Stage<Image<'T>,Image<'T>> =      
    failTypeMismatch<'T> "sqrt" floatNintTypes
    liftOperatorUnaryReleaseAfter "sqrt" "UnaryImageFunction" ImageFunctions.sqrtImage
let sqrtWindowed<'T when 'T: equality> (winSz: uint) : Stage<Image<'T>,Image<'T>> =
    failTypeMismatch<'T> "sqrtWindowed" floatNintTypes
    let win = max 1u winSz
    let memoryNeed nPixels = 2UL * nPixels * uint64 win * getBytesPerComponent<'T>
    let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed
    let costUnits input = float (inputValue input * uint64 win)
    liftWindowedUnaryReleaseAfter "sqrtWindowed" win ImageFunctions.sqrtImage memoryNeed id
    |> withCostModel (imageOperatorCost<'T> "SqrtWindowed" Map memoryModel (Some(float win)) None None None [] costUnits)
let square<'T when 'T: equality> : Stage<Image<'T>,Image<'T>> =      
    failTypeMismatch<'T> "square" floatNintTypes
    liftOperatorUnaryReleaseAfter "square" "UnaryImageFunction" ImageFunctions.squareImage

let clamp<'T when 'T: equality> lower upper : Stage<Image<'T>, Image<'T>> =
    liftUnaryReleaseAfter "clamp" (ImageFunctions.clampImage lower upper) id id

let shiftScale<'T when 'T: equality> shift scale : Stage<Image<'T>, Image<'T>> =
    liftOperatorUnaryReleaseAfter "shiftScale" "ShiftScale" (ImageFunctions.shiftScale shift scale)

let intensityStretch<'T when 'T: equality> inputMinimum inputMaximum outputMinimum outputMaximum : Stage<Image<'T>, Image<'T>> =
    if inputMinimum = inputMaximum then
        invalidArg "inputMaximum" "Cannot stretch intensities from an input range with zero width."
    if outputMinimum = outputMaximum then
        invalidArg "outputMaximum" "Cannot stretch intensities to an output range with zero width using shiftScale."

    let scale = (outputMaximum - outputMinimum) / (inputMaximum - inputMinimum)
    let shift = outputMinimum / scale - inputMinimum
    liftOperatorUnaryReleaseAfter "intensityStretch" "IntensityStretch" (ImageFunctions.shiftScale shift scale)

let private defaultConvolutionWindowSize (kernelDepth: uint) =
    max 1u kernelDepth

let private effectiveWindowSize stageName minimumWindowSize requestedWindowSize =
    let minimumWindowSize = max 1u minimumWindowSize
    match requestedWindowSize with
    | None -> minimumWindowSize
    | Some windowSize when windowSize >= minimumWindowSize -> windowSize
    | Some windowSize ->
        stopWithConfigurationError
            $"{stageName}: requested window size {windowSize} is smaller than the minimum {minimumWindowSize} required by the stage parameters."

let private requestedWindowTag (winSz: uint option) =
    winSz |> Option.map (fun value -> "requestedWindowSize", string value)

let private windowCostTags ksz winSz =
    [ yield "kernelSize", string ksz
      yield "minimumWindowSize", string ksz
      match requestedWindowTag winSz with
      | Some tag -> yield tag
      | None -> () ]

let private effectiveWindowTags win =
    [ "windowSize", string win
      "effectiveWindowSize", string win ]

let private makeWindowedLocalOp name ksz winSz core =
    let pad = ksz / 2u
    let win = effectiveWindowSize name ksz winSz
    let stride = win - ksz + 1u
    let stg = liftSlabReleaseAfter name core id id
    windowedViaSlabRequired win pad stride ksz pad stride stg
    |> Stage.withSliceCardinality (SlimPipeline.Domain(sameSliceDomainForKernelDepth ksz))

let private makeWindowedOperatorOp<'T when 'T: equality>
    (name: string)
    (operator: string)
    (radius: float option)
    (ksz: uint)
    (winSz: uint option)
    (core: Image<'T> -> Image<'T>)
    : Stage<Image<'T>, Image<'T>> =
    let pad = ksz / 2u
    let win = effectiveWindowSize name ksz winSz
    let stride = win - ksz + 1u
    let memoryNeed nPixels = 2UL * nPixels * uint64 win * getBytesPerComponent<'T>
    let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed
    let costUnits input = float (inputValue input * uint64 win)
    let stg =
        liftSlabReleaseAfter name core memoryNeed id
        |> withCostModel
            (imageOperatorCost<'T>
                operator
                Map
                memoryModel
                (Some(float win))
                radius
                (Some(float ksz))
                None
                [ yield! windowCostTags ksz winSz
                  yield! effectiveWindowTags win
                  yield "stride", string stride
                  yield "pad", string pad ]
                costUnits)

    windowedViaSlabRequired win pad stride ksz pad stride stg
    |> Stage.withSliceCardinality (SlimPipeline.Domain(sameSliceDomainForKernelDepth ksz))

let smoothWMedian<'T when 'T: equality> radius winSz : Stage<Image<'T>, Image<'T>> =
    let ksz = 2u * radius + 1u
    makeWindowedOperatorOp<'T> "smoothWMedian" "SmoothWMedian" (Some(float radius)) ksz winSz (ImageFunctions.median radius)

let smoothWBilateral<'T when 'T: equality> domainSigma rangeSigma winSz : Stage<Image<'T>, Image<'T>> =
    makeWindowedOperatorOp<'T> "smoothWBilateral" "SmoothWBilateral" None 7u winSz (ImageFunctions.bilateral domainSigma rangeSigma)

let gradientMagnitude<'T when 'T: equality> winSz : Stage<Image<'T>, Image<'T>> =
    makeWindowedOperatorOp<'T> "gradientMagnitude" "GradientMagnitude" None 3u winSz ImageFunctions.gradientMagnitude

let private finiteDiffKernelDepth order =
    let kernel = ImageFunctions.finiteDiffFilter3D 2u order
    try
        max 1u (kernel.GetDepth())
    finally
        kernel.decRefCount()

let gradient (order: uint) (winSz: uint option) : Stage<Image<float>, Image<float list>> =
    let ksz = finiteDiffKernelDepth order
    let pad = ksz / 2u
    let win = effectiveWindowSize "gradient" ksz winSz
    let stride = win - ksz + 1u
    let memoryNeed nPixels = (4UL * nPixels * uint64 win) * getBytesPerComponent<float>
    let stg = liftSlabReleaseAfter "gradient" (ImageFunctions.gradientVector3D order) memoryNeed id
    windowedViaSlabRequired win pad stride ksz pad stride stg
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
        let memoryNeed nPixels = (8UL * nPixels * uint64 win) * getBytesPerComponent<float>
        let stg = liftSlabReleaseAfter "gaussianVectorElements" (ImageFunctions.smoothVectorElements3D sigma) memoryNeed id
        windowedViaSlabRequired win pad stride ksz pad stride stg
        |> Stage.withSliceCardinality (SlimPipeline.Domain(sameSliceDomainForKernelDepth ksz))

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

let private pcaImages (state: PcaAccumulator) : Image<float list> list =
    let eigen = pcaEigenSystem state
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
    makeWindowedOperatorOp<'T> "sobelEdge" "SobelEdge" None 3u winSz ImageFunctions.sobelEdge

let laplacian<'T when 'T: equality> winSz : Stage<Image<'T>, Image<'T>> =
    makeWindowedOperatorOp<'T> "laplacian" "Laplacian" None 3u winSz ImageFunctions.laplacian

let grayscaleErode<'T when 'T: equality> radius winSz : Stage<Image<'T>, Image<'T>> =
    let ksz = 2u * radius + 1u
    makeWindowedOperatorOp<'T> "grayscaleErode" "GrayscaleErode" (Some(float radius)) ksz winSz (ImageFunctions.grayscaleErode radius)

let grayscaleDilate<'T when 'T: equality> radius winSz : Stage<Image<'T>, Image<'T>> =
    let ksz = 2u * radius + 1u
    makeWindowedOperatorOp<'T> "grayscaleDilate" "GrayscaleDilate" (Some(float radius)) ksz winSz (ImageFunctions.grayscaleDilate radius)

let grayscaleOpening<'T when 'T: equality> radius winSz : Stage<Image<'T>, Image<'T>> =
    let ksz = 2u * radius + 1u
    makeWindowedOperatorOp<'T> "grayscaleOpening" "GrayscaleOpening" (Some(float radius)) ksz winSz (ImageFunctions.grayscaleOpening radius)

let grayscaleClosing<'T when 'T: equality> radius winSz : Stage<Image<'T>, Image<'T>> =
    let ksz = 2u * radius + 1u
    makeWindowedOperatorOp<'T> "grayscaleClosing" "GrayscaleClosing" (Some(float radius)) ksz winSz (ImageFunctions.grayscaleClosing radius)

let whiteTopHat<'T when 'T: equality> radius winSz : Stage<Image<'T>, Image<'T>> =
    let ksz = 2u * radius + 1u
    makeWindowedOperatorOp<'T> "whiteTopHat" "WhiteTopHat" (Some(float radius)) ksz winSz (ImageFunctions.whiteTopHat radius)

let blackTopHat<'T when 'T: equality> radius winSz : Stage<Image<'T>, Image<'T>> =
    let ksz = 2u * radius + 1u
    makeWindowedOperatorOp<'T> "blackTopHat" "BlackTopHat" (Some(float radius)) ksz winSz (ImageFunctions.blackTopHat radius)

let morphologicalGradient<'T when 'T: equality> radius winSz : Stage<Image<'T>, Image<'T>> =
    let ksz = 2u * radius + 1u
    makeWindowedOperatorOp<'T> "morphologicalGradient" "MorphologicalGradient" (Some(float radius)) ksz winSz (ImageFunctions.morphologicalGradient radius)

let binaryContour (fullyConnected: bool) winSz =
    makeWindowedOperatorOp<uint8> "binaryContour" "BinaryContour" None 3u winSz (fun image -> ImageFunctions.binaryContour fullyConnected image)

let binaryMedian radius winSz =
    let ksz = 2u * radius + 1u
    makeWindowedOperatorOp<uint8> "binaryMedian" "BinaryMedian" (Some(float radius)) ksz winSz (fun image -> ImageFunctions.binaryMedian radius image)

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
    let interpolation = ImageFunctions.ResampleInterpolation.parse interpolationName

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

    let interpolation = ImageFunctions.ResampleInterpolation.parse interpolationName
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
                            let pixels = image.toFlatArray()

                            for y in 0 .. step .. height - 1 do
                                for x in 0 .. step .. width - 1 do
                                    let value = pixels[flatIndex2 width x y]
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

    let memoryModel = StageMemoryModel.fromSinglePeak Reduce id
    let costUnits input = inputValue input |> float
    Stage.reduce $"histogramEstimate.{typeof<'T>.Name}" reducer Streaming id id
    |> withCostModel (imageOperatorCost<'T> "EstimateHistogram" Reduce memoryModel None None None None [] costUnits)

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

let histogramEqualization<'T, 'H when 'T: equality and 'H: comparison> (histogram: Histogram<'H>) =
    let lookup = histogramEqualizationLookup histogram.Counts

    let equalize (_debug: bool) (image: Image<'T>) =
        try
            let width = int (image.GetWidth())
            let height = int (image.GetHeight())
            let pixels = image.toFlatArray()
            let outputPixels = Array.zeroCreate<float> pixels.Length
            let mutable i = 0
            while i < pixels.Length do
                outputPixels[i] <- pixels[i] |> histogramKeyToFloat |> lookup
                i <- i + 1

            let output = Image<float>.ofFlatArray([ uint width; uint height ], outputPixels, "histogramEqualization", image.index)

            output.index <- image.index
            output
        finally
            image.decRefCount()

    let memoryModel = StageMemoryModel.fromSinglePeak Map imageBytes<float>
    let costUnits input = inputValue input |> float
    Stage.map "histogramEqualization" equalize imageBytes<float> id
    |> withCostModel (imageOperatorCost<float> "HistogramEqualization" Map memoryModel None None None None [] costUnits)

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
                                let pixels = image.toFlatArray()
                                let accumulator =
                                    match state with
                                    | None ->
                                        Some(width, height, Array.zeroCreate<float> (width * height))
                                    | Some(expectedWidth, expectedHeight, values) ->
                                        if expectedWidth <> width || expectedHeight <> height then
                                            invalidOp $"sumProjection requires constant x-y slice size; got {width}x{height}, expected {expectedWidth}x{expectedHeight}."
                                        Some(expectedWidth, expectedHeight, values)

                                match accumulator with
                                | None -> return None
                                | Some(_, _, values) ->
                                    for y in 0 .. height - 1 do
                                        for x in 0 .. width - 1 do
                                            let i = flatIndex2 width x y
                                            values[i] <- values[i] + transform (Convert.ToDouble pixels[i])
                                    return accumulator
                            finally
                                image.decRefCount()
                        })
                    None

            match state with
            | None ->
                return raise (InvalidOperationException "sumProjection cannot reduce an empty image stream.")
            | Some(width, height, values) ->
                let image = Image<float>.ofFlatArray([ uint width; uint height ], values, "sumProjection")
                image.index <- 0
                return image
        }

    let memoryNeed nElems =
        imageBytes<float> nElems

    let memoryModel = StageMemoryModel.fromSinglePeak Reduce memoryNeed
    let costUnits input = inputValue input |> float
    Stage.reduce $"sumProjection {transformName}" reducer Streaming memoryNeed id
    |> withCostModel (imageOperatorCost<'T> "SumProjection" Reduce memoryModel None None None None [] costUnits)

let volume xUnit yUnit zUnit : Stage<Image<uint8>, float> =
    if xUnit <= 0.0 then invalidArg (nameof xUnit) "xUnit must be positive."
    if yUnit <= 0.0 then invalidArg (nameof yUnit) "yUnit must be positive."
    if zUnit <= 0.0 then invalidArg (nameof zUnit) "zUnit must be positive."

    let voxelVolume = xUnit * yUnit * zUnit

    let folder volume (image: Image<uint8>) =
        try
            let width = int (image.GetWidth())
            let height = int (image.GetHeight())
            let pixels = image.toFlatArray()
            let mutable foreground = 0UL
            for y in 0 .. height - 1 do
                for x in 0 .. width - 1 do
                    match pixels[flatIndex2 width x y] with
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
    let memoryModel = StageMemoryModel.fromSinglePeak Map id
    let costUnits input = inputValue input |> float
    Stage.map<Image<'T>,ImageStats> "computeStats: map" (fun _ -> releaseAfter ImageFunctions.computeStats) id id
    |> withCostModel (imageOperatorCost<'T> "ComputeStats" Map memoryModel None None None None [] costUnits)

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

let smoothWGauss (sigma: float) (outputRegionMode: ImageFunctions.OutputRegionMode option) (boundaryCondition: ImageFunctions.BoundaryCondition option) (winSz: uint option) : Stage<Image<float>, Image<float>> =
    let ksz = ImageFunctions.defaultGaussWindowSize sigma
    let kernel = ImageFunctions.gauss 3u sigma (Some ksz)
    let win = effectiveWindowSize "smoothWGauss" (defaultConvolutionWindowSize ksz) winSz
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
    let memoryNeed nPixels = (2UL*nPixels*(uint64 win) + (uint64 ksz))*getBytesPerComponent<float>
    let elementTransformation = id
    let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed
    let costUnits input =
        let kernelVoxels = uint64 ksz * uint64 ksz * uint64 ksz
        float (inputValue input * uint64 win * kernelVoxels)
    let stg =
        liftSlabReleaseAfter "smoothWGauss" (fun image3D -> ImageFunctions.convolve outputRegionMode boundaryCondition image3D kernel) memoryNeed elementTransformation
        |> withCostModel
            (imageOperatorCost<float>
                "SmoothWGauss"
                Map
                memoryModel
                (Some(float win))
                None
                (Some(float ksz))
                (Some sigma)
                [ yield! windowCostTags ksz winSz
                  yield! effectiveWindowTags win
                  yield "stride", string stride
                  yield "pad", string pad
                  yield "sigma", Convert.ToString(sigma, CultureInfo.InvariantCulture) ]
                costUnits)
    windowedViaSlabRequired win pad stride ksz outputStart stride stg
    --> cleanStage "smoothWGauss.cleanup" (fun () -> kernel.decRefCount())
    |> Stage.withSliceCardinality (sliceCardinalityForConvolution ksz outputRegionMode)

let structureTensor (sigma: float) (rho: float) : Stage<Image<float>, Image<float list>> =
    let preSmooth =
        if sigma <= 0.0 then identityStage "structureTensor.preSmooth.identity"
        else smoothWGauss sigma None None None

    preSmooth
    --> gradient 1u None
    --> liftUnaryReleaseAfter "structureTensorOuterProduct" ImageFunctions.structureTensorOuterProduct id id
    --> gaussianVectorElements rho
    --> liftUnaryReleaseAfter "structureTensorEigenMatrix" ImageFunctions.structureTensorEigenMatrix id id

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
    let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed
    let costUnits input = inputValue input |> float
    let pipe = { Name = name; Apply = apply; Profile = Streaming }
    Stage.fromPipe name transition memoryNeed id pipe
    |> withCostModel (imageOperatorCost<'T> "CreatePadding" Map memoryModel None None None None [] costUnits)
    |> Stage.withSliceCardinality (SlimPipeline.Domain(SlimPipeline.SliceDomain.expand beforeZ afterZ))

let crop<'T when 'T: equality> beforeX afterX beforeY afterY beforeZ afterZ : Stage<Image<'T>, Image<'T>> =
    let name = "crop"
    let cropXY (image: Image<'T>) =
        ImageFunctions.crop2D [ beforeX; beforeY ] [ afterX; afterY ] image

    let memoryNeed nPixels = nPixels * getBytesPerComponent<'T>
    let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed
    let costUnits input = inputValue input |> float

    let cropXYStage =
        liftUnaryReleaseAfter name cropXY memoryNeed id
        |> withCostModel (imageOperatorCost<'T> "Crop" Map memoryModel None None None None [] costUnits)

    Stage.trim "cropZ" beforeZ afterZ (fun image -> image.decRefCount())
    --> cropXYStage

let private convolveSliceOp (name: string) (kernel: Image<'T>) (outputRegionMode: ImageFunctions.OutputRegionMode option) (bc: ImageFunctions.BoundaryCondition option) =
    let memoryNeed nPixels =
        2UL * nPixels * getBytesPerComponent<'T>

    let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed
    let costUnits input =
        let kernelVoxels = uint64 (kernel.GetWidth()) * uint64 (kernel.GetHeight())
        float (inputValue input * kernelVoxels)

    liftUnaryReleaseAfter name (fun image -> ImageFunctions.convolve outputRegionMode bc image kernel) memoryNeed id
    |> withCostModel (imageOperatorCost<'T> "Convolve" Map memoryModel None None (Some(float (max (kernel.GetWidth()) (kernel.GetHeight())))) None [] costUnits)

let private convolveSlabOp (name: string) (kernel: Image<'T>) (outputRegionMode: ImageFunctions.OutputRegionMode option) (bc: ImageFunctions.BoundaryCondition option) (winSz: uint option) =
    let ksz = max 1u (kernel.GetDepth())
    let win = effectiveWindowSize name (defaultConvolutionWindowSize ksz) winSz
    let stride = win - ksz + 1u
    let pad =
        match outputRegionMode with
        | Some Valid -> 0u
        | _ -> ksz / 2u

    let outputStart =
        match outputRegionMode with
        | Some Valid -> 0u
        | _ -> pad

    let memoryNeed nPixels =
        (2UL * nPixels * uint64 win + uint64 ksz) * getBytesPerComponent<'T>

    let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed
    let costUnits input =
        let kernelVoxels = uint64 (kernel.GetWidth()) * uint64 (kernel.GetHeight()) * uint64 (kernel.GetDepth())
        float (inputValue input * uint64 win * kernelVoxels)

    let stg =
        liftSlabReleaseAfter name (fun image3D -> ImageFunctions.convolve outputRegionMode bc image3D kernel) memoryNeed id
        |> withCostModel
            (imageOperatorCost<'T>
                "Convolve"
                Map
                memoryModel
                (Some(float win))
                None
                (Some(float ksz))
                None
                [ yield! windowCostTags ksz winSz
                  yield! effectiveWindowTags win
                  yield "stride", string stride
                  yield "pad", string pad ]
                costUnits)

    (window win pad stride)
    --> requireWindowSize ksz
    --> windowToSlabWithRange
    --> mapSlabWithStage stg
    --> slabWithRangeToWindow
    --> slabSkipTakeM outputStart stride
    --> flattenList ()
    |> Stage.withSliceCardinality (sliceCardinalityForConvolution ksz outputRegionMode)

let convolveOp (name: string) (kernel: Image<'T>) (outputRegionMode: ImageFunctions.OutputRegionMode option) (bc: ImageFunctions.BoundaryCondition option) (winSz: uint option) : Stage<Image<'T>, Image<'T>> =
    if kernel.GetDimensions() < 3u then
        convolveSliceOp name kernel outputRegionMode bc
    else
        convolveSlabOp name kernel outputRegionMode bc winSz

let convolve kernel outputRegionMode boundaryCondition winSz = convolveOp "convolve" kernel outputRegionMode boundaryCondition winSz
let conv kernel = convolveOp "conv" kernel None None None

let finiteDiff (direction: uint) (order: uint) =
    let kernel = ImageFunctions.finiteDiffFilter3D direction order
    convolveOp "finiteDiff" kernel None None None

// these only works on uint8
let private makeMorphOp (name: string) (operator: string) (radius: uint) (winSz: uint option) (core: uint -> Image<'T> -> Image<'T>) : Stage<Image<'T>,Image<'T>> when 'T: equality =
    let ksz   = 2u * radius + 1u
    let pad = ksz/2u
    let win = effectiveWindowSize name (defaultConvolutionWindowSize ksz) winSz
    let stride = win - ksz + 1u

    let memoryNeed nPixels = 2UL * nPixels * uint64 win * getBytesPerComponent<'T>
    let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed
    let costUnits input = float (inputValue input * uint64 win)
    let stg =
        liftSlabReleaseAfter name (core radius) memoryNeed id
        |> withCostModel
            (imageOperatorCost<'T>
                operator
                Map
                memoryModel
                (Some(float win))
                (Some(float radius))
                (Some(float ksz))
                None
                [ yield! windowCostTags ksz winSz
                  yield! effectiveWindowTags win
                  yield "stride", string stride
                  yield "pad", string pad ]
                costUnits)
    windowedViaSlabRequired win pad stride ksz pad stride stg
    |> Stage.withSliceCardinality (SlimPipeline.Domain(sameSliceDomainForKernelDepth ksz))

let erode radius = makeMorphOp "binaryErode" "Erode" radius None ImageFunctions.binaryErode
let dilate radius = makeMorphOp "binaryDilate" "Dilate" radius None ImageFunctions.binaryDilate
let opening radius = makeMorphOp "binaryOpening" "Opening" radius None ImageFunctions.binaryOpening
let closing radius = makeMorphOp "binaryClosing" "Closing" radius None ImageFunctions.binaryClosing

type private FlatSlice =
    { Index: int
      Pixels: uint8[] }

let private lineHalo dz length =
    let left = length - length / 2 - 1
    let right = length / 2
    let a = -left * dz
    let b = right * dz
    max 0 (-min a b), max 0 (max a b)

let private dilateLineSlice width height (window: FlatSlice[]) center dx dy dz length =
    let output = Array.zeroCreate<uint8> (width * height)
    let centerIndex = center
    let left = length - length / 2 - 1
    let right = length / 2

    for y in 0 .. height - 1 do
        let row = y * width
        for x in 0 .. width - 1 do
            let mutable found = false
            let mutable t = -left
            while not found && t <= right do
                let xx = x + t * dx
                let yy = y + t * dy
                let zz = centerIndex + t * dz
                if xx >= 0 && xx < width && yy >= 0 && yy < height && zz >= 0 && zz < window.Length then
                    if window[zz].Pixels[flatIndex2 width xx yy] = 1uy then
                        found <- true
                t <- t + 1
            if found then
                output[row + x] <- 1uy

    output

let private erodeLineSlice width height (window: FlatSlice[]) center dx dy dz length =
    let output = Array.zeroCreate<uint8> (width * height)
    let centerIndex = center
    let left = length - length / 2 - 1
    let right = length / 2

    for y in 0 .. height - 1 do
        let row = y * width
        for x in 0 .. width - 1 do
            let mutable inside = true
            let mutable t = -left
            while inside && t <= right do
                let xx = x + t * dx
                let yy = y + t * dy
                let zz = centerIndex + t * dz
                if xx < 0 || xx >= width || yy < 0 || yy >= height || zz < 0 || zz >= window.Length || window[zz].Pixels[flatIndex2 width xx yy] <> 1uy then
                    inside <- false
                t <- t + 1
            if inside then
                output[row + x] <- 1uy

    output

let private streamingZonohedralLineStage operationName operatorName lineOperator radius (lineIndex: int) (dx: int, dy: int, dz: int, length: int) =
    let prePad, postPad = lineHalo dz length
    let windowLength = prePad + 1 + postPad
    let memoryNeed nPixels =
        uint64 (windowLength + 1) * nPixels * getBytesPerComponent<uint8>

    let transition = ProfileTransition.create Streaming Streaming
    let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed
    let costUnits input =
        float (inputValue input * uint64 (max 1 length))

    let apply (_debug: bool) (input: AsyncSeq<Image<uint8>>) =
        asyncSeq {
            let queue = ResizeArray<FlatSlice>()
            let mutable width = 0
            let mutable height = 0
            let mutable plane = 0
            let mutable initialized = false
            let mutable realCount = 0
            let mutable emittedCount = 0
            let mutable lastIndex = -1

            let ensureInitialized (image: Image<uint8>) =
                if not initialized then
                    width <- int (image.GetWidth())
                    height <- int (image.GetHeight())
                    plane <- width * height
                    for i in prePad .. -1 .. 1 do
                        queue.Add({ Index = -i; Pixels = Array.zeroCreate<uint8> plane })
                    initialized <- true
                elif int (image.GetWidth()) <> width || int (image.GetHeight()) <> height then
                    invalidArg "input" $"All slices in streaming zonohedral {operationName} must have the same width and height."

            let tryEmit () =
                if initialized && queue.Count >= windowLength && emittedCount < realCount then
                    let centerSlice = queue[prePad]
                    let window = queue |> Seq.truncate windowLength |> Seq.toArray
                    let pixels = lineOperator width height window prePad dx dy dz length
                    queue.RemoveAt(0)
                    emittedCount <- emittedCount + 1
                    Some(Image<uint8>.ofSimpleITKNDispose(importScalarImage [ uint width; uint height ] pixels, $"binary{operatorName}Zonohedral.line{lineIndex}", centerSlice.Index))
                else
                    None

            for image in input do
                ensureInitialized image
                let pixels =
                    try
                        copyScalarPixels<uint8> image.Image plane
                    finally
                        image.decRefCount()
                queue.Add({ Index = realCount; Pixels = pixels })
                lastIndex <- realCount
                realCount <- realCount + 1

                match tryEmit () with
                | Some output -> yield output
                | None -> ()

            if initialized then
                for i in 1 .. postPad do
                    queue.Add({ Index = lastIndex + i; Pixels = Array.zeroCreate<uint8> plane })
                    match tryEmit () with
                    | Some output -> yield output
                    | None -> ()
        }

    Stage.fromAsyncSeq
        $"binary{operatorName}Zonohedral.line{lineIndex}"
        apply
        transition
        memoryModel
        id
    |> withCostModel
        (imageOperatorCost<uint8>
            $"{operatorName}ZonohedralLine"
            Map
            memoryModel
            (Some(float windowLength))
            (Some(float radius))
            (Some(float length))
            None
            [ "direction", $"{dx},{dy},{dz}"
              "lineLength", string length
              "prePad", string prePad
              "postPad", string postPad
              "approximation", "zonohedral" ]
            costUnits)

let dilateZonohedral radius (_winSz: uint option) =
    let lines = ImageFunctions.zonohedralBestLines radius
    let stage =
        lines
        |> Array.mapi (streamingZonohedralLineStage "dilation" "Dilate" dilateLineSlice radius)
        |> Array.fold (fun acc lineStage -> acc --> lineStage) (identityStage "binaryDilateZonohedral.start")
    stage
    |> Stage.withSliceCardinality (SlimPipeline.Domain SlimPipeline.SliceDomain.preserve)

let erodeZonohedral radius (_winSz: uint option) =
    let lines = ImageFunctions.zonohedralBestLines radius
    let stage =
        lines
        |> Array.mapi (streamingZonohedralLineStage "erosion" "Erode" erodeLineSlice radius)
        |> Array.fold (fun acc lineStage -> acc --> lineStage) (identityStage "binaryErodeZonohedral.start")
    stage
    |> Stage.withSliceCardinality (SlimPipeline.Domain SlimPipeline.SliceDomain.preserve)

let openingZonohedral radius winSz =
    let stage = erodeZonohedral radius winSz --> dilateZonohedral radius winSz
    stage
    |> Stage.withSliceCardinality (SlimPipeline.Domain SlimPipeline.SliceDomain.preserve)

let closingZonohedral radius winSz =
    let stage = dilateZonohedral radius winSz --> erodeZonohedral radius winSz
    stage
    |> Stage.withSliceCardinality (SlimPipeline.Domain SlimPipeline.SliceDomain.preserve)

let connectedComponents winSz =
    let winSz = effectiveWindowSize "connectedComponents" 1u winSz
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

    let mapper (_debug: bool) (chunkIndex: int64) (slab: Image<uint8>) : Image<uint64> * uint64 =
        let result =
            try
                ImageFunctions.connectedComponents slab
            finally
                slab.decRefCount()
        result.Labels.index <- int chunkIndex * int stride
        result.Labels, result.ObjectCount

    let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed
    let costUnits input = float (inputValue input * uint64 winSz)
    let stg =
        Stage.mapi "connectedComponents" mapper memoryNeed id
        |> withCostModel (imageOperatorCost<uint8> "ConnectedComponents" Map memoryModel (Some(float winSz)) None None None [] costUnits)

    (window winSz pad stride) --> windowToSlab<uint8> --> stg

let connectedComponentsLabels winSz =
    connectedComponents winSz
    --> Stage.map "connectedComponentsLabels" (fun _ (labels, _objectCount) -> labels) id id

let connectedComponentsFullVolumeMemoryBytes (width: uint) (height: uint) (depth: uint) =
    let voxels = uint64 width * uint64 height * uint64 depth
    let inputMask = voxels * getBytesPerComponent<uint8>
    let thresholdMask = voxels * getBytesPerComponent<uint8>
    let labels = voxels * getBytesPerComponent<uint64>
    let outputMask = voxels * getBytesPerComponent<uint8>
    inputMask + thresholdMask + (2UL * labels) + outputMask

let connectedComponentsFullVolumeFits availableMemory (width: uint) (height: uint) (depth: uint) =
    connectedComponentsFullVolumeMemoryBytes width height depth <= availableMemory

let relabelComponents a winSz = 
    let winSz = effectiveWindowSize "relabelComponents" 1u winSz
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
    let memoryNeed nPixels = 2UL * nPixels * uint64 winSz * getBytesPerComponent<uint8>
    let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed
    let costUnits input = float (inputValue input * uint64 winSz)
    let stg =
        mapWindow "signedDistanceBand" f memoryNeed id
        |> withCostModel (imageOperatorCost<uint8> "SignedDistanceBand" Map memoryModel (Some(float winSz)) (Some(float bandRadius)) None None [] costUnits)
    (window winSz pad stride) --> stg --> flattenList ()
let threshold<'T when 'T: equality> a b : Stage<Image<'T>, Image<uint8>> =
    let memoryNeed nPixels = 2UL * nPixels * getBytesPerComponent<'T>
    let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed
    let costUnits input = inputValue input |> float
    liftUnaryReleaseAfter "threshold" (ImageFunctions.threshold a b) memoryNeed id
    |> withCostModel (imageOperatorCost<'T> "Threshold" Map memoryModel None None None None [] costUnits)

let windowedViaSlab<'S, 'T when 'S: equality and 'T: equality> (windowSize: uint) (stage: Stage<Image<'S>, Image<'T>>) : Stage<Image<'S>, Image<'T>> =
    let win = max 1u windowSize
    (window win 0u win)
    --> windowToSlabWithRange<'S>
    --> mapSlabWithStage stage
    --> slabWithRangeToWindow<'T>
    --> slabSkipTakeM 0u win
    --> flattenList ()

let windowSlabRoundtrip<'T when 'T: equality> windowSize : Stage<Image<'T>, Image<'T>> =
    windowedViaSlab windowSize (identityStage "windowSlabRoundtrip.inner")

let windowedCast<'S, 'T when 'S: equality and 'T: equality> windowSize : Stage<Image<'S>, Image<'T>> =
    windowedViaSlab windowSize (cast<'S, 'T>)

let windowedThreshold<'T when 'T: equality> windowSize a b : Stage<Image<'T>, Image<uint8>> =
    windowedViaSlab windowSize (threshold<'T> a b)

let addNormalNoise a b = liftOperatorUnaryReleaseAfter "addNormalNoise" "AddNormalNoise" (ImageFunctions.addNormalNoise a b)
let addSaltAndPepperNoise probability = liftOperatorUnaryReleaseAfter "addSaltAndPepperNoise" "AddSaltAndPepperNoise" (ImageFunctions.addSaltAndPepperNoise probability)
let addShotNoise scale = liftOperatorUnaryReleaseAfter "addShotNoise" "AddShotNoise" (ImageFunctions.addShotNoise scale)
let addSpeckleNoise stddev = liftOperatorUnaryReleaseAfter "addSpeckleNoise" "AddSpeckleNoise" (ImageFunctions.addSpeckleNoise stddev)

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

let srcPlan (pl: Plan<unit, unit>) (width: uint) (height: uint) (depth: uint) (stage: Stage<unit,Image<'T>> option) =
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
    Plan.createWithOptimizer stage pl.memAvail memPeak nElems (uint64 depth) pl.debug pl.optimize
    |> Plan.withRuntimeOptionsFrom pl
    |> Plan.withSourcePeek sourcePeek

let private validateCoordinateDimensions name width height depth =
    if width = 0u then invalidArg "width" $"{name} width must be positive."
    if height = 0u then invalidArg "height" $"{name} height must be positive."
    if depth = 0u then invalidArg "depth" $"{name} depth must be positive."

let private repeatedImageSource<'T when 'T: equality>
    name
    sliceName
    depth
    cleaning
    (image: Image<'T>)
    (pl: Plan<unit, unit>)
    : Plan<unit, Image<'T>> =
    if depth = 0u then invalidArg "depth" $"{name} requires a positive depth."
    if image.GetDimensions() <> 2u then
        invalidArg "image" $"{name} expects a 2D image, got {image.GetDimensions()}D."

    let width = image.GetWidth()
    let height = image.GetHeight()

    let mapper (i: int) : Image<'T> =
        let output = image.copy($"{sliceName}[{i}]", i)
        if pl.debug && DebugLevel.current() >= 1u then printfn "[%s] Created slice %A" name i
        output

    let stage = { srcStage name width height depth mapper with Cleaning = cleaning } |> Some
    srcPlan pl width height depth stage

let private coordinatePlan<'T when 'T: equality> name width height depth makeSlice cleaning (pl: Plan<unit, unit>) : Plan<unit, Image<'T>> =
    validateCoordinateDimensions name width height depth
    let stage = { srcStage name width height depth makeSlice with Cleaning = cleaning } |> Some
    srcPlan pl width height depth stage

let coordinateX<'T when 'T: equality> width height depth =
    let name = "coordinateX"
    let baseImage = Image<'T>.coordinateAxis2D(width, height, 0, $"{name}.base", 0)
    let cleanup () =
        baseImage.decRefCount()
    repeatedImageSource name name depth [ cleanup ] baseImage

let coordinateY<'T when 'T: equality> width height depth =
    let name = "coordinateY"
    let baseImage = Image<'T>.coordinateAxis2D(width, height, 1, $"{name}.base", 0)
    let cleanup () =
        baseImage.decRefCount()
    repeatedImageSource name name depth [ cleanup ] baseImage

let coordinateZ<'T when 'T: equality> width height depth =
    let name = "coordinateZ"
    let makeSlice (z: int) =
        let value = Convert.ChangeType(z, typeof<'T>) :?> 'T
        Image<'T>.constant2D(width, height, value, $"{name}[{z}]", z)
    coordinatePlan name width height depth makeSlice []

let zero<'T when 'T: equality> (width: uint) (height: uint) (depth: uint) (pl: Plan<unit, unit>) : Plan<unit, Image<'T>> =
    let mapper (i: int) : Image<'T> = 
        let image = new Image<'T>([width; height], 1u,$"zero[{i}]", i)
        if pl.debug && DebugLevel.current() >= 1u then printfn "[zero] Created slice %A" i
        image
    let stage = srcStage "zero" width height depth mapper |> Some
    srcPlan pl width height depth stage

let polygonMask (width: uint) (height: uint) (polygon: Polygon2D) : Image<uint8> =
    let vertices = polygon |> List.map (fun p -> p.X, p.Y)
    Image<uint8>.polygonMask(width, height, vertices, "polygonMask", 0)

let repeat<'T when 'T: equality> (image: Image<'T>) (depth: uint) (pl: Plan<unit, unit>) : Plan<unit, Image<'T>> =
    repeatedImageSource "repeat" "repeat" depth [] image pl

let repeatStage<'T when 'T: equality> (depth: uint) : Stage<Image<'T>, Image<'T>> =
    if depth = 0u then invalidArg "depth" "repeatStage requires a positive depth."

    let copySlice (image: Image<'T>) =
        if image.GetDimensions() <> 2u then
            invalidArg "image" $"repeatStage expects 2D images, got {image.GetDimensions()}D."

        try
            [ for i in 0 .. int depth - 1 ->
                let outputIndex = image.index * int depth + i
                image.copy($"repeat[{outputIndex}]", outputIndex) ]
        finally
            image.decRefCount()

    let memoryNeed input = input * uint64 depth
    Stage.map $"repeatStage {depth}" (fun _ image -> copySlice image) memoryNeed id
    --> flattenList ()
    |> Stage.withSliceCardinality SliceCardinality.unknown

let euler2DTransformPath (width: uint) (height: uint) (depth: uint) (transform: string) =
    if width = 0u then invalidArg "width" "euler2DTransformPath requires a positive width."
    if height = 0u then invalidArg "height" "euler2DTransformPath requires a positive height."
    if depth = 0u then invalidArg "depth" "euler2DTransformPath requires a positive depth."

    let centerX = float width / 2.0 - 0.5
    let centerY = float height / 2.0 - 0.5

    fun (i: uint) ->
        let dx = float i
        let angle = 2.0 * Math.PI * float i / float depth

        match transform.Trim().ToLowerInvariant() with
        | "antidiagonal"
        | "anti diagonal"
        | "anti-diagonal" ->
            (centerX, centerY, angle), (float width - dx - centerX, dx - centerY)
        | "topdown"
        | "top down"
        | "top-down" ->
            (centerX, centerY, angle), (0.0, dx - centerY)
        | _ ->
            (centerX, centerY, angle), (0.0, 0.0)

let createByEuler2DTransformFromImage<'T when 'T: equality> (depth: uint) (transform: uint -> (float*float*float) * (float*float)) : Stage<Image<'T>, Image<'T>> =
    if depth = 0u then invalidArg "depth" "createByEuler2DTransformFromImage requires a positive depth."

    let mapper (_debug: bool) (_idx: int64) (img: Image<'T>) =
        try
            if img.GetDimensions() <> 2u then
                invalidArg "img" $"createByEuler2DTransformFromImage expects 2D images, got {img.GetDimensions()}D."

            [ for i in 0 .. int depth - 1 ->
                let rot, trans = transform (uint i)
                ImageFunctions.euler2DTransform img rot trans ]
        finally
            img.decRefCount()

    let memoryNeed input = input * uint64 depth

    Stage.mapi $"createByEuler2DTransformFromImage {depth}" mapper memoryNeed id
    --> flattenList ()
    |> Stage.withSliceCardinality (SliceCardinality.reduceTo (uint64 depth))

let normalNoise<'T when 'T: equality> (width: uint) (height: uint) (depth: uint) (mean: float) (stddev: float) (pl: Plan<unit, unit>) : Plan<unit, Image<'T>> =
    let mapper (i: int) : Image<'T> =
        let zero = new Image<'T>([width; height], 1u, $"normalNoise.zero[{i}]", i)
        let image = ImageFunctions.addNormalNoise mean stddev zero
        zero.decRefCount()
        if pl.debug && DebugLevel.current() >= 1u then printfn "[normalNoise] Created slice %A" i
        image
    let stage = srcStage "normalNoise" width height depth mapper |> Some
    srcPlan pl width height depth stage

let private noiseSource<'T when 'T: equality> name width height depth addNoise (pl: Plan<unit, unit>) : Plan<unit, Image<'T>> =
    let mapper (i: int) : Image<'T> =
        let zero = new Image<'T>([width; height], 1u, $"{name}.zero[{i}]", i)
        let image = addNoise zero
        zero.decRefCount()
        if pl.debug && DebugLevel.current() >= 1u then printfn "[%s] Created slice %A" name i
        image
    let stage = srcStage name width height depth mapper |> Some
    srcPlan pl width height depth stage

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
    srcPlan pl width height depth stage


let empty (pl: Plan<unit, unit>) : Plan<unit, unit> =
    let stage = "empty" |> Stage.empty |> Some
    Plan.createWithOptimizer stage pl.memAvail 0UL 0UL 0UL pl.debug pl.optimize
    |> Plan.withRuntimeOptionsFrom pl

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
      BoundaryEquivalences: (uint * uint64 * uint64) list
      SlabCounts: Map<uint, uint64>
      Statistics: ComponentStatistics list }

type private ComponentLabelKey = uint * uint64

type private ComponentLabelUnionFind() =
    let parent = Dictionary<ComponentLabelKey, ComponentLabelKey>()
    let rank = Dictionary<ComponentLabelKey, byte>()

    member _.Add(key: ComponentLabelKey) =
        if not (parent.ContainsKey key) then
            parent.[key] <- key
            rank.[key] <- 0uy

    member this.Find(key: ComponentLabelKey) =
        let mutable p = Unchecked.defaultof<ComponentLabelKey>

        if not (parent.TryGetValue(key, &p)) then
            parent.[key] <- key
            rank.[key] <- 0uy
            key
        elif p = key then
            key
        else
            let root = this.Find p
            parent.[key] <- root
            root

    member this.Union(left: ComponentLabelKey, right: ComponentLabelKey) =
        let leftRoot = this.Find left
        let rightRoot = this.Find right

        if leftRoot <> rightRoot then
            let leftRank = rank.[leftRoot]
            let rightRank = rank.[rightRoot]

            if leftRank < rightRank then
                parent.[leftRoot] <- rightRoot
            elif leftRank > rightRank then
                parent.[rightRoot] <- leftRoot
            elif leftRoot < rightRoot then
                parent.[rightRoot] <- leftRoot
                rank.[leftRoot] <- leftRank + 1uy
            else
                parent.[leftRoot] <- rightRoot
                rank.[rightRoot] <- rightRank + 1uy

    member _.Keys =
        parent.Keys :> seq<ComponentLabelKey>

let private slabBaseLabels slabCounts =
    let _, _, bases =
        slabCounts
        |> Map.toList
        |> List.sortBy fst
        |> List.fold
            (fun (_, nextLabel, acc) (slabIndex, objectCount) ->
                slabIndex, nextLabel + objectCount, acc |> Map.add slabIndex nextLabel)
            (0u, 1UL, Map.empty<uint, uint64>)
    bases

let private defaultComponentLabel bases slabIndex oldLabel =
    if oldLabel = 0UL then
        0UL
    else
        (bases |> Map.find slabIndex) + oldLabel - 1UL

let private makeConnectedComponentLocalRelabels slabCounts boundaryEquivalences =
    let bases = slabBaseLabels slabCounts
    let unionFind = ComponentLabelUnionFind()
    let touched = HashSet<ComponentLabelKey>()

    boundaryEquivalences
    |> List.iter (fun (previousSlab, previousLabel, currentLabel) ->
        if previousLabel <> 0UL && currentLabel <> 0UL then
            let previousKey = previousSlab, previousLabel
            let currentKey = previousSlab + 1u, currentLabel
            touched.Add(previousKey) |> ignore
            touched.Add(currentKey) |> ignore
            unionFind.Union(previousKey, currentKey))

    let rootLabels = Dictionary<ComponentLabelKey, uint64>()

    // Reverse relabelling reads labelled slabs from the end of the stack, so
    // crossing components use the latest/default-largest label as canonical.
    touched
    |> Seq.iter (fun ((slabIndex, label) as key) ->
        let root = unionFind.Find key
        let candidate = defaultComponentLabel bases slabIndex label
        let mutable existing = 0UL
        if rootLabels.TryGetValue(root, &existing) then
            if candidate > existing then
                rootLabels.[root] <- candidate
        else
            rootLabels.[root] <- candidate)

    let slabRelabels = Dictionary<uint, Dictionary<uint64, uint64>>()

    touched
    |> Seq.iter (fun ((slabIndex, label) as key) ->
        let root = unionFind.Find key
        let newLabel = rootLabels.[root]
        let defaultNewLabel = defaultComponentLabel bases slabIndex label
        if newLabel <> defaultNewLabel then
            let mutable slabMap = Unchecked.defaultof<Dictionary<uint64, uint64>>
            if not (slabRelabels.TryGetValue(slabIndex, &slabMap)) then
                slabMap <- Dictionary<uint64, uint64>()
                slabRelabels.[slabIndex] <- slabMap
            slabMap.[label] <- newLabel)

    slabRelabels

let private flattenConnectedComponentLocalRelabels (slabRelabels: Dictionary<uint, Dictionary<uint64, uint64>>) =
    [ for KeyValue(slabIndex, slabMap) in slabRelabels do
          for KeyValue(oldLabel, newLabel) in slabMap do
              yield slabIndex, oldLabel, newLabel ]
    |> List.sortBy (fun (slabIndex, oldLabel, _) -> slabIndex, oldLabel)

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

let private labelSlabStatistics slabIndex (labelSlab: Image<uint64>) =
    Image.foldi
        (fun index acc label ->
            if label = 0UL then
                acc
            else
                let x = index |> List.tryItem 0 |> Option.defaultValue 0u
                let y = index |> List.tryItem 1 |> Option.defaultValue 0u
                let localZ = index |> List.tryItem 2 |> Option.defaultValue 0u
                let z = uint (labelSlab.index + int localZ)
                let stats = ComponentStatistics.create label x y z
                acc |> ComponentStatistics.addToMap (slabIndex, label) stats)
        Map.empty
        labelSlab

let makeConnectedComponentTranslationTable winSz : Stage<Image<uint64> * uint64, ConnectedComponentTranslationTable> =
    let name = "makeConnectedComponentTranslationTable"
    let winSz = effectiveWindowSize name 1u winSz

    let addBoundaryEquivalences (boundaryEquivalences: ResizeArray<uint * uint64 * uint64>) (unionFind: ComponentLabelUnionFind) previousSlab (previous: Image<uint64>) (current: Image<uint64>) =
        Image.fold2
            (fun () p1 p2 ->
                if p1 <> 0UL && p2 <> 0UL then
                    boundaryEquivalences.Add(previousSlab, p1, p2)
                    unionFind.Union((previousSlab, p1), (previousSlab + 1u, p2)))
            ()
            previous
            current

    let slabBaseLabels slabCounts =
        let _, _, bases =
            slabCounts
            |> Map.toList
            |> List.sortBy fst
            |> List.fold
                (fun (lastSlab, nextLabel, acc) (slabIndex, objectCount) ->
                    slabIndex, nextLabel + objectCount, acc |> Map.add slabIndex nextLabel)
                (0u, 1UL, Map.empty<uint, uint64>)
        bases

    let baseLabel bases slabIndex oldLabel =
        if oldLabel = 0UL then
            0UL
        else
            (bases |> Map.find slabIndex) + oldLabel - 1UL

    let collapseEquivalences (unionFind: ComponentLabelUnionFind) slabCounts =
        let bases = slabBaseLabels slabCounts
        let rootLabels = Dictionary<ComponentLabelKey, uint64>()

        slabCounts
        |> Map.toList
        |> List.iter (fun (slabIndex, objectCount) ->
            for label in 1UL .. objectCount do
                let key = slabIndex, label
                let root = unionFind.Find key
                let candidate = baseLabel bases slabIndex label
                let mutable existing = 0UL
                if rootLabels.TryGetValue(root, &existing) then
                    if candidate < existing then
                        rootLabels.[root] <- candidate
                else
                    rootLabels.[root] <- candidate)

        // Emit later slabs first. This gives the second pass a reverse-ordered
        // translation table, matching the direction in which slab supersets are
        // resolved when temporary label slices are read back from the end.
        [ for slabIndex, objectCount in slabCounts |> Map.toList |> List.sortByDescending fst do
              yield slabIndex, 0UL, 0UL
              for label in 1UL .. objectCount do
                  let root = unionFind.Find(slabIndex, label)
                  yield slabIndex, label, rootLabels.[root] ]

    let mergeTranslatedStats translation localStatistics =
        let translation = translation |> Map.ofList
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
            let unionFind = ComponentLabelUnionFind()
            let boundaryEquivalences = ResizeArray<uint * uint64 * uint64>()
            let mutable slabCounts = Map.empty<uint,uint64>
            let mutable localStatistics = Map.empty<uint * uint64, ComponentStatistics>

            do!
                input
                |> AsyncSeq.iterAsync (fun (labelSlab, objectCount) ->
                    async {
                        let slabIndex = uint (labelSlab.index / int winSz)
                        slabCounts <- slabCounts |> Map.add slabIndex objectCount
                        localStatistics <-
                            labelSlab
                            |> labelSlabStatistics slabIndex
                            |> Map.fold (fun acc key stats -> acc |> ComponentStatistics.addToMap key stats) localStatistics

                        let firstSlice = ImageFunctions.extractSlice 2u 0 labelSlab

                        match previousBoundary with
                        | Some (previousSlab, previousSlice) when slabIndex = previousSlab + 1u ->
                            addBoundaryEquivalences boundaryEquivalences unionFind previousSlab previousSlice firstSlice
                            previousSlice.decRefCount()
                        | Some (_, previousSlice) ->
                            previousSlice.decRefCount()
                        | None -> ()

                        firstSlice.decRefCount()

                        let depth = labelSlab.GetDepth() |> int
                        let lastSlice = ImageFunctions.extractSlice 2u (depth - 1) labelSlab
                        labelSlab.decRefCount()
                        previousBoundary <- Some (slabIndex, lastSlice)
                    })

            previousBoundary |> Option.iter (fun (_, image) -> image.decRefCount())

            let translation =
                collapseEquivalences unionFind slabCounts

            return
                { Labels = translation
                  BoundaryEquivalences = boundaryEquivalences |> Seq.toList
                  SlabCounts = slabCounts
                  Statistics =
                      localStatistics
                      |> mergeTranslatedStats (translation |> List.map (fun (slabIndex, oldLabel, newLabel) -> (slabIndex, oldLabel), newLabel)) }
        }

    let memoryNeed nPixels = 2UL * nPixels * uint64 sizeof<uint64>
    let elementTransformation = fun _ -> 1UL
    Stage.reduce name reducer Streaming memoryNeed elementTransformation

let makeConnectedComponentLabelTranslationTable winSz : Stage<Image<uint64> * uint64, ConnectedComponentTranslationTable> =
    let name = "makeConnectedComponentLabelTranslationTable"
    let winSz = effectiveWindowSize name 1u winSz

    let addBoundaryEquivalences (boundaryEquivalences: ResizeArray<uint * uint64 * uint64>) previousSlab (previous: Image<uint64>) (current: Image<uint64>) =
        Image.fold2
            (fun () p1 p2 ->
                if p1 <> 0UL && p2 <> 0UL then
                    boundaryEquivalences.Add(previousSlab, p1, p2))
            ()
            previous
            current

    let reducer (_debug: bool) (input: AsyncSeq<Image<uint64> * uint64>) =
        async {
            let mutable previousBoundary : (uint * Image<uint64>) option = None
            let boundaryEquivalences = ResizeArray<uint * uint64 * uint64>()
            let mutable slabCounts = Map.empty<uint,uint64>

            do!
                input
                |> AsyncSeq.iterAsync (fun (labelSlab, objectCount) ->
                    async {
                        let slabIndex = uint (labelSlab.index / int winSz)
                        slabCounts <- slabCounts |> Map.add slabIndex objectCount

                        let firstSlice = ImageFunctions.extractSlice 2u 0 labelSlab

                        match previousBoundary with
                        | Some (previousSlab, previousSlice) when slabIndex = previousSlab + 1u ->
                            addBoundaryEquivalences boundaryEquivalences previousSlab previousSlice firstSlice
                            previousSlice.decRefCount()
                        | Some (_, previousSlice) ->
                            previousSlice.decRefCount()
                        | None -> ()

                        firstSlice.decRefCount()

                        let depth = labelSlab.GetDepth() |> int
                        let lastSlice = ImageFunctions.extractSlice 2u (depth - 1) labelSlab
                        labelSlab.decRefCount()
                        previousBoundary <- Some (slabIndex, lastSlice)
                    })

            previousBoundary |> Option.iter (fun (_, image) -> image.decRefCount())
            let boundaryEquivalences = boundaryEquivalences |> Seq.toList
            let localRelabels =
                makeConnectedComponentLocalRelabels slabCounts boundaryEquivalences
                |> flattenConnectedComponentLocalRelabels

            return
                { Labels = localRelabels
                  BoundaryEquivalences = boundaryEquivalences
                  SlabCounts = slabCounts
                  Statistics = [] }
        }

    let memoryNeed nPixels = 2UL * nPixels * uint64 sizeof<uint64>
    let elementTransformation = fun _ -> 1UL
    Stage.reduce name reducer Streaming memoryNeed elementTransformation

let updateConnectedComponents winSz (translationTable: ConnectedComponentTranslationTable) : Stage<Image<uint64>,Image<uint64>> =
    let name = "updateConnectedComponents"
    let winSz = effectiveWindowSize name 1u winSz

    let slabBases = slabBaseLabels translationTable.SlabCounts
    let localRelabels =
        if translationTable.BoundaryEquivalences.IsEmpty then
            translationTable.Labels
        else
            makeConnectedComponentLocalRelabels translationTable.SlabCounts translationTable.BoundaryEquivalences
            |> flattenConnectedComponentLocalRelabels

    let translationTableSlabbed = List.groupBy (fun (slabIndex,_,_) -> slabIndex) localRelabels
    let translationMap =
        translationTableSlabbed
        |> List.map (fun (slabIndex,lst) ->
            let slabTranslation = Dictionary<uint64, uint64>()
            lst
            |> List.iter (fun (_, oldLabel, newLabel) ->
                slabTranslation.[oldLabel] <- newLabel)
            slabIndex, slabTranslation)
        |> Map.ofList

    let mapper (debug: bool) (_sliceIndex: int64) (image: Image<uint64>) : Image<uint64> =
        let slabIndex = uint (image.index / int winSz)
        let baseLabel = slabBases |> Map.tryFind slabIndex |> Option.defaultValue 1UL
        let trans = translationMap |> Map.tryFind slabIndex
        let res =
            Image.map
                (fun oldLabel ->
                    if oldLabel = 0UL then
                        0UL
                    else
                        match trans with
                        | Some slabTranslation ->
                            let mutable newLabel = 0UL
                            if slabTranslation.TryGetValue(oldLabel, &newLabel) then
                                newLabel
                            else
                                baseLabel + oldLabel - 1UL
                        | None ->
                            baseLabel + oldLabel - 1UL)
                image
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

        let mutable chunkInfo : ChunkInfo = {chunks = [0;0;0] ; size = [0UL;0UL;0UL]; topLeftInfo = {dimensions = 0u; size = [0UL;0UL;0UL]; componentType = ""; numberOfComponents = 0u}}
        let memPeak = 256UL // surrugate string length
        let memoryNeed = fun _ -> memPeak
        let elementTransformation = fun _ -> chunkInfo.chunks[int k] |> uint64

        let readSlabStage =
            Stage.map name (fun _ idx -> _readSlabStacked<'T> tmpDir tmpSuffix chunkInfo k idx) memoryNeed id

        let transposeSlicesStage =
            Stage.map
                name
                (fun _ stack ->
                    stack
                    |> List.map (fun im ->
                        let trnsp = ImageFunctions.permuteAxes [1u;0u;2u] im
                        im.decRefCount()
                        trnsp))
                memoryNeed
                id

        (writeChunks tmpDir tmpSuffix winSz winSz winSz)
        --> cleanStage name (fun () -> StackIO.deleteIfExists tmpDir) 
        --> StackCore.ignoreSingles () // force calculation of full stream and decrease references
        --> Stage.map name (fun _ _ -> chunkInfo <- getChunkInfo tmpDir tmpSuffix) memoryNeed elementTransformation // insert side-effect
        --> Stage.map name (fun _ _ -> [0..(chunkInfo.chunks[int k]-1)]) memoryNeed elementTransformation
        --> flattenList () // expand to a new, non-empty stream
        --> readSlabStage
        --> slabToWindowAlong<'T> k
        --> windowItems ()
        --> (if j < i then transposeSlicesStage else identityStage name)
        --> flattenList ()
