module StackImageFunctions

open FSharp.Control
open SlimPipeline // Core processing model
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

let private nativeImageStageCost name memoryModel workUnits =
    StageCostModel.create
        memoryModel
        (StageWorkModel.native Map (Some name) workUnits)

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

let failTypeMismatch<'T> name lst =
    let t = typeof<'T>
    if lst |> List.exists ((=) t) |> not then
        let names = List.map (fun (t: System.Type) -> t.Name) lst
        failwith $"[{name}] wrong type. Type {t} must be one of {names}"

/// Simple functions
let private floatNintTypes = [typeof<float>;typeof<float32>;typeof<int>]
let private floatTypes = [typeof<float>;typeof<float32>]
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
    let workUnits input = float (inputValue input * uint64 win)
    liftWindowedUnaryReleaseAfter "sqrtWindowed" win ImageFunctions.sqrtImage memoryNeed id
    |> withCostModel (nativeImageStageCost $"sqrtWindowed.{typeof<'T>.Name}" memoryModel workUnits)
let square<'T when 'T: equality> : Stage<Image<'T>,Image<'T>> =      
    failTypeMismatch<'T> "square" floatNintTypes
    liftUnaryReleaseAfter "square" ImageFunctions.squareImage id id

//let histogram<'T when 'T: comparison> = histogramOp<'T> "histogram"
let imageHistogram () =
    Stage.map<Image<'T>,Map<'T,uint64>> "histogram: map" (fun _ -> releaseAfter ImageFunctions.histogram) id id// Assumed max for uint8, can be done better

let imageHistogramFold () =
    Stage.fold<Map<'T,uint64>, Map<'T,uint64>> "histogram: fold" ImageFunctions.addHistogram (Map.empty<'T, uint64>) id (fun _ -> 1UL)

let histogram () =
    imageHistogram () --> imageHistogramFold ()

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

let discreteGaussianOp (name: string) (sigma: float) (outputRegionMode: ImageFunctions.OutputRegionMode option) (boundaryCondition: ImageFunctions.BoundaryCondition option) (winSz: uint option) : Stage<Image<float>, Image<float>> =
    let roundFloatToUint v = uint (v+0.5)

    let ksz = 4.0 * sigma + 1.0 |> roundFloatToUint
    let win = Option.defaultValue ksz winSz |> max ksz // max should be found by memory availability
    let stride = win - ksz + 1u
    let pad = 
        match outputRegionMode with
            | Some Valid -> 0u
            | _ -> ksz/2u //floor
    // length-2*pad = n*stride+windowSize
    // stride = windowSize-2*pad
    // => n = (windowSize-length-2*pad)/(2*pad-windowSize)
    // e.g., integer solutions for 
    // windowSize = 1, 6, 15, or 26, pad = 2, length = 22, => n = 21, 10, 1, or 0
    let f debug = 
        if debug then printfn $"discreteGaussianOp: sigma {sigma}, ksz {ksz}, win {win}, stride {stride}, pad {pad}"
        volFctToWindowFctReleaseAfterDebug debug (ImageFunctions.discreteGaussian 3u sigma (ksz |> Some) outputRegionMode boundaryCondition) ksz 0u stride
    let memoryNeed nPixels = (2UL*nPixels*(uint64 win) + (uint64 ksz))*getBytesPerComponent<float>
    let elementTransformation = id
    let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed
    let workUnits input =
        let kernelVoxels = uint64 ksz * uint64 ksz * uint64 ksz
        float (inputValue input * uint64 win * kernelVoxels)
    let stg =
        mapWindow name f memoryNeed elementTransformation
        |> withCostModel (nativeImageStageCost $"discreteGaussian.Float64" memoryModel workUnits)
    (window win pad stride) --> stg --> flattenList ()
    |> Stage.withSliceCardinality (sliceCardinalityForConvolution ksz outputRegionMode)

let discreteGaussian = discreteGaussianOp "discreteGaussian"
let convGauss sigma = discreteGaussianOp "convGauss" sigma None None None

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

let createPadding name pad: Stage<unit,Image<'S>>=
    let transition = ProfileTransition.create Streaming Streaming
    let memoryNeed nPixels = nPixels*getBytesPerComponent<'S>
    let elementTransformation = id
    let zeroMaker i = Image<'S>([0u;0u],1u,"Padding",i)
    Stage.init "padding" pad zeroMaker transition memoryNeed elementTransformation

let convolveOp (name: string) (kernel: Image<'T>) (outputRegionMode: ImageFunctions.OutputRegionMode option) (bc: ImageFunctions.BoundaryCondition option) (winSz: uint option) : Stage<Image<'T>, Image<'T>> =
    let windowFromKernel (k: Image<'T>) : uint =
        max 1u (k.GetDepth())
    let ksz = windowFromKernel kernel
    let win = Option.defaultValue ksz winSz |> max ksz
    let stride = win-ksz+1u
    let pad = 
        match outputRegionMode with
            | Some Valid -> 0u
            | _ -> ksz/2u //floor
    let f debug =  volFctToWindowFctReleaseAfterDebug debug (fun image3D -> ImageFunctions.convolve outputRegionMode bc image3D kernel) ksz 0u stride
    let memoryNeed nPixels = (2UL*nPixels*(uint64 win) + (uint64 ksz))*getBytesPerComponent<'T>
    let elementTransformation = id
    let memoryModel = StageMemoryModel.fromSinglePeak Map memoryNeed
    let workUnits input =
        let kernelVoxels = uint64 (kernel.GetWidth()) * uint64 (kernel.GetHeight()) * uint64 (kernel.GetDepth())
        float (inputValue input * uint64 win * kernelVoxels)
    let stg =
        mapWindow name f memoryNeed elementTransformation
        |> withCostModel (nativeImageStageCost $"convolve.{typeof<'T>.Name}" memoryModel workUnits)
    (window win pad stride) --> stg --> flattenList ()
    |> Stage.withSliceCardinality (sliceCardinalityForConvolution ksz outputRegionMode)

let convolve kernel outputRegionMode boundaryCondition winSz = convolveOp "convolve" kernel outputRegionMode boundaryCondition winSz
let conv kernel = convolveOp "conv" kernel None None None

let finiteDiff (sigma: float) (direction: uint) (order: uint) =
    let kernel = ImageFunctions.finiteDiffFilter3D sigma direction order
    convolveOp "finiteDiff" kernel None None None

// these only works on uint8
let private makeMorphOp (name: string) (radius: uint) (winSz: uint option) (core: uint -> Image<'T> -> Image<'T>) : Stage<Image<'T>,Image<'T>> when 'T: equality =
    let ksz   = 2u * radius + 1u
    let pad = ksz/2u
    let win = Option.defaultValue ksz winSz |> min ksz
    let stride = win - ksz + 1u

    let f debug = volFctToWindowFctReleaseAfterDebug debug (core radius) ksz pad stride
    let stg = mapWindow name f id id
    (window win pad stride) --> stg --> flattenList ()
    |> Stage.withSliceCardinality (SlimPipeline.Domain(sameSliceDomainForKernelDepth ksz))

let erode radius = makeMorphOp "binaryErode"  radius None ImageFunctions.binaryErode
let dilate radius = makeMorphOp "binaryErode"  radius None ImageFunctions.binaryDilate
let opening radius = makeMorphOp "binaryErode"  radius None ImageFunctions.binaryOpening
let closing radius = makeMorphOp "binaryErode"  radius None ImageFunctions.binaryClosing

/// Full stack operators
let binaryFillHoles (winSz: uint)= 
    let pad, stride = 0u, winSz
    let f debug = volFctToWindowFctReleaseAfterDebug debug (ImageFunctions.binaryFillHoles) 1u pad stride
    let stg = mapWindow "fillHoles" f id id
    (window winSz pad stride) --> stg --> flattenList ()

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

let watershed a (winSz: uint) =
    let pad, stride = 0u, winSz
    let f debug = volFctToWindowFctReleaseAfterDebug debug (ImageFunctions.watershed a) 1u pad stride
    let stg = mapWindow "watershed" f id id
    (window winSz pad stride) --> stg --> flattenList ()
let signedDistanceMap (winSz: uint) =
    let pad, stride = 0u, winSz
    let f debug = volFctToWindowFctReleaseAfterDebug debug (ImageFunctions.signedDistanceMap 0uy 1uy) 1u pad stride
    let stg = mapWindow "signedDistanceMap" f id id
    (window winSz pad stride) --> stg --> flattenList ()
let otsuThreshold (winSz: uint) =
    let pad, stride = 0u, winSz
    let f debug = volFctToWindowFctReleaseAfterDebug debug (ImageFunctions.otsuThreshold) 1u pad stride
    let stg = mapWindow "otsuThreshold" f id id
    (window winSz pad stride) --> stg --> flattenList ()

let momentsThreshold (winSz: uint) =
    let pad, stride = 0u, winSz
    let f debug = volFctToWindowFctReleaseAfterDebug debug (ImageFunctions.momentsThreshold) 1u pad stride
    let stg = mapWindow "momentsThreshold" f id id
    (window winSz pad stride) --> stg --> flattenList ()

let threshold a b = liftUnaryReleaseAfter "threshold" (ImageFunctions.threshold a b) id id

let addNormalNoise a b = liftUnaryReleaseAfter "addNormalNoise" (ImageFunctions.addNormalNoise a b) id id

let ImageConstantPad<'T when 'T: equality> (padLower: uint list) (padUpper: uint list) (c: double) =
    liftUnaryReleaseAfter "constantPad2D" (ImageFunctions.constantPad2D padLower padUpper c) id id // Check that constantPad2D makes a new image!!!

let show (plt: Image<'T> -> unit) : Stage<Image<'T>, unit> =
    let consumer (debug: bool) (idx: int) (image: Image<'T>) =
        if debug then printfn "[show] Showing image %d" idx
        let width = image.GetWidth() |> int
        let height = image.GetHeight() |> int
        plt image
        image.decRefCount()
    let memoryNeed = id
    Stage.consumeWith "show" consumer memoryNeed

let plot (plt: (float list)->(float list)->unit) : Stage<(float * float) list, unit> = // better be (float*float) list
    let consumer (debug: bool) (idx: int) (points: (float*float) list) =
        if debug then printfn $"[plot] Plotting {points.Length} 2D points"
        let x,y = points |> List.unzip
        plt x y
    let memoryNeed = id
    Stage.consumeWith "plot" consumer memoryNeed

let print () : Stage<'T, unit> =
    let consumer (debug: bool) (idx: int) (elm: 'T) =
        if debug then printfn "[print]"
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
    if pl.debug then printfn $"[zero] {width}x{height}x{depth}"
    let mapper (i: int) : Image<'T> = 
        let image = new Image<'T>([width; height], 1u,$"zero[{i}]", i)
        if pl.debug then printfn "[zero] Created image %A" i
        image
    let stage = srcStage "zero" width height depth mapper |> Some
    srcPlan pl.debug pl.memAvail width height depth stage

let createByEuler2DTransform<'T when 'T: equality> (img: Image<'T>) (depth: uint) (transform: uint -> (float*float*float) * (float*float)) (pl: Plan<unit, unit>) : Plan<unit, Image<'T>> =
    // width, heigth, depth should be replaced with shape and shapeUpdate, and mapper should be deferred to outside Core!!!
    let width= img.GetWidth()
    let height = img.GetHeight()
    if pl.debug then printfn $"[createByTranslation] {width}x{height}x{depth}"
    let mapper (i: int) : Image<'T> =
        let rot, trans = transform (uint i)
        let image = ImageFunctions.euler2DTransform img rot trans
        if pl.debug then printfn "[createByTranslation] Created image %A" i
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

let makeConnectedComponentTranslationTable (winSz: uint) : Stage<Image<uint64> * uint64,(uint*uint64*uint64) list> =
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

    let reducer (debug: bool) (input: AsyncSeq<Image<uint64> * uint64>) =
        async {
            let mutable previousBoundary : (uint * Image<uint64>) option = None
            let mutable graph = simpleGraph.empty
            let mutable chunkCounts = Map.empty<uint,uint64>

            do!
                input
                |> AsyncSeq.iterAsync (fun (labelChunk, objectCount) ->
                    async {
                        let chunk = uint (labelChunk.index / int winSz)
                        chunkCounts <- chunkCounts |> Map.add chunk objectCount

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

            return
                chunkCounts
                |> makeBaseMap
                |> collapseTouchingComponents graph
                |> toTranslationList
        }

    let memoryNeed nPixels = 2UL * nPixels * uint64 sizeof<uint64>
    let elementTransformation = fun _ -> 1UL
    Stage.reduce name reducer Streaming memoryNeed elementTransformation

let trd (_,_,c) = c

let updateConnectedComponents (winSz: uint) (translationTable: (uint*uint64*uint64) list) : Stage<Image<uint64>,Image<uint64>> =
    let name = "updateConnectedComponents"
    let translationTableChunked = List.groupBy (fun (c,_,_) -> c) translationTable
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
        Stage.idStage<Image<'T>> name
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

        (writeInSlabs tmpDir tmpSuffix winSz winSz winSz)
        --> Stage.clean name (fun () -> StackIO.deleteIfExists tmpDir) 
        --> StackCore.ignoreSingles () // force calculation of full stream and decrease references
        --> Stage.map name (fun _ _ -> chunkInfo <- getChunkInfo tmpDir tmpSuffix) memoryNeed elementTransformation // insert side-effect
        --> Stage.map name (fun _ _ -> [0..(chunkInfo.chunks[int k]-1)]) memoryNeed elementTransformation
        --> flattenList () // expand to a new, non-empty stream
        --> Stage.map name (fun debug idx -> mapper chunkInfo debug idx) memoryNeed elementTransformation // mapper chunkInfo does not work, since argument is copied at compile time
        --> flattenList ()
