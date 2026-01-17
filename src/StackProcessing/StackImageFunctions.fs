module StackImageFunctions

open SlimPipeline // Core processing model
open StackCore
open StackIO

let liftUnary name  = Stage.liftReleaseUnary name ignore
let liftUnaryReleaseAfter (name: string) (f: Image<'S> -> Image<'T>) (memoryNeed: MemoryNeed) (lengthTransformation: LengthTransformation) = 
    Stage.liftReleaseUnary name (decIfImage>>ignore) f memoryNeed lengthTransformation

let getBytesPerComponent<'T> = (typeof<'T> |> Image.getBytesPerComponent |> uint64)

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
        volFctToLstFctReleaseAfter (ImageFunctions.discreteGaussian 3u sigma (ksz |> Some) outputRegionMode boundaryCondition) pad stride
    let memoryNeed nPixels = (2UL*nPixels*(uint64 win) + (uint64 ksz))*(typeof<float> |> Image.getBytesPerComponent |> uint64)
    let lengthTransformation nElems = 
        match outputRegionMode with
            | Some Valid -> nElems - 2UL * uint64 pad
            |_ -> nElems
    let stg = Stage.map name f memoryNeed lengthTransformation // wrong for Valid, where the sequences becomes shorter
    (window win pad stride) --> stg --> flatten ()

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
    let memoryNeed nPixels = nPixels*(typeof<'S> |> Image.getBytesPerComponent |> uint64)
    let lengthTransformation nElems = nElems + (uint64 pad)
    let zeroMaker i = Image<'S>([0u;0u],1u,"Padding",i)
    Stage.init "padding" pad zeroMaker transition memoryNeed lengthTransformation

let convolveOp (name: string) (kernel: Image<'T>) (outputRegionMode: ImageFunctions.OutputRegionMode option) (bc: ImageFunctions.BoundaryCondition option) (winSz: uint option) : Stage<Image<'T>, Image<'T>> =
    let windowFromKernel (k: Image<'T>) : uint =
        max 1u (k.GetDepth())
    let ksz = windowFromKernel kernel
    let win = Option.defaultValue ksz winSz |> min ksz
    let stride = win-ksz+1u
    let pad = 
        match outputRegionMode with
            | Some Valid -> 0u
            | _ -> ksz/2u //floor
    let f debug =  volFctToLstFctReleaseAfter (fun image3D -> ImageFunctions.convolve outputRegionMode bc image3D kernel) pad stride
    let memoryNeed nPixels = (2UL*nPixels*(uint64 win) + (uint64 ksz))*(typeof<'T> |> Image.getBytesPerComponent |> uint64)
    let lengthTransformation nElems = nElems - 2UL*(uint64 pad) 
    let stg = Stage.map name f memoryNeed lengthTransformation
    let padding = createPadding "padding" pad 
    (Stage.prepend "prepend" padding) --> (Stage.append "append" padding) --> (window win 0u stride) --> stg --> flatten ()

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

    let f debug = volFctToLstFctReleaseAfter (core radius) pad stride
    let stg = Stage.map name f id id
    (window win pad stride) --> stg --> flatten ()

let erode radius = makeMorphOp "binaryErode"  radius None ImageFunctions.binaryErode
let dilate radius = makeMorphOp "binaryErode"  radius None ImageFunctions.binaryDilate
let opening radius = makeMorphOp "binaryErode"  radius None ImageFunctions.binaryOpening
let closing radius = makeMorphOp "binaryErode"  radius None ImageFunctions.binaryClosing

/// Full stack operators
let binaryFillHoles (winSz: uint)= 
    let pad, stride = 0u, winSz
    let f debug = volFctToLstFctReleaseAfter (ImageFunctions.binaryFillHoles) pad stride
    let stg = Stage.map "fillHoles" f id id
    (window winSz pad stride) --> stg --> flatten ()

let connectedComponents (winSz: uint) = 
    let pad, stride = 0u, winSz
    let mapper debug = volFctToLstFctReleaseAfter (ImageFunctions.connectedComponents) pad stride
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
    let lengthTransformation = id
    let stg = Stage.map "connectedComponents" mapper memoryNeed lengthTransformation
    (window winSz pad stride) --> stg --> flatten ()

let relabelComponents a (winSz: uint) = 
    let pad, stride = 0u, winSz
    let f debug = volFctToLstFctReleaseAfter (ImageFunctions.relabelComponents a) pad stride
    let stg = Stage.map "relabelComponents" f id id
    (window winSz pad stride) --> stg --> flatten ()

let watershed a (winSz: uint) =
    let pad, stride = 0u, winSz
    let f debug = volFctToLstFctReleaseAfter (ImageFunctions.watershed a) pad stride
    let stg = Stage.map "watershed" f id id
    (window winSz pad stride) --> stg --> flatten ()
let signedDistanceMap (winSz: uint) =
    let pad, stride = 0u, winSz
    let f debug = volFctToLstFctReleaseAfter (ImageFunctions.signedDistanceMap 0uy 1uy) pad stride
    let stg = Stage.map "signedDistanceMap" f id id
    (window winSz pad stride) --> stg --> flatten ()
let otsuThreshold (winSz: uint) =
    let pad, stride = 0u, winSz
    let f debug = volFctToLstFctReleaseAfter (ImageFunctions.otsuThreshold) pad stride
    let stg = Stage.map "otsuThreshold" f id id
    (window winSz pad stride) --> stg --> flatten ()

let momentsThreshold (winSz: uint) =
    let pad, stride = 0u, winSz
    let f debug = volFctToLstFctReleaseAfter (ImageFunctions.momentsThreshold) pad stride
    let stg = Stage.map "momentsThreshold" f id id
    (window winSz pad stride) --> stg --> flatten ()

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
    let lengthTransformation = fun _ -> uint64 depth
    Stage.init name depth mapper transition memoryNeed lengthTransformation

let srcPlan (debug: bool) (memAvail: uint64) (width: uint) (height: uint) (depth: uint) (stage: Stage<unit,Image<'T>> option) =
    let nElems = (uint64 width) * (uint64 height)
    let memPeak = Image<'T>.memoryEstimate width height
    Plan.create stage memAvail memPeak nElems (uint64 depth)  debug

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

let getConnectedChunkNeighbours (inputDir: string) (suffix: string) (winSz: uint) (pl: Plan<unit, unit>) : Plan<unit, Image<uint64>*Image<uint64>> =
    let name = "getConnectedChunkNeighbours"
    let filenames = _getFilenames inputDir ("*"+suffix) Array.sort
    let depth = (uint64 filenames.Length) / (uint64 winSz) - 1UL

    let mapper (i: int) : string*string = 
        let j = (i + 1)*(int winSz)
        filenames[j - 1], filenames[j]

    let transition = ProfileTransition.create Unit Streaming
    let memPeak = 2*sizeof<uint> |> uint64
    let memoryNeed = fun _ -> 2*sizeof<uint> |> uint64
    let lengthTransformation = fun _ -> depth
    let stage = Stage.init "getConnectedChunkNeighbours" (uint depth) mapper transition memoryNeed lengthTransformation |> Some

    let memPerElem = 256UL // surrugate string length
    let length = depth
    Plan.create stage pl.memAvail memPeak memPerElem length pl.debug
    >=> readFilePairs<uint64> pl.debug

let makeAdjacencyGraph (): Stage<Image<uint64>*Image<uint64>,uint*simpleGraph.Graph<uint*uint64>> =
    let name = "makeAdjacencyGraph"
    let folder (i: uint, graph:simpleGraph.Graph<uint*uint64>) (image1: Image<uint64>, image2: Image<uint64>) : uint*simpleGraph.Graph<uint*uint64> = 
        let sliceFolder (i: uint) (g: simpleGraph.Graph<uint*uint64>) (p1: uint64) (p2: uint64) : simpleGraph.Graph<uint*uint64> =
            if p1 <> 0UL && p2 <> 0UL then
                simpleGraph.addEdge (i,p1) (i+1u,p2) g
            else
                g
        let res = (i+1u,Image.fold2 (sliceFolder i) graph image1 image2)
        image1.decRefCount()
        image2.decRefCount()
        res

    let memoryNeed = id
    let lengthTransformation = fun _ -> 1UL
    let init = (0u, simpleGraph.empty)
    Stage.fold $"{name}" folder init memoryNeed lengthTransformation

let makeTranslationTable () : Stage<uint*simpleGraph.Graph<uint*uint64>,(uint*uint64*uint64) list> =
    let name = "makeTranslationTable"
    let mapper (debug: bool) (i: uint, graph:simpleGraph.Graph<uint*uint64>) = 
        let cc = simpleGraph.connectedComponents graph
        List.zip cc [1UL .. uint64 cc.Length]
        |> List.collect (fun ( nodeLst, newVal) -> List.map (fun (chunk, oldVal) -> (chunk, oldVal, newVal)) nodeLst)
        |> List.sort

    let memoryNeed = id
    let lengthTransformation = fun _ -> 1UL
    let init = (0u, simpleGraph.empty)
    Stage.map $"{name}" mapper memoryNeed lengthTransformation

let trd (_,_,c) = c

let updateConnectedComponents (winSz: uint) (translationTable: (uint*uint64*uint64) list) : Stage<Image<uint64>,Image<uint64>> =
    let name = "updateConnectedComponents"
    let translationTableChunked = List.groupBy (fun (c,_,_) -> c) translationTable
    let translationMap = List.map (fun (_,lst) -> (0u,0UL,0UL)::lst |> List.map (fun (_,i,j)->(i,j)) |> Map.ofList) translationTableChunked

    let mapper (debug: bool) (image: Image<uint64>) : Image<uint64> = 
        let chunk = image.index/int winSz
        //let _,trans = translationTableChunked[chunk]
        //let res = Image.map (fun v -> if v=0UL then 0UL else trans |> List.find (fun (_,w,_) -> v = w) |> trd) image
        let trans = translationMap[chunk]
        let res = Image.map (fun v -> trans[v]) image
        image.decRefCount()
        res

    let memoryNeed = fun _ -> 2*sizeof<uint> |> uint64
    let lengthTransformation = id
    Stage.map "updateConnectedComponents" mapper memoryNeed lengthTransformation

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
        let lengthTransformation = id
        Stage.map name (fun _ -> ImageFunctions.permuteAxes [i;j;k]) memoryNeed lengthTransformation
    else // k neq 2u
        // writechunks and reread in permuted order
        let tmpDir = getUnusedDirectoryName "tmp"
        let tmpSuffix = ".tiff"

        let mapper (chunkInfo: ChunkInfo) (debug: bool) (idx: int): Image<'T> list = 
            let chunkSlice = _readChunkSlice<'T> tmpDir tmpSuffix chunkInfo k (int idx)
            let sz = chunkSlice.GetSize ()
            let stack =  ImageFunctions.unstack k chunkSlice
            chunkSlice.decRefCount()
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
        let lengthTransformation = fun _ -> chunkInfo.chunks[int k] |> uint64

        (writeInChunks tmpDir tmpSuffix winSz winSz winSz)
        --> tap "writeInChunks"
        --> Stage.clean name (fun () -> printfn "******** going to delete ***********") 
        --> Stage.clean name (fun () -> StackIO.deleteIfExists tmpDir) 
        --> Stage.clean name (fun () -> printfn "******** deleted it ***********") 
        --> tap "cleaning added"
        --> StackCore.ignoreSingles () // force calculation of full stream and decrease references
        --> tap "ignoreSingles"
        --> Stage.map name (fun _ _ -> chunkInfo <- getChunkInfo tmpDir tmpSuffix) memoryNeed lengthTransformation // insert side-effect
        --> tap "sideEffect"
        --> Stage.map name (fun _ _ -> [0..(chunkInfo.chunks[int k]-1)]) memoryNeed lengthTransformation
        --> tapIt (fun elm -> sprintf $"reinitialize\n{chunkInfo}")
        --> flatten () // expand to a new, non-empty stream
        --> tap "flatten 1"
        --> Stage.map name (fun debug idx -> mapper chunkInfo debug idx) memoryNeed lengthTransformation // mapper chunkInfo does not work, since argument is copied at compile time
        --> tap "mapper"
        --> flatten ()
        --> tap "flatten 2"
