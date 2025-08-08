module StackProcessing

open SlimPipeline // Core processing model
open System.IO

type Stage<'S,'T> = SlimPipeline.Stage<'S,'T>
type Profile = SlimPipeline.Profile
type ProfileTransition = SlimPipeline.ProfileTransition
//type Slice<'S when 'S: equality> = Slice.Slice<'S>
type Image<'S when 'S: equality> = Image.Image<'S>

let releaseAfter (f: Image<'S>->'T) (I:Image<'S>) = 
    let v = f I
    I.decRefCount()
    v
(*
let releaseAfterStage (debug: bool) (stage: Stage<'S, 'T>) : Stage<'S, 'T> = 
    let f (I:Image<'T>): Image<'T> = 
        let R = stage.Pipe.Apply debug I
        I.decRefCount()
        R
    Stage.liftUnary 
*)

let releaseNAfter (n: int) (f: Image<'S> list->'T list) (sLst: Image<'S> list) : 'T list =
    let tLst = f sLst;
    sLst |> List.take (int n) |> List.map (fun I -> I.decRefCount()) |> ignore
    tLst 

let (-->) = Stage.(-->)
let source = Pipeline.source 
let debug = Pipeline.debug 
let zip = Pipeline.zip
let (>=>) pl (stage: Stage<'b,'c>) = Pipeline.(>=>) pl stage //(stage |> disposeInputAfter "read+dispose" )
let (>=>>) = Pipeline.(>=>>)
let (>>=>) = Pipeline.(>>=>)
let (>>=>>) = Pipeline.(>>=>>)
let sink (pl: Pipeline<unit,unit>) : unit = Pipeline.sink pl
let sinkList (plLst: Pipeline<unit,unit> list) : unit = Pipeline.sinkList plLst
//let combineIgnore = Pipeline.combineIgnore
let drainSingle pl = Pipeline.drainSingle "drainSingle" pl
let drainList pl = Pipeline.drainList "drainList" pl
let drainLast pl = Pipeline.drainLast "drainLast" pl
let tap = Stage.tap
let tapIt = Stage.tapIt
let ignoreAll = Stage.ignore<_>
let zeroMaker<'S when 'S: equality> (ex:Image<'S>) : Image<'S> = new Image<'S>(ex.GetSize(), 1u, "zero", 0u)

let liftUnary = Stage.liftUnary
let liftUnaryReleaseAfter 
    (name: string)
    (f: Image<'S> -> Image<'T>)
    (memoryNeed: MemoryNeed)
    (nElemsTransformation: NElemsTransformation) = 
    liftUnary name (releaseAfter f) memoryNeed nElemsTransformation

let liftWindowed = Stage.liftWindowed
let liftWindowedReleaseAfter 
    (name: string) 
    (window: uint) 
    (pad: uint) 
    (zeroMaker: Image<'S>->Image<'S>) 
    (stride: uint) 
    (emitStart: uint) 
    (emitCount: uint) 
    (f: Image<'S> list -> Image<'T> list) 
    (memoryNeed: MemoryNeed)
    (nElemsTransformation: NElemsTransformation)
    : Stage<Image<'S>, Image<'T>> =
    Stage.liftWindowed<Image<'S>,Image<'T>> name window pad zeroMaker stride emitStart emitCount (releaseNAfter (int stride) f) memoryNeed nElemsTransformation

let getBytesPerComponent<'T> = (typeof<'T> |> Image.getBytesPerComponent |> uint64)

let write (outputDir: string) (suffix: string) : Stage<Image<'T>, unit> =
    if not (Directory.Exists(outputDir)) then
        Directory.CreateDirectory(outputDir) |> ignore
    let consumer (debug: bool) (idx: int) (image:Image<'T>) =
        let fileName = Path.Combine(outputDir, sprintf "image_%03d%s" idx suffix)
        if debug then printfn "[write] Saved image %d to %s as %s" idx fileName (typeof<'T>.Name) 
        image.toFile(fileName)
        image.decRefCount()
    Stage.consumeWith $"write \"{outputDir}/*{suffix}\"" consumer 

let show (plt: Image<'T> -> unit) : Stage<Image<'T>, unit> =
    let consumer (debug: bool) (idx: int) (image:Image<'T>) =
        if debug then printfn "[show] Showing image %d" idx
        let width = image.GetWidth() |> int
        let height = image.GetHeight() |> int
        plt image
        image.decRefCount()
    Stage.consumeWith "show" consumer 

let plot (plt: (float list)->(float list)->unit) : Stage<(float * float) list, unit> = // better be (float*float) list
    let consumer (debug: bool) (idx: int) (points: (float*float) list) =
        if debug then printfn "[plot] Plotting {points.Length} 2D points"
        let x,y = points |> List.unzip
        plt x y
    Stage.consumeWith "plot" consumer 

let print () : Stage<'T, unit> =
    let consumer (debug: bool) (idx: int) (elm: 'T) =
        if debug then printfn "[print]"
        printfn "%d -> %A" idx elm
    Stage.consumeWith "print" consumer 

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
    (pl : Pipeline<unit, unit>) 
    : Pipeline<unit, Image<uint>> =
    let img = Image.generateCoordinateAxis axis size
    let sz = Image.GetSize img
    let shapeUpdate = fun (s: Shape) -> s
    let op : Stage<unit, Image<uint>> =
        {
            Name = "axisSource"
            Pipe = img |> liftImageSource "axisSource"
            Transition = ProfileTransition.create Constant Streaming
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
let cast<'S,'T when 'S: equality and 'T: equality> = Stage.cast<Image<'S>,Image<'T>> (sprintf "cast(%s->%s)" typeof<'S>.Name typeof<'T>.Name) (fun (I: Image<'S>) -> I.castTo<'T> ())

/// Basic arithmetic
let memNeeded<'T> nTimes nElems = nElems*nTimes*getBytesPerComponent<'T> // Assuming source and target in memory simultaneously
let add (image: Image<'T>) = 
    liftUnaryReleaseAfter "add" ((+) image) id id
let inline scalarAddImage<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) = 
    liftUnaryReleaseAfter "scalarAddImage" (fun (s:Image<^T>)->ImageFunctions.scalarAddImage<^T> i s) id id
let inline imageAddScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    liftUnaryReleaseAfter "imageAddScalar" (fun (s:Image<^T>)->ImageFunctions.imageAddScalar<^T> s i) id id

let sub (image: Image<'T>) = 
    liftUnaryReleaseAfter "sub" ((-) image) id id
let inline scalarSubImage<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    liftUnaryReleaseAfter "scalarSubImage" (fun (s:Image<^T>)->ImageFunctions.scalarSubImage<^T> i s) id id
let inline imageSubScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    liftUnaryReleaseAfter "imageSubScalar" (fun (s:Image<^T>)->ImageFunctions.imageSubScalar<^T> s i) id id

let mul (image: Image<'T>) = liftUnaryReleaseAfter "mul" (( * ) image) id id
let inline scalarMulImage<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    liftUnaryReleaseAfter "scalarMulImage" (fun (s:Image<^T>)->ImageFunctions.scalarMulImage<^T> i s) id id
let inline imageMulScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    liftUnaryReleaseAfter "imageMulScalar" (fun (s:Image<^T>)->ImageFunctions.imageMulScalar<^T> s i) id id

let div (image: Image<'T>) = liftUnaryReleaseAfter "div" ((/) image) id id
let inline scalarDivImage<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    liftUnaryReleaseAfter "scalarDivImage" (fun (s:Image<^T>)->ImageFunctions.scalarDivImage<^T> i s) id id
let inline imageDivScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    liftUnaryReleaseAfter "imageDivScalar" (fun (s:Image<^T>)->ImageFunctions.imageDivScalar<^T> s i) id id

/// Simple functions
let abs<'T when 'T: equality> : Stage<Image<'T>,Image<'T>> = liftUnaryReleaseAfter "abs"    ImageFunctions.absImage id id
let absFloat      : Stage<Image<float>,Image<float>> =      liftUnaryReleaseAfter "abs"    ImageFunctions.absImage id id
let absFloat32    : Stage<Image<float32>,Image<float32>> =  liftUnaryReleaseAfter "abs"    ImageFunctions.absImage id id
let absInt        : Stage<Image<int>,Image<int>> =          liftUnaryReleaseAfter "abs"    ImageFunctions.absImage id id
let acosFloat     : Stage<Image<float>,Image<float>> =      liftUnaryReleaseAfter "acos"   ImageFunctions.acosImage id id
let acosFloat32   : Stage<Image<float32>,Image<float32>> =  liftUnaryReleaseAfter "acos"   ImageFunctions.acosImage id id
let asinFloat     : Stage<Image<float>,Image<float>> =      liftUnaryReleaseAfter "asin"   ImageFunctions.asinImage id id
let asinFloat32   : Stage<Image<float32>,Image<float32>> =  liftUnaryReleaseAfter "asin"   ImageFunctions.asinImage id id
let atanFloat     : Stage<Image<float>,Image<float>> =      liftUnaryReleaseAfter "atan"   ImageFunctions.atanImage id id
let atanFloat32   : Stage<Image<float32>,Image<float32>> =  liftUnaryReleaseAfter "atan"   ImageFunctions.atanImage id id
let cosFloat      : Stage<Image<float>,Image<float>> =      liftUnaryReleaseAfter "cos"    ImageFunctions.cosImage id id
let cosFloat32    : Stage<Image<float32>,Image<float32>> =  liftUnaryReleaseAfter "cos"    ImageFunctions.cosImage id id
let sinFloat      : Stage<Image<float>,Image<float>> =      liftUnaryReleaseAfter "sin"    ImageFunctions.sinImage id id
let sinFloat32    : Stage<Image<float32>,Image<float32>> =  liftUnaryReleaseAfter "sin"    ImageFunctions.sinImage id id
let tanFloat      : Stage<Image<float>,Image<float>> =      liftUnaryReleaseAfter "tan"    ImageFunctions.tanImage id id
let tanFloat32    : Stage<Image<float32>,Image<float32>> =  liftUnaryReleaseAfter "tan"    ImageFunctions.tanImage id id
let expFloat      : Stage<Image<float>,Image<float>> =      liftUnaryReleaseAfter "exp"    ImageFunctions.expImage id id
let expFloat32    : Stage<Image<float32>,Image<float32>> =  liftUnaryReleaseAfter "exp"    ImageFunctions.expImage id id
let log10Float    : Stage<Image<float>,Image<float>> =      liftUnaryReleaseAfter "log10"  ImageFunctions.log10Image id id
let log10Float32  : Stage<Image<float32>,Image<float32>> =  liftUnaryReleaseAfter "log10"  ImageFunctions.log10Image id id
let logFloat      : Stage<Image<float>,Image<float>> =      liftUnaryReleaseAfter "log"    ImageFunctions.logImage id id
let logFloat32    : Stage<Image<float32>,Image<float32>> =  liftUnaryReleaseAfter "log"    ImageFunctions.logImage id id
let roundFloat    : Stage<Image<float>,Image<float>> =      liftUnaryReleaseAfter "round"  ImageFunctions.roundImage id id
let roundFloat32  : Stage<Image<float32>,Image<float32>> =  liftUnaryReleaseAfter "round"  ImageFunctions.roundImage id id
let sqrtFloat     : Stage<Image<float>,Image<float>> =      liftUnaryReleaseAfter "sqrt"   ImageFunctions.sqrtImage id id
let sqrtFloat32   : Stage<Image<float32>,Image<float32>> =  liftUnaryReleaseAfter "sqrt"   ImageFunctions.sqrtImage id id
let sqrtInt       : Stage<Image<int>,Image<int>> =          liftUnaryReleaseAfter "sqrt"   ImageFunctions.sqrtImage id id
let squareFloat   : Stage<Image<float>,Image<float>> =      liftUnaryReleaseAfter "square" ImageFunctions.squareImage id id
let squareFloat32 : Stage<Image<float32>,Image<float32>> =  liftUnaryReleaseAfter "square" ImageFunctions.squareImage id id
let squareInt     : Stage<Image<int>,Image<int>> =          liftUnaryReleaseAfter "square" ImageFunctions.squareImage id id

//let histogram<'T when 'T: comparison> = histogramOp<'T> "histogram"
let imageHistogram () =
    Stage.map<Image<'T>,Map<'T,uint64>> "histogram:map" (releaseAfter ImageFunctions.histogram) id id// Assumed max for uint8, can be done better

let imageHistogramFold () =
    Stage.fold<Map<'T,uint64>, Map<'T,uint64>> "histogram:fold" ImageFunctions.addHistogram (Map.empty<'T, uint64>) id id

let histogram () =
    imageHistogram () --> imageHistogramFold ()

let inline map2pairs< ^T, ^S when ^T: comparison and ^T : (static member op_Explicit : ^T -> float) and  ^S : (static member op_Explicit : ^S -> float) > = 
    let map2pairs (map: Map<'T, 'S>): ('T * 'S) list =
        map |> Map.toList
    liftUnary "map2pairs" map2pairs id id
let inline pairs2floats< ^T, ^S when ^T : (static member op_Explicit : ^T -> float) and  ^S : (static member op_Explicit : ^S -> float) > = 
    let pairs2floats (pairs: (^T * ^S) list) : (float * float) list =
        pairs |> List.map (fun (k, v) -> (float k, float v)) 
    liftUnary "pairs2floats" pairs2floats id id
let inline pairs2ints< ^T, ^S when ^T : (static member op_Explicit : ^T -> int) and  ^S : (static member op_Explicit : ^S -> int) > = 
    let pairs2ints (pairs: (^T * ^S) list) : (int * int) list =
        pairs |> List.map (fun (k, v) -> (int k, int v)) 
    liftUnary "pairs2ints" pairs2ints id id

type ImageStats = ImageFunctions.ImageStats
let imageComputeStats () =
    Stage.map<Image<'T>,ImageStats> "computeStats:map" (releaseAfter ImageFunctions.computeStats) id id

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
    Stage.fold<ImageStats, ImageStats> "computeStats:fold" ImageFunctions.addComputeStats zeroStats id id

let computeStats () =
    imageComputeStats () --> imageComputeStatsFold ()

////////////////////////////////////////////////
/// Convolution like operators

// Chained type definitions do expose the originals
open type ImageFunctions.OutputRegionMode
open type ImageFunctions.BoundaryCondition

let stackFUnstack f (images : Image<'T> list) =
    let stck = images |> ImageFunctions.stack 
    stck |> releaseAfter (f >> ImageFunctions.unstack)

let skipNTakeM (n: uint) (m: uint) (lst: 'a list) : 'a list =
    let m = uint lst.Length - 2u*n;
    if m = 0u then []
    else lst |> List.skip (int n) |> List.take (int m) // This needs releaseAfter!!!

let stackFUnstackTrim trim f (images : Image<'T> list) =
    let m = uint images.Length - 2u*trim 
    let stck = images |> ImageFunctions.stack 
    stck |> (f >> ImageFunctions.unstack >> skipNTakeM trim m)

let discreteGaussianOp (name:string) (sigma:float) (outputRegionMode: ImageFunctions.OutputRegionMode option) (boundaryCondition: ImageFunctions.BoundaryCondition option) (winSz: uint option): Stage<Image<float>, Image<float>> =
    let roundFloatToUint v = uint (v+0.5)

    let ksz = 4.0 * sigma + 1.0 |> roundFloatToUint
    let win = Option.defaultValue ksz winSz |> max ksz // max should be found by memory availability
    let stride = win - ksz + 1u
    let pad = 
        match outputRegionMode with
            | Some Valid -> 0u
            | _ -> ksz/2u //floor
    let f images = images |> stackFUnstackTrim pad (fun image3D -> ImageFunctions.discreteGaussian 3u sigma (ksz |> Some) outputRegionMode boundaryCondition image3D)
    liftWindowedReleaseAfter name win pad zeroMaker<float> stride pad stride f id id

let discreteGaussian = discreteGaussianOp "discreteGaussian"
let convGauss sigma = discreteGaussianOp "convGauss" sigma None None None

// stride calculation example
// ker = 3, win = 7
// Image position:  2 1 0 1 2 3 4 5 6 7 8 9 
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

let convolveOp (name: string) (kernel: Image<'T>) (outputRegionMode: ImageFunctions.OutputRegionMode option) (bc: ImageFunctions.BoundaryCondition option) (winSz: uint option): Stage<Image<'T>, Image<'T>> =
    let windowFromKernel (k: Image<'T>) : uint =
        max 1u (k.GetDepth())
    let ksz = windowFromKernel kernel
    let win = Option.defaultValue ksz winSz |> min ksz
    let stride = win-ksz+1u
    let pad = 
        match outputRegionMode with
            | Some Valid -> 0u
            | _ -> ksz/2u //floor
    let f images = images |> stackFUnstackTrim (ksz - 1u) (fun image3D -> ImageFunctions.convolve outputRegionMode bc image3D kernel)
    liftWindowedReleaseAfter name win pad zeroMaker<'T> stride (win-1u) (1u) f id id

let convolve kernel outputRegionMode boundaryCondition winSz = convolveOp "convolve" kernel outputRegionMode boundaryCondition winSz
let conv kernel = convolveOp "conv" kernel None None None

let finiteDiff (direction: uint) (order: uint) =
    let kernel = ImageFunctions.finiteDiffFilter3D direction order
    convolveOp "finiteDiff" kernel None None None

// these only works on uint8
let private makeMorphOp (name:string) (radius:uint) (winSz: uint option) (core: uint -> Image<'T> -> Image<'T>) : Stage<Image<'T>,Image<'T>> when 'T: equality =
    let ksz   = 2u * radius + 1u
    let win = Option.defaultValue ksz winSz |> min ksz
    let stride = win - ksz + 1u
    let f images = images |> stackFUnstackTrim radius (core radius)
    liftWindowedReleaseAfter name win 0u zeroMaker<'T> stride (stride - 1u) stride f id id

let erode radius = makeMorphOp "binaryErode"  radius None ImageFunctions.binaryErode
let dilate radius = makeMorphOp "binaryErode"  radius None ImageFunctions.binaryDilate
let opening radius = makeMorphOp "binaryErode"  radius None ImageFunctions.binaryOpening
let closing radius = makeMorphOp "binaryErode"  radius None ImageFunctions.binaryClosing

/// Full stack operators
let binaryFillHoles (winSz: uint)= 
    let f images = images |> stackFUnstack ImageFunctions.binaryFillHoles
    liftWindowedReleaseAfter "fillHoles" winSz 0u zeroMaker<uint8> winSz 0u winSz f id id

let connectedComponents (winSz: uint) = 
    let f images = images |> stackFUnstack ImageFunctions.connectedComponents
    liftWindowedReleaseAfter "connectedComponents" winSz 0u zeroMaker<uint8> winSz 0u winSz f id id

let relabelComponents a (winSz: uint) = 
    let f images = images |> stackFUnstack (ImageFunctions.relabelComponents a)
    liftWindowedReleaseAfter "relabelComponents" winSz 0u zeroMaker<uint64> winSz 0u winSz f id id

let watershed a (winSz: uint) =
    let f images = images |> stackFUnstack (ImageFunctions.watershed a)
    liftWindowedReleaseAfter "watershed" winSz 0u zeroMaker<uint8> winSz 0u winSz f id id

let signedDistanceMap (winSz: uint) =
    let f images = images |> stackFUnstack (ImageFunctions.signedDistanceMap 0uy 1uy)
    liftWindowedReleaseAfter "signedDistanceMap" winSz 0u zeroMaker<uint8> winSz 0u winSz f id id

let otsuThreshold (winSz: uint) =
    let f images = images |> stackFUnstack (ImageFunctions.otsuThreshold)
    liftWindowedReleaseAfter "otsuThreshold" winSz 0u zeroMaker<uint8> winSz 0u winSz f id id

let momentsThreshold (winSz: uint) =
    let f images = images |> stackFUnstack (ImageFunctions.momentsThreshold)
    liftWindowedReleaseAfter "momentsThreshold" winSz 0u zeroMaker<uint8> winSz 0u winSz f id id

let threshold a b = liftUnaryReleaseAfter "threshold" (ImageFunctions.threshold a b) id id

let addNormalNoise a b = liftUnaryReleaseAfter "addNormalNoise" (ImageFunctions.addNormalNoise a b) id id

let ImageConstantPad<'T when 'T : equality> (padLower : uint list) (padUpper : uint list) (c : double) =
    liftUnaryReleaseAfter "constantPad2D" (ImageFunctions.constantPad2D padLower padUpper c) id id // Check that constantPad2D makes a new image!!!

// Not Pipes nor Operators
type FileInfo = ImageFunctions.FileInfo
let getStackDepth (inputDir: string) (suffix: string) : uint =
    let files = Directory.GetFiles(inputDir, "*"+suffix) |> Array.sort
    files.Length |> uint

let getStackInfo (inputDir: string) (suffix: string): FileInfo =
    let files = Directory.GetFiles(inputDir, "*"+suffix) |> Array.sort
    let depth = files.Length |> uint64
    if depth = 0uL then
        failwith $"No {suffix} files found in directory: {inputDir}"
    let fi = ImageFunctions.getFileInfo(files[0])
    {fi with dimensions = fi.dimensions+1u; size = fi.size @ [depth]}

let getStackSize (inputDir: string) (suffix: string): uint*uint*uint =
    let fi = getStackInfo inputDir suffix 
    (uint fi.size[0],uint fi.size[1],uint fi.size[2])

let getStackWidth (inputDir: string) (suffix: string): uint64 =
    let fi = getStackInfo inputDir suffix
    fi.size[0]

let getStackHeight (inputDir: string) (suffix: string): uint64 =
    let fi = getStackInfo inputDir suffix
    printfn "%A" fi
    fi.size[1]

let zero<'T when 'T: equality> 
    (width: uint) 
    (height: uint) 
    (depth: uint) 
    (pl : Pipeline<unit, unit>) 
    : Pipeline<unit, Image<'T>> =
    // width, heigth, depth should be replaced with shape and shapeUpdate, and mapper should be deferred to outside Core!!!
    if pl.debug then printfn $"[createAs] {width}x{height}x{depth}"
    let mapper (i: uint) : Image<'T> = 
        let image = new Image<'T>([width; height;1u], 1u,$"zero[{i}]", i)
        if pl.debug then printfn "[create] Created image %A" i
        image
    let transition = ProfileTransition.create Constant Streaming
    let shapeUpdate = id
    let stage = Stage.init "create" depth mapper transition id id |> Some
    //let flow = Flow.returnM stage
    let nElems = (uint64 width) * (uint64 height)
    let context = id
    Pipeline.create stage pl.memAvail nElems (uint64 depth)  pl.debug

let readFilteredOp<'T when 'T: equality> (name:string) (inputDir : string) (suffix : string) (filter: string[]->string[]) (pl : Pipeline<unit, unit>) : Pipeline<unit, Image<'T>> =
    // much should be deferred to outside Core!!!
    if pl.debug then printfn $"[{name}]"
    let (width,height,depth) = getStackSize inputDir suffix
    let filenames = Directory.GetFiles(inputDir, "*"+suffix) |> filter
    let depth = uint filenames.Length
    let mapper (i: uint) : Image<'T> = 
        let fileName = filenames[int i]; 
        let image = Image<'T>.ofFile (fileName, fileName, uint i)
        if pl.debug then printfn "[%s] Reading image %A from %s as %s" name i fileName (typeof<'T>.Name)
        image
    let transition = ProfileTransition.create Constant Streaming
    let stage = Stage.init $"{name}" (uint depth) mapper transition id id |> Some
    //let flow = Flow.returnM stage
    let memPerElem = (uint64 width)*(uint64 height)*getBytesPerComponent<'T>
    let length = (uint64 depth)
    Pipeline.create stage pl.memAvail memPerElem length  pl.debug

let readFiltered<'T when 'T: equality> (inputDir : string) (suffix : string)  (filter: string[]->string[]) (pl : Pipeline<unit, unit>) : Pipeline<unit, Image<'T>> =
    readFilteredOp<'T> $"readFiltered \"{inputDir}/*{suffix}\"" inputDir suffix filter pl

let read<'T when 'T: equality> (inputDir : string) (suffix : string) (pl : Pipeline<unit, unit>) : Pipeline<unit, Image<'T>> =
    readFilteredOp<'T> $"read \"{inputDir}/*{suffix}\"" inputDir suffix Array.sort pl

let readRandom<'T when 'T: equality> (count: uint) (inputDir : string) (suffix : string) (pl : Pipeline<unit, unit>) : Pipeline<unit, Image<'T>> =
    readFilteredOp<'T> $"readRandom \"{inputDir}/*{suffix}\"" inputDir suffix (Array.randomChoices (int count)) pl
