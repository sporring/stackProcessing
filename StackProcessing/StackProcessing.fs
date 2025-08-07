module StackProcessing

open SlimPipeline // Core processing model
open System.IO

type Stage<'S,'T> = SlimPipeline.Stage<'S,'T>
type Profile = SlimPipeline.Profile
type ProfileTransition = SlimPipeline.ProfileTransition
type Slice<'S when 'S: equality> = Slice.Slice<'S>

let (-->) = Stage.(-->)
let source = Pipeline.source 
let debug = Pipeline.debug 
let zip = Pipeline.zip
let (>=>) = Pipeline.(>=>)
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
let liftUnary 
    (f: Slice.Slice<'S> -> Slice.Slice<'T>)
    (memoryNeed: MemoryNeed)
    (nElemsTransformation: NElemsTransformation) = 
    Stage.liftUnary<Slice.Slice<'S>,Slice.Slice<'T>> "liftUnary" f memoryNeed nElemsTransformation
let zeroMaker<'S when 'S: equality> (ex:Slice.Slice<'S>) : Slice.Slice<'S> = Slice.create<'S> (Slice.GetWidth ex) (Slice.GetHeight ex) 1u 0u
let liftWindowed 
    (name: string) 
    (window: uint) 
    (pad: uint) 
    (zeroMaker: Slice.Slice<'S>->Slice.Slice<'S>) 
    (stride: uint) 
    (emitStart: uint) 
    (emitCount: uint) 
    (f: Slice.Slice<'S> list -> Slice.Slice<'T> list) 
    (memoryNeed: MemoryNeed)
    (nElemsTransformation: NElemsTransformation)
    : Stage<Slice.Slice<'S>, Slice.Slice<'T>> =
    Stage.liftWindowed<Slice.Slice<'S>,Slice.Slice<'T>> name window pad zeroMaker stride emitStart emitCount f memoryNeed nElemsTransformation
let getBytesPerComponent<'T> = (typeof<'T> |> Image.getBytesPerComponent |> uint64)

let write (outputDir: string) (suffix: string) : Stage<Slice.Slice<'T>, unit> =
    if not (Directory.Exists(outputDir)) then
        Directory.CreateDirectory(outputDir) |> ignore
    let consumer (debug: bool) (idx: int) (slice:Slice.Slice<'T>) =
        let fileName = Path.Combine(outputDir, sprintf "slice_%03d%s" idx suffix)
        if debug then printfn "[write] Saved slice %d to %s as %s" idx fileName (typeof<'T>.Name) 
        slice.Image.toFile(fileName)
    Stage.consumeWith $"write to \"{outputDir}\"" consumer 

let show (plt: Slice.Slice<'T> -> unit) : Stage<Slice.Slice<'T>, unit> =
    let consumer (debug: bool) (idx: int) (slice:Slice.Slice<'T>) =
        if debug then printfn "[show] Showing slice %d" idx
        let width = slice |> Slice.GetWidth |> int
        let height = slice |> Slice.GetHeight |> int
        plt slice
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
let liftImageSource (name: string) (img: Slice.Slice<'T>) : Pipe<unit, Slice.Slice<'T>> =
    {
        Name = name
        Profile = Streaming
        Apply = fun _ -> img |> Slice.unstack |> AsyncSeq.ofSeq
    }

let axisSource
    (axis: int) 
    (size: int list)
    (pl : Pipeline<unit, unit>) 
    : Pipeline<unit, Slice.Slice<uint>> =
    let img = Slice.generateCoordinateAxis axis size
    let sz = Slice.GetSize img
    let shapeUpdate = fun (s: Shape) -> s
    let op : Stage<unit, Slice.Slice<uint>> =
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
let cast<'S,'T when 'S: equality and 'T: equality> = Stage.cast<Slice.Slice<'S>,Slice.Slice<'T>> (sprintf "cast(%s->%s)" typeof<'S>.Name typeof<'T>.Name) Slice.cast<'S,'T>

/// Basic arithmetic
let memNeeded<'T> nTimes nElems = nElems*nTimes*getBytesPerComponent<'T> // Assuming source and target in memory simultaneously
let add slice = 
    Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>> "add" (Slice.add slice) id id
let inline scalarAddSlice<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) = 
    Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>> "scalarAddSlice" (fun (s:Slice.Slice<^T>)->Slice.scalarAddSlice<^T> i s) id id
let inline sliceAddScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>> "sliceAddScalar" (fun (s:Slice.Slice<^T>)->Slice.sliceAddScalar<^T> s i) id id

let sub slice = 
    Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>> "sub" (Slice.sub slice) id id
let inline scalarSubSlice<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>> "scalarSubSlice" (fun (s:Slice.Slice<^T>)->Slice.scalarSubSlice<^T> i s) id id
let inline sliceSubScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>> "sliceSubScalar" (fun (s:Slice.Slice<^T>)->Slice.sliceSubScalar<^T> s i) id id

let mul slice = Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>> "mul" (Slice.mul slice) id id
let inline scalarMulSlice<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>> "scalarMulSlice" (fun (s:Slice.Slice<^T>)->Slice.scalarMulSlice<^T> i s) id id
let inline sliceMulScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>> "sliceMulScalar" (fun (s:Slice.Slice<^T>)->Slice.sliceMulScalar<^T> s i) id id

let div slice = Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>> "div" (Slice.div slice) id id
let inline scalarDivSlice<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>> "scalarDivSlice" (fun (s:Slice.Slice<^T>)->Slice.scalarDivSlice<^T> i s) id id
let inline sliceDivScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>> "sliceDivScalar" (fun (s:Slice.Slice<^T>)->Slice.sliceDivScalar<^T> s i) id id

/// Simple functions
let abs<'T when 'T: equality> : Stage<Slice.Slice<'T>,Slice.Slice<'T>> =      Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>> "abs"    Slice.absSlice id id
let absFloat      : Stage<Slice.Slice<float>,Slice.Slice<float>> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>> "abs"    Slice.absSlice id id
let absFloat32    : Stage<Slice.Slice<float32>,Slice.Slice<float32>> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>> "abs"    Slice.absSlice id id
let absInt        : Stage<Slice.Slice<int>,Slice.Slice<int>> =          Stage.liftUnary<Slice.Slice<int>,Slice.Slice<int>> "abs"    Slice.absSlice id id
let acosFloat     : Stage<Slice.Slice<float>,Slice.Slice<float>> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>> "acos"   Slice.acosSlice id id
let acosFloat32   : Stage<Slice.Slice<float32>,Slice.Slice<float32>> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>> "acos"   Slice.acosSlice id id
let asinFloat     : Stage<Slice.Slice<float>,Slice.Slice<float>> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>> "asin"   Slice.asinSlice id id
let asinFloat32   : Stage<Slice.Slice<float32>,Slice.Slice<float32>> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>> "asin"   Slice.asinSlice id id
let atanFloat     : Stage<Slice.Slice<float>,Slice.Slice<float>> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>> "atan"   Slice.atanSlice id id
let atanFloat32   : Stage<Slice.Slice<float32>,Slice.Slice<float32>> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>> "atan"   Slice.atanSlice id id
let cosFloat      : Stage<Slice.Slice<float>,Slice.Slice<float>> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>> "cos"    Slice.cosSlice id id
let cosFloat32    : Stage<Slice.Slice<float32>,Slice.Slice<float32>> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>> "cos"    Slice.cosSlice id id
let sinFloat      : Stage<Slice.Slice<float>,Slice.Slice<float>> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>> "sin"    Slice.sinSlice id id
let sinFloat32    : Stage<Slice.Slice<float32>,Slice.Slice<float32>> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>> "sin"    Slice.sinSlice id id
let tanFloat      : Stage<Slice.Slice<float>,Slice.Slice<float>> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>> "tan"    Slice.tanSlice id id
let tanFloat32    : Stage<Slice.Slice<float32>,Slice.Slice<float32>> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>> "tan"    Slice.tanSlice id id
let expFloat      : Stage<Slice.Slice<float>,Slice.Slice<float>> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>> "exp"    Slice.expSlice id id
let expFloat32    : Stage<Slice.Slice<float32>,Slice.Slice<float32>> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>> "exp"    Slice.expSlice id id
let log10Float    : Stage<Slice.Slice<float>,Slice.Slice<float>> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>> "log10"  Slice.log10Slice id id
let log10Float32  : Stage<Slice.Slice<float32>,Slice.Slice<float32>> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>> "log10"  Slice.log10Slice id id
let logFloat      : Stage<Slice.Slice<float>,Slice.Slice<float>> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>> "log"    Slice.logSlice id id
let logFloat32    : Stage<Slice.Slice<float32>,Slice.Slice<float32>> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>> "log"    Slice.logSlice id id
let roundFloat    : Stage<Slice.Slice<float>,Slice.Slice<float>> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>> "round"  Slice.roundSlice id id
let roundFloat32  : Stage<Slice.Slice<float32>,Slice.Slice<float32>> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>> "round"  Slice.roundSlice id id
let sqrtFloat     : Stage<Slice.Slice<float>,Slice.Slice<float>> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>> "sqrt"   Slice.sqrtSlice id id
let sqrtFloat32   : Stage<Slice.Slice<float32>,Slice.Slice<float32>> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>> "sqrt"   Slice.sqrtSlice id id
let sqrtInt       : Stage<Slice.Slice<int>,Slice.Slice<int>> =          Stage.liftUnary<Slice.Slice<int>,Slice.Slice<int>> "sqrt"   Slice.sqrtSlice id id
let squareFloat   : Stage<Slice.Slice<float>,Slice.Slice<float>> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>> "square" Slice.squareSlice id id
let squareFloat32 : Stage<Slice.Slice<float32>,Slice.Slice<float32>> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>> "square" Slice.squareSlice id id
let squareInt     : Stage<Slice.Slice<int>,Slice.Slice<int>> =          Stage.liftUnary<Slice.Slice<int>,Slice.Slice<int>> "square" Slice.squareSlice id id

//let histogram<'T when 'T: comparison> = histogramOp<'T> "histogram"
let sliceHistogram () =
    Stage.map<Slice.Slice<'T>,Map<'T,uint64>> "histogram:map" Slice.histogram id id // Assumed max for uint8, can be done better

let sliceHistogramFold () =
    Stage.fold<Map<'T,uint64>, Map<'T,uint64>> "histogram:fold" Slice.addHistogram (Map.empty<'T, uint64>) id id

let histogram () =
    sliceHistogram () --> sliceHistogramFold ()

let inline map2pairs< ^T, ^S when ^T: comparison and ^T : (static member op_Explicit : ^T -> float) and  ^S : (static member op_Explicit : ^S -> float) > = 
    let map2pairs (map: Map<'T, 'S>): ('T * 'S) list =
        map |> Map.toList
    Stage.liftUnary<Map<^T, ^S>,(^T * ^S) list> "map2pairs" map2pairs id id
let inline pairs2floats< ^T, ^S when ^T : (static member op_Explicit : ^T -> float) and  ^S : (static member op_Explicit : ^S -> float) > = 
    let pairs2floats (pairs: (^T * ^S) list) : (float * float) list =
        pairs |> List.map (fun (k, v) -> (float k, float v)) 
    Stage.liftUnary<(^T * ^S) list,(float*float) list> "pairs2floats" pairs2floats id id
let inline pairs2ints< ^T, ^S when ^T : (static member op_Explicit : ^T -> int) and  ^S : (static member op_Explicit : ^S -> int) > = 
    let pairs2ints (pairs: (^T * ^S) list) : (int * int) list =
        pairs |> List.map (fun (k, v) -> (int k, int v)) 
    Stage.liftUnary<(^T * ^S) list,(int*int) list> "pairs2ints" pairs2ints id id

type ImageStats = ImageFunctions.ImageStats
let sliceComputeStats () =
    Stage.map<Slice.Slice<'T>,ImageStats> "computeStats:map" Slice.computeStats id id

let sliceComputeStatsFold () =
    let zeroStats: ImageStats = { 
        NumPixels = 0u
        Mean = 0.0
        Std = 0.0
        Min = infinity
        Max = -infinity
        Sum = 0.0
        Var = 0.0
    }
    Stage.fold<ImageStats, ImageStats> "computeStats:fold" Slice.addComputeStats zeroStats id id

let computeStats () =
    sliceComputeStats () --> sliceComputeStatsFold ()

////////////////////////////////////////////////
/// Convolution like operators

// Chained type definitions do expose the originals
open type ImageFunctions.OutputRegionMode
open type ImageFunctions.BoundaryCondition

let stackFUnstack f (slices : Slice.Slice<'T> list) =
    slices |> Slice.stack |> f |> Slice.unstack

let skipNTakeM (n: uint) (m: uint) (lst: 'a list) : 'a list =
    let m = uint lst.Length - 2u*n;
    if m = 0u then []
    else lst |> List.skip (int n) |> List.take (int m) 

let stackFUnstackTrim trim f (slices : Slice.Slice<'T> list) =
    let m = uint slices.Length - 2u*trim 
    slices |> Slice.stack |> f |> Slice.unstack |> skipNTakeM trim m

let takeEveryNth (n: uint) = 
    liftWindowed "nth" 1u 0u zeroMaker<float> n 0u 100000u id id

let discreteGaussianOp (name:string) (sigma:float) (outputRegionMode: Slice.OutputRegionMode option) (boundaryCondition: ImageFunctions.BoundaryCondition option) (winSz: uint option): Stage<Slice.Slice<float>, Slice.Slice<float>> =
    let roundFloatToUint v = uint (v+0.5)

    let ksz = 4.0 * sigma + 1.0 |> roundFloatToUint
    let win = Option.defaultValue ksz winSz |> max ksz // max should be found by memory availability
    let stride = win - ksz + 1u
    let pad = 
        match outputRegionMode with
            | Some Valid -> 0u
            | _ -> ksz/2u //floor
    let f slices = slices |> stackFUnstackTrim pad (fun slice3D -> Slice.discreteGaussian 3u sigma (ksz |> Some) outputRegionMode boundaryCondition slice3D)
    liftWindowed name win pad zeroMaker<float> stride pad stride f id id

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

let convolveOp (name: string) (kernel: Slice.Slice<'T>) (outputRegionMode: Slice.OutputRegionMode option) (bc: Slice.BoundaryCondition option) (winSz: uint option): Stage<Slice.Slice<'T>, Slice.Slice<'T>> =
    let windowFromKernel (k: Slice.Slice<'T>) : uint =
        max 1u (k |> Slice.GetDepth)
    let ksz = windowFromKernel kernel
    let win = Option.defaultValue ksz winSz |> min ksz
    let stride = win-ksz+1u
    let pad = 
        match outputRegionMode with
            | Some Valid -> 0u
            | _ -> ksz/2u //floor
    let f slices = slices |> stackFUnstackTrim (ksz - 1u) (fun slice3D -> Slice.convolve outputRegionMode bc slice3D kernel)
    liftWindowed name win pad zeroMaker<'T> stride (win-1u) (1u) f id id

let convolve kernel outputRegionMode boundaryCondition winSz = convolveOp "convolve" kernel outputRegionMode boundaryCondition winSz
let conv kernel = convolveOp "conv" kernel None None None

let finiteDiff (direction: uint) (order: uint) =
    let kernel = Slice.finiteDiffFilter3D direction order
    convolveOp "finiteDiff" kernel None None None

// these only works on uint8
let private makeMorphOp (name:string) (radius:uint) (winSz: uint option) (core: uint -> Slice.Slice<'T> -> Slice.Slice<'T>) : Stage<Slice.Slice<'T>,Slice.Slice<'T>> when 'T: equality =
    let ksz   = 2u * radius + 1u
    let win = Option.defaultValue ksz winSz |> min ksz
    let stride = win - ksz + 1u
    let f slices = slices |> stackFUnstackTrim radius (core radius)
    liftWindowed name win 0u zeroMaker<'T> stride (stride - 1u) stride f id id

let erode radius = makeMorphOp "binaryErode"  radius None Slice.binaryErode
let dilate radius = makeMorphOp "binaryErode"  radius None Slice.binaryDilate
let opening radius = makeMorphOp "binaryErode"  radius None Slice.binaryOpening
let closing radius = makeMorphOp "binaryErode"  radius None Slice.binaryClosing

/// Full stack operators
let binaryFillHoles (winSz: uint)= 
    let f slices = slices |> stackFUnstack Slice.binaryFillHoles
    liftWindowed "fillHoles" winSz 0u zeroMaker<uint8> winSz 0u winSz f id id

let connectedComponents (winSz: uint) = 
    let f slices = slices |> stackFUnstack Slice.connectedComponents
    liftWindowed "connectedComponents" winSz 0u zeroMaker<uint8> winSz 0u winSz f id id

let relabelComponents a (winSz: uint) = 
    let f slices = slices |> stackFUnstack (Slice.relabelComponents a)
    liftWindowed "relabelComponents" winSz 0u zeroMaker<uint64> winSz 0u winSz f id id

let watershed a (winSz: uint) =
    let f slices = slices |> stackFUnstack (Slice.watershed a)
    liftWindowed "watershed" winSz 0u zeroMaker<uint8> winSz 0u winSz f id id

let signedDistanceMap (winSz: uint) =
    let f slices = slices |> stackFUnstack (Slice.signedDistanceMap 0uy 1uy)
    liftWindowed "signedDistanceMap" winSz 0u zeroMaker<uint8> winSz 0u winSz f id id

let otsuThreshold (winSz: uint) =
    let f slices = slices |> stackFUnstack (Slice.otsuThreshold)
    liftWindowed "otsuThreshold" winSz 0u zeroMaker<uint8> winSz 0u winSz f id id

let momentsThreshold (winSz: uint) =
    let f slices = slices |> stackFUnstack (Slice.momentsThreshold)
    liftWindowed "momentsThreshold" winSz 0u zeroMaker<uint8> winSz 0u winSz f id id

let threshold a b = Stage.liftUnary "threshold" (Slice.threshold a b) id id

let addNormalNoise a b = Stage.liftUnary "addNormalNoise" (Slice.addNormalNoise a b) id id

let SliceConstantPad<'T when 'T : equality> (padLower : uint list) (padUpper : uint list) (c : double) =
    Stage.liftUnary "constantPad2D" (Slice.constantPad2D padLower padUpper c) id id

// Not Pipes nor Operators
type FileInfo = Slice.FileInfo
let getStackDepth = Slice.getStackDepth
let getStackHeight = Slice.getStackHeight
let getStackInfo = Slice.getStackInfo
let getStackSize = Slice.getStackSize
let getStackWidth = Slice.getStackWidth

let zero<'T when 'T: equality> 
    (width: uint) 
    (height: uint) 
    (depth: uint) 
    (pl : Pipeline<unit, unit>) 
    : Pipeline<unit, Slice.Slice<'T>> =
    // width, heigth, depth should be replaced with shape and shapeUpdate, and mapper should be deferred to outside Core!!!
    if pl.debug then printfn $"[createAs] {width}x{height}x{depth}"
    let mapper (i: uint) : Slice.Slice<'T> = 
        let slice = Slice.create<'T> width height 1u i
        if pl.debug then printfn "[create] Created slice %A" i
        slice
    let transition = ProfileTransition.create Constant Streaming
    let shapeUpdate = id
    let stage = Stage.init "create" depth mapper transition id id |> Some
    //let flow = Flow.returnM stage
    let nElems = (uint64 width) * (uint64 height)
    let context = id
    Pipeline.create stage pl.memAvail nElems (uint64 depth)  pl.debug

let readFilteredOp<'T when 'T: equality> (name:string) (inputDir : string) (suffix : string) (filter: string[]->string[]) (pl : Pipeline<unit, unit>) : Pipeline<unit, Slice.Slice<'T>> =
    // much should be deferred to outside Core!!!
    if pl.debug then printfn $"[{name}] {inputDir}/*{suffix}"
    let (width,height,depth) = Slice.getStackSize inputDir suffix
    let filenames = Directory.GetFiles(inputDir, "*"+suffix) |> filter
    let depth = uint filenames.Length
    let mapper (i: uint) : Slice.Slice<'T> = 
        let fileName = filenames[int i]; 
        let slice = Slice.readSlice<'T> (uint i) fileName
        if pl.debug then printfn "[%s] Reading slice %A from %s as %s" name i fileName (typeof<'T>.Name)
        slice
    let transition = ProfileTransition.create Constant Streaming
    let stage = Stage.init $"{name}" (uint depth) mapper transition id id |> Some
    //let flow = Flow.returnM stage
    let memPerElem = (uint64 width)*(uint64 height)*getBytesPerComponent<'T>
    let length = (uint64 depth)
    Pipeline.create stage pl.memAvail memPerElem length  pl.debug

let readFiltered<'T when 'T: equality> (inputDir : string) (suffix : string)  (filter: string[]->string[]) (pl : Pipeline<unit, unit>) : Pipeline<unit, Slice.Slice<'T>> =
    readFilteredOp<'T> "filterReadAs" inputDir suffix filter pl

let read<'T when 'T: equality> (inputDir : string) (suffix : string) (pl : Pipeline<unit, unit>) : Pipeline<unit, Slice.Slice<'T>> =
    readFilteredOp<'T> "readAs" inputDir suffix Array.sort pl

let readRandom<'T when 'T: equality> (count: uint) (inputDir : string) (suffix : string) (pl : Pipeline<unit, unit>) : Pipeline<unit, Slice.Slice<'T>> =
    readFilteredOp<'T> "readAs" inputDir suffix (Array.randomChoices (int count)) pl
