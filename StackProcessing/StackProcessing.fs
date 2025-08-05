module StackProcessing

open SlimPipeline // Core processing model
open System.IO

type Stage<'S,'T> = SlimPipeline.Stage<'S,'T>
type Profile = SlimPipeline.Profile
type ProfileTransition = SlimPipeline.ProfileTransition
type Slice<'S when 'S: equality> = Slice.Slice<'S>

let (-->) = Stage.(-->)
let source = Pipeline.source (fun _ -> 0UL)
let debug = Pipeline.debug (fun _ -> 0UL)
let sink (pl: Pipeline<unit,unit>) : unit = Pipeline.sink pl
let sinkList (plLst: Pipeline<unit,unit> list) : unit = Pipeline.sinkList plLst
let (>=>) = Pipeline.(>=>)
let (>=>>) = Pipeline.(>=>>)
let (>>=>) = Pipeline.(>>=>)
//let combineIgnore = Pipeline.combineIgnore
let drainSingle pl = Pipeline.drainSingle "drainSingle" pl
let drainList pl = Pipeline.drainList "drainList" pl
let drainLast pl = Pipeline.drainLast "drainLast" pl
let tap = Stage.tap
let tapIt = Stage.tapIt
let ignoreAll = Stage.ignore<_>
let liftUnary (f: Slice.Slice<'S> -> Slice.Slice<'T>) = 
    Stage.liftUnary<Slice.Slice<'S>,Slice.Slice<'T>> "liftUnary" f
let zeroMaker<'S when 'S: equality> (ex:Slice.Slice<'S>) : Slice.Slice<'S> = Slice.create<'S> (Slice.GetWidth ex) (Slice.GetHeight ex) 1u 0u
let liftWindowed (name: string) (updateId: uint->Slice.Slice<'S>->Slice.Slice<'S>) (window: uint) (pad: uint) (zeroMaker: Slice.Slice<'S>->Slice.Slice<'S>) (stride: uint) (emitStart: uint) (emitCount: uint) (f: Slice.Slice<'S> list -> Slice.Slice<'T> list) (shapeUpdate: uint64 -> uint64) : Stage<Slice.Slice<'S>, Slice.Slice<'T>> =
    Stage.liftWindowed<Slice.Slice<'S>,Slice.Slice<'T>> name updateId window pad zeroMaker stride emitStart emitCount f shapeUpdate
let getBytesPerComponent<'T> = (typeof<'T> |> Image.getBytesPerComponent |> uint64)

let write (outputDir: string) (suffix: string) : Stage<Slice.Slice<'T>, unit> =
    if not (Directory.Exists(outputDir)) then
        Directory.CreateDirectory(outputDir) |> ignore
    let consumer (slice:Slice.Slice<'T>) =
        let fileName = Path.Combine(outputDir, sprintf "slice_%03d%s" slice.Index suffix)
        slice.Image.toFile(fileName)
        printfn "[Write] Saved slice %d to %s as %s" slice.Index fileName (typeof<'T>.Name) 
    Stage.consumeWith $"write:{outputDir}" consumer 

let show (plt: Slice.Slice<'T> -> unit) : Stage<Slice.Slice<'T>, unit> =
    let consumer (slice:Slice.Slice<'T>) =
        let width = slice |> Slice.GetWidth |> int
        let height = slice |> Slice.GetHeight |> int
        plt slice
        printfn "[Show] Showing slice %d" slice.Index
    Stage.consumeWith "show" consumer 

let plot (plt: (float list)->(float list)->unit) : Stage<(float * float) list, unit> = // better be (float*float) list
    let consumer (points: (float*float) list) =
        let x,y = points |> List.unzip
        plt x y
        printfn "[Plot] Plotting {points.Length} 2D points"
    Stage.consumeWith "plot" consumer 

let print () : Stage<'T, unit> =
    let consumer (elm: 'T) =
        printfn "%A" elm
        printfn "[Plot] Plotting {points.Length} 2D points"
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
    Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>> "add" (Slice.add slice) (memNeeded<'T> 2UL) id
let inline scalarAddSlice<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) = 
    Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>> "scalarAddSlice" (fun (s:Slice.Slice<^T>)->Slice.scalarAddSlice<^T> i s) id
let inline sliceAddScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>> "sliceAddScalar" (fun (s:Slice.Slice<^T>)->Slice.sliceAddScalar<^T> s i) id

let sub slice = 
    Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>> "sub" (Slice.sub slice) id
let inline scalarSubSlice<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>> "scalarSubSlice" (fun (s:Slice.Slice<^T>)->Slice.scalarSubSlice<^T> i s) id
let inline sliceSubScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>> "sliceSubScalar" (fun (s:Slice.Slice<^T>)->Slice.sliceSubScalar<^T> s i) id

let mul slice = Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>> "mul" (Slice.mul slice) id
let inline scalarMulSlice<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>> "scalarMulSlice" (fun (s:Slice.Slice<^T>)->Slice.scalarMulSlice<^T> i s) id
let inline sliceMulScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>> "sliceMulScalar" (fun (s:Slice.Slice<^T>)->Slice.sliceMulScalar<^T> s i) id

let div slice = Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>> "div" (Slice.div slice) id
let inline scalarDivSlice<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>> "scalarDivSlice" (fun (s:Slice.Slice<^T>)->Slice.scalarDivSlice<^T> i s) id
let inline sliceDivScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>> "sliceDivScalar" (fun (s:Slice.Slice<^T>)->Slice.sliceDivScalar<^T> s i) id

/// Simple functions
let abs<'T when 'T: equality> : Stage<Slice.Slice<'T>,Slice.Slice<'T>> =      Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>> "abs"    Slice.absSlice id
let absFloat      : Stage<Slice.Slice<float>,Slice.Slice<float>> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>> "abs"    Slice.absSlice id
let absFloat32    : Stage<Slice.Slice<float32>,Slice.Slice<float32>> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>> "abs"    Slice.absSlice id
let absInt        : Stage<Slice.Slice<int>,Slice.Slice<int>> =          Stage.liftUnary<Slice.Slice<int>,Slice.Slice<int>> "abs"    Slice.absSlice id
let acosFloat     : Stage<Slice.Slice<float>,Slice.Slice<float>> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>> "acos"   Slice.acosSlice id
let acosFloat32   : Stage<Slice.Slice<float32>,Slice.Slice<float32>> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>> "acos"   Slice.acosSlice id
let asinFloat     : Stage<Slice.Slice<float>,Slice.Slice<float>> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>> "asin"   Slice.asinSlice id
let asinFloat32   : Stage<Slice.Slice<float32>,Slice.Slice<float32>> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>> "asin"   Slice.asinSlice id
let atanFloat     : Stage<Slice.Slice<float>,Slice.Slice<float>> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>> "atan"   Slice.atanSlice id
let atanFloat32   : Stage<Slice.Slice<float32>,Slice.Slice<float32>> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>> "atan"   Slice.atanSlice id
let cosFloat      : Stage<Slice.Slice<float>,Slice.Slice<float>> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>> "cos"    Slice.cosSlice id
let cosFloat32    : Stage<Slice.Slice<float32>,Slice.Slice<float32>> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>> "cos"    Slice.cosSlice id
let sinFloat      : Stage<Slice.Slice<float>,Slice.Slice<float>> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>> "sin"    Slice.sinSlice id
let sinFloat32    : Stage<Slice.Slice<float32>,Slice.Slice<float32>> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>> "sin"    Slice.sinSlice id
let tanFloat      : Stage<Slice.Slice<float>,Slice.Slice<float>> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>> "tan"    Slice.tanSlice id
let tanFloat32    : Stage<Slice.Slice<float32>,Slice.Slice<float32>> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>> "tan"    Slice.tanSlice id
let expFloat      : Stage<Slice.Slice<float>,Slice.Slice<float>> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>> "exp"    Slice.expSlice id
let expFloat32    : Stage<Slice.Slice<float32>,Slice.Slice<float32>> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>> "exp"    Slice.expSlice id
let log10Float    : Stage<Slice.Slice<float>,Slice.Slice<float>> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>> "log10"  Slice.log10Slice id
let log10Float32  : Stage<Slice.Slice<float32>,Slice.Slice<float32>> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>> "log10"  Slice.log10Slice id
let logFloat      : Stage<Slice.Slice<float>,Slice.Slice<float>> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>> "log"    Slice.logSlice id
let logFloat32    : Stage<Slice.Slice<float32>,Slice.Slice<float32>> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>> "log"    Slice.logSlice id
let roundFloat    : Stage<Slice.Slice<float>,Slice.Slice<float>> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>> "round"  Slice.roundSlice id
let roundFloat32  : Stage<Slice.Slice<float32>,Slice.Slice<float32>> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>> "round"  Slice.roundSlice id
let sqrtFloat     : Stage<Slice.Slice<float>,Slice.Slice<float>> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>> "sqrt"   Slice.sqrtSlice id
let sqrtFloat32   : Stage<Slice.Slice<float32>,Slice.Slice<float32>> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>> "sqrt"   Slice.sqrtSlice id
let sqrtInt       : Stage<Slice.Slice<int>,Slice.Slice<int>> =          Stage.liftUnary<Slice.Slice<int>,Slice.Slice<int>> "sqrt"   Slice.sqrtSlice id
let squareFloat   : Stage<Slice.Slice<float>,Slice.Slice<float>> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>> "square" Slice.squareSlice id
let squareFloat32 : Stage<Slice.Slice<float32>,Slice.Slice<float32>> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>> "square" Slice.squareSlice id
let squareInt     : Stage<Slice.Slice<int>,Slice.Slice<int>> =          Stage.liftUnary<Slice.Slice<int>,Slice.Slice<int>> "square" Slice.squareSlice id

//let histogram<'T when 'T: comparison> = histogramOp<'T> "histogram"
let sliceHistogram () =
    Stage.map<Slice.Slice<'T>,Map<'T,uint64>> "histogram:map" Slice.histogram (fun s -> 256UL) // Assumed max for uint8, can be done better

let sliceHistogramFold () =
    Stage.fold<Map<'T,uint64>, Map<'T,uint64>> "histogram:fold" Slice.addHistogram (Map.empty<'T, uint64>) id

let histogram () =
    sliceHistogram () --> sliceHistogramFold ()

let inline map2pairs< ^T, ^S when ^T: comparison and ^T : (static member op_Explicit : ^T -> float) and  ^S : (static member op_Explicit : ^S -> float) > = 
    let map2pairs (map: Map<'T, 'S>): ('T * 'S) list =
        map |> Map.toList
    Stage.liftUnary<Map<^T, ^S>,(^T * ^S) list> "map2pairs" map2pairs id
let inline pairs2floats< ^T, ^S when ^T : (static member op_Explicit : ^T -> float) and  ^S : (static member op_Explicit : ^S -> float) > = 
    let pairs2floats (pairs: (^T * ^S) list) : (float * float) list =
        pairs |> List.map (fun (k, v) -> (float k, float v)) 
    Stage.liftUnary<(^T * ^S) list,(float*float) list> "pairs2floats" pairs2floats id
let inline pairs2ints< ^T, ^S when ^T : (static member op_Explicit : ^T -> int) and  ^S : (static member op_Explicit : ^S -> int) > = 
    let pairs2ints (pairs: (^T * ^S) list) : (int * int) list =
        pairs |> List.map (fun (k, v) -> (int k, int v)) 
    Stage.liftUnary<(^T * ^S) list,(int*int) list> "pairs2ints" pairs2ints id

type ImageStats = ImageFunctions.ImageStats
let sliceComputeStats () =
    Stage.map<Slice.Slice<'T>,ImageStats> "computeStats:map" Slice.computeStats id

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
    Stage.fold<ImageStats, ImageStats> "computeStats:fold" Slice.addComputeStats zeroStats id

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
    liftWindowed "nth" Slice.updateId 1u 0u zeroMaker<float> n 0u 100000u id id

let discreteGaussianOp (name:string) (sigma:float) (outputRegionMode: Slice.OutputRegionMode option) (boundaryCondition: ImageFunctions.BoundaryCondition option) (winSz: uint option): Stage<Slice.Slice<float>, Slice.Slice<float>> =
    let roundFloatToUint v = uint (v+0.5)

    let ksz = 2.0 * sigma + 1.0 |> roundFloatToUint
    let win = Option.defaultValue ksz winSz |> min ksz // max should be found by memory availability
    let stride = win - ksz + 1u
    let pad = 
        match outputRegionMode with
            | Some Valid -> 0u
            | _ -> ksz/2u //floor
    let f slices = slices |> stackFUnstack (fun slice3D -> Slice.discreteGaussian 3u sigma (ksz |> Some) outputRegionMode boundaryCondition slice3D)
    liftWindowed name Slice.updateId win pad zeroMaker<float> stride (stride - 1u) stride f id

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
    let f slices = slices |> stackFUnstack (fun slice3D -> Slice.convolve outputRegionMode bc slice3D kernel)
    liftWindowed name Slice.updateId win pad zeroMaker<'T> stride (stride - 1u) stride f id

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
    liftWindowed name Slice.updateId win 0u zeroMaker<'T> stride (stride - 1u) stride f id

let erode radius = makeMorphOp "binaryErode"  radius None Slice.binaryErode
let dilate radius = makeMorphOp "binaryErode"  radius None Slice.binaryDilate
let opening radius = makeMorphOp "binaryErode"  radius None Slice.binaryOpening
let closing radius = makeMorphOp "binaryErode"  radius None Slice.binaryClosing

/// Full stack operators
let binaryFillHoles (winSz: uint)= 
    let f slices = slices |> stackFUnstack Slice.binaryFillHoles
    liftWindowed "fillHoles" Slice.updateId winSz 0u zeroMaker<uint8> winSz 0u winSz f

let connectedComponents (winSz: uint) = 
    let f slices = slices |> stackFUnstack Slice.connectedComponents
    liftWindowed "fillHoles" Slice.updateId winSz 0u zeroMaker<uint8> winSz 0u winSz f id

let relabelComponents a (winSz: uint) = 
    let f slices = slices |> stackFUnstack (Slice.relabelComponents a)
    liftWindowed "relabelComponents" Slice.updateId winSz 0u zeroMaker<uint64> winSz 0u winSz f id

let watershed a (winSz: uint) =
    let f slices = slices |> stackFUnstack (Slice.watershed a)
    liftWindowed "watershed" Slice.updateId winSz 0u zeroMaker<uint8> winSz 0u winSz f id

let signedDistanceMap (winSz: uint) =
    let f slices = slices |> stackFUnstack (Slice.signedDistanceMap 0uy 1uy)
    liftWindowed "signedDistanceMap" Slice.updateId winSz 0u zeroMaker<uint8> winSz 0u winSz f id

let otsuThreshold (winSz: uint) =
    let f slices = slices |> stackFUnstack (Slice.otsuThreshold)
    liftWindowed "otsuThreshold" Slice.updateId winSz 0u zeroMaker<uint8> winSz 0u winSz f id

let momentsThreshold (winSz: uint) =
    let f slices = slices |> stackFUnstack (Slice.momentsThreshold)
    liftWindowed "momentsThreshold" Slice.updateId winSz 0u zeroMaker<uint8> winSz 0u winSz f id

let threshold a b = Stage.liftUnary "threshold" (Slice.threshold a b) id

let addNormalNoise a b = Stage.liftUnary "addNormalNoise" (Slice.addNormalNoise a b) id

let SliceConstantPad<'T when 'T : equality> (padLower : uint list) (padUpper : uint list) (c : double) =
    Stage.liftUnary "constantPad2D" (Slice.constantPad2D padLower padUpper c) id

// Not Pipes nor Operators
type FileInfo = Slice.FileInfo
let getStackDepth = Slice.getStackDepth
let getStackHeight = Slice.getStackHeight
let getStackInfo = Slice.getStackInfo
let getStackSize = Slice.getStackSize
let getStackWidth = Slice.getStackWidth

let createAs<'T when 'T: equality> (width: uint) (height: uint) (depth: uint) (pl : Pipeline<unit, unit>) : Pipeline<unit, Slice.Slice<'T>> =
    // width, heigth, depth should be replaced with shape and shapeUpdate, and mapper should be deferred to outside Core!!!
    if pl.debug then printfn $"[createAs] {width}x{height}x{depth}"
    let mapper (i: uint) : Slice.Slice<'T> = 
        let slice = Slice.create<'T> width height 1u i
        if pl.debug then printfn "[create] Created slice %A" i
        slice
    let transition = ProfileTransition.create Constant Streaming
    let shapeUpdate = id
    let stage = Stage.init "create" depth mapper transition shapeUpdate 
    let flow = Flow.returnM stage
    let shape = (uint64 width) * (uint64 height)
    let context = id
    Pipeline.create flow pl.memAvail shape (uint64 depth) context pl.debug

let readAs<'T when 'T: equality> (inputDir : string) (suffix : string) (pl : Pipeline<unit, unit>) : Pipeline<unit, Slice.Slice<'T>> =
    // much should be deferred to outside Core!!!
    if pl.debug then printfn $"[readAs] {inputDir}/*{suffix}"
    let (width,height,depth) = Slice.getStackSize inputDir suffix
    let filenames = Directory.GetFiles(inputDir, "*"+suffix) |> Array.sort
    let depth = uint filenames.Length
    let mapper (i: uint) : Slice.Slice<'T> = 
        let fileName = filenames[int i]; 
        let slice = Slice.readSlice<'T> (uint i) fileName
        printfn "[readAs] Reading slice %A from %s as %s" i fileName (typeof<'T>.Name)
        slice
    let transition = ProfileTransition.create Constant Streaming
    let shapeUpdate = id
    let stage = Stage.init $"read: {inputDir}" (uint depth) mapper transition shapeUpdate 
    let flow = Flow.returnM stage
    let memPerElem = (uint64 width)*(uint64 height)*(typeof<'T> |> getBytesPerComponent |> uint64)
    let length = (uint64 depth)
    Pipeline.create flow pl.memAvail memPerElem length context pl.debug

let readRandomAs<'T when 'T: equality> (count: uint) (inputDir : string) (suffix : string) (pl : Pipeline<unit, unit>) : Pipeline<unit, Slice.Slice<'T>> =
    if pl.debug then printfn $"[readRandomAs] {count} slices from {inputDir}/*{suffix}"
    let (width,height,depth) = Slice.getStackSize inputDir suffix
    let filenames = Directory.GetFiles(inputDir, "*"+suffix) |> Array.randomChoices (int count)
    let depth = filenames.Length
    let mapper (i: uint) : Slice.Slice<'T> = 
        let fileName = filenames[int i]; 
        let slice = Slice.readSlice<'T> (uint i) fileName
        if pl.debug then printfn "[readRandomSlices] Reading slice %A from %s" i fileName
        slice
    let transition = ProfileTransition.create Constant Streaming
    let shapeUpdate = id
    let stage = Stage.init $"read: {inputDir}" (uint depth) mapper transition shapeUpdate 
    let flow = Flow.returnM stage
    let shape = (uint64 width)*(uint64 height)
    let context = id
    Pipeline.create flow pl.memAvail shape (uint64 count) context pl.debug
