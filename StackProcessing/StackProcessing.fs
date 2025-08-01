module StackProcessing

open SlimPipeline // Core processing model
open Routing // Combinators and routing logic
open Processing // Common image operators
//open Slice // Image and slice types
open System.IO

type Stage<'S,'T,'Shape> = SlimPipeline.Stage<'S,'T,'Shape>
type MemoryProfile = SlimPipeline.MemoryProfile
type MemoryTransition = SlimPipeline.MemoryTransition
type Slice<'S when 'S: equality> = Slice.Slice<'S>

type Shape = Zero | Slice of uint*uint
let shapeContext = {
    memPerElement = fun sh -> match sh with Zero -> 0UL | Slice (width,height) -> (uint64 width)*(uint64 height); 
    depth = fun _ -> 0u}

let idOp = Stage.id
let (-->) = Stage.(-->)
let source = Pipeline.source<Shape> shapeContext
let debug = Pipeline.debug<Shape> shapeContext
let sink (pl: Pipeline<unit,unit,Shape>) : unit = Pipeline.sink pl
let sinkList (plLst: Pipeline<unit,unit,Shape> list) : unit = Pipeline.sinkList plLst
let (>=>) = Pipeline.(>=>)
let (>=>>) = Routing.(>=>>)
let (>>=>) = Routing.(>>=>)
//let combineIgnore = Routing.combineIgnore
let drainSingle pl = Routing.drainSingle "drainSingle" pl
let drainList pl = Routing.drainList "drainList" pl
let drainLast pl = Routing.drainLast "drainLast" pl
let tap = Stage.tap
let tapIt = Stage.tapIt
let ignoreAll = Stage.ignore<_,Shape>
let liftUnary (f: Slice.Slice<'S> -> Slice.Slice<'T>) = Stage.liftUnary<Slice.Slice<'S>,Slice.Slice<'T>,Shape> "liftUnary" f

let write (outputDir: string) (suffix: string) : Stage<Slice.Slice<'T>, unit, Shape> =
    if not (Directory.Exists(outputDir)) then
        Directory.CreateDirectory(outputDir) |> ignore
    let consumer (slice:Slice.Slice<'T>) =
        let fileName = Path.Combine(outputDir, sprintf "slice_%03d%s" slice.Index suffix)
        slice.Image.toFile(fileName)
        printfn "[Write] Saved slice %d to %s as %s" slice.Index fileName (typeof<'T>.Name) 
    Stage.consumeWith $"write:{outputDir}" consumer 

let show (plt: Slice.Slice<'T> -> unit) : Stage<Slice.Slice<'T>, unit, Shape> =
    let consumer (slice:Slice.Slice<'T>) =
        let width = slice |> Slice.GetWidth |> int
        let height = slice |> Slice.GetHeight |> int
        plt slice
        printfn "[Show] Showing slice %d" slice.Index
    Stage.consumeWith "show" consumer 

let plot (plt: (float list)->(float list)->unit) : Stage<(float * float) list, unit, Shape> = // better be (float*float) list
    let consumer (points: (float*float) list) =
        let x,y = points |> List.unzip
        plt x y
        printfn "[Plot] Plotting {points.Length} 2D points"
    Stage.consumeWith "plot" consumer 

let print () : Stage<'T, unit, Shape> =
    let consumer (elm: 'T) =
        printfn "%A" elm
        printfn "[Plot] Plotting {points.Length} 2D points"
    Stage.consumeWith "print" consumer 

(*
let write = Processing.writeOp
let print = Processing.printOp
let plot = Processing.plotOp
let show = Processing.showOp
*)

let finiteDiffFilter3D 
    (direction: uint) 
    (order: uint)
    (pl : Pipeline<unit, unit, Shape>) 
    : Pipeline<unit, Slice.Slice<float>, Shape> =
    let img = Slice.finiteDiffFilter3D direction order
    let sz = Slice.GetSize img
    let shapeUpdate = fun (s: Shape) -> s
    let op : Stage<unit, Slice.Slice<float>, Shape> =
        {
            Name = "gaussSource"
            Pipe = img |> liftImageSource "gaussSource"
            Transition = MemoryTransition.create Constant Streaming
            ShapeUpdate = shapeUpdate
        }
    let width, height, depth = sz[0], sz[1], sz[2]
    let context = ShapeContext.create (fun _ -> width*height |> uint64) (fun _ -> depth)
    {
        flow = MemFlow.returnM op
        mem = pl.mem
        shape = Slice (width,height) |> Some
        context = context
        debug = pl.debug
    }

let axisSource
    (axis: int) 
    (size: int list)
    (pl : Pipeline<unit, unit, Shape>) 
    : Pipeline<unit, Slice.Slice<uint>, Shape> =
    let img = Slice.generateCoordinateAxis axis size
    let sz = Slice.GetSize img
    let shapeUpdate = fun (s: Shape) -> s
    let op : Stage<unit, Slice.Slice<uint>, Shape> =
        {
            Name = "axisSource"
            Pipe = img |> liftImageSource "axisSource"
            Transition = MemoryTransition.create Constant Streaming
            ShapeUpdate = shapeUpdate
        }
    let width, height, depth = sz[0], sz[1], sz[2]
    let context = ShapeContext.create (fun _ -> width*height |> uint64) (fun _ -> depth)
    {
        flow = MemFlow.returnM op
        mem = pl.mem
        shape = Slice (width,height) |> Some
        context = context
        debug = pl.debug
    }

/// Pixel type casting
let cast<'S,'T when 'S: equality and 'T: equality> = Stage.cast<Slice.Slice<'S>,Slice.Slice<'T>,Shape> (sprintf "cast(%s->%s)" typeof<'S>.Name typeof<'T>.Name) Slice.cast<'S,'T>

/// Basic arithmetic
let add slice = 
    Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>,Shape> "add" (Slice.add slice)
let inline scalarAddSlice<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) = 
    Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>,Shape> "scalarAddSlice" (fun (s:Slice.Slice<^T>)->Slice.scalarAddSlice<^T> i s)
let inline sliceAddScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>,Shape> "sliceAddScalar" (fun (s:Slice.Slice<^T>)->Slice.sliceAddScalar<^T> s i)

let sub slice = 
    Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>,Shape> "sub" (Slice.sub slice)
let inline scalarSubSlice<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>,Shape> "scalarSubSlice" (fun (s:Slice.Slice<^T>)->Slice.scalarSubSlice<^T> i s)
let inline sliceSubScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>,Shape> "sliceSubScalar" (fun (s:Slice.Slice<^T>)->Slice.sliceSubScalar<^T> s i)

let mul slice = Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>,Shape> "mul" (Slice.mul slice)
let inline scalarMulSlice<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>,Shape> "scalarMulSlice" (fun (s:Slice.Slice<^T>)->Slice.scalarMulSlice<^T> i s)
let inline sliceMulScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>,Shape> "sliceMulScalar" (fun (s:Slice.Slice<^T>)->Slice.sliceMulScalar<^T> s i)

let div slice = Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>,Shape> "div" (Slice.div slice)
let inline scalarDivSlice<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>,Shape> "scalarDivSlice" (fun (s:Slice.Slice<^T>)->Slice.scalarDivSlice<^T> i s)
let inline sliceDivScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>,Shape> "sliceDivScalar" (fun (s:Slice.Slice<^T>)->Slice.sliceDivScalar<^T> s i)

/// Simple functions
let abs<'T when 'T: equality>      : Stage<Slice.Slice<'T>,Slice.Slice<'T>, Shape> =      Stage.liftUnary<Slice.Slice<'T>,Slice.Slice<'T>,Shape> "abs"    Slice.absSlice
let absFloat      : Stage<Slice.Slice<float>,Slice.Slice<float>, Shape> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>,Shape> "abs"    Slice.absSlice
let absFloat32    : Stage<Slice.Slice<float32>,Slice.Slice<float32>, Shape> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>,Shape> "abs"    Slice.absSlice
let absInt        : Stage<Slice.Slice<int>,Slice.Slice<int>, Shape> =          Stage.liftUnary<Slice.Slice<int>,Slice.Slice<int>,Shape> "abs"    Slice.absSlice
let acosFloat     : Stage<Slice.Slice<float>,Slice.Slice<float>, Shape> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>,Shape> "acos"   Slice.acosSlice
let acosFloat32   : Stage<Slice.Slice<float32>,Slice.Slice<float32>, Shape> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>,Shape> "acos"   Slice.acosSlice
let asinFloat     : Stage<Slice.Slice<float>,Slice.Slice<float>, Shape> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>,Shape> "asin"   Slice.asinSlice
let asinFloat32   : Stage<Slice.Slice<float32>,Slice.Slice<float32>, Shape> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>,Shape> "asin"   Slice.asinSlice
let atanFloat     : Stage<Slice.Slice<float>,Slice.Slice<float>, Shape> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>,Shape> "atan"   Slice.atanSlice
let atanFloat32   : Stage<Slice.Slice<float32>,Slice.Slice<float32>, Shape> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>,Shape> "atan"   Slice.atanSlice
let cosFloat      : Stage<Slice.Slice<float>,Slice.Slice<float>, Shape> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>,Shape> "cos"    Slice.cosSlice
let cosFloat32    : Stage<Slice.Slice<float32>,Slice.Slice<float32>, Shape> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>,Shape> "cos"    Slice.cosSlice
let sinFloat      : Stage<Slice.Slice<float>,Slice.Slice<float>, Shape> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>,Shape> "sin"    Slice.sinSlice
let sinFloat32    : Stage<Slice.Slice<float32>,Slice.Slice<float32>, Shape> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>,Shape> "sin"    Slice.sinSlice
let tanFloat      : Stage<Slice.Slice<float>,Slice.Slice<float>, Shape> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>,Shape> "tan"    Slice.tanSlice
let tanFloat32    : Stage<Slice.Slice<float32>,Slice.Slice<float32>, Shape> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>,Shape> "tan"    Slice.tanSlice
let expFloat      : Stage<Slice.Slice<float>,Slice.Slice<float>, Shape> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>,Shape> "exp"    Slice.expSlice
let expFloat32    : Stage<Slice.Slice<float32>,Slice.Slice<float32>, Shape> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>,Shape> "exp"    Slice.expSlice
let log10Float    : Stage<Slice.Slice<float>,Slice.Slice<float>, Shape> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>,Shape> "log10"  Slice.log10Slice
let log10Float32  : Stage<Slice.Slice<float32>,Slice.Slice<float32>, Shape> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>,Shape> "log10"  Slice.log10Slice
let logFloat      : Stage<Slice.Slice<float>,Slice.Slice<float>, Shape> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>,Shape> "log"    Slice.logSlice
let logFloat32    : Stage<Slice.Slice<float32>,Slice.Slice<float32>, Shape> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>,Shape> "log"    Slice.logSlice
let roundFloat    : Stage<Slice.Slice<float>,Slice.Slice<float>, Shape> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>,Shape> "round"  Slice.roundSlice
let roundFloat32  : Stage<Slice.Slice<float32>,Slice.Slice<float32>, Shape> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>,Shape> "round"  Slice.roundSlice
let sqrtFloat     : Stage<Slice.Slice<float>,Slice.Slice<float>, Shape> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>,Shape> "sqrt"   Slice.sqrtSlice
let sqrtFloat32   : Stage<Slice.Slice<float32>,Slice.Slice<float32>, Shape> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>,Shape> "sqrt"   Slice.sqrtSlice
let sqrtInt       : Stage<Slice.Slice<int>,Slice.Slice<int>, Shape> =          Stage.liftUnary<Slice.Slice<int>,Slice.Slice<int>,Shape> "sqrt"   Slice.sqrtSlice
let squareFloat   : Stage<Slice.Slice<float>,Slice.Slice<float>, Shape> =      Stage.liftUnary<Slice.Slice<float>,Slice.Slice<float>,Shape> "square" Slice.squareSlice
let squareFloat32 : Stage<Slice.Slice<float32>,Slice.Slice<float32>, Shape> =  Stage.liftUnary<Slice.Slice<float32>,Slice.Slice<float32>,Shape> "square" Slice.squareSlice
let squareInt     : Stage<Slice.Slice<int>,Slice.Slice<int>, Shape> =          Stage.liftUnary<Slice.Slice<int>,Slice.Slice<int>,Shape> "square" Slice.squareSlice

//let histogram<'T when 'T: comparison> = histogramOp<'T,Shape> "histogram"
let sliceHistogram () =
    Stage.map<Slice.Slice<'T>,Map<'T,uint64>, Shape> "histogram:map" Slice.histogram

let sliceHistogramFold () =
    Stage.fold<Map<'T,uint64>, Map<'T,uint64>, Shape> "histogram:fold" Slice.addHistogram (Map.empty<'T, uint64>)

let histogram () =
    sliceHistogram () --> sliceHistogramFold ()

let inline map2pairs< ^T, ^S when ^T: comparison and ^T : (static member op_Explicit : ^T -> float) and  ^S : (static member op_Explicit : ^S -> float) > = 
    let map2pairs (map: Map<'T, 'S>): ('T * 'S) list =
        map |> Map.toList
    Stage.liftUnary<Map<^T, ^S>,(^T * ^S) list,Shape> "map2pairs" map2pairs
let inline pairs2floats< ^T, ^S when ^T : (static member op_Explicit : ^T -> float) and  ^S : (static member op_Explicit : ^S -> float) > = 
    let pairs2floats (pairs: (^T * ^S) list) : (float * float) list =
        pairs |> List.map (fun (k, v) -> (float k, float v)) 
    Stage.liftUnary<(^T * ^S) list,(float*float) list,Shape> "pairs2floats" pairs2floats
let inline pairs2ints< ^T, ^S when ^T : (static member op_Explicit : ^T -> int) and  ^S : (static member op_Explicit : ^S -> int) > = 
    let pairs2ints (pairs: (^T * ^S) list) : (int * int) list =
        pairs |> List.map (fun (k, v) -> (int k, int v)) 
    Stage.liftUnary<(^T * ^S) list,(int*int) list,Shape> "pairs2ints" pairs2ints

type ImageStats = ImageFunctions.ImageStats
let computeStats<'T when 'T: equality and 'T: comparison> = computeStatsOp<'T,Shape> "computeStats"

/// Convolution like operators
// Chained type definitions do expose the originals
let zeroPad = Processing.zeroPad
let periodicPad = Processing.periodicPad
let zeroFluxNeumannPad = Processing.zeroFluxNeumannPad
let valid = Processing.valid
let same = Processing.same

let discreteGaussian = discreteGaussianOp<Shape> "discreteGaussian"
let convGauss = convGaussOp<Shape> "convGauss"

let convolve kernel outputRegionMode boundaryCondition winSz = convolveOp "convolve" kernel outputRegionMode boundaryCondition winSz
let conv kernel = convOp "conv" kernel

// these only works on uint8
let erode            r       = binaryErodeOp   "binaryErode"   r None
let dilate           r       = binaryDilateOp  "binaryDilate"  r None
let opening          r       = binaryOpeningOp "binaryOpening" r None
let closing          r       = binaryClosingOp "binaryClosing" r None

/// Full stack operators
let binaryFillHoles = binaryFillHolesOp<Shape> "fillHoles"
let connectedComponents = connectedComponentsOp<Shape> "components"
let piecewiseConnectedComponents wz = piecewiseConnectedComponentsOp "piecewiseConnectedComponents" wz

// Annoying F# value restriction requires explicit types here, sigh
let otsuThreshold<'T when 'T: equality> = (otsuThresholdOp "otsuThreshold")
let otsuMultiThreshold n = otsuMultiThresholdOp "otsuMultiThreshold" n
let momentsThreshold<'T when 'T: equality> = momentsThresholdOp "momentsThreshold"
let signedDistanceMap = signedDistanceMapOp<Shape> "signedDistanceMap"
let watershed a = watershedOp "watershed" a
let threshold a b = thresholdOp "threshold" a b
let addNormalNoise a b = addNormalNoiseOp "addNormalNoise" a b
let relabelComponents a = relabelComponentsOp "relabelComponents" a

let constantPad2D<'T when 'T : equality> padLower padUpper c = constantPad2DOp "constantPad2D" padLower padUpper c

// Not Pipes nor Operators
type FileInfo = Slice.FileInfo
let getStackDepth = Slice.getStackDepth
let getStackHeight = Slice.getStackHeight
let getStackInfo = Slice.getStackInfo
let getStackSize = Slice.getStackSize
let getStackWidth = Slice.getStackWidth

let createAs<'T when 'T: equality> (width: uint) (height: uint) (depth: uint) (pl : Pipeline<unit, unit, Shape>) : Pipeline<unit, Slice.Slice<'T>,Shape> =
    // width, heigth, depth should be replaced with shape and shapeUpdate, and mapper should be deferred to outside Core!!!
    if pl.debug then printfn $"[createAs] {width}x{height}x{depth}"
    let mapper (i: uint) : Slice.Slice<'T> = 
        let slice = Slice.create<'T> width height 1u i
        if pl.debug then printfn "[create] Created slice %A" i
        slice
    let transition = MemoryTransition.create Constant Streaming
    let shapeUpdate = id
    let stage = Stage.init "create" depth mapper transition shapeUpdate 
    let flow = MemFlow.returnM stage
    let shape = Slice (width,height)
    let context = ShapeContext.create (fun _ -> width*height |> uint64) (fun _ -> depth)
    Pipeline.create flow pl.mem (Some shape) context pl.debug

let readAs<'T when 'T: equality> (inputDir : string) (suffix : string) (pl : Pipeline<unit, unit, Shape>) : Pipeline<unit, Slice.Slice<'T>,Shape> =
    // much should be deferred to outside Core!!!
    if pl.debug then printfn $"[readAs] {inputDir}/*{suffix}"
    let (width,height,depth) = Slice.getStackSize inputDir suffix
    let filenames = Directory.GetFiles(inputDir, "*"+suffix) |> Array.sort
    let depth = filenames.Length
    let mapper (i: uint) : Slice.Slice<'T> = 
        let fileName = filenames[int i]; 
        let slice = Slice.readSlice<'T> (uint i) fileName
        printfn "[readAs] Reading slice %A from %s as %s" i fileName (typeof<'T>.Name)
        slice
    let transition = MemoryTransition.create Constant Streaming
    let shapeUpdate = id
    let stage = Stage.init $"read: {inputDir}" (uint depth) mapper transition shapeUpdate 
    let flow = MemFlow.returnM stage
    let shape = Slice (width,height)
    let context = ShapeContext.create (fun _ -> width*height |> uint64) (fun _ -> uint depth)
    Pipeline.create flow pl.mem (Some shape) context pl.debug

let readRandomAs<'T when 'T: equality> (count: uint) (inputDir : string) (suffix : string) (pl : Pipeline<unit, unit, Shape>) : Pipeline<unit, Slice.Slice<'T>,Shape> =
    if pl.debug then printfn $"[readRandomAs] {count} slices from {inputDir}/*{suffix}"
    let (width,height,depth) = Slice.getStackSize inputDir suffix
    let filenames = Directory.GetFiles(inputDir, "*"+suffix) |> Array.randomChoices (int count)
    let depth = filenames.Length
    let mapper (i: uint) : Slice.Slice<'T> = 
        let fileName = filenames[int i]; 
        let slice = Slice.readSlice<'T> (uint i) fileName
        if pl.debug then printfn "[readRandomSlices] Reading slice %A from %s" i fileName
        slice
    let transition = MemoryTransition.create Constant Streaming
    let shapeUpdate = id
    let stage = Stage.init $"read: {inputDir}" (uint depth) mapper transition shapeUpdate 
    let flow = MemFlow.returnM stage
    let shape = Slice (width,height)
    let context = ShapeContext.create (fun _ -> width*height |> uint64) (fun _ -> uint depth)
    Pipeline.create flow pl.mem (Some shape) context pl.debug
