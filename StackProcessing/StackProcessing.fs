module StackProcessing

open SlimPipeline // Core processing model
open Routing // Combinators and routing logic
open Processing // Common image operators
open Slice // Image and slice types
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
let sink (pl: Pipeline<unit,unit,'Shape>) : unit = Pipeline.sink pl
let sinkList (plLst: Pipeline<unit,unit,'Shape> list) : unit = Pipeline.sinkList plLst
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
let liftUnary (f: Slice<'T> -> Slice<'T>) = Stage.liftUnary "liftUnary" f

let write = Processing.writeOp
let print = Processing.printOp
let plot = Processing.plotOp
let show = Processing.showOp

let finiteDiffFilter3D 
    (direction: uint) 
    (order: uint)
    (pl : Pipeline<unit, unit, Shape>) 
    : Pipeline<unit, Slice<float>, Shape> =
    let img = finiteDiffFilter3D direction order
    let sz = GetSize img
    let shapeUpdate = fun (s: Shape) -> s
    let op : Stage<unit, Slice<float>, Shape> =
        {
            Name = "gaussSource"
            Pipe = img |> liftImageSource "gaussSource"
            Transition = Stage.transition Constant Streaming
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
    : Pipeline<unit, Slice<uint>, Shape> =
    let img = Slice.generateCoordinateAxis axis size
    let sz = GetSize img
    let shapeUpdate = fun (s: Shape) -> s
    let op : Stage<unit, Slice<uint>, Shape> =
        {
            Name = "axisSource"
            Pipe = img |> liftImageSource "axisSource"
            Transition = Stage.transition Constant Streaming
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
let cast<'S,'T when 'S: equality and 'T: equality> = castOp<'S,'T,Shape> (sprintf "cast(%s->%s)" typeof<'S>.Name typeof<'T>.Name) Slice.cast<'S,'T>

/// Basic arithmetic
let add slice = addOp "add" slice
let inline scalarAddSlice<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) = scalarAddSliceOp "scalarAddSlice" i
let inline sliceAddScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) = sliceAddScalarOp<'T,Shape> "sliceAddScalar" i

let sub slice = subOp "sub" slice
let inline scalarSubSlice<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) = scalarSubSliceOp "scalarSubSlice" i
let inline sliceSubScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) = sliceSubScalarOp "sliceSubScalar" i

let mul slice = mulOp "mul" slice
let inline scalarMulSlice<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) = scalarMulSliceOp "scalarMulSlice" i
let inline sliceMulScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) = sliceMulScalarOp "sliceMulScalar" i

let div slice = divOp "div" slice
let inline scalarDivSlice<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) = scalarDivSliceOp "scalarDivSlice" i
let inline sliceDivScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) = sliceDivScalarOp "sliceDivScalar" i

/// Simple functions
let absFloat      : Stage<Slice<float>,Slice<float>, Shape> =      Stage.liftUnary "abs"    absSlice
let absFloat32    : Stage<Slice<float32>,Slice<float32>, Shape> =  Stage.liftUnary "abs"    absSlice
let absInt        : Stage<Slice<int>,Slice<int>, Shape> =          Stage.liftUnary "abs"    absSlice
let acosFloat     : Stage<Slice<float>,Slice<float>, Shape> =      Stage.liftUnary "acos"   acosSlice
let acosFloat32   : Stage<Slice<float32>,Slice<float32>, Shape> =  Stage.liftUnary "acos"   acosSlice
let asinFloat     : Stage<Slice<float>,Slice<float>, Shape> =      Stage.liftUnary "asin"   asinSlice
let asinFloat32   : Stage<Slice<float32>,Slice<float32>, Shape> =  Stage.liftUnary "asin"   asinSlice
let atanFloat     : Stage<Slice<float>,Slice<float>, Shape> =      Stage.liftUnary "atan"   atanSlice
let atanFloat32   : Stage<Slice<float32>,Slice<float32>, Shape> =  Stage.liftUnary "atan"   atanSlice
let cosFloat      : Stage<Slice<float>,Slice<float>, Shape> =      Stage.liftUnary "cos"    cosSlice
let cosFloat32    : Stage<Slice<float32>,Slice<float32>, Shape> =  Stage.liftUnary "cos"    cosSlice
let sinFloat      : Stage<Slice<float>,Slice<float>, Shape> =      Stage.liftUnary "sin"    sinSlice
let sinFloat32    : Stage<Slice<float32>,Slice<float32>, Shape> =  Stage.liftUnary "sin"    sinSlice
let tanFloat      : Stage<Slice<float>,Slice<float>, Shape> =      Stage.liftUnary "tan"    tanSlice
let tanFloat32    : Stage<Slice<float32>,Slice<float32>, Shape> =  Stage.liftUnary "tan"    tanSlice
let expFloat      : Stage<Slice<float>,Slice<float>, Shape> =      Stage.liftUnary "exp"    expSlice
let expFloat32    : Stage<Slice<float32>,Slice<float32>, Shape> =  Stage.liftUnary "exp"    expSlice
let log10Float    : Stage<Slice<float>,Slice<float>, Shape> =      Stage.liftUnary "log10"  log10Slice
let log10Float32  : Stage<Slice<float32>,Slice<float32>, Shape> =  Stage.liftUnary "log10"  log10Slice
let logFloat      : Stage<Slice<float>,Slice<float>, Shape> =      Stage.liftUnary "log"    logSlice
let logFloat32    : Stage<Slice<float32>,Slice<float32>, Shape> =  Stage.liftUnary "log"    logSlice
let roundFloat    : Stage<Slice<float>,Slice<float>, Shape> =      Stage.liftUnary "round"  roundSlice
let roundFloat32  : Stage<Slice<float32>,Slice<float32>, Shape> =  Stage.liftUnary "round"  roundSlice
let sqrtFloat     : Stage<Slice<float>,Slice<float>, Shape> =      Stage.liftUnary "sqrt"   sqrtSlice
let sqrtFloat32   : Stage<Slice<float32>,Slice<float32>, Shape> =  Stage.liftUnary "sqrt"   sqrtSlice
let sqrtInt       : Stage<Slice<int>,Slice<int>, Shape> =          Stage.liftUnary "sqrt"   sqrtSlice
let squareFloat   : Stage<Slice<float>,Slice<float>, Shape> =      Stage.liftUnary "square" squareSlice
let squareFloat32 : Stage<Slice<float32>,Slice<float32>, Shape> =  Stage.liftUnary "square" squareSlice
let squareInt     : Stage<Slice<int>,Slice<int>, Shape> =          Stage.liftUnary "square" squareSlice

let histogram<'T when 'T: comparison> = histogramOp<'T,Shape> "histogram"
let inline map2pairs< ^T, ^S when ^T: comparison and ^T : (static member op_Explicit : ^T -> float) and  ^S : (static member op_Explicit : ^S -> float) > = map2pairsOp<'T,'S,Shape> "map2pairs"
let inline pairs2floats< ^T, ^S when ^T : (static member op_Explicit : ^T -> float) and  ^S : (static member op_Explicit : ^S -> float) > = pairs2floatsOp<'T,'S,Shape> "pairs2floats"
let inline pairs2ints< ^T, ^S when ^T : (static member op_Explicit : ^T -> int) and  ^S : (static member op_Explicit : ^S -> int) > = pairs2intsOp<'T,'S,Shape> "pairs2ints"

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

let createAs<'T when 'T: equality> (width: uint) (height: uint) (depth: uint) (pl : Pipeline<unit, unit, Shape>) : Pipeline<unit, Slice<'T>,Shape> =
    // width, heigth, depth should be replaced with shape and shapeUpdate, and mapper should be deferred to outside Core!!!
    if pl.debug then printfn $"[createAs] {width}x{height}x{depth}"
    let mapper (i: uint) : Slice<'T> = 
        let slice = Slice.create<'T> width height 1u i
        if pl.debug then printfn "[create] Created slice %A" i
        slice
    let transition = Stage.transition Constant Streaming
    let shapeUpdate = id
    let stage = Stage.init "create" depth mapper transition shapeUpdate 
    let flow = MemFlow.returnM stage
    let shape = Slice (width,height)
    let context = ShapeContext.create (fun _ -> width*height |> uint64) (fun _ -> depth)
    Pipeline.create flow pl.mem (Some shape) context pl.debug

let readAs<'T when 'T: equality> (inputDir : string) (suffix : string) (pl : Pipeline<unit, unit, Shape>) : Pipeline<unit, Slice<'T>,Shape> =
    // much should be deferred to outside Core!!!
    if pl.debug then printfn $"[readAs] {inputDir}/*{suffix}"
    let (width,height,depth) = Slice.getStackSize inputDir suffix
    let filenames = Directory.GetFiles(inputDir, "*"+suffix) |> Array.sort
    let depth = filenames.Length
    let mapper (i: uint) : Slice<'T> = 
        let fileName = filenames[int i]; 
        let slice = Slice.readSlice<'T> (uint i) fileName
        if pl.debug then printfn "[readSlices] Reading slice %A from %s" i fileName
        slice
    let transition = Stage.transition Constant Streaming
    let shapeUpdate = id
    let stage = Stage.init $"read: {inputDir}" (uint depth) mapper transition shapeUpdate 
    let flow = MemFlow.returnM stage
    let shape = Slice (width,height)
    let context = ShapeContext.create (fun _ -> width*height |> uint64) (fun _ -> uint depth)
    Pipeline.create flow pl.mem (Some shape) context pl.debug

let readRandomAs<'T when 'T: equality> (count: uint) (inputDir : string) (suffix : string) (pl : Pipeline<unit, unit, Shape>) : Pipeline<unit, Slice<'T>,Shape> =
    if pl.debug then printfn $"[readRandomAs] {count} slices from {inputDir}/*{suffix}"
    let (width,height,depth) = Slice.getStackSize inputDir suffix
    let filenames = Directory.GetFiles(inputDir, "*"+suffix) |> Array.randomChoices (int count)
    let depth = filenames.Length
    let mapper (i: uint) : Slice<'T> = 
        let fileName = filenames[int i]; 
        let slice = Slice.readSlice<'T> (uint i) fileName
        if pl.debug then printfn "[readRandomSlices] Reading slice %A from %s" i fileName
        slice
    let transition = Stage.transition Constant Streaming
    let shapeUpdate = id
    let stage = Stage.init $"read: {inputDir}" (uint depth) mapper transition shapeUpdate 
    let flow = MemFlow.returnM stage
    let shape = Slice (width,height)
    let context = ShapeContext.create (fun _ -> width*height |> uint64) (fun _ -> uint depth)
    Pipeline.create flow pl.mem (Some shape) context pl.debug
