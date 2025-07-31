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
let sink (pl: Pipeline<unit,unit,'Shape>) : unit = Pipeline.sink pl
let sinkList (plLst: Pipeline<unit,unit,'Shape> list) : unit = Pipeline.sinkList plLst
let (>=>) = Pipeline.(>=>)
let (>=>>) = Routing.(>=>>)
let (>>=>) = Routing.(>>=>)
let combineIgnore = Routing.combineIgnore
let drainSingle pl = Routing.drainSingle "drainSingle" pl
let drainList pl = Routing.drainList "drainList" pl
let drainLast pl = Routing.drainLast "drainLast" pl
let tap = Stage.tap
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
    let context = MemFlow.create (fun _ -> width*height |> uint64) (fun _ -> depth)
    {
        flow = MemFlow.returnM op
        mem = pl.mem
        shape = Slice (width,height) |> Some
        context = context
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
    let context = MemFlow.create (fun _ -> width*height |> uint64) (fun _ -> depth)
    {
        flow = MemFlow.returnM op
        mem = pl.mem
        shape = Slice (width,height) |> Some
        context = context
    }

(*
let castUInt8ToInt8 = castUInt8ToInt8Op "castUInt8ToInt8"
let castUInt8ToUInt16 = castUInt8ToUInt16Op "castUInt8ToUInt16"
let castUInt8ToInt16 = castUInt8ToInt16Op "castUInt8ToInt16"
let castUInt8ToUInt = castUInt8ToUIntOp "castUInt8ToUInt"
let castUInt8ToInt = castUInt8ToIntOp "castUInt8ToInt"
let castUInt8ToUInt64 = castUInt8ToUInt64Op "castUInt8ToUInt64"
let castUInt8ToInt64 = castUInt8ToInt64Op "castUInt8ToInt64"
let castUInt8ToFloat32 = castUInt8ToFloat32Op "castUInt8ToFloat32"
let castUInt8ToFloat = castUInt8ToFloatOp "castUInt8ToFloat"
let castInt8ToUInt8 = castInt8ToUInt8Op "castInt8ToUInt8"
let castInt8ToUInt16 = castInt8ToUInt16Op "castInt8ToUInt16"
let castInt8ToInt16 = castInt8ToInt16Op "castInt8ToInt16"
let castInt8ToUInt = castInt8ToUIntOp "castInt8ToUInt"
let castInt8ToInt = castInt8ToIntOp "castInt8ToInt"
let castInt8ToUInt64 = castInt8ToUInt64Op "castInt8ToUInt64"
let castInt8ToInt64 = castInt8ToInt64Op "castInt8ToInt64"
let castInt8ToFloat32 = castInt8ToFloat32Op "castInt8ToFloat32"
let castInt8ToFloat = castInt8ToFloatOp "castInt8ToFloat"

let castUInt16ToUInt8 = castUInt16ToUInt8Op "castUInt16ToUInt8"
let castUInt16ToInt8 = castUInt16ToInt8Op "castUInt16ToInt8"
let castUInt16ToInt16 = castUInt16ToInt16Op "castUInt16ToInt16"
let castUInt16ToUInt = castUInt16ToUIntOp "castUInt16ToUInt"
let castUInt16ToInt = castUInt16ToIntOp "castUInt16ToInt"
let castUInt16ToUInt64 = castUInt16ToUInt64Op "castUInt16ToUInt64"
let castUInt16ToInt64 = castUInt16ToInt64Op "castUInt16ToInt64"
let castUInt16ToFloat32 = castUInt16ToFloat32Op "castUInt16ToFloat32"
let castUInt16ToFloat = castUInt16ToFloatOp "castUInt16ToFloat"
let castInt16ToUInt8 = castInt16ToUInt8Op "castInt16ToUInt8"
let castInt16ToInt8 = castInt16ToInt8Op "castInt16ToInt8"
let castInt16ToUInt16 = castInt16ToUInt16Op "castInt16ToUInt16"
let castInt16ToUInt = castInt16ToUIntOp "castInt16ToUInt"
let castInt16ToInt = castInt16ToIntOp "castInt16ToInt"
let castInt16ToUInt64 = castInt16ToUInt64Op "castInt16ToUInt64"
let castInt16ToInt64 = castInt16ToInt64Op "castInt16ToInt64"
let castInt16ToFloat32 = castInt16ToFloat32Op "castInt16ToFloat32"
let castInt16ToFloat = castInt16ToFloatOp "castInt16ToFloat"

let castUIntToUInt8 = castUIntToUInt8Op "castUIntToUInt8"
let castUIntToInt8 = castUIntToInt8Op "castUIntToInt8"
let castUIntToUInt16 = castUIntToUInt16Op "castUIntToUInt16"
let castUIntToInt16 = castUIntToInt16Op "castUIntToInt16"
let castUIntToInt = castUIntToIntOp "castUIntToInt"
let castUIntToUInt64 = castUIntToUInt64Op "castUIntToUInt64"
let castUIntToInt64 = castUIntToInt64Op "castUIntToInt64"
let castUIntToFloat32 = castUIntToFloat32Op "castUIntToFloat32"
let castUIntToFloat = castUIntToFloatOp "castUIntToFloat"
let castIntToUInt8 = castIntToUInt8Op "castIntToUInt8"
let castIntToInt8 = castIntToInt8Op "castIntToInt8"
let castIntToUInt16 = castIntToUInt16Op "castIntToUInt16"
let castIntToInt16 = castIntToInt16Op "castIntToInt16"
let castIntToUInt = castIntToUIntOp "castIntToUInt"
let castIntToUInt64 = castIntToUInt64Op "castIntToUInt64"
let castIntToInt64 = castIntToInt64Op "castIntToInt64"
let castIntToFloat32 = castIntToFloat32Op "castIntToFloat32"
let castIntToFloat = castIntToFloatOp "castIntToFloat"

let castUInt64ToUInt8 = castUInt64ToUInt8Op "castUInt64ToUInt8"
let castUInt64ToInt8 = castUInt64ToInt8Op "castUInt64ToInt8"
let castUInt64ToUInt16 = castUInt64ToUInt16Op "castUInt64ToUInt16"
let castUInt64ToInt16 = castUInt64ToInt16Op "castUInt64ToInt16"
let castUInt64ToUInt = castUInt64ToUIntOp "castUInt64ToUInt"
let castUInt64ToInt = castUInt64ToIntOp "castUInt64ToInt"
let castUInt64ToInt64 = castUInt64ToInt64Op "castUInt64ToInt64"
let castUInt64ToFloat32 = castUInt64ToFloat32Op "castUInt64ToFloat32"
let castUInt64ToFloat = castUInt64ToFloatOp "castUInt64ToFloat"
let castInt64ToUInt8 = castInt64ToUInt8Op "castInt64ToUInt8"
let castInt64ToInt8 = castInt64ToInt8Op "castInt64ToInt8"
let castInt64ToUInt16 = castInt64ToUInt16Op "castInt64ToUInt16"
let castInt64ToInt16 = castInt64ToInt16Op "castInt64ToInt16"
let castInt64ToUInt = castInt64ToUIntOp "castInt64ToUInt"
let castInt64ToInt = castInt64ToIntOp "castInt64ToInt"
let castInt64ToUInt64 = castInt64ToUInt64Op "castInt64ToUInt64"
let castInt64ToFloat32 = castInt64ToFloat32Op "castInt64ToFloat32"
let castInt64ToFloat = castInt64ToFloatOp "castInt64ToFloat"

let castFloat32ToUInt8 = castFloat32ToUInt8Op "castFloat32ToUInt8"
let castFloat32ToInt8 = castFloat32ToInt8Op "castFloat32ToInt8"
let castFloat32ToUInt16 = castFloat32ToUInt16Op "castFloat32ToUInt16"
let castFloat32ToInt16 = castFloat32ToInt16Op "castFloat32ToInt16"
let castFloat32ToUInt = castFloat32ToUIntOp "castFloat32ToUInt"
let castFloat32ToInt = castFloat32ToIntOp "castFloat32ToInt"
let castFloat32ToUInt64 = castFloat32ToUInt64Op "castFloat32ToUInt64"
let castFloat32ToInt64 = castFloat32ToInt64Op "castFloat32ToInt64"
let castFloat32ToFloat = castFloat32ToFloatOp "castFloat32ToFloat"
*)
//let castFloatToUInt8 = castFloatToUInt8Op "castFloatToUInt8"
let castFloatToUInt8 = castOp<float,uint8,Shape> "castFloatToUInt8" Slice.cast<float,uint8>
(*
let castFloatToInt8 = castFloatToInt8Op "castFloatToInt8"
let castFloatToUInt16 = castFloatToUInt16Op "castFloatToUInt16"
let castFloatToInt16 = castFloatToInt16Op "castFloatToInt16"
let castFloatToUInt = castFloatToUIntOp "castFloatToUInt"
let castFloatToInt = castFloatToIntOp "castFloatToInt"
let castFloatToUIn64 = castFloatToUIn64Op "castFloatToUIn64"
let castFloatToInt64 = castFloatToInt64Op "castFloatToInt64"
let castFloatToFloat32 = castFloatToFloat32Op "castFloatToFloat32"
*)
//let cast<'S,'T,Shape> = castOp<'S,'T,Shape> (sprintf "cast(%s->%s)" typeof<'S>.Name typeof<'T>.Name) Slice.cast<'S,'T>

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
    let mapper (i: uint) : Slice<'T> = 
        let slice = Slice.create<'T> width height 1u i
        printfn "[create] Created slice %A" i
        slice
    let transition = Stage.transition Constant Streaming
    let shapeUpdate = id
    let stage = Stage.init "create" depth mapper transition shapeUpdate 
    let flow = MemFlow.returnM stage
    let shape = Slice (width,height)
    let context = MemFlow.create (fun _ -> width*height |> uint64) (fun _ -> depth)
    Pipeline.create flow pl.mem (Some shape) context

let readAs<'T when 'T: equality> (inputDir : string) (suffix : string) (pl : Pipeline<unit, unit, Shape>) : Pipeline<unit, Slice<'T>,Shape> =
    // much should be deferred to outside Core!!!
    let (width,height,depth) = Slice.getStackSize inputDir suffix
    let filenames = Directory.GetFiles(inputDir, "*"+suffix) |> Array.sort
    let depth = filenames.Length
    let mapper (i: uint) : Slice<'T> = 
        let fileName = filenames[int i]; 
        let slice = Slice.readSlice<'T> (uint i) fileName
        printfn "[readSlices] Reading slice %A from %s" i fileName
        slice
    let transition = Stage.transition Constant Streaming
    let shapeUpdate = id
    let stage = Stage.init $"read: {inputDir}" (uint depth) mapper transition shapeUpdate 
    let flow = MemFlow.returnM stage
    let shape = Slice (width,height)
    let context = MemFlow.create (fun _ -> width*height |> uint64) (fun _ -> uint depth)
    Pipeline.create flow pl.mem (Some shape) context

let readRandomAs<'T when 'T: equality> (count: uint) (inputDir : string) (suffix : string) (pl : Pipeline<unit, unit, Shape>) : Pipeline<unit, Slice<'T>,Shape> =
    let (width,height,depth) = Slice.getStackSize inputDir suffix
    let filenames = Directory.GetFiles(inputDir, "*"+suffix) |> Array.randomChoices (int count)
    let depth = filenames.Length
    let mapper (i: uint) : Slice<'T> = 
        let fileName = filenames[int i]; 
        let slice = Slice.readSlice<'T> (uint i) fileName
        printfn "[readRandomSlices] Reading slice %A from %s" i fileName
        slice
    let transition = Stage.transition Constant Streaming
    let shapeUpdate = id
    let stage = Stage.init $"read: {inputDir}" (uint depth) mapper transition shapeUpdate 
    let flow = MemFlow.returnM stage
    let shape = Slice (width,height)
    let context = MemFlow.create (fun _ -> width*height |> uint64) (fun _ -> uint depth)
    Pipeline.create flow pl.mem (Some shape) context
