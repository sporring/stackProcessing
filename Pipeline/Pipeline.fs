module Pipeline

open Core // Core processing model
open Routing // Combinators and routing logic
open SourceSink // Sources and sinks (file IO, streaming)
open Processing // Common image operators
open Slice // Image and slice types

type Pipe<'S,'T> = Core.Pipe<'S,'T>
type Operation<'S,'T> = Core.Operation<'S,'T>
type MemoryProfile = Core.MemoryProfile
type MemoryTransition = Core.MemoryTransition
type Slice<'S when 'S: equality> = Slice.Slice<'S>

let idOp = Core.idOp
let (-->) = Core.(-->)
let source = Core.sourceOp
let sink (pl: Pipeline<unit,unit>) : unit = Core.sinkOp pl
let sinkList (plLst: Pipeline<unit,unit> list) : unit = Core.sinkListOp plLst
let (>=>) = Core.(>=>)
let (>=>>) = Routing.(>=>>)
let (>>=>) = Routing.(>>=>)
let combineIgnore = Routing.combineIgnore
let drainSingle pl = Routing.drainSingle "drainSingle" pl
let drainList pl = Routing.drainList "drainList" pl
let drainLast pl = Routing.drainLast "drainLast" pl
let tap = Routing.tapOp
let liftUnary (f: Slice<'T> -> Slice<'T>) = Routing.liftUnaryOp "liftUnary" f

let create<'T when 'T: equality> = SourceSink.createOp<'T>
let readAs<'T when 'T: equality> = SourceSink.readOp<'T>
let readRandomAs<'T when 'T: equality> = SourceSink.readRandomOp<'T>
let write = SourceSink.writeOp
let print = SourceSink.printOp
let plot = SourceSink.plotOp
let show = SourceSink.showOp

let finiteDiffFilter3D = SourceSink.finiteDiffFilter3DOp
let axisSource = SourceSink.axisSourceOp

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
let castFloatToUInt8 = castFloatToUInt8Op "castFloatToUInt8"
let castFloatToInt8 = castFloatToInt8Op "castFloatToInt8"
let castFloatToUInt16 = castFloatToUInt16Op "castFloatToUInt16"
let castFloatToInt16 = castFloatToInt16Op "castFloatToInt16"
let castFloatToUInt = castFloatToUIntOp "castFloatToUInt"
let castFloatToInt = castFloatToIntOp "castFloatToInt"
let castFloatToUIn64 = castFloatToUIn64Op "castFloatToUIn64"
let castFloatToInt64 = castFloatToInt64Op "castFloatToInt64"
let castFloatToFloat32 = castFloatToFloat32Op "castFloatToFloat32"

/// Basic arithmetic
let add slice = addOp "add" slice
let inline scalarAddSlice<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) = scalarAddSliceOp "scalarAddSlice" i
let inline sliceAddScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) = sliceAddScalarOp "sliceAddScalar" i

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
let absFloat      = absFloatOp "abs"
let absFloat32    = absFloat32Op "abs"
let absInt        = absIntOp "abs"
let acosFloat     = acosFloatOp "acos"
let acosFloat32   = acosFloat32Op "acos"
let asinFloat     = asinFloatOp "asin"
let asinFloat32   = asinFloat32Op "asin"
let atanFloat     = atanFloatOp "atan"
let atanFloat32   = atanFloat32Op "atan"
let cosFloat      = cosFloatOp "cos"
let cosFloat32    = cosFloat32Op "cos"
let expFloat      = expFloatOp "exp"
let expFloat32    = expFloat32Op "exp"
let log10Float    = log10FloatOp "log10"
let log10Float32  = log10Float32Op "log10"
let logFloat      = logFloatOp "log"
let logFloat32    = logFloat32Op "log"
let roundFloat    = roundFloatOp "round"
let roundFloat32  = roundFloat32Op "round"
let sinFloat      = sinFloatOp "sin"
let sinFloat32    = sinFloat32Op "sin"
let sqrtFloat     = sqrtFloatOp "sqrt"
let sqrtFloat32   = sqrtFloat32Op "sqrt"
let sqrtInt       = sqrtIntOp "sqrt"
let squareFloat   = squareFloatOp "square"
let squareFloat32 = squareFloat32Op "square"
let squareInt     = squareIntOp "square"
let tanFloat      = tanFloatOp "tan"
let tanFloat32    = tanFloat32Op "tan"

let histogram<'T when 'T: comparison> = histogramOp<'T> "histogram"
let inline map2pairs< ^T, ^S when ^T: comparison and ^T : (static member op_Explicit : ^T -> float) and  ^S : (static member op_Explicit : ^S -> float) > = map2pairsOp<'T,'S> "map2pairs"
let inline pairs2floats< ^T, ^S when ^T : (static member op_Explicit : ^T -> float) and  ^S : (static member op_Explicit : ^S -> float) > = pairs2floatsOp<'T,'S> "pairs2floats"
let inline pairs2ints< ^T, ^S when ^T : (static member op_Explicit : ^T -> int) and  ^S : (static member op_Explicit : ^S -> int) > = pairs2intsOp<'T,'S> "pairs2ints"

type ImageStats = ImageFunctions.ImageStats
let computeStats<'T when 'T: equality and 'T: comparison> = computeStatsOp<'T> "computeStats"

/// Convolution like operators
// Chained type definitions do expose the originals
let zeroPad = Processing.zeroPad
let periodicPad = Processing.periodicPad
let zeroFluxNeumannPad = Processing.zeroFluxNeumannPad
let valid = Processing.valid
let same = Processing.same

let discreteGaussian sigma bc winSz = discreteGaussianOp "discreteGaussian" sigma bc winSz
let convGauss sigma bc       = discreteGaussianOp "convGauss" sigma bc None
let convolve kernel bc winSz = convolveOp "convolve" kernel bc winSz
let conv kernel              = convolveOp "conv" kernel None None
let convGaussOp sigma bc     = discreteGaussianOp "convGauss" sigma bc None

// these only works on uint8
let erode            r       = binaryErodeOp   "binaryErode"   r None
let dilate           r       = binaryDilateOp  "binaryDilate"  r None
let opening          r       = binaryOpeningOp "binaryOpening" r None
let closing          r       = binaryClosingOp "binaryClosing" r None

/// Full stack operators
let binaryFillHoles = binaryFillHolesOp "fillHoles"
let connectedComponents = connectedComponentsOp "components"
let piecewiseConnectedComponents wz = piecewiseConnectedComponentsOp "piecewiseConnectedComponents" wz

// Annoying F# value restriction requires explicit types here, sigh
let otsuThreshold<'T when 'T: equality> = (otsuThresholdOp "otsuThreshold")
let otsuMultiThreshold n = otsuMultiThresholdOp "otsuMultiThreshold" n
let momentsThreshold<'T when 'T: equality> = momentsThresholdOp "momentsThreshold"
let signedDistanceMap = signedDistanceMapOp "signedDistanceMap"
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

