module Ops 
open Processing
open Core
open Core.Helpers
open Slice

// Cast
let castUInt8ToInt8 = asPipe (castUInt8ToInt8Op "castUInt8ToInt8")
let castUInt8ToUInt16 = asPipe (castUInt8ToUInt16Op "castUInt8ToUInt16")
let castUInt8ToInt16 = asPipe (castUInt8ToInt16Op "castUInt8ToInt16")
let castUInt8ToUInt = asPipe (castUInt8ToUIntOp "castUInt8ToUInt")
let castUInt8ToInt = asPipe (castUInt8ToIntOp "castUInt8ToInt")
let castUInt8ToUInt64 = asPipe (castUInt8ToUInt64Op "castUInt8ToUInt64")
let castUInt8ToInt64 = asPipe (castUInt8ToInt64Op "castUInt8ToInt64")
let castUInt8ToFloat32 = asPipe (castUInt8ToFloat32Op "castUInt8ToFloat32")
let castUInt8ToFloat = asPipe (castUInt8ToFloatOp "castUInt8ToFloat")
let castInt8ToUInt8 = asPipe (castInt8ToUInt8Op "castInt8ToUInt8")
let castInt8ToUInt16 = asPipe (castInt8ToUInt16Op "castInt8ToUInt16")
let castInt8ToInt16 = asPipe (castInt8ToInt16Op "castInt8ToInt16")
let castInt8ToUInt = asPipe (castInt8ToUIntOp "castInt8ToUInt")
let castInt8ToInt = asPipe (castInt8ToIntOp "castInt8ToInt")
let castInt8ToUInt64 = asPipe (castInt8ToUInt64Op "castInt8ToUInt64")
let castInt8ToInt64 = asPipe (castInt8ToInt64Op "castInt8ToInt64")
let castInt8ToFloat32 = asPipe (castInt8ToFloat32Op "castInt8ToFloat32")
let castInt8ToFloat = asPipe (castInt8ToFloatOp "castInt8ToFloat")

let castUInt16ToUInt8 = asPipe (castUInt16ToUInt8Op "castUInt16ToUInt8")
let castUInt16ToInt8 = asPipe (castUInt16ToInt8Op "castUInt16ToInt8")
let castUInt16ToInt16 = asPipe (castUInt16ToInt16Op "castUInt16ToInt16")
let castUInt16ToUInt = asPipe (castUInt16ToUIntOp "castUInt16ToUInt")
let castUInt16ToInt = asPipe (castUInt16ToIntOp "castUInt16ToInt")
let castUInt16ToUInt64 = asPipe (castUInt16ToUInt64Op "castUInt16ToUInt64")
let castUInt16ToInt64 = asPipe (castUInt16ToInt64Op "castUInt16ToInt64")
let castUInt16ToFloat32 = asPipe (castUInt16ToFloat32Op "castUInt16ToFloat32")
let castUInt16ToFloat = asPipe (castUInt16ToFloatOp "castUInt16ToFloat")
let castInt16ToUInt8 = asPipe (castInt16ToUInt8Op "castInt16ToUInt8")
let castInt16ToInt8 = asPipe (castInt16ToInt8Op "castInt16ToInt8")
let castInt16ToUInt16 = asPipe (castInt16ToUInt16Op "castInt16ToUInt16")
let castInt16ToUInt = asPipe (castInt16ToUIntOp "castInt16ToUInt")
let castInt16ToInt = asPipe (castInt16ToIntOp "castInt16ToInt")
let castInt16ToUInt64 = asPipe (castInt16ToUInt64Op "castInt16ToUInt64")
let castInt16ToInt64 = asPipe (castInt16ToInt64Op "castInt16ToInt64")
let castInt16ToFloat32 = asPipe (castInt16ToFloat32Op "castInt16ToFloat32")
let castInt16ToFloat = asPipe (castInt16ToFloatOp "castInt16ToFloat")

let castUIntToUInt8 = asPipe (castUIntToUInt8Op "castUIntToUInt8")
let castUIntToInt8 = asPipe (castUIntToInt8Op "castUIntToInt8")
let castUIntToUInt16 = asPipe (castUIntToUInt16Op "castUIntToUInt16")
let castUIntToInt16 = asPipe (castUIntToInt16Op "castUIntToInt16")
let castUIntToInt = asPipe (castUIntToIntOp "castUIntToInt")
let castUIntToUInt64 = asPipe (castUIntToUInt64Op "castUIntToUInt64")
let castUIntToInt64 = asPipe (castUIntToInt64Op "castUIntToInt64")
let castUIntToFloat32 = asPipe (castUIntToFloat32Op "castUIntToFloat32")
let castUIntToFloat = asPipe (castUIntToFloatOp "castUIntToFloat")
let castIntToUInt8 = asPipe (castIntToUInt8Op "castIntToUInt8")
let castIntToInt8 = asPipe (castIntToInt8Op "castIntToInt8")
let castIntToUInt16 = asPipe (castIntToUInt16Op "castIntToUInt16")
let castIntToInt16 = asPipe (castIntToInt16Op "castIntToInt16")
let castIntToUInt = asPipe (castIntToUIntOp "castIntToUInt")
let castIntToUInt64 = asPipe (castIntToUInt64Op "castIntToUInt64")
let castIntToInt64 = asPipe (castIntToInt64Op "castIntToInt64")
let castIntToFloat32 = asPipe (castIntToFloat32Op "castIntToFloat32")
let castIntToFloat = asPipe (castIntToFloatOp "castIntToFloat")

let castUInt64ToUInt8 = asPipe (castUInt64ToUInt8Op "castUInt64ToUInt8")
let castUInt64ToInt8 = asPipe (castUInt64ToInt8Op "castUInt64ToInt8")
let castUInt64ToUInt16 = asPipe (castUInt64ToUInt16Op "castUInt64ToUInt16")
let castUInt64ToInt16 = asPipe (castUInt64ToInt16Op "castUInt64ToInt16")
let castUInt64ToUInt = asPipe (castUInt64ToUIntOp "castUInt64ToUInt")
let castUInt64ToInt = asPipe (castUInt64ToIntOp "castUInt64ToInt")
let castUInt64ToInt64 = asPipe (castUInt64ToInt64Op "castUInt64ToInt64")
let castUInt64ToFloat32 = asPipe (castUInt64ToFloat32Op "castUInt64ToFloat32")
let castUInt64ToFloat = asPipe (castUInt64ToFloatOp "castUInt64ToFloat")
let castInt64ToUInt8 = asPipe (castInt64ToUInt8Op "castInt64ToUInt8")
let castInt64ToInt8 = asPipe (castInt64ToInt8Op "castInt64ToInt8")
let castInt64ToUInt16 = asPipe (castInt64ToUInt16Op "castInt64ToUInt16")
let castInt64ToInt16 = asPipe (castInt64ToInt16Op "castInt64ToInt16")
let castInt64ToUInt = asPipe (castInt64ToUIntOp "castInt64ToUInt")
let castInt64ToInt = asPipe (castInt64ToIntOp "castInt64ToInt")
let castInt64ToUInt64 = asPipe (castInt64ToUInt64Op "castInt64ToUInt64")
let castInt64ToFloat32 = asPipe (castInt64ToFloat32Op "castInt64ToFloat32")
let castInt64ToFloat = asPipe (castInt64ToFloatOp "castInt64ToFloat")

let castFloat32ToUInt8 = asPipe (castFloat32ToUInt8Op "castFloat32ToUInt8")
let castFloat32ToInt8 = asPipe (castFloat32ToInt8Op "castFloat32ToInt8")
let castFloat32ToUInt16 = asPipe (castFloat32ToUInt16Op "castFloat32ToUInt16")
let castFloat32ToInt16 = asPipe (castFloat32ToInt16Op "castFloat32ToInt16")
let castFloat32ToUInt = asPipe (castFloat32ToUIntOp "castFloat32ToUInt")
let castFloat32ToInt = asPipe (castFloat32ToIntOp "castFloat32ToInt")
let castFloat32ToUInt64 = asPipe (castFloat32ToUInt64Op "castFloat32ToUInt64")
let castFloat32ToInt64 = asPipe (castFloat32ToInt64Op "castFloat32ToInt64")
let castFloat32ToFloat = asPipe (castFloat32ToFloatOp "castFloat32ToFloat")
let castFloatToUInt8 = asPipe (castFloatToUInt8Op "castFloatToUInt8")
let castFloatToInt8 = asPipe (castFloatToInt8Op "castFloatToInt8")
let castFloatToUInt16 = asPipe (castFloatToUInt16Op "castFloatToUInt16")
let castFloatToInt16 = asPipe (castFloatToInt16Op "castFloatToInt16")
let castFloatToUInt = asPipe (castFloatToUIntOp "castFloatToUInt")
let castFloatToInt = asPipe (castFloatToIntOp "castFloatToInt")
let castFloatToUIn64 = asPipe (castFloatToUIn64Op "castFloatToUIn64")
let castFloatToInt64 = asPipe (castFloatToInt64Op "castFloatToInt64")
let castFloatToFloat32 = asPipe (castFloatToFloat32Op "castFloatToFloat32")

/// Basic arithmetic
let add slice = asPipe (addOp "add" slice)
let inline scalarAddSlice<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) = asPipe (scalarAddSliceOp "scalarAddSlice" i)
let inline sliceAddScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) = asPipe (sliceAddScalarOp "sliceAddScalar" i)

let sub slice = asPipe (subOp "sub" slice)
let inline scalarSubSlice<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) = asPipe (scalarSubSliceOp "scalarSubSlice" i)
let inline sliceSubScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) = asPipe (sliceSubScalarOp "sliceSubScalar" i)

let mul slice = asPipe (mulOp "mul" slice)
let inline scalarMulSlice<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) = asPipe (scalarMulSliceOp "scalarMulSlice" i)
let inline sliceMulScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) = asPipe (sliceMulScalarOp "sliceMulScalar" i)

let div slice = asPipe (divOp "div" slice)
let inline scalarDivSlice<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) = asPipe (scalarDivSliceOp "scalarDivSlice" i)
let inline sliceDivScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) = asPipe (sliceDivScalarOp "sliceDivScalar" i)

/// Simple functions
let absFloat      = asPipe (absFloatOp "abs")
let absFloat32    = asPipe (absFloat32Op "abs")
let absInt        = asPipe (absIntOp "abs")
let acosFloat     = asPipe (acosFloatOp "acos")
let acosFloat32   = asPipe (acosFloat32Op "acos")
let asinFloat     = asPipe (asinFloatOp "asin")
let asinFloat32   = asPipe (asinFloat32Op "asin")
let atanFloat     = asPipe (atanFloatOp "atan")
let atanFloat32   = asPipe (atanFloat32Op "atan")
let cosFloat      = asPipe (cosFloatOp "cos")
let cosFloat32    = asPipe (cosFloat32Op "cos")
let expFloat      = asPipe (expFloatOp "exp")
let expFloat32    = asPipe (expFloat32Op "exp")
let log10Float    = asPipe (log10FloatOp "log10")
let log10Float32  = asPipe (log10Float32Op "log10")
let logFloat      = asPipe (logFloatOp "log")
let logFloat32    = asPipe (logFloat32Op "log")
let roundFloat    = asPipe (roundFloatOp "round")
let roundFloat32  = asPipe (roundFloat32Op "round")
let sinFloat      = asPipe (sinFloatOp "sin")
let sinFloat32    = asPipe (sinFloat32Op "sin")
let sqrtFloat     = asPipe (sqrtFloatOp "sqrt")
let sqrtFloat32   = asPipe (sqrtFloat32Op "sqrt")
let sqrtInt       = asPipe (sqrtIntOp "sqrt")
let squareFloat   = asPipe (squareFloatOp "square")
let squareFloat32 = asPipe (squareFloat32Op "square")
let squareInt     = asPipe (squareIntOp "square")
let tanFloat      = asPipe (tanFloatOp "tan")
let tanFloat32    = asPipe (tanFloat32Op "tan")

let histogram<'T when 'T: comparison> = asPipe (histogramOp<'T> "histogram")
let inline map2pairs< ^T, ^S when ^T: comparison and ^T : (static member op_Explicit : ^T -> float) and  ^S : (static member op_Explicit : ^S -> float) > = asPipe (map2pairsOp<'T,'S> "map2pairs")
let inline pairs2floats< ^T, ^S when ^T : (static member op_Explicit : ^T -> float) and  ^S : (static member op_Explicit : ^S -> float) > = asPipe (pairs2floatsOp<'T,'S> "pairs2floats")
let inline pairs2ints< ^T, ^S when ^T : (static member op_Explicit : ^T -> int) and  ^S : (static member op_Explicit : ^S -> int) > = asPipe (pairs2intsOp<'T,'S> "pairs2ints")

type ImageStats = ImageFunctions.ImageStats
let computeStats<'T when 'T: equality and 'T: comparison> = asPipe (computeStatsOp<'T> "computeStats")

/// Convolution like operators
// Chained type definitions do expose the originals
let zeroPad = Processing.zeroPad
let periodicPad = Processing.periodicPad
let zeroFluxNeumannPad = Processing.zeroFluxNeumannPad
let valid = Processing.valid
let same = Processing.same

let discreteGaussian sigma bc winSz = asPipe (discreteGaussianOp "discreteGaussian" sigma bc winSz)
let convGauss sigma bc       = asPipe (discreteGaussianOp "convGauss" sigma bc None)
let convolve kernel bc winSz = asPipe (convolveOp "convolve" kernel bc winSz)
let conv kernel              = asPipe (convolveOp "conv" kernel None None)
let convGaussOp sigma bc     = discreteGaussianOp "convGauss" sigma bc None

// these only works on uint8
let erode            r       = asPipe (binaryErodeOp   "binaryErode"   r None)
let dilate           r       = asPipe (binaryDilateOp  "binaryDilate"  r None)
let opening          r       = asPipe (binaryOpeningOp "binaryOpening" r None)
let closing          r       = asPipe (binaryClosingOp "binaryClosing" r None)

/// Full stack operators
let binaryFillHoles = asPipe (binaryFillHolesOp "fillHoles")
let connectedComponents = asPipe (connectedComponentsOp "components")
let piecewiseConnectedComponents wz = asPipe (piecewiseConnectedComponentsOp "piecewiseConnectedComponents" wz)

// Annoying F# value restriction requires explicit types here, sigh
let otsuThreshold<'T when 'T: equality> = asPipe ((otsuThresholdOp "otsuThreshold"))
let otsuMultiThreshold n = asPipe (otsuMultiThresholdOp "otsuMultiThreshold" n)
let momentsThreshold<'T when 'T: equality> = asPipe (momentsThresholdOp "momentsThreshold")
let signedDistanceMap = asPipe (signedDistanceMapOp "signedDistanceMap")
let watershed a = asPipe (watershedOp "watershed" a)
let threshold a b = asPipe (thresholdOp "threshold" a b)
let addNormalNoise a b = asPipe (addNormalNoiseOp "addNormalNoise" a b)
let relabelComponents a = asPipe (relabelComponentsOp "relabelComponents" a)

let constantPad2D<'T when 'T : equality> padLower padUpper c = asPipe (constantPad2DOp "constantPad2D" padLower padUpper c)
