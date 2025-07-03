module Ops 
open Processing
open Core
open Core.Helpers
open Slice

// Float
let sqrtFloat     = asPipe (sqrtFloatOp "sqrt")
let absFloat      = asPipe (absFloatOp "abs")
let logFloat      = asPipe (logFloatOp "log")
let log10Float    = asPipe (log10FloatOp "log10")
let expFloat      = asPipe (expFloatOp "exp")
let squareFloat   = asPipe (squareFloatOp "square")
let sinFloat      = asPipe (sinFloatOp "sin")
let cosFloat      = asPipe (cosFloatOp "cos")
let tanFloat      = asPipe (tanFloatOp "tan")
let asinFloat     = asPipe (asinFloatOp "asin")
let acosFloat     = asPipe (acosFloatOp "acos")
let atanFloat     = asPipe (atanFloatOp "atan")
let roundFloat    = asPipe (roundFloatOp "round")

// Float32
let sqrtFloat32   = asPipe (sqrtFloat32Op "sqrt")
let absFloat32    = asPipe (absFloat32Op "abs")
let logFloat32    = asPipe (logFloat32Op "log")
let log10Float32  = asPipe (log10Float32Op "log10")
let expFloat32    = asPipe (expFloat32Op "exp")
let squareFloat32 = asPipe (squareFloat32Op "square")
let sinFloat32    = asPipe (sinFloat32Op "sin")
let cosFloat32    = asPipe (cosFloat32Op "cos")
let tanFloat32    = asPipe (tanFloat32Op "tan")
let asinFloat32   = asPipe (asinFloat32Op "asin")
let acosFloat32   = asPipe (acosFloat32Op "acos")
let atanFloat32   = asPipe (atanFloat32Op "atan")
let roundFloat32  = asPipe (roundFloat32Op "round")

// Int
let sqrtInt       = asPipe (sqrtIntOp "sqrt")
let absInt        = asPipe (absIntOp "abs")
let squareInt     = asPipe (squareIntOp "square")


let discreteGaussian sigma boundaryCondition winSz = 
    asPipe (discreteGaussianOp "discreteGaussian" sigma boundaryCondition winSz)
let convGauss sigma =
    asPipe (discreteGaussianOp "convGauss" sigma None None)

let convolve kernel bc winSz =
    asPipe (convolveOp "convolve" kernel bc winSz)

let conv kernel =
    asPipe (convolveOp "conv" kernel None None)

/// these only works on uint8
let binaryErode      r winSz = asPipe (binaryErodeOp   "binaryErode"   r winSz)
let erode            r       = asPipe (binaryErodeOp   "binaryErode"   r None)
let binaryDilate     r winSz = asPipe (binaryDilateOp  "binaryDilate"  r winSz)
let dilate           r       = asPipe (binaryDilateOp  "binaryDilate"  r None)
let binaryOpening    r winSz = asPipe (binaryOpeningOp "binaryOpening" r winSz)
let opening          r       = asPipe (binaryOpeningOp "binaryOpening" r None)
let binaryClosing    r winSz = asPipe (binaryClosingOp "binaryClosing" r winSz)
let closing          r       = asPipe (binaryClosingOp "binaryClosing" r None)
let binaryFillHoles = asPipe (binaryFillHolesOp "fillHoles")
let connectedComponents = asPipe (connectedComponentsOp "components")

/// simple one sided arithmatic operators
let addUInt8   scalar = asPipe (addUInt8Op "addUInt8" scalar)
let addInt8    scalar = asPipe (addInt8Op "addInt8" scalar)
let addUInt16  scalar = asPipe (addUInt16Op "addUInt16" scalar)
let addInt16   scalar = asPipe (addInt16Op "addInt16" scalar)
let addUInt    scalar = asPipe (addUIntOp "addUInt" scalar)
let addInt     scalar = asPipe (addIntOp "addInt" scalar)
let addUInt64  scalar = asPipe (addUInt64Op "addUInt64" scalar)
let addInt64   scalar = asPipe (addInt64Op "addInt64" scalar)
let addFloat32 scalar = asPipe (addFloat32Op "addFloat32" scalar)
let addFloat   scalar = asPipe (addFloatOp "addFloat" scalar)

let subUInt8   scalar = asPipe (subUInt8Op "subUInt8" scalar)
let subInt8    scalar = asPipe (subInt8Op "subInt8" scalar)
let subUInt16  scalar = asPipe (subUInt16Op "subUInt16" scalar)
let subInt16   scalar = asPipe (subInt16Op "subInt16" scalar)
let subUInt    scalar = asPipe (subUIntOp "subUInt" scalar)
let subInt     scalar = asPipe (subIntOp "subInt" scalar)
let subUInt64  scalar = asPipe (subUInt64Op "subUInt64" scalar)
let subInt64   scalar = asPipe (subInt64Op "subInt64" scalar)
let subFloat32 scalar = asPipe (subFloat32Op "subFloat32" scalar)
let subFloat   scalar = asPipe (subFloatOp "subFloat" scalar)

let mulUInt8   scalar = asPipe (mulUInt8Op "mulUInt8" scalar)
let mulInt8    scalar = asPipe (mulInt8Op "mulInt8" scalar)
let mulUInt16  scalar = asPipe (mulUInt16Op "mulUInt16" scalar)
let mulInt16   scalar = asPipe (mulInt16Op "mulInt16" scalar)
let mulUInt    scalar = asPipe (mulUIntOp "mulUInt" scalar)
let mulInt     scalar = asPipe (mulIntOp "mulInt" scalar)
let mulUInt64  scalar = asPipe (mulUInt64Op "mulUInt64" scalar)
let mulInt64   scalar = asPipe (mulInt64Op "mulInt64" scalar)
let mulFloat32 scalar = asPipe (mulFloat32Op "mulFloat32" scalar)
let mulFloat   scalar = asPipe (mulFloatOp "mulFloat" scalar)

let divUInt8   scalar = asPipe (divUInt8Op "divUInt8" scalar)
let divInt8    scalar = asPipe (divInt8Op "divInt8" scalar)
let divUInt16  scalar = asPipe (divUInt16Op "divUInt16" scalar)
let divInt16   scalar = asPipe (divInt16Op "divInt16" scalar)
let divUInt    scalar = asPipe (divUIntOp "divUInt" scalar)
let divInt     scalar = asPipe (divIntOp "divInt" scalar)
let divUInt64  scalar = asPipe (divUInt64Op "divUInt64" scalar)
let divInt64   scalar = asPipe (divInt64Op "divInt64" scalar)
let divFloat32 scalar = asPipe (divFloat32Op "divFloat32" scalar)
let divFloat   scalar = asPipe (divFloatOp "divFloat" scalar)

let add (im1: Slice<'T>) (im2: Slice<'T>) : Slice<'T> = Slice.add im1 im2
let sub (im1: Slice<'T>) (im2: Slice<'T>) : Slice<'T> = Slice.sub im1 im2
let mul (im1: Slice<'T>) (im2: Slice<'T>) : Slice<'T> = Slice.mul im1 im2
let div (im1: Slice<'T>) (im2: Slice<'T>) : Slice<'T> = Slice.div im1 im2

let isGreaterEqual (im1: Slice<'T>) (im2: Slice<'T>) : Slice<'T> = Slice.isGreaterEqual im1 im2
let isGreater (im1: Slice<'T>) (im2: Slice<'T>) : Slice<'T> = Slice.isGreater im1 im2
let isEqual (im1: Slice<'T>) (im2: Slice<'T>) : Slice<'T> = Slice.isEqual im1 im2
let isNotEqual (im1: Slice<'T>) (im2: Slice<'T>) : Slice<'T> = Slice.isNotEqual im1 im2
let isLessThanEqual (im1: Slice<'T>) (im2: Slice<'T>) : Slice<'T> = Slice.isLessThanEqual im1 im2
let isLessThan (im1: Slice<'T>) (im2: Slice<'T>) : Slice<'T> = Slice.isLessThan im1 im2
let sAnd (im1: Slice<'T>) (im2: Slice<'T>) : Slice<'T> = Slice.sAnd im1 im2
let sOr (im1: Slice<'T>) (im2: Slice<'T>) : Slice<'T> = Slice.sOr im1 im2
let sXor (im1: Slice<'T>) (im2: Slice<'T>) : Slice<'T> = Slice.sXor im1 im2

let sNot () = asPipe (sNotOp  "divInt")

let pow (im1: Slice<'T>) (im2: Slice<'T>) : Slice<'T> = Slice.pow im1 im2
// Add missing types

// Annoying F# value restriction requires explicit types here, sigh
let otsuThreshold<'T when 'T: equality> = asPipe ((otsuThresholdOp "otsuThreshold"))
let otsuMultiThreshold n = asPipe (otsuMultiThresholdOp "otsuMultiThreshold" n)
let momentsThreshold<'T when 'T: equality> = asPipe (momentsThresholdOp "momentsThreshold")
let signedDistanceMap = asPipe (signedDistanceMapOp "signedDistanceMap")
let watershed a = asPipe (watershedOp "watershed" a)
let threshold a b = asPipe (thresholdOp "threshold" a b)
let addNormalNoise a b = asPipe (addNormalNoiseOp "addNormalNoise" a b)
let relabelComponents a = asPipe (relabelComponentsOp "relabelComponents" a)

let histPipe = asPipe (histogramOp<float> "hist")

type ImageStats = ImageFunctions.ImageStats
let computeStats<'T when 'T: equality and 'T: comparison> = asPipe (computeStatsOp<'T> "stats")

let constantPad2D<'T when 'T : equality> padLower padUpper c = asPipe (constantPad2DOp "constantPad2D" padLower padUpper c)
