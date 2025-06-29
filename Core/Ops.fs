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

/// selected simple arithmatic operators
let addInt             scalar = asPipe (addIntOp     "addInt"     scalar)
let addUInt8           scalar = asPipe (addUInt8Op   "addUInt8"   scalar)
let addFloat           scalar = asPipe (addFloatOp   "addFloat"   scalar)
// Add missing types

let subInt             scalar = asPipe (subIntOp  "subInt"  scalar)
let subFloat           scalar = asPipe (subFloatOp "subFloat" scalar)
// Add missing types

let mulInt             scalar = asPipe (mulIntOp    "mulInt"    scalar)
let mulUInt8           scalar = asPipe (mulUInt8Op  "mulUInt8"  scalar)
let mulFloat           scalar = asPipe (mulFloatOp  "mulFloat"  scalar)
// Add missing types

let divInt             scalar = asPipe (divIntOp  "divInt"  scalar)
let divFloat           scalar = asPipe (divFloatOp "divFloat" scalar)
// Add missing types

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

let modulus (im1: Slice<'T>) (im2: Slice<'T>) : Slice<'T> = Slice.modulus im1 im2
// Add missing types
let pow (im1: Slice<'T>) (im2: Slice<'T>) : Slice<'T> = Slice.pow im1 im2
// Add missing types

let otsuThreshold = asPipe ((otsuThresholdOp "otsu"))
let otsuMultiThreshold n = asPipe (otsuMultiThresholdOp "otsuMulti" n)
let momentsThreshold = asPipe (momentsThresholdOp "moments")

