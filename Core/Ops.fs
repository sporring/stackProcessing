module Ops 
open Processing
open Core.Helpers

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

let binaryErode      r winSz = asPipe (binaryErodeOp   "binaryErode"   r winSz)
let erode            r       = asPipe (binaryErodeOp   "binaryErode"   r None)
let binaryDilate     r winSz = asPipe (binaryDilateOp  "binaryDilate"  r winSz)
let dilate           r       = asPipe (binaryDilateOp  "binaryDilate"  r None)
let binaryOpening    r winSz = asPipe (binaryOpeningOp "binaryOpening" r winSz)
let opening          r       = asPipe (binaryOpeningOp "binaryOpening" r None)
let binaryClosing    r winSz = asPipe (binaryClosingOp "binaryClosing" r winSz)
let closing          r       = asPipe (binaryClosingOp "binaryClosing" r None)
