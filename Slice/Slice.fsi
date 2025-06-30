namespace FSharp
module Slice
/// <summary>
/// Represents a slice of a stack of 2d images. 
/// </summary>
type Slice<'T when 'T: equality> =
    {
      Index: uint
      Image: Image.Image<'T>
    }
    member EstimateUsage: unit -> uint
val create:
  width: uint -> height: uint -> depth: uint -> idx: uint -> Slice<'T>
    when 'T: equality
val GetDepth: s: Slice<'T> -> uint32 when 'T: equality
val GetDimensions: s: Slice<'T> -> uint32 when 'T: equality
val GetHeight: s: Slice<'T> -> uint32 when 'T: equality
val GetWidth: s: Slice<'T> -> uint32 when 'T: equality
val GetSize: s: Slice<'T> -> uint list when 'T: equality
val ToString: s: Slice<'T> -> string when 'T: equality
val toArray2D: s: Slice<'T> -> 'T array2d when 'T: equality
val toArray3D: s: Slice<'T> -> 'T array3d when 'T: equality
val toArray4D: s: Slice<'T> -> 'T array4d when 'T: equality
val toImage: s: Slice<'T> -> Image.Image<'T> when 'T: equality
val toSimpleITK: s: Slice<'T> -> itk.simple.Image when 'T: equality
val cast<'T when 'T: equality> : s: Slice<obj> -> Slice<obj> when 'T: equality
val castUInt8ToFloat: s: Slice<uint8> -> Slice<float>
val castFloatToUInt8: s: Slice<float> -> Slice<uint8>
val toFloat: value: obj -> float
val toSeqSeq: s: Slice<'T> -> float seq seq when 'T: equality
val private liftSource:
  f: (unit -> Image.Image<'T>) -> unit -> Slice<'T> when 'T: equality
val private liftSource1:
  f: ('a -> Image.Image<'T>) -> a: 'a -> Slice<'T> when 'T: equality
val private liftSource2:
  f: ('a -> 'b -> Image.Image<'T>) -> a: 'a -> b: 'b -> Slice<'T>
    when 'T: equality
val private liftSource3:
  f: ('a -> 'b -> 'c -> Image.Image<'T>) -> a: 'a -> b: 'b -> c: 'c -> Slice<'T>
    when 'T: equality
val private liftUnary:
  f: (Image.Image<'T> -> Image.Image<'T>) -> s: Slice<'T> -> Slice<'T>
    when 'T: equality
val private liftUnary1:
  f: ('a -> Image.Image<'T> -> Image.Image<'T>) ->
    a: 'a -> s: Slice<'T> -> Slice<'T> when 'T: equality
val private liftUnary2:
  f: ('a -> 'b -> Image.Image<'T> -> Image.Image<'T>) ->
    a: 'a -> b: 'b -> s: Slice<'T> -> Slice<'T> when 'T: equality
val private liftUnary3:
  f: ('a -> 'b -> 'c -> Image.Image<'T> -> Image.Image<'T>) ->
    a: 'a -> b: 'b -> c: 'c -> s: Slice<'T> -> Slice<'T> when 'T: equality
val private liftUnary4:
  f: ('a -> 'b -> 'c -> 'd -> Image.Image<'T> -> Image.Image<'T>) ->
    a: 'a -> b: 'b -> c: 'c -> d: 'd -> s: Slice<'T> -> Slice<'T>
    when 'T: equality
val private liftBinary:
  f: (Image.Image<'T> -> Image.Image<'T> -> Image.Image<'T>) ->
    s1: Slice<'T> -> s2: Slice<'T> -> Slice<'T> when 'T: equality
val private liftBinary1:
  f: ('a -> Image.Image<'T> -> Image.Image<'T> -> Image.Image<'T>) ->
    a: 'a -> s1: Slice<'T> -> s2: Slice<'T> -> Slice<'T> when 'T: equality
val private liftBinary2:
  f: ('a -> 'b -> Image.Image<'T> -> Image.Image<'T> -> Image.Image<'T>) ->
    a: 'a -> b: 'b -> s1: Slice<'T> -> s2: Slice<'T> -> Slice<'T>
    when 'T: equality
val private liftBinary3:
  f: ('a -> 'b -> 'c -> Image.Image<'T> -> Image.Image<'T> -> Image.Image<'T>) ->
    a: 'a -> b: 'b -> c: 'c -> s1: Slice<'T> -> s2: Slice<'T> -> Slice<'T>
    when 'T: equality
val private liftBinaryOp:
  f: (Image.Image<'T> * Image.Image<'T> -> Image.Image<'T>) ->
    s1: Slice<'T> * s2: Slice<'T> -> Slice<'T> when 'T: equality
val private liftBinaryOpInt:
  f: (Image.Image<int> * int -> Image.Image<int>) ->
    s1: Slice<int> * s2: int -> Slice<int>
val private liftBinaryOpUInt8:
  f: (Image.Image<uint8> * uint8 -> Image.Image<uint8>) ->
    s1: Slice<uint8> * s2: uint8 -> Slice<uint8>
val private liftBinaryOpFloat:
  f: (Image.Image<float> * float -> Image.Image<float>) ->
    s1: Slice<float> * s2: float -> Slice<float>
val private liftImageScalarOp:
  f: (Image.Image<'T> -> 'T -> Image.Image<'T>) ->
    s: Slice<'T> -> i: 'T -> Slice<'T> when 'T: equality
val private liftBinaryCmp:
  f: (Image.Image<'T> * Image.Image<'T> -> bool) ->
    s1: Slice<'T> * s2: Slice<'T> -> bool when 'T: equality
val absSlice: s: Slice<'T> -> Slice<'T> when 'T: equality
val logSlice: s: Slice<'T> -> Slice<'T> when 'T: equality
val log10Slice: s: Slice<'T> -> Slice<'T> when 'T: equality
val expSlice: s: Slice<'T> -> Slice<'T> when 'T: equality
val sqrtSlice: s: Slice<'T> -> Slice<'T> when 'T: equality
val squareSlice: s: Slice<'T> -> Slice<'T> when 'T: equality
val sinSlice: s: Slice<'T> -> Slice<'T> when 'T: equality
val cosSlice: s: Slice<'T> -> Slice<'T> when 'T: equality
val tanSlice: s: Slice<'T> -> Slice<'T> when 'T: equality
val asinSlice: s: Slice<'T> -> Slice<'T> when 'T: equality
val acosSlice: s: Slice<'T> -> Slice<'T> when 'T: equality
val atanSlice: s: Slice<'T> -> Slice<'T> when 'T: equality
val roundSlice: s: Slice<'T> -> Slice<'T> when 'T: equality
type BoundaryCondition = ImageFunctions.BoundaryCondition
type OutputRegionMode = ImageFunctions.OutputRegionMode
val convolve:
  a: ImageFunctions.BoundaryCondition option ->
    s: Slice<'T> -> t: Slice<'T> -> Slice<'T> when 'T: equality
val conv: s: Slice<'T> -> t: Slice<'T> -> Slice<'T> when 'T: equality
val discreteGaussian:
  a: float ->
    b: uint option ->
    c: ImageFunctions.BoundaryCondition option -> s: Slice<'T> -> Slice<'T>
    when 'T: equality
val gauss: dim: uint -> sigma: float -> kernelSize: uint option -> Slice<float>
val finiteDiffFilter1D: order: uint -> Slice<float>
val finiteDiffFilter2D: direction: uint -> order: uint -> Slice<float>
val finiteDiffFilter3D: direction: uint -> order: uint -> Slice<float>
val finiteDiffFilter4D: direction: uint -> order: uint -> Slice<float>
val gradientConvolve:
  a: 'a -> b: 'b -> s: Slice<'T> -> (uint -> uint32 -> Slice<'c> -> Slice<'c>)
    when 'T: equality and 'c: equality
val binaryErode: a: uint -> s: Slice<uint8> -> Slice<uint8>
val binaryDilate: a: uint -> s: Slice<uint8> -> Slice<uint8>
val binaryOpening: a: uint -> s: Slice<uint8> -> Slice<uint8>
val binaryClosing: a: uint -> s: Slice<uint8> -> Slice<uint8>
val binaryFillHoles: s: Slice<uint8> -> Slice<uint8>
val squeeze: s: Slice<'T> -> Slice<'T> when 'T: equality
val concatAlong:
  a: uint -> s: Slice<'T> -> t: Slice<'T> -> Slice<'T> when 'T: equality
val constantPad2D:
  a: uint list -> b: uint list -> c: double -> s: Slice<'T> -> Slice<'T>
    when 'T: equality
val connectedComponents: s: Slice<uint8> -> Slice<uint64>
val relabelComponents: a: uint -> s: Slice<uint64> -> Slice<uint64>
val watershed: a: float -> s: Slice<'T> -> Slice<'T> when 'T: equality
val otsuThreshold: s: Slice<'T> -> Slice<'T> when 'T: equality
val otsuMultiThreshold: a: byte -> s: Slice<'T> -> Slice<'T> when 'T: equality
val momentsThreshold: s: Slice<'T> -> Slice<'T> when 'T: equality
val signedDistanceMap:
  inside: uint8 -> outside: uint8 -> s: Slice<uint8> -> Slice<float>
val generateCoordinateAxis: axis: int -> size: int list -> Slice<uint32>
val unique: s: Slice<'T> -> 'T list when 'T: comparison
val labelShapeStatistics:
  s: Slice<'T> -> Map<int64,ImageFunctions.LabelShapeStatistics>
    when 'T: equality
type ImageStats = ImageFunctions.ImageStats
val computeStats: s: Slice<'T> -> ImageStats when 'T: equality
val addComputeStats: s1: ImageStats -> s2: ImageStats -> ImageStats
val histogram: s: Slice<'T> -> Map<'T,uint64> when 'T: comparison
val addHistogram:
  h1: Map<'T,uint64> -> h2: Map<'T,uint64> -> Map<'T,uint64> when 'T: comparison
val map2pairs: map: Map<'T,'S> -> ('T * 'S) list when 'T: comparison
val inline pairs2floats:
  pairs: (^T * ^S) list -> (float * float) list
    when ^T: (static member op_Explicit: ^T -> float) and
         ^S: (static member op_Explicit: ^S -> float)
val inline pairs2ints:
  pairs: (^T * ^S) list -> (int * int) list
    when ^T: (static member op_Explicit: ^T -> int) and
         ^S: (static member op_Explicit: ^S -> int)
val swap: f: ('a -> 'b -> 'c) -> a: 'b -> b: 'a -> 'c
val add: a: Slice<'T> -> b: Slice<'T> -> Slice<'T> when 'T: equality
val inline sliceAddScalar:
  s: Slice<^T> -> i: ^T -> Slice<^T>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val inline scalarAddSlice:
  i: ^T -> s: Slice<^T> -> Slice<^T>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val sub: a: Slice<'T> -> b: Slice<'T> -> Slice<'T> when 'T: equality
val inline sliceSubScalar:
  s: Slice<^T> -> i: ^T -> Slice<^T>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val inline scalarSubSlice:
  i: ^T -> s: Slice<^T> -> Slice<^T>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val mul: a: Slice<'T> -> b: Slice<'T> -> Slice<'T> when 'T: equality
val inline sliceMulScalar:
  s: Slice<^T> -> i: ^T -> Slice<^T>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val inline scalarMulSlice:
  i: ^T -> s: Slice<^T> -> Slice<^T>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val div: a: Slice<'T> -> b: Slice<'T> -> Slice<'T> when 'T: equality
val inline sliceDivScalar:
  s: Slice<^T> -> i: ^T -> Slice<^T>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val inline scalarDivSlice:
  i: ^T -> s: Slice<^T> -> Slice<^T>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val modulus: a: Slice<'T> -> b: Slice<'T> -> Slice<'T> when 'T: equality
val pow: a: Slice<'T> -> b: Slice<'T> -> Slice<'T> when 'T: equality
val isGreaterEqual: a: Slice<'T> -> b: Slice<'T> -> Slice<'T> when 'T: equality
val gte: a: Slice<'T> -> b: Slice<'T> -> bool when 'T: equality
val isGreater: a: Slice<'T> -> b: Slice<'T> -> Slice<'T> when 'T: equality
val ge: a: Slice<'T> -> b: Slice<'T> -> bool when 'T: equality
val isEqual: a: Slice<'T> -> b: Slice<'T> -> Slice<'T> when 'T: equality
val eq: a: Slice<'T> -> b: Slice<'T> -> bool when 'T: equality
val isNotEqual: a: Slice<'T> -> b: Slice<'T> -> Slice<'T> when 'T: equality
val neq: a: Slice<'T> -> b: Slice<'T> -> bool when 'T: equality
val isLessThanEqual: a: Slice<'T> -> b: Slice<'T> -> Slice<'T> when 'T: equality
val lte: a: Slice<'T> -> b: Slice<'T> -> bool when 'T: equality
val isLessThan: a: Slice<'T> -> b: Slice<'T> -> Slice<'T> when 'T: equality
val lt: a: Slice<'T> -> b: Slice<'T> -> bool when 'T: equality
val sAnd: a: Slice<'T> -> b: Slice<'T> -> Slice<'T> when 'T: equality
val sOr: a: Slice<'T> -> b: Slice<'T> -> Slice<'T> when 'T: equality
val sXor: a: Slice<'T> -> b: Slice<'T> -> Slice<'T> when 'T: equality
val sNot: s: Slice<'T> -> Slice<'T> when 'T: equality
val addNormalNoise:
  a: float -> b: float -> s: Slice<'T> -> Slice<'T> when 'T: equality
val threshold:
  a: float -> b: float -> s: Slice<'T> -> Slice<'T> when 'T: equality
val stack: sLst: Slice<'T> list -> Slice<'T> when 'T: equality
val extractSlice: a: uint -> s: Slice<'T> -> Slice<'T> when 'T: equality
val unstack: s: Slice<'T> -> Slice<'T> list when 'T: equality
type FileInfo = ImageFunctions.FileInfo
val getFileInfo: filename: string -> FileInfo
val readSlice: idx: uint -> filename: string -> Slice<'T> when 'T: equality
val writeSlice: filename: string -> s: Slice<'T> -> unit when 'T: equality
val getStackDepth: inputDir: string -> suffix: string -> uint
val getStackInfo: inputDir: string -> suffix: string -> FileInfo
val getStackSize: inputDir: string -> suffix: string -> uint64 list
val getStackWidth: inputDir: string -> suffix: string -> uint64
val getStackHeight: inputDir: string -> suffix: string -> uint64
