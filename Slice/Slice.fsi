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
val create:
  width: uint -> height: uint -> depth: uint -> idx: uint -> Slice<'T>
    when 'T: equality
val GetDepth: slice: Slice<'T> -> uint32 when 'T: equality
val GetDimension: slice: Slice<'T> -> uint32 when 'T: equality
val GetHeight: slice: Slice<'T> -> uint32 when 'T: equality
val GetWidth: slice: Slice<'T> -> uint32 when 'T: equality
val GetSize: slice: Slice<'T> -> uint list when 'T: equality
val ToString: slice: Slice<'T> -> string when 'T: equality
val toArray2D: slice: Slice<'T> -> 'T array2d when 'T: equality
val toArray3D: slice: Slice<'T> -> 'T array3d when 'T: equality
val toArray4D: slice: Slice<'T> -> 'T array4d when 'T: equality
val cast: slice: Slice<obj> -> Image.Image<'S> when 'S: equality
val toFloat: value: obj -> float
val toSeqSeq: slice: Slice<'T> -> float seq seq when 'T: equality
val private liftUnary:
  f: (Image.Image<'T> -> Image.Image<'T>) -> s: Slice<'T> -> Slice<'T>
    when 'T: equality
val private liftUnary1:
  f: ('a -> Image.Image<'T> -> Image.Image<'T>) ->
    a: 'a -> s: Slice<'T> -> Slice<'T> when 'T: equality
val private liftUnary2:
  f: ('a -> 'b -> Image.Image<'T> -> Image.Image<'T>) ->
    a: 'a -> b: 'b -> s: Slice<'T> -> Slice<'T> when 'T: equality
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
val convolve:
  a: ImageFunctions.OutputRegionMode ->
    b: ImageFunctions.BoundaryCondition ->
    c: bool -> s: Slice<'T> -> t: Slice<'T> -> Slice<'T> when 'T: equality
val conv: s: Slice<'T> -> t: Slice<'T> -> Slice<'T> when 'T: equality
val discreteGaussian: a: float -> s: Slice<'T> -> Slice<'T> when 'T: equality
val recursiveGaussian:
  a: float -> b: uint -> s: Slice<'T> -> Slice<'T> when 'T: equality
val laplacianConvolve: a: float -> s: Slice<'T> -> Slice<'T> when 'T: equality
val gradientConvolve:
  a: 'a -> b: 'b -> s: Slice<'T> -> (uint -> uint32 -> Slice<'c> -> Slice<'c>)
    when 'T: equality and 'c: equality
val binaryErode:
  a: uint -> b: float -> s: Slice<'T> -> Slice<'T> when 'T: equality
val binaryDilate:
  a: uint -> b: float -> s: Slice<'T> -> Slice<'T> when 'T: equality
val binaryOpening:
  a: uint -> b: float -> s: Slice<'T> -> Slice<'T> when 'T: equality
val binaryClosing:
  a: uint -> b: float -> s: Slice<'T> -> Slice<'T> when 'T: equality
val binaryFillHoles: a: float -> s: Slice<'T> -> Slice<'T> when 'T: equality
val squeeze: s: Slice<'T> -> Slice<'T> when 'T: equality
val concatAlong:
  a: uint -> s: Slice<'T> -> t: Slice<'T> -> Slice<'T> when 'T: equality
val connectedComponents: s: Slice<'T> -> Slice<'T> when 'T: equality
val relabelComponents: a: uint -> s: Slice<'T> -> Slice<'T> when 'T: equality
val watershed: a: float -> s: Slice<'T> -> Slice<'T> when 'T: equality
val otsuThreshold: s: Slice<'T> -> Slice<'T> when 'T: equality
val otsuMultiThreshold: a: byte -> s: Slice<'T> -> Slice<'T> when 'T: equality
val momentsThreshold: s: Slice<'T> -> Slice<'T> when 'T: equality
val signedDistanceMap:
  inside: uint8 -> outside: uint8 -> img: Slice<uint8> -> Slice<float>
val generateCoordinateAxis: axis: int -> size: int list -> Slice<uint32>
val unique: img: Slice<'T> -> 'T list when 'T: comparison
val labelShapeStatistics:
  img: Slice<'T> -> Map<int64,ImageFunctions.LabelShapeStatistics>
    when 'T: equality
type ImageStats = ImageFunctions.ImageStats
val computeStats: img: Slice<'T> -> ImageStats when 'T: equality
val addComputeStats: s1: ImageStats -> s2: ImageStats -> ImageStats
val histogram: img: Slice<'T> -> Map<'T,uint64> when 'T: comparison
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
val addInt: a: Slice<int> -> b: int -> Slice<int>
val addUInt8: a: Slice<uint8> -> b: uint8 -> Slice<uint8>
val addFloat: a: Slice<float> -> b: float -> Slice<float>
val sub: a: Slice<'T> -> b: Slice<'T> -> Slice<'T> when 'T: equality
val subInt: a: Slice<int> -> b: int -> Slice<int>
val subFloat: a: Slice<float> -> b: float -> Slice<float>
val mul: a: Slice<'T> -> b: Slice<'T> -> Slice<'T> when 'T: equality
val mulInt: a: Slice<int> -> b: int -> Slice<int>
val mulUInt8: a: Slice<uint8> -> b: uint8 -> Slice<uint8>
val mulFloat: a: Slice<float> -> b: float -> Slice<float>
val div: a: Slice<'T> -> b: Slice<'T> -> Slice<'T> when 'T: equality
val divInt: a: Slice<int> -> b: int -> Slice<int>
val divFloat: a: Slice<float> -> b: float -> Slice<float>
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
val stack: slices: Slice<'T> list -> Slice<'T> when 'T: equality
val extractSlice: a: uint -> s: Slice<'T> -> Slice<'T> when 'T: equality
val getDepth: inputDir: string -> suffix: string -> uint
type FileInfo = ImageFunctions.FileInfo
val getFileInfo: fname: string -> FileInfo
val getVolumeSize: inputDir: string -> suffix: string -> uint * uint * uint
val readSlice: idx: uint -> filename: string -> Slice<'T> when 'T: equality
val writeSlice: filename: string -> slice: Slice<'T> -> unit when 'T: equality
