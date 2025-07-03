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
val toSeqSeq: s: Slice<'T> -> float seq seq when 'T: equality
val castUInt8ToInt8: s: Slice<uint8> -> Slice<int8>
val castUInt8ToUInt16: s: Slice<uint8> -> Slice<uint16>
val castUInt8ToInt16: s: Slice<uint8> -> Slice<int16>
val castUInt8ToUInt: s: Slice<uint8> -> Slice<uint>
val castUInt8ToInt: s: Slice<uint8> -> Slice<int>
val castUInt8ToUInt64: s: Slice<uint8> -> Slice<uint64>
val castUInt8ToInt64: s: Slice<uint8> -> Slice<int64>
val castUInt8ToFloat32: s: Slice<uint8> -> Slice<float32>
val castUInt8ToFloat: s: Slice<uint8> -> Slice<float>
val castInt8ToUInt8: s: Slice<int8> -> Slice<uint8>
val castInt8ToUInt16: s: Slice<int8> -> Slice<uint16>
val castInt8ToInt16: s: Slice<int8> -> Slice<int16>
val castInt8ToUInt: s: Slice<int8> -> Slice<uint>
val castInt8ToInt: s: Slice<int8> -> Slice<int>
val castInt8ToUInt64: s: Slice<int8> -> Slice<uint64>
val castInt8ToInt64: s: Slice<int8> -> Slice<int64>
val castInt8ToFloat32: s: Slice<int8> -> Slice<float32>
val castInt8ToFloat: s: Slice<int8> -> Slice<float>
val castUInt16ToUInt8: s: Slice<uint16> -> Slice<uint8>
val castUInt16ToInt8: s: Slice<uint16> -> Slice<int8>
val castUInt16ToInt16: s: Slice<uint16> -> Slice<int16>
val castUInt16ToUInt: s: Slice<uint16> -> Slice<uint>
val castUInt16ToInt: s: Slice<uint16> -> Slice<int>
val castUInt16ToUInt64: s: Slice<uint16> -> Slice<uint64>
val castUInt16ToInt64: s: Slice<uint16> -> Slice<int64>
val castUInt16ToFloat32: s: Slice<uint16> -> Slice<float32>
val castUInt16ToFloat: s: Slice<uint16> -> Slice<float>
val castInt16ToUInt8: s: Slice<int16> -> Slice<uint8>
val castInt16ToInt8: s: Slice<int16> -> Slice<int8>
val castInt16ToUInt16: s: Slice<int16> -> Slice<uint16>
val castInt16ToUInt: s: Slice<int16> -> Slice<uint>
val castInt16ToInt: s: Slice<int16> -> Slice<int>
val castInt16ToUInt64: s: Slice<int16> -> Slice<uint64>
val castInt16ToInt64: s: Slice<int16> -> Slice<int64>
val castInt16ToFloat32: s: Slice<int16> -> Slice<float32>
val castInt16ToFloat: s: Slice<int16> -> Slice<float>
val castUIntToUInt8: s: Slice<uint> -> Slice<uint8>
val castUIntToInt8: s: Slice<uint> -> Slice<int8>
val castUIntToUInt16: s: Slice<uint> -> Slice<uint16>
val castUIntToInt16: s: Slice<uint> -> Slice<int16>
val castUIntToInt: s: Slice<uint> -> Slice<int>
val castUIntToUInt64: s: Slice<uint> -> Slice<uint64>
val castUIntToInt64: s: Slice<uint> -> Slice<int64>
val castUIntToFloat32: s: Slice<uint> -> Slice<float32>
val castUIntToFloat: s: Slice<uint> -> Slice<float>
val castIntToUInt8: s: Slice<int> -> Slice<uint8>
val castIntToInt8: s: Slice<int> -> Slice<int8>
val castIntToUInt16: s: Slice<int> -> Slice<uint16>
val castIntToInt16: s: Slice<int> -> Slice<int16>
val castIntToUInt: s: Slice<int> -> Slice<uint>
val castIntToUInt64: s: Slice<int> -> Slice<uint64>
val castIntToInt64: s: Slice<int> -> Slice<int64>
val castIntToFloat32: s: Slice<int> -> Slice<float32>
val castIntToFloat: s: Slice<int> -> Slice<float>
val castFloat32ToUInt8: s: Slice<float32> -> Slice<uint8>
val castFloat32ToInt8: s: Slice<float32> -> Slice<int8>
val castFloat32ToUInt16: s: Slice<float32> -> Slice<uint16>
val castFloat32ToInt16: s: Slice<float32> -> Slice<int16>
val castFloat32ToUInt: s: Slice<float32> -> Slice<uint>
val castFloat32ToInt: s: Slice<float32> -> Slice<int>
val castFloat32ToUInt64: s: Slice<float32> -> Slice<uint64>
val castFloat32ToInt64: s: Slice<float32> -> Slice<int64>
val castFloat32Tofloat: s: Slice<float32> -> Slice<float>
val castFloatToUInt8: s: Slice<float> -> Slice<uint8>
val castFloatToInt8: s: Slice<float> -> Slice<int8>
val castFloatToUInt16: s: Slice<float> -> Slice<uint16>
val castFloatToInt16: s: Slice<float> -> Slice<int16>
val castFloatToUInt: s: Slice<float> -> Slice<uint>
val castFloatToInt: s: Slice<float> -> Slice<int>
val castFloatToUIn64: s: Slice<float> -> Slice<uint64>
val castFloatToInt64: s: Slice<float> -> Slice<int64>
val castFloatToFloat32: s: Slice<float> -> Slice<float32>
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
val private liftUnary5:
  f: ('a -> 'b -> 'c -> 'd -> 'e -> Image.Image<'T> -> Image.Image<'T>) ->
    a: 'a -> b: 'b -> c: 'c -> d: 'd -> e: 'e -> s: Slice<'T> -> Slice<'T>
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
  a: ImageFunctions.OutputRegionMode option ->
    b: ImageFunctions.BoundaryCondition option ->
    s: Slice<'T> -> t: Slice<'T> -> Slice<'T> when 'T: equality
val conv: s: Slice<'T> -> t: Slice<'T> -> Slice<'T> when 'T: equality
val discreteGaussian:
  a: uint ->
    b: float ->
    c: uint option ->
    d: ImageFunctions.OutputRegionMode option ->
    e: ImageFunctions.BoundaryCondition option -> s: Slice<'T> -> Slice<'T>
    when 'T: equality
val gauss: dim: uint -> sigma: float -> kernelSize: uint option -> Slice<float>
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
val generateCoordinateAxis: axis: int -> size: int list -> Slice<uint>
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
val getStackSize: inputDir: string -> suffix: string -> uint * uint * uint
val getStackWidth: inputDir: string -> suffix: string -> uint64
val getStackHeight: inputDir: string -> suffix: string -> uint64
