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
val updateId: id: uint -> s: Slice<'S> -> Slice<'S> when 'S: equality
val cast: s: Slice<'S> -> Slice<'T> when 'S: equality and 'T: equality
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
val castUInt64ToUInt8: s: Slice<uint64> -> Slice<uint8>
val castUInt64ToInt8: s: Slice<uint64> -> Slice<int8>
val castUInt64ToUInt16: s: Slice<uint64> -> Slice<uint16>
val castUInt64ToInt16: s: Slice<uint64> -> Slice<int16>
val castUInt64ToUInt: s: Slice<uint64> -> Slice<uint>
val castUInt64ToInt: s: Slice<uint64> -> Slice<int>
val castUInt64ToInt64: s: Slice<uint64> -> Slice<int64>
val castUInt64ToFloat32: s: Slice<uint64> -> Slice<float32>
val castUInt64ToFloat: s: Slice<uint64> -> Slice<float>
val castInt64ToUInt8: s: Slice<int64> -> Slice<uint8>
val castInt64ToInt8: s: Slice<int64> -> Slice<int8>
val castInt64ToUInt16: s: Slice<int64> -> Slice<uint16>
val castInt64ToInt16: s: Slice<int64> -> Slice<int16>
val castInt64ToUInt: s: Slice<int64> -> Slice<uint>
val castInt64ToInt: s: Slice<int64> -> Slice<int>
val castInt64ToUInt64: s: Slice<int64> -> Slice<uint64>
val castInt64ToFloat32: s: Slice<int64> -> Slice<float32>
val castInt64ToFloat: s: Slice<int64> -> Slice<float>
val castFloat32ToUInt8: s: Slice<float32> -> Slice<uint8>
val castFloat32ToInt8: s: Slice<float32> -> Slice<int8>
val castFloat32ToUInt16: s: Slice<float32> -> Slice<uint16>
val castFloat32ToInt16: s: Slice<float32> -> Slice<int16>
val castFloat32ToUInt: s: Slice<float32> -> Slice<uint>
val castFloat32ToInt: s: Slice<float32> -> Slice<int>
val castFloat32ToUInt64: s: Slice<float32> -> Slice<uint64>
val castFloat32ToInt64: s: Slice<float32> -> Slice<int64>
val castFloat32ToFloat: s: Slice<float32> -> Slice<float>
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
module Processing
module internal InternalHelpers =
    val plotListAsync:
      plt: (float list -> float list -> unit) ->
        vectorSeq: FSharp.Control.AsyncSeq<(float * float) list> -> Async<unit>
    val showSliceAsync:
      plt: (Slice.Slice<'T> -> unit) ->
        slices: FSharp.Control.AsyncSeq<Slice.Slice<'T>> -> Async<unit>
        when 'T: equality
    val printAsync: slices: FSharp.Control.AsyncSeq<'T> -> Async<unit>
    val writeSlicesAsync:
      outputDir: string ->
        suffix: string ->
        slices: FSharp.Control.AsyncSeq<Slice.Slice<'T>> -> Async<unit>
        when 'T: equality
/// Source parts
val writeOp:
  path: string ->
    suffix: string -> SlimPipeline.Stage<Slice.Slice<'a>,unit,'Shape>
    when 'a: equality
val showOp:
  plt: (Slice.Slice<'T> -> unit) ->
    SlimPipeline.Stage<Slice.Slice<'T>,unit,'Shape> when 'T: equality
val plotOp:
  plt: (float list -> float list -> unit) ->
    SlimPipeline.Stage<(float * float) list,unit,'Shape>
val printOp: unit -> SlimPipeline.Stage<'T,unit,'Shape>
val liftImageSource:
  name: string ->
    img: Slice.Slice<'T> -> SlimPipeline.Pipe<unit,Slice.Slice<'T>>
    when 'T: equality
val axisSourceOp:
  axis: int ->
    size: int list ->
    pl: SlimPipeline.Pipeline<unit,unit,uint list> ->
    SlimPipeline.Pipeline<unit,Slice.Slice<uint>,uint list>
val finiteDiffFilter3DOp:
  direction: uint ->
    order: uint ->
    pl: SlimPipeline.Pipeline<unit,unit,uint list> ->
    SlimPipeline.Pipeline<unit,Slice.Slice<float>,uint list>
val skipNTakeM: n: uint -> m: uint -> lst: 'a list -> 'a list
val internal liftWindowedOp:
  name: string ->
    window: uint ->
    pad: uint ->
    zeroMaker: (Slice.Slice<'S> -> Slice.Slice<'S>) ->
    stride: uint ->
    emitStart: uint ->
    emitCount: uint ->
    f: (Slice.Slice<'S> -> Slice.Slice<'T>) ->
    SlimPipeline.Stage<Slice.Slice<'S>,Slice.Slice<'T>,'Shape>
    when 'S: equality and 'T: equality
val internal liftWindowedTrimOp:
  name: string ->
    window: uint ->
    pad: uint ->
    zeroMaker: (Slice.Slice<'S> -> Slice.Slice<'S>) ->
    stride: uint ->
    emitStart: uint ->
    emitCount: uint ->
    trim: uint ->
    f: (Slice.Slice<'S> -> Slice.Slice<'T>) ->
    SlimPipeline.Stage<Slice.Slice<'S>,Slice.Slice<'T>,'Shape>
    when 'S: equality and 'T: equality
/// quick constructor for Streaming→Streaming unary ops
val internal liftUnaryOpInt:
  name: string ->
    f: (Slice.Slice<int> -> Slice.Slice<int>) ->
    SlimPipeline.Stage<Slice.Slice<int>,Slice.Slice<int>,'Shape>
val internal liftUnaryOpFloat32:
  name: string ->
    f: (Slice.Slice<float32> -> Slice.Slice<float32>) ->
    SlimPipeline.Stage<Slice.Slice<float32>,Slice.Slice<float32>,'Shape>
val internal liftUnaryOpFloat:
  name: string ->
    f: (Slice.Slice<float> -> Slice.Slice<float>) ->
    SlimPipeline.Stage<Slice.Slice<float>,Slice.Slice<float>,'Shape>
val internal liftBinaryOp:
  name: string ->
    f: (Slice.Slice<'T> -> Slice.Slice<'T> -> Slice.Slice<'T>) ->
    SlimPipeline.Stage<(Slice.Slice<'T> * Slice.Slice<'T>),Slice.Slice<'T>,
                       'Shape> when 'T: equality
val internal liftBinaryOpFloat:
  name: string ->
    f: (Slice.Slice<float> -> Slice.Slice<float> -> Slice.Slice<float>) ->
    SlimPipeline.Stage<(Slice.Slice<float> * Slice.Slice<float>),
                       Slice.Slice<float>,'Shape>
val internal liftFullOp:
  name: string ->
    f: (Slice.Slice<'T> -> Slice.Slice<'T>) ->
    SlimPipeline.Stage<Slice.Slice<'T>,Slice.Slice<'T>,'Shape> when 'T: equality
val internal liftFullParamOp:
  name: string ->
    f: ('P -> Slice.Slice<'T> -> Slice.Slice<'T>) ->
    param: 'P -> SlimPipeline.Stage<Slice.Slice<'T>,Slice.Slice<'T>,'Shape>
    when 'T: equality
val internal liftFullParam2Op:
  name: string ->
    f: ('P -> 'Q -> Slice.Slice<'T> -> Slice.Slice<'T>) ->
    param1: 'P ->
    param2: 'Q -> SlimPipeline.Stage<Slice.Slice<'T>,Slice.Slice<'T>,'Shape>
    when 'T: equality
val inline castOp:
  name: string ->
    f: (Slice.Slice<'S> -> Slice.Slice<'T>) ->
    SlimPipeline.Stage<Slice.Slice<'S>,Slice.Slice<'T>,'Shape>
    when 'S: equality and 'T: equality
val castUInt8ToInt8Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<uint8>,Slice.Slice<int8>,'a>
val castUInt8ToUInt16Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<uint8>,Slice.Slice<uint16>,'a>
val castUInt8ToInt16Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<uint8>,Slice.Slice<int16>,'a>
val castUInt8ToUIntOp:
  name: string -> SlimPipeline.Stage<Slice.Slice<uint8>,Slice.Slice<uint>,'a>
val castUInt8ToIntOp:
  name: string -> SlimPipeline.Stage<Slice.Slice<uint8>,Slice.Slice<int>,'a>
val castUInt8ToUInt64Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<uint8>,Slice.Slice<uint64>,'a>
val castUInt8ToInt64Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<uint8>,Slice.Slice<int64>,'a>
val castUInt8ToFloat32Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<uint8>,Slice.Slice<float32>,'a>
val castUInt8ToFloatOp:
  name: string -> SlimPipeline.Stage<Slice.Slice<uint8>,Slice.Slice<float>,'a>
val castInt8ToUInt8Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<int8>,Slice.Slice<uint8>,'a>
val castInt8ToUInt16Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<int8>,Slice.Slice<uint16>,'a>
val castInt8ToInt16Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<int8>,Slice.Slice<int16>,'a>
val castInt8ToUIntOp:
  name: string -> SlimPipeline.Stage<Slice.Slice<int8>,Slice.Slice<uint>,'a>
val castInt8ToIntOp:
  name: string -> SlimPipeline.Stage<Slice.Slice<int8>,Slice.Slice<int>,'a>
val castInt8ToUInt64Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<int8>,Slice.Slice<uint64>,'a>
val castInt8ToInt64Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<int8>,Slice.Slice<int64>,'a>
val castInt8ToFloat32Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<int8>,Slice.Slice<float32>,'a>
val castInt8ToFloatOp:
  name: string -> SlimPipeline.Stage<Slice.Slice<int8>,Slice.Slice<float>,'a>
val castUInt16ToUInt8Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<uint16>,Slice.Slice<uint8>,'a>
val castUInt16ToInt8Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<uint16>,Slice.Slice<int8>,'a>
val castUInt16ToInt16Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<uint16>,Slice.Slice<int16>,'a>
val castUInt16ToUIntOp:
  name: string -> SlimPipeline.Stage<Slice.Slice<uint16>,Slice.Slice<uint>,'a>
val castUInt16ToIntOp:
  name: string -> SlimPipeline.Stage<Slice.Slice<uint16>,Slice.Slice<int>,'a>
val castUInt16ToUInt64Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<uint16>,Slice.Slice<uint64>,'a>
val castUInt16ToInt64Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<uint16>,Slice.Slice<int64>,'a>
val castUInt16ToFloat32Op:
  name: string ->
    SlimPipeline.Stage<Slice.Slice<uint16>,Slice.Slice<float32>,'a>
val castUInt16ToFloatOp:
  name: string -> SlimPipeline.Stage<Slice.Slice<uint16>,Slice.Slice<float>,'a>
val castInt16ToUInt8Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<int16>,Slice.Slice<uint8>,'a>
val castInt16ToInt8Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<int16>,Slice.Slice<int8>,'a>
val castInt16ToUInt16Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<int16>,Slice.Slice<uint16>,'a>
val castInt16ToUIntOp:
  name: string -> SlimPipeline.Stage<Slice.Slice<int16>,Slice.Slice<uint>,'a>
val castInt16ToIntOp:
  name: string -> SlimPipeline.Stage<Slice.Slice<int16>,Slice.Slice<int>,'a>
val castInt16ToUInt64Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<int16>,Slice.Slice<uint64>,'a>
val castInt16ToInt64Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<int16>,Slice.Slice<int64>,'a>
val castInt16ToFloat32Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<int16>,Slice.Slice<float32>,'a>
val castInt16ToFloatOp:
  name: string -> SlimPipeline.Stage<Slice.Slice<int16>,Slice.Slice<float>,'a>
val castUIntToUInt8Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<uint>,Slice.Slice<uint8>,'a>
val castUIntToInt8Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<uint>,Slice.Slice<int8>,'a>
val castUIntToUInt16Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<uint>,Slice.Slice<uint16>,'a>
val castUIntToInt16Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<uint>,Slice.Slice<int16>,'a>
val castUIntToIntOp:
  name: string -> SlimPipeline.Stage<Slice.Slice<uint>,Slice.Slice<int>,'a>
val castUIntToUInt64Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<uint>,Slice.Slice<uint64>,'a>
val castUIntToInt64Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<uint>,Slice.Slice<int64>,'a>
val castUIntToFloat32Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<uint>,Slice.Slice<float32>,'a>
val castUIntToFloatOp:
  name: string -> SlimPipeline.Stage<Slice.Slice<uint>,Slice.Slice<float>,'a>
val castIntToUInt8Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<int>,Slice.Slice<uint8>,'a>
val castIntToInt8Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<int>,Slice.Slice<int8>,'a>
val castIntToUInt16Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<int>,Slice.Slice<uint16>,'a>
val castIntToInt16Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<int>,Slice.Slice<int16>,'a>
val castIntToUIntOp:
  name: string -> SlimPipeline.Stage<Slice.Slice<int>,Slice.Slice<uint>,'a>
val castIntToUInt64Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<int>,Slice.Slice<uint64>,'a>
val castIntToInt64Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<int>,Slice.Slice<int64>,'a>
val castIntToFloat32Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<int>,Slice.Slice<float32>,'a>
val castIntToFloatOp:
  name: string -> SlimPipeline.Stage<Slice.Slice<int>,Slice.Slice<float>,'a>
val castUInt64ToUInt8Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<uint64>,Slice.Slice<uint8>,'a>
val castUInt64ToInt8Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<uint64>,Slice.Slice<int8>,'a>
val castUInt64ToUInt16Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<uint64>,Slice.Slice<uint16>,'a>
val castUInt64ToInt16Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<uint64>,Slice.Slice<int16>,'a>
val castUInt64ToUIntOp:
  name: string -> SlimPipeline.Stage<Slice.Slice<uint64>,Slice.Slice<uint>,'a>
val castUInt64ToIntOp:
  name: string -> SlimPipeline.Stage<Slice.Slice<uint64>,Slice.Slice<int>,'a>
val castUInt64ToInt64Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<uint64>,Slice.Slice<int64>,'a>
val castUInt64ToFloat32Op:
  name: string ->
    SlimPipeline.Stage<Slice.Slice<uint64>,Slice.Slice<float32>,'a>
val castUInt64ToFloatOp:
  name: string -> SlimPipeline.Stage<Slice.Slice<uint64>,Slice.Slice<float>,'a>
val castInt64ToUInt8Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<int64>,Slice.Slice<uint8>,'a>
val castInt64ToInt8Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<int64>,Slice.Slice<int8>,'a>
val castInt64ToUInt16Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<int64>,Slice.Slice<uint16>,'a>
val castInt64ToInt16Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<int64>,Slice.Slice<int16>,'a>
val castInt64ToUIntOp:
  name: string -> SlimPipeline.Stage<Slice.Slice<int64>,Slice.Slice<uint>,'a>
val castInt64ToIntOp:
  name: string -> SlimPipeline.Stage<Slice.Slice<int64>,Slice.Slice<int>,'a>
val castInt64ToUInt64Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<int64>,Slice.Slice<uint64>,'a>
val castInt64ToFloat32Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<int64>,Slice.Slice<float32>,'a>
val castInt64ToFloatOp:
  name: string -> SlimPipeline.Stage<Slice.Slice<int64>,Slice.Slice<float>,'a>
val castFloat32ToUInt8Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<float32>,Slice.Slice<uint8>,'a>
val castFloat32ToInt8Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<float32>,Slice.Slice<int8>,'a>
val castFloat32ToUInt16Op:
  name: string ->
    SlimPipeline.Stage<Slice.Slice<float32>,Slice.Slice<uint16>,'a>
val castFloat32ToInt16Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<float32>,Slice.Slice<int16>,'a>
val castFloat32ToUIntOp:
  name: string -> SlimPipeline.Stage<Slice.Slice<float32>,Slice.Slice<uint>,'a>
val castFloat32ToIntOp:
  name: string -> SlimPipeline.Stage<Slice.Slice<float32>,Slice.Slice<int>,'a>
val castFloat32ToUInt64Op:
  name: string ->
    SlimPipeline.Stage<Slice.Slice<float32>,Slice.Slice<uint64>,'a>
val castFloat32ToInt64Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<float32>,Slice.Slice<int64>,'a>
val castFloat32ToFloatOp:
  name: string -> SlimPipeline.Stage<Slice.Slice<float32>,Slice.Slice<float>,'a>
val castFloatToUInt8Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<float>,Slice.Slice<uint8>,'a>
val castFloatToInt8Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<float>,Slice.Slice<int8>,'a>
val castFloatToUInt16Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<float>,Slice.Slice<uint16>,'a>
val castFloatToInt16Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<float>,Slice.Slice<int16>,'a>
val castFloatToUIntOp:
  name: string -> SlimPipeline.Stage<Slice.Slice<float>,Slice.Slice<uint>,'a>
val castFloatToIntOp:
  name: string -> SlimPipeline.Stage<Slice.Slice<float>,Slice.Slice<int>,'a>
val castFloatToUIn64Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<float>,Slice.Slice<uint64>,'a>
val castFloatToInt64Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<float>,Slice.Slice<int64>,'a>
val castFloatToFloat32Op:
  name: string -> SlimPipeline.Stage<Slice.Slice<float>,Slice.Slice<float32>,'a>
/// Basic arithmetic
val addOp:
  name: string ->
    slice: Slice.Slice<'a> ->
    SlimPipeline.Stage<Slice.Slice<'a>,Slice.Slice<'a>,'b> when 'a: equality
val inline scalarAddSliceOp:
  name: string ->
    i: ^T -> SlimPipeline.Stage<Slice.Slice<^T>,Slice.Slice<^T>,obj>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val inline sliceAddScalarOp:
  name: string ->
    i: ^T -> SlimPipeline.Stage<Slice.Slice<^T>,Slice.Slice<^T>,obj>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val subOp:
  name: string ->
    slice: Slice.Slice<'a> ->
    SlimPipeline.Stage<Slice.Slice<'a>,Slice.Slice<'a>,'b> when 'a: equality
val inline scalarSubSliceOp:
  name: string ->
    i: ^T -> SlimPipeline.Stage<Slice.Slice<^T>,Slice.Slice<^T>,obj>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val inline sliceSubScalarOp:
  name: string ->
    i: ^T -> SlimPipeline.Stage<Slice.Slice<^T>,Slice.Slice<^T>,obj>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val mulOp:
  name: string ->
    slice: Slice.Slice<'a> ->
    SlimPipeline.Stage<Slice.Slice<'a>,Slice.Slice<'a>,'b> when 'a: equality
val inline scalarMulSliceOp:
  name: string ->
    i: ^T -> SlimPipeline.Stage<Slice.Slice<^T>,Slice.Slice<^T>,obj>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val inline sliceMulScalarOp:
  name: string ->
    i: ^T -> SlimPipeline.Stage<Slice.Slice<^T>,Slice.Slice<^T>,obj>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val divOp:
  name: string ->
    slice: Slice.Slice<'a> ->
    SlimPipeline.Stage<Slice.Slice<'a>,Slice.Slice<'a>,'b> when 'a: equality
val inline scalarDivSliceOp:
  name: string ->
    i: ^T -> SlimPipeline.Stage<Slice.Slice<^T>,Slice.Slice<^T>,obj>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val inline sliceDivScalarOp:
  name: string ->
    i: ^T -> SlimPipeline.Stage<Slice.Slice<^T>,Slice.Slice<^T>,obj>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
/// Histogram related functions
val histogramOp:
  name: string -> SlimPipeline.Stage<Slice.Slice<'T>,Map<'T,uint64>,'Shape>
    when 'T: comparison
val map2pairsOp:
  name: string -> SlimPipeline.Stage<Map<'T,'S>,('T * 'S) list,'Shape>
    when 'T: comparison
val inline pairs2floatsOp:
  name: string -> SlimPipeline.Stage<(^T * ^S) list,(float * float) list,^Shape>
    when ^T: (static member op_Explicit: ^T -> float) and
         ^S: (static member op_Explicit: ^S -> float)
val inline pairs2intsOp:
  name: string -> SlimPipeline.Stage<(^T * ^S) list,(int * int) list,^Shape>
    when ^T: (static member op_Explicit: ^T -> int) and
         ^S: (static member op_Explicit: ^S -> int)
type ImageStats = Slice.ImageStats
val computeStatsOp:
  name: string -> SlimPipeline.Stage<Slice.Slice<'T>,ImageStats,'Shape>
    when 'T: equality
/// Convolution like operators
val zeroPad: ImageFunctions.BoundaryCondition
val periodicPad: ImageFunctions.BoundaryCondition
val zeroFluxNeumannPad: ImageFunctions.BoundaryCondition
val valid: ImageFunctions.OutputRegionMode
val same: ImageFunctions.OutputRegionMode
val zeroMaker: ex: Slice.Slice<'S> -> Slice.Slice<'S> when 'S: equality
val discreteGaussianOp:
  name: string ->
    sigma: float ->
    outputRegionMode: Slice.OutputRegionMode option ->
    boundaryCondition: ImageFunctions.BoundaryCondition option ->
    winSz: uint option ->
    SlimPipeline.Stage<Slice.Slice<float>,Slice.Slice<float>,'Shape>
val convGaussOp:
  name: string ->
    sigma: float ->
    SlimPipeline.Stage<Slice.Slice<float>,Slice.Slice<float>,'Shape>
val convolveOp:
  name: string ->
    kernel: Slice.Slice<'T> ->
    outputRegionMode: Slice.OutputRegionMode option ->
    bc: Slice.BoundaryCondition option ->
    winSz: uint option ->
    SlimPipeline.Stage<Slice.Slice<'T>,Slice.Slice<'T>,'Shape> when 'T: equality
val convOp:
  name: string ->
    kernel: Slice.Slice<'a> ->
    (uint option -> SlimPipeline.Stage<Slice.Slice<'a>,Slice.Slice<'a>,'b>)
    when 'a: equality
val private makeMorphOp:
  name: string ->
    radius: uint ->
    winSz: uint option ->
    core: (uint -> Slice.Slice<'T> -> Slice.Slice<'T>) ->
    SlimPipeline.Stage<Slice.Slice<'T>,Slice.Slice<'T>,'Shape> when 'T: equality
val binaryErodeOp:
  name: string ->
    radius: uint ->
    winSz: uint option ->
    SlimPipeline.Stage<Slice.Slice<uint8>,Slice.Slice<uint8>,'a>
val binaryDilateOp:
  name: string ->
    radius: uint ->
    winSz: uint option ->
    SlimPipeline.Stage<Slice.Slice<uint8>,Slice.Slice<uint8>,'a>
val binaryOpeningOp:
  name: string ->
    radius: uint ->
    winSz: uint option ->
    SlimPipeline.Stage<Slice.Slice<uint8>,Slice.Slice<uint8>,'a>
val binaryClosingOp:
  name: string ->
    radius: uint ->
    winSz: uint option ->
    SlimPipeline.Stage<Slice.Slice<uint8>,Slice.Slice<uint8>,'a>
val binaryFillHolesOp<'Shape> :
  name: string -> SlimPipeline.Stage<Slice.Slice<uint8>,Slice.Slice<uint8>,obj>
val connectedComponentsOp:
  name: string ->
    SlimPipeline.Stage<Slice.Slice<uint8>,Slice.Slice<uint64>,'Shape>
val piecewiseConnectedComponentsOp:
  name: string ->
    windowSize: uint option ->
    SlimPipeline.Stage<Slice.Slice<uint8>,Slice.Slice<uint64>,'Shape>
val otsuThresholdOp:
  name: string -> SlimPipeline.Stage<Slice.Slice<'T>,Slice.Slice<'T>,'a>
    when 'T: equality
val otsuMultiThresholdOp:
  name: string ->
    n: byte -> SlimPipeline.Stage<Slice.Slice<'a>,Slice.Slice<'a>,'b>
    when 'a: equality
val momentsThresholdOp:
  name: string -> SlimPipeline.Stage<Slice.Slice<'a>,Slice.Slice<'a>,'b>
    when 'a: equality
val signedDistanceMapOp:
  name: string ->
    SlimPipeline.Stage<Slice.Slice<uint8>,Slice.Slice<float>,'Shape>
val watershedOp:
  name: string ->
    a: float -> SlimPipeline.Stage<Slice.Slice<'a>,Slice.Slice<'a>,'b>
    when 'a: equality
val thresholdOp:
  name: string ->
    a: float ->
    b: float -> SlimPipeline.Stage<Slice.Slice<'a>,Slice.Slice<'a>,'b>
    when 'a: equality
val addNormalNoiseOp:
  name: string ->
    a: float ->
    b: float -> SlimPipeline.Stage<Slice.Slice<'a>,Slice.Slice<'a>,'b>
    when 'a: equality
val relabelComponentsOp:
  name: string ->
    a: uint -> SlimPipeline.Stage<Slice.Slice<uint64>,Slice.Slice<uint64>,'a>
val constantPad2DOp<'T when 'T: equality> :
  name: string ->
    padLower: uint list ->
    padUpper: uint list ->
    c: double -> SlimPipeline.Stage<Slice.Slice<obj>,Slice.Slice<obj>,obj>
    when 'T: equality
type FileInfo = Slice.FileInfo
val getStackDepth: (string -> string -> uint)
val getStackInfo: (string -> string -> Slice.FileInfo)
val getStackSize: (string -> string -> uint * uint * uint)
val getStackWidth: (string -> string -> uint64)
val getStackHeight: (string -> string -> uint64)
module StackProcessing
type Stage<'S,'T,'Shape> = SlimPipeline.Stage<'S,'T,'Shape>
type MemoryProfile = SlimPipeline.MemoryProfile
type MemoryTransition = SlimPipeline.MemoryTransition
type Slice<'S when 'S: equality> = Slice.Slice<'S>
type Shape =
    | Zero
    | Slice of uint * uint
val shapeContext: SlimPipeline.ShapeContext<Shape>
val idOp: (unit -> SlimPipeline.Stage<'a,'a,'b>)
val (-->) :
  (SlimPipeline.Stage<'a,'b,'c> ->
     SlimPipeline.Stage<'b,'d,'c> -> SlimPipeline.Stage<'a,'d,'c>)
val source: (uint64 -> SlimPipeline.Pipeline<unit,unit,Shape>)
val sink: pl: SlimPipeline.Pipeline<unit,unit,'Shape> -> unit
val sinkList: plLst: SlimPipeline.Pipeline<unit,unit,'Shape> list -> unit
val (>=>) :
  (SlimPipeline.Pipeline<'a,'b,'c> ->
     SlimPipeline.Stage<'b,'d,'c> -> SlimPipeline.Pipeline<'a,'d,'c>)
val (>=>>) :
  (SlimPipeline.Pipeline<'a,'a,'b> ->
     SlimPipeline.Stage<'a,'a,'b> * SlimPipeline.Stage<'a,'c,'b> ->
       SlimPipeline.SharedPipeline<'a,'a,'c,'b>) when 'a: equality
val (>>=>) :
  (SlimPipeline.SharedPipeline<unit,'a,'b,'c> ->
     (SlimPipeline.Stage<unit,'a,'c> * SlimPipeline.Stage<unit,'b,'c> ->
        SlimPipeline.Stage<unit,'d,'c>) -> SlimPipeline.Pipeline<unit,'d,'c>)
    when 'd: equality
val combineIgnore:
  (SlimPipeline.Stage<'a,'b,'c> * SlimPipeline.Stage<'a,'d,'c> ->
     SlimPipeline.Stage<'a,unit,'c>)
val drainSingle: pl: SlimPipeline.Pipeline<'a,'b,'c> -> 'b
val drainList: pl: SlimPipeline.Pipeline<'a,'b,'c> -> 'b list
val drainLast: pl: SlimPipeline.Pipeline<'a,'b,'c> -> 'b
val tap: (string -> SlimPipeline.Stage<'a,'a,'b>)
val liftUnary:
  f: (Slice<'T> -> Slice<'T>) -> SlimPipeline.Stage<Slice<'T>,Slice<'T>,'a>
    when 'T: equality
val write:
  (string -> string -> SlimPipeline.Stage<Slice.Slice<'a>,unit,'b>)
    when 'a: equality
val print: (unit -> SlimPipeline.Stage<'a,unit,'b>)
val plot:
  ((float list -> float list -> unit) ->
     SlimPipeline.Stage<(float * float) list,unit,'a>)
val show:
  ((Slice.Slice<'a> -> unit) -> SlimPipeline.Stage<Slice.Slice<'a>,unit,'b>)
    when 'a: equality
val finiteDiffFilter3D:
  (uint ->
     uint ->
     SlimPipeline.Pipeline<unit,unit,uint list> ->
     SlimPipeline.Pipeline<unit,Slice.Slice<float>,uint list>)
val axisSource:
  (int ->
     int list ->
     SlimPipeline.Pipeline<unit,unit,uint list> ->
     SlimPipeline.Pipeline<unit,Slice.Slice<uint>,uint list>)
val castFloatToUInt8:
  SlimPipeline.Stage<Slice.Slice<float>,Slice.Slice<uint8>,Shape>
/// Basic arithmetic
val add:
  slice: Slice.Slice<'a> ->
    SlimPipeline.Stage<Slice.Slice<'a>,Slice.Slice<'a>,'b> when 'a: equality
val inline scalarAddSlice:
  i: ^T -> SlimPipeline.Stage<Slice.Slice<^T>,Slice.Slice<^T>,obj>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val inline sliceAddScalar:
  i: ^T -> SlimPipeline.Stage<Slice.Slice<^T>,Slice.Slice<^T>,obj>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val sub:
  slice: Slice.Slice<'a> ->
    SlimPipeline.Stage<Slice.Slice<'a>,Slice.Slice<'a>,'b> when 'a: equality
val inline scalarSubSlice:
  i: ^T -> SlimPipeline.Stage<Slice.Slice<^T>,Slice.Slice<^T>,obj>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val inline sliceSubScalar:
  i: ^T -> SlimPipeline.Stage<Slice.Slice<^T>,Slice.Slice<^T>,obj>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val mul:
  slice: Slice.Slice<'a> ->
    SlimPipeline.Stage<Slice.Slice<'a>,Slice.Slice<'a>,'b> when 'a: equality
val inline scalarMulSlice:
  i: ^T -> SlimPipeline.Stage<Slice.Slice<^T>,Slice.Slice<^T>,obj>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val inline sliceMulScalar:
  i: ^T -> SlimPipeline.Stage<Slice.Slice<^T>,Slice.Slice<^T>,obj>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val div:
  slice: Slice.Slice<'a> ->
    SlimPipeline.Stage<Slice.Slice<'a>,Slice.Slice<'a>,'b> when 'a: equality
val inline scalarDivSlice:
  i: ^T -> SlimPipeline.Stage<Slice.Slice<^T>,Slice.Slice<^T>,obj>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
val inline sliceDivScalar:
  i: ^T -> SlimPipeline.Stage<Slice.Slice<^T>,Slice.Slice<^T>,obj>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)
/// Simple functions
val absFloat: Stage<Slice<float>,Slice<float>,Shape>
val absFloat32: Stage<Slice<float32>,Slice<float32>,Shape>
val absInt: Stage<Slice<int>,Slice<int>,Shape>
val acosFloat: Stage<Slice<float>,Slice<float>,Shape>
val acosFloat32: Stage<Slice<float32>,Slice<float32>,Shape>
val asinFloat: Stage<Slice<float>,Slice<float>,Shape>
val asinFloat32: Stage<Slice<float32>,Slice<float32>,Shape>
val atanFloat: Stage<Slice<float>,Slice<float>,Shape>
val atanFloat32: Stage<Slice<float32>,Slice<float32>,Shape>
val cosFloat: Stage<Slice<float>,Slice<float>,Shape>
val cosFloat32: Stage<Slice<float32>,Slice<float32>,Shape>
val sinFloat: Stage<Slice<float>,Slice<float>,Shape>
val sinFloat32: Stage<Slice<float32>,Slice<float32>,Shape>
val tanFloat: Stage<Slice<float>,Slice<float>,Shape>
val tanFloat32: Stage<Slice<float32>,Slice<float32>,Shape>
val expFloat: Stage<Slice<float>,Slice<float>,Shape>
val expFloat32: Stage<Slice<float32>,Slice<float32>,Shape>
val log10Float: Stage<Slice<float>,Slice<float>,Shape>
val log10Float32: Stage<Slice<float32>,Slice<float32>,Shape>
val logFloat: Stage<Slice<float>,Slice<float>,Shape>
val logFloat32: Stage<Slice<float32>,Slice<float32>,Shape>
val roundFloat: Stage<Slice<float>,Slice<float>,Shape>
val roundFloat32: Stage<Slice<float32>,Slice<float32>,Shape>
val sqrtFloat: Stage<Slice<float>,Slice<float>,Shape>
val sqrtFloat32: Stage<Slice<float32>,Slice<float32>,Shape>
val sqrtInt: Stage<Slice<int>,Slice<int>,Shape>
val squareFloat: Stage<Slice<float>,Slice<float>,Shape>
val squareFloat32: Stage<Slice<float32>,Slice<float32>,Shape>
val squareInt: Stage<Slice<int>,Slice<int>,Shape>
val histogram<'T when 'T: comparison> :
  SlimPipeline.Stage<Slice.Slice<'T>,Map<'T,uint64>,Shape> when 'T: comparison
val inline map2pairs<^T,^S
                       when ^T: comparison and
                            ^T: (static member op_Explicit: ^T -> float) and
                            ^S: (static member op_Explicit: ^S -> float)> :
  SlimPipeline.Stage<Map<^T,^S>,(^T * ^S) list,Shape>
    when ^T: comparison and ^T: (static member op_Explicit: ^T -> float) and
         ^S: (static member op_Explicit: ^S -> float)
val inline pairs2floats<^T,^S
                          when ^T: (static member op_Explicit: ^T -> float) and
                               ^S: (static member op_Explicit: ^S -> float)> :
  SlimPipeline.Stage<(^T * ^S) list,(float * float) list,Shape>
    when ^T: (static member op_Explicit: ^T -> float) and
         ^S: (static member op_Explicit: ^S -> float)
val inline pairs2ints<^T,^S
                        when ^T: (static member op_Explicit: ^T -> int) and
                             ^S: (static member op_Explicit: ^S -> int)> :
  SlimPipeline.Stage<(^T * ^S) list,(int * int) list,Shape>
    when ^T: (static member op_Explicit: ^T -> int) and
         ^S: (static member op_Explicit: ^S -> int)
type ImageStats = ImageFunctions.ImageStats
val computeStats<'T when 'T: comparison> :
  SlimPipeline.Stage<Slice.Slice<'T>,Processing.ImageStats,Shape>
    when 'T: comparison
/// Convolution like operators
val zeroPad: ImageFunctions.BoundaryCondition
val periodicPad: ImageFunctions.BoundaryCondition
val zeroFluxNeumannPad: ImageFunctions.BoundaryCondition
val valid: ImageFunctions.OutputRegionMode
val same: ImageFunctions.OutputRegionMode
val discreteGaussian:
  (float ->
     Slice.OutputRegionMode option ->
     ImageFunctions.BoundaryCondition option ->
     uint option ->
     SlimPipeline.Stage<Slice.Slice<float>,Slice.Slice<float>,Shape>)
val convGauss:
  (float -> SlimPipeline.Stage<Slice.Slice<float>,Slice.Slice<float>,Shape>)
val convolve:
  kernel: Slice.Slice<'a> ->
    outputRegionMode: Slice.OutputRegionMode option ->
    boundaryCondition: Slice.BoundaryCondition option ->
    winSz: uint option -> SlimPipeline.Stage<Slice.Slice<'a>,Slice.Slice<'a>,'b>
    when 'a: equality
val conv:
  kernel: Slice.Slice<'a> ->
    (uint option -> SlimPipeline.Stage<Slice.Slice<'a>,Slice.Slice<'a>,'b>)
    when 'a: equality
val erode:
  r: uint -> SlimPipeline.Stage<Slice.Slice<uint8>,Slice.Slice<uint8>,'a>
val dilate:
  r: uint -> SlimPipeline.Stage<Slice.Slice<uint8>,Slice.Slice<uint8>,'a>
val opening:
  r: uint -> SlimPipeline.Stage<Slice.Slice<uint8>,Slice.Slice<uint8>,'a>
val closing:
  r: uint -> SlimPipeline.Stage<Slice.Slice<uint8>,Slice.Slice<uint8>,'a>
/// Full stack operators
val binaryFillHoles:
  SlimPipeline.Stage<Slice.Slice<uint8>,Slice.Slice<uint8>,obj>
val connectedComponents:
  SlimPipeline.Stage<Slice.Slice<uint8>,Slice.Slice<uint64>,Shape>
val piecewiseConnectedComponents:
  wz: uint option ->
    SlimPipeline.Stage<Slice.Slice<uint8>,Slice.Slice<uint64>,'a>
val otsuThreshold<'T when 'T: equality> :
  SlimPipeline.Stage<Slice.Slice<obj>,Slice.Slice<obj>,obj> when 'T: equality
val otsuMultiThreshold:
  n: byte -> SlimPipeline.Stage<Slice.Slice<'a>,Slice.Slice<'a>,'b>
    when 'a: equality
val momentsThreshold<'T when 'T: equality> :
  SlimPipeline.Stage<Slice.Slice<obj>,Slice.Slice<obj>,obj> when 'T: equality
val signedDistanceMap:
  SlimPipeline.Stage<Slice.Slice<uint8>,Slice.Slice<float>,Shape>
val watershed:
  a: float -> SlimPipeline.Stage<Slice.Slice<'a>,Slice.Slice<'a>,'b>
    when 'a: equality
val threshold:
  a: float -> b: float -> SlimPipeline.Stage<Slice.Slice<'a>,Slice.Slice<'a>,'b>
    when 'a: equality
val addNormalNoise:
  a: float -> b: float -> SlimPipeline.Stage<Slice.Slice<'a>,Slice.Slice<'a>,'b>
    when 'a: equality
val relabelComponents:
  a: uint -> SlimPipeline.Stage<Slice.Slice<uint64>,Slice.Slice<uint64>,'a>
val constantPad2D<'T when 'T: equality> :
  padLower: uint list ->
    padUpper: uint list ->
    c: double -> SlimPipeline.Stage<Slice.Slice<obj>,Slice.Slice<obj>,obj>
    when 'T: equality
type FileInfo = Slice.FileInfo
val getStackDepth: (string -> string -> uint)
val getStackHeight: (string -> string -> uint64)
val getStackInfo: (string -> string -> Slice.FileInfo)
val getStackSize: (string -> string -> uint * uint * uint)
val getStackWidth: (string -> string -> uint64)
val createAs:
  width: uint ->
    height: uint ->
    depth: uint ->
    pl: SlimPipeline.Pipeline<unit,unit,Shape> ->
    SlimPipeline.Pipeline<unit,Slice<'T>,Shape> when 'T: equality
val readAs:
  inputDir: string ->
    suffix: string ->
    pl: SlimPipeline.Pipeline<unit,unit,Shape> ->
    SlimPipeline.Pipeline<unit,Slice<'T>,Shape> when 'T: equality
val readRandomAs:
  count: uint ->
    inputDir: string ->
    suffix: string ->
    pl: SlimPipeline.Pipeline<unit,unit,Shape> ->
    SlimPipeline.Pipeline<unit,Slice<'T>,Shape> when 'T: equality
