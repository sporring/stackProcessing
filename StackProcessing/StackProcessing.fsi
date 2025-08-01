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

val liftImageSource:
  name: string ->
    img: Slice.Slice<'T> -> SlimPipeline.Pipe<unit,Slice.Slice<'T>>
    when 'T: equality

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

/// Histogram related functions
val histogramOp:
  name: string -> SlimPipeline.Stage<Slice.Slice<'T>,Map<'T,uint64>,'Shape>
    when 'T: comparison

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

val debug: (uint64 -> SlimPipeline.Pipeline<unit,unit,Shape>)

val sink: pl: SlimPipeline.Pipeline<unit,unit,Shape> -> unit

val sinkList: plLst: SlimPipeline.Pipeline<unit,unit,Shape> list -> unit

val (>=>) :
  (SlimPipeline.Pipeline<'a,'b,'c> ->
     SlimPipeline.Stage<'b,'d,'c> -> SlimPipeline.Pipeline<'a,'d,'c>)
    when 'd: equality

val (>=>>) :
  (SlimPipeline.Pipeline<'a,'b,'c> ->
     SlimPipeline.Stage<'b,'b,'c> * SlimPipeline.Stage<'b,'d,'c> ->
       SlimPipeline.SharedPipeline<'a,'b,'d,'c>) when 'a: equality

val (>>=>) :
  (SlimPipeline.SharedPipeline<'a,'b,'c,'d> ->
     ('b -> 'c -> 'e) -> SlimPipeline.Pipeline<'a,'e,'d>) when 'e: equality

val drainSingle: pl: SlimPipeline.Pipeline<'a,'b,'c> -> 'b

val drainList: pl: SlimPipeline.Pipeline<'a,'b,'c> -> 'b list

val drainLast: pl: SlimPipeline.Pipeline<'a,'b,'c> -> 'b

val tap: (string -> SlimPipeline.Stage<'a,'a,'b>)

val tapIt: (('a -> string) -> SlimPipeline.Stage<'a,'a,'b>)

val ignoreAll: (unit -> SlimPipeline.Stage<'a,unit,Shape>)

val liftUnary:
  f: (Slice.Slice<'S> -> Slice.Slice<'T>) ->
    SlimPipeline.Stage<Slice.Slice<'S>,Slice.Slice<'T>,Shape>
    when 'S: equality and 'T: equality

val write:
  outputDir: string -> suffix: string -> Stage<Slice.Slice<'T>,unit,Shape>
    when 'T: equality

val show:
  plt: (Slice.Slice<'T> -> unit) -> Stage<Slice.Slice<'T>,unit,Shape>
    when 'T: equality

val plot:
  plt: (float list -> float list -> unit) ->
    Stage<(float * float) list,unit,Shape>

val print: unit -> Stage<'T,unit,Shape>

val finiteDiffFilter3D:
  direction: uint ->
    order: uint ->
    pl: SlimPipeline.Pipeline<unit,unit,Shape> ->
    SlimPipeline.Pipeline<unit,Slice.Slice<float>,Shape>

val axisSource:
  axis: int ->
    size: int list ->
    pl: SlimPipeline.Pipeline<unit,unit,Shape> ->
    SlimPipeline.Pipeline<unit,Slice.Slice<uint>,Shape>

/// Pixel type casting
val cast<'S,'T when 'S: equality and 'T: equality> :
  SlimPipeline.Stage<Slice.Slice<'S>,Slice.Slice<'T>,Shape>
    when 'S: equality and 'T: equality

/// Basic arithmetic
val add:
  slice: Slice.Slice<'T> ->
    SlimPipeline.Stage<Slice.Slice<'T>,Slice.Slice<'T>,Shape> when 'T: equality

val inline scalarAddSlice:
  i: ^T -> SlimPipeline.Stage<Slice.Slice<^T>,Slice.Slice<^T>,Shape>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val inline sliceAddScalar:
  i: ^T -> SlimPipeline.Stage<Slice.Slice<^T>,Slice.Slice<^T>,Shape>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val sub:
  slice: Slice.Slice<'T> ->
    SlimPipeline.Stage<Slice.Slice<'T>,Slice.Slice<'T>,Shape> when 'T: equality

val inline scalarSubSlice:
  i: ^T -> SlimPipeline.Stage<Slice.Slice<^T>,Slice.Slice<^T>,Shape>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val inline sliceSubScalar:
  i: ^T -> SlimPipeline.Stage<Slice.Slice<^T>,Slice.Slice<^T>,Shape>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val mul:
  slice: Slice.Slice<'T> ->
    SlimPipeline.Stage<Slice.Slice<'T>,Slice.Slice<'T>,Shape> when 'T: equality

val inline scalarMulSlice:
  i: ^T -> SlimPipeline.Stage<Slice.Slice<^T>,Slice.Slice<^T>,Shape>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val inline sliceMulScalar:
  i: ^T -> SlimPipeline.Stage<Slice.Slice<^T>,Slice.Slice<^T>,Shape>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val div:
  slice: Slice.Slice<'T> ->
    SlimPipeline.Stage<Slice.Slice<'T>,Slice.Slice<'T>,Shape> when 'T: equality

val inline scalarDivSlice:
  i: ^T -> SlimPipeline.Stage<Slice.Slice<^T>,Slice.Slice<^T>,Shape>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val inline sliceDivScalar:
  i: ^T -> SlimPipeline.Stage<Slice.Slice<^T>,Slice.Slice<^T>,Shape>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

/// Simple functions
val abs<'T when 'T: equality> :
  Stage<Slice.Slice<'T>,Slice.Slice<'T>,Shape> when 'T: equality

val absFloat: Stage<Slice.Slice<float>,Slice.Slice<float>,Shape>

val absFloat32: Stage<Slice.Slice<float32>,Slice.Slice<float32>,Shape>

val absInt: Stage<Slice.Slice<int>,Slice.Slice<int>,Shape>

val acosFloat: Stage<Slice.Slice<float>,Slice.Slice<float>,Shape>

val acosFloat32: Stage<Slice.Slice<float32>,Slice.Slice<float32>,Shape>

val asinFloat: Stage<Slice.Slice<float>,Slice.Slice<float>,Shape>

val asinFloat32: Stage<Slice.Slice<float32>,Slice.Slice<float32>,Shape>

val atanFloat: Stage<Slice.Slice<float>,Slice.Slice<float>,Shape>

val atanFloat32: Stage<Slice.Slice<float32>,Slice.Slice<float32>,Shape>

val cosFloat: Stage<Slice.Slice<float>,Slice.Slice<float>,Shape>

val cosFloat32: Stage<Slice.Slice<float32>,Slice.Slice<float32>,Shape>

val sinFloat: Stage<Slice.Slice<float>,Slice.Slice<float>,Shape>

val sinFloat32: Stage<Slice.Slice<float32>,Slice.Slice<float32>,Shape>

val tanFloat: Stage<Slice.Slice<float>,Slice.Slice<float>,Shape>

val tanFloat32: Stage<Slice.Slice<float32>,Slice.Slice<float32>,Shape>

val expFloat: Stage<Slice.Slice<float>,Slice.Slice<float>,Shape>

val expFloat32: Stage<Slice.Slice<float32>,Slice.Slice<float32>,Shape>

val log10Float: Stage<Slice.Slice<float>,Slice.Slice<float>,Shape>

val log10Float32: Stage<Slice.Slice<float32>,Slice.Slice<float32>,Shape>

val logFloat: Stage<Slice.Slice<float>,Slice.Slice<float>,Shape>

val logFloat32: Stage<Slice.Slice<float32>,Slice.Slice<float32>,Shape>

val roundFloat: Stage<Slice.Slice<float>,Slice.Slice<float>,Shape>

val roundFloat32: Stage<Slice.Slice<float32>,Slice.Slice<float32>,Shape>

val sqrtFloat: Stage<Slice.Slice<float>,Slice.Slice<float>,Shape>

val sqrtFloat32: Stage<Slice.Slice<float32>,Slice.Slice<float32>,Shape>

val sqrtInt: Stage<Slice.Slice<int>,Slice.Slice<int>,Shape>

val squareFloat: Stage<Slice.Slice<float>,Slice.Slice<float>,Shape>

val squareFloat32: Stage<Slice.Slice<float32>,Slice.Slice<float32>,Shape>

val squareInt: Stage<Slice.Slice<int>,Slice.Slice<int>,Shape>

val sliceHistogram:
  unit -> SlimPipeline.Stage<Slice.Slice<'T>,Map<'T,uint64>,Shape>
    when 'T: comparison

val sliceHistogramFold:
  unit -> SlimPipeline.Stage<Map<'T,uint64>,Map<'T,uint64>,Shape>
    when 'T: comparison

val histogram:
  unit -> SlimPipeline.Stage<Slice.Slice<'a>,Map<'a,uint64>,Shape>
    when 'a: comparison

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
    SlimPipeline.Pipeline<unit,Slice.Slice<'T>,Shape> when 'T: equality

val readAs:
  inputDir: string ->
    suffix: string ->
    pl: SlimPipeline.Pipeline<unit,unit,Shape> ->
    SlimPipeline.Pipeline<unit,Slice.Slice<'T>,Shape> when 'T: equality

val readRandomAs:
  count: uint ->
    inputDir: string ->
    suffix: string ->
    pl: SlimPipeline.Pipeline<unit,unit,Shape> ->
    SlimPipeline.Pipeline<unit,Slice.Slice<'T>,Shape> when 'T: equality

