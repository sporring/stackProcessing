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


module StackProcessing

type Stage<'S,'T,'ShapeS,'ShapeT> = SlimPipeline.Stage<'S,'T,'ShapeS,'ShapeT>

type MemoryProfile = SlimPipeline.MemoryProfile

type MemoryTransition = SlimPipeline.MemoryTransition

type Slice<'S when 'S: equality> = Slice.Slice<'S>

type Shape =
    | Zero
    | Slice of uint * uint
    | List of uint

val shapeContext: SlimPipeline.ShapeContext<Shape>

val (-->) :
  (SlimPipeline.Stage<'a,'b,'c,'d> ->
     SlimPipeline.Stage<'b,'e,'d,'f> -> SlimPipeline.Stage<'a,'e,'c,'f>)

val source: (uint64 -> SlimPipeline.Pipeline<unit,unit,Shape,Shape>)

val debug: (uint64 -> SlimPipeline.Pipeline<unit,unit,Shape,Shape>)

val sink: pl: SlimPipeline.Pipeline<unit,unit,Shape,Shape> -> unit

val sinkList: plLst: SlimPipeline.Pipeline<unit,unit,Shape,Shape> list -> unit

val (>=>) :
  (SlimPipeline.Pipeline<'a,'b,'c,'d> ->
     SlimPipeline.Stage<'b,'e,'d,'f> -> SlimPipeline.Pipeline<'a,'e,'c,'f>)
    when 'e: equality

val (>=>>) :
  (SlimPipeline.Pipeline<'a,'b,'c,'d> ->
     SlimPipeline.Stage<'b,'e,'d,'f> * SlimPipeline.Stage<'b,'g,'d,'h> ->
       SlimPipeline.Pipeline<'a,('e * 'g),'c,('f * 'h)>)
    when 'e: equality and 'g: equality

val (>>=>) :
  (('a -> 'b -> 'c) ->
     SlimPipeline.Pipeline<'d,'e,'f,'g> ->
     SlimPipeline.Stage<'e,'a,'g,'h> * SlimPipeline.Stage<'e,'b,'g,'i> ->
       ('g option -> 'j option) -> SlimPipeline.Pipeline<'d,'c,'f,'j>)
    when 'c: equality

val drainSingle: pl: SlimPipeline.Pipeline<'a,'b,'c,'d> -> 'b

val drainList: pl: SlimPipeline.Pipeline<'a,'b,'c,'d> -> 'b list

val drainLast: pl: SlimPipeline.Pipeline<'a,'b,'c,'d> -> 'b

val tap: (string -> SlimPipeline.Stage<'a,'a,'b,'b>)

val tapIt: (('a -> string) -> SlimPipeline.Stage<'a,'a,'b,'b>)

val ignoreAll: (unit -> SlimPipeline.Stage<'a,unit,Shape,Shape>)

val liftUnary:
  f: (Slice.Slice<'S> -> Slice.Slice<'T>) ->
    ((Shape option -> Shape option) ->
       SlimPipeline.Stage<Slice.Slice<'S>,Slice.Slice<'T>,Shape,Shape>)
    when 'S: equality and 'T: equality

val zeroMaker: ex: Slice.Slice<'S> -> Slice.Slice<'S> when 'S: equality

val liftWindowed:
  name: string ->
    updateId: (uint -> Slice.Slice<'S> -> Slice.Slice<'S>) ->
    window: uint ->
    pad: uint ->
    zeroMaker: (Slice.Slice<'S> -> Slice.Slice<'S>) ->
    stride: uint ->
    emitStart: uint ->
    emitCount: uint ->
    f: (Slice.Slice<'S> list -> Slice.Slice<'T> list) ->
    shapeUpdate: (Shape option -> Shape option) ->
    Stage<Slice.Slice<'S>,Slice.Slice<'T>,Shape,Shape>
    when 'S: equality and 'T: equality

val write:
  outputDir: string -> suffix: string -> Stage<Slice.Slice<'T>,unit,Shape,Shape>
    when 'T: equality

val show:
  plt: (Slice.Slice<'T> -> unit) -> Stage<Slice.Slice<'T>,unit,Shape,Shape>
    when 'T: equality

val plot:
  plt: (float list -> float list -> unit) ->
    Stage<(float * float) list,unit,Shape,Shape>

val print: unit -> Stage<'T,unit,Shape,Shape>

/// Pixel type casting
val cast<'S,'T when 'S: equality and 'T: equality> :
  SlimPipeline.Stage<Slice.Slice<'S>,Slice.Slice<'T>,Shape,Shape>
    when 'S: equality and 'T: equality

/// Basic arithmetic
val add:
  slice: Slice.Slice<'T> ->
    ((Shape option -> Shape option) ->
       SlimPipeline.Stage<Slice.Slice<'T>,Slice.Slice<'T>,Shape,Shape>)
    when 'T: equality

val inline scalarAddSlice:
  i: ^T ->
    ((Shape option -> Shape option) ->
       SlimPipeline.Stage<Slice.Slice<^T>,Slice.Slice<^T>,Shape,Shape>)
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val inline sliceAddScalar:
  i: ^T ->
    ((Shape option -> Shape option) ->
       SlimPipeline.Stage<Slice.Slice<^T>,Slice.Slice<^T>,Shape,Shape>)
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val sub:
  slice: Slice.Slice<'T> ->
    ((Shape option -> Shape option) ->
       SlimPipeline.Stage<Slice.Slice<'T>,Slice.Slice<'T>,Shape,Shape>)
    when 'T: equality

val inline scalarSubSlice:
  i: ^T ->
    ((Shape option -> Shape option) ->
       SlimPipeline.Stage<Slice.Slice<^T>,Slice.Slice<^T>,Shape,Shape>)
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val inline sliceSubScalar:
  i: ^T ->
    ((Shape option -> Shape option) ->
       SlimPipeline.Stage<Slice.Slice<^T>,Slice.Slice<^T>,Shape,Shape>)
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val mul:
  slice: Slice.Slice<'T> ->
    ((Shape option -> Shape option) ->
       SlimPipeline.Stage<Slice.Slice<'T>,Slice.Slice<'T>,Shape,Shape>)
    when 'T: equality

val inline scalarMulSlice:
  i: ^T ->
    ((Shape option -> Shape option) ->
       SlimPipeline.Stage<Slice.Slice<^T>,Slice.Slice<^T>,Shape,Shape>)
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val inline sliceMulScalar:
  i: ^T ->
    ((Shape option -> Shape option) ->
       SlimPipeline.Stage<Slice.Slice<^T>,Slice.Slice<^T>,Shape,Shape>)
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val div:
  slice: Slice.Slice<'T> ->
    ((Shape option -> Shape option) ->
       SlimPipeline.Stage<Slice.Slice<'T>,Slice.Slice<'T>,Shape,Shape>)
    when 'T: equality

val inline scalarDivSlice:
  i: ^T ->
    ((Shape option -> Shape option) ->
       SlimPipeline.Stage<Slice.Slice<^T>,Slice.Slice<^T>,Shape,Shape>)
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val inline sliceDivScalar:
  i: ^T ->
    ((Shape option -> Shape option) ->
       SlimPipeline.Stage<Slice.Slice<^T>,Slice.Slice<^T>,Shape,Shape>)
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

/// Simple functions
val abs<'T when 'T: equality> :
  Stage<Slice.Slice<'T>,Slice.Slice<'T>,Shape,Shape> when 'T: equality

val absFloat: Stage<Slice.Slice<float>,Slice.Slice<float>,Shape,Shape>

val absFloat32: Stage<Slice.Slice<float32>,Slice.Slice<float32>,Shape,Shape>

val absInt: Stage<Slice.Slice<int>,Slice.Slice<int>,Shape,Shape>

val acosFloat: Stage<Slice.Slice<float>,Slice.Slice<float>,Shape,Shape>

val acosFloat32: Stage<Slice.Slice<float32>,Slice.Slice<float32>,Shape,Shape>

val asinFloat: Stage<Slice.Slice<float>,Slice.Slice<float>,Shape,Shape>

val asinFloat32: Stage<Slice.Slice<float32>,Slice.Slice<float32>,Shape,Shape>

val atanFloat: Stage<Slice.Slice<float>,Slice.Slice<float>,Shape,Shape>

val atanFloat32: Stage<Slice.Slice<float32>,Slice.Slice<float32>,Shape,Shape>

val cosFloat: Stage<Slice.Slice<float>,Slice.Slice<float>,Shape,Shape>

val cosFloat32: Stage<Slice.Slice<float32>,Slice.Slice<float32>,Shape,Shape>

val sinFloat: Stage<Slice.Slice<float>,Slice.Slice<float>,Shape,Shape>

val sinFloat32: Stage<Slice.Slice<float32>,Slice.Slice<float32>,Shape,Shape>

val tanFloat: Stage<Slice.Slice<float>,Slice.Slice<float>,Shape,Shape>

val tanFloat32: Stage<Slice.Slice<float32>,Slice.Slice<float32>,Shape,Shape>

val expFloat: Stage<Slice.Slice<float>,Slice.Slice<float>,Shape,Shape>

val expFloat32: Stage<Slice.Slice<float32>,Slice.Slice<float32>,Shape,Shape>

val log10Float: Stage<Slice.Slice<float>,Slice.Slice<float>,Shape,Shape>

val log10Float32: Stage<Slice.Slice<float32>,Slice.Slice<float32>,Shape,Shape>

val logFloat: Stage<Slice.Slice<float>,Slice.Slice<float>,Shape,Shape>

val logFloat32: Stage<Slice.Slice<float32>,Slice.Slice<float32>,Shape,Shape>

val roundFloat: Stage<Slice.Slice<float>,Slice.Slice<float>,Shape,Shape>

val roundFloat32: Stage<Slice.Slice<float32>,Slice.Slice<float32>,Shape,Shape>

val sqrtFloat: Stage<Slice.Slice<float>,Slice.Slice<float>,Shape,Shape>

val sqrtFloat32: Stage<Slice.Slice<float32>,Slice.Slice<float32>,Shape,Shape>

val sqrtInt: Stage<Slice.Slice<int>,Slice.Slice<int>,Shape,Shape>

val squareFloat: Stage<Slice.Slice<float>,Slice.Slice<float>,Shape,Shape>

val squareFloat32: Stage<Slice.Slice<float32>,Slice.Slice<float32>,Shape,Shape>

val squareInt: Stage<Slice.Slice<int>,Slice.Slice<int>,Shape,Shape>

val sliceHistogram:
  unit -> SlimPipeline.Stage<Slice.Slice<'T>,Map<'T,uint64>,Shape,Shape>
    when 'T: comparison

val sliceHistogramFold:
  unit -> SlimPipeline.Stage<Map<'T,uint64>,Map<'T,uint64>,Shape,Shape>
    when 'T: comparison

val histogram:
  unit -> SlimPipeline.Stage<Slice.Slice<'a>,Map<'a,uint64>,Shape,Shape>
    when 'a: comparison

val inline map2pairs<^T,^S
                       when ^T: comparison and
                            ^T: (static member op_Explicit: ^T -> float) and
                            ^S: (static member op_Explicit: ^S -> float)> :
  ((Shape option -> Shape option) ->
     SlimPipeline.Stage<Map<^T,^S>,(^T * ^S) list,Shape,Shape>)
    when ^T: comparison and ^T: (static member op_Explicit: ^T -> float) and
         ^S: (static member op_Explicit: ^S -> float)

val inline pairs2floats<^T,^S
                          when ^T: (static member op_Explicit: ^T -> float) and
                               ^S: (static member op_Explicit: ^S -> float)> :
  ((Shape option -> Shape option) ->
     SlimPipeline.Stage<(^T * ^S) list,(float * float) list,Shape,Shape>)
    when ^T: (static member op_Explicit: ^T -> float) and
         ^S: (static member op_Explicit: ^S -> float)

val inline pairs2ints<^T,^S
                        when ^T: (static member op_Explicit: ^T -> int) and
                             ^S: (static member op_Explicit: ^S -> int)> :
  ((Shape option -> Shape option) ->
     SlimPipeline.Stage<(^T * ^S) list,(int * int) list,Shape,Shape>)
    when ^T: (static member op_Explicit: ^T -> int) and
         ^S: (static member op_Explicit: ^S -> int)

type ImageStats = ImageFunctions.ImageStats

val sliceComputeStats:
  unit -> SlimPipeline.Stage<Slice.Slice<'T>,ImageStats,Shape,Shape>
    when 'T: equality

val sliceComputeStatsFold:
  unit -> SlimPipeline.Stage<ImageStats,ImageStats,Shape,Shape>

val computeStats:
  unit -> SlimPipeline.Stage<Slice.Slice<'a>,ImageStats,Shape,Shape>
    when 'a: equality

val stackFUnstack:
  f: (Slice.Slice<'T> -> Slice.Slice<'a>) ->
    slices: Slice.Slice<'T> list -> Slice.Slice<'a> list
    when 'T: equality and 'a: equality

val skipNTakeM: n: uint -> m: uint -> lst: 'a list -> 'a list

val stackFUnstackTrim:
  trim: uint32 ->
    f: (Slice.Slice<'T> -> Slice.Slice<'a>) ->
    slices: Slice.Slice<'T> list -> Slice.Slice<'a> list
    when 'T: equality and 'a: equality

val discreteGaussianOp:
  name: string ->
    sigma: float ->
    outputRegionMode: Slice.OutputRegionMode option ->
    boundaryCondition: ImageFunctions.BoundaryCondition option ->
    winSz: uint option ->
    Stage<Slice.Slice<float>,Slice.Slice<float>,Shape,Shape>

val discreteGaussian:
  (float ->
     Slice.OutputRegionMode option ->
     ImageFunctions.BoundaryCondition option ->
     uint option -> Stage<Slice.Slice<float>,Slice.Slice<float>,Shape,Shape>)

val convGauss:
  sigma: float -> Stage<Slice.Slice<float>,Slice.Slice<float>,Shape,Shape>

val convolveOp:
  name: string ->
    kernel: Slice.Slice<'T> ->
    outputRegionMode: Slice.OutputRegionMode option ->
    bc: Slice.BoundaryCondition option ->
    winSz: uint option -> Stage<Slice.Slice<'T>,Slice.Slice<'T>,Shape,Shape>
    when 'T: equality

val convolve:
  kernel: Slice.Slice<'a> ->
    outputRegionMode: Slice.OutputRegionMode option ->
    boundaryCondition: Slice.BoundaryCondition option ->
    winSz: uint option -> Stage<Slice.Slice<'a>,Slice.Slice<'a>,Shape,Shape>
    when 'a: equality

val conv:
  kernel: Slice.Slice<'a> -> Stage<Slice.Slice<'a>,Slice.Slice<'a>,Shape,Shape>
    when 'a: equality

val finiteDiff:
  direction: uint ->
    order: uint -> Stage<Slice.Slice<float>,Slice.Slice<float>,Shape,Shape>

val private makeMorphOp:
  name: string ->
    radius: uint ->
    winSz: uint option ->
    core: (uint -> Slice.Slice<'T> -> Slice.Slice<'T>) ->
    Stage<Slice.Slice<'T>,Slice.Slice<'T>,Shape,Shape> when 'T: equality

val erode:
  radius: uint -> Stage<Slice.Slice<uint8>,Slice.Slice<uint8>,Shape,Shape>

val dilate:
  radius: uint -> Stage<Slice.Slice<uint8>,Slice.Slice<uint8>,Shape,Shape>

val opening:
  radius: uint -> Stage<Slice.Slice<uint8>,Slice.Slice<uint8>,Shape,Shape>

val closing:
  radius: uint -> Stage<Slice.Slice<uint8>,Slice.Slice<uint8>,Shape,Shape>

/// Full stack operators
val binaryFillHoles:
  winSz: uint ->
    ((Shape option -> Shape option) ->
       Stage<Slice.Slice<uint8>,Slice.Slice<uint8>,Shape,Shape>)

val connectedComponents:
  winSz: uint ->
    ((Shape option -> Shape option) ->
       Stage<Slice.Slice<uint8>,Slice.Slice<uint64>,Shape,Shape>)

val relabelComponents:
  a: uint ->
    winSz: uint ->
    ((Shape option -> Shape option) ->
       Stage<Slice.Slice<uint64>,Slice.Slice<uint64>,Shape,Shape>)

val watershed:
  a: float ->
    winSz: uint ->
    ((Shape option -> Shape option) ->
       Stage<Slice.Slice<uint8>,Slice.Slice<uint8>,Shape,Shape>)

val signedDistanceMap:
  winSz: uint ->
    ((Shape option -> Shape option) ->
       Stage<Slice.Slice<uint8>,Slice.Slice<float>,Shape,Shape>)

val otsuThreshold:
  winSz: uint ->
    ((Shape option -> Shape option) ->
       Stage<Slice.Slice<uint8>,Slice.Slice<uint8>,Shape,Shape>)

val momentsThreshold:
  winSz: uint ->
    ((Shape option -> Shape option) ->
       Stage<Slice.Slice<uint8>,Slice.Slice<uint8>,Shape,Shape>)

val threshold:
  a: float ->
    b: float ->
    (('a option -> 'b option) ->
       SlimPipeline.Stage<Slice.Slice<'c>,Slice.Slice<'c>,'a,'b>)
    when 'c: equality

val addNormalNoise:
  a: float ->
    b: float ->
    (('a option -> 'b option) ->
       SlimPipeline.Stage<Slice.Slice<'c>,Slice.Slice<'c>,'a,'b>)
    when 'c: equality

val SliceConstantPad<'T when 'T: equality> :
  padLower: uint list ->
    padUpper: uint list ->
    c: double ->
    ((obj option -> obj option) ->
       SlimPipeline.Stage<Slice.Slice<obj>,Slice.Slice<obj>,obj,obj>)
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
    pl: SlimPipeline.Pipeline<unit,unit,Shape,Shape> ->
    SlimPipeline.Pipeline<unit,Slice.Slice<'T>,Shape,Shape> when 'T: equality

val readAs:
  inputDir: string ->
    suffix: string ->
    pl: SlimPipeline.Pipeline<unit,unit,Shape,Shape> ->
    SlimPipeline.Pipeline<unit,Slice.Slice<'T>,Shape,Shape> when 'T: equality

val readRandomAs:
  count: uint ->
    inputDir: string ->
    suffix: string ->
    pl: SlimPipeline.Pipeline<unit,unit,Shape,Shape> ->
    SlimPipeline.Pipeline<unit,Slice.Slice<'T>,Shape,Shape> when 'T: equality

