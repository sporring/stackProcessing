namespace FSharp




module StackProcessing

type Stage<'S,'T> = SlimPipeline.Stage<'S,'T>

type Profile = SlimPipeline.Profile

type ProfileTransition = SlimPipeline.ProfileTransition

type Image<'S when 'S: equality> = Image.Image<'S>

val releaseAfter: f: (Image<'S> -> 'T) -> I: Image<'S> -> 'T when 'S: equality

val releaseAfter2:
  f: (Image<'S> -> Image<'S> -> 'T) -> I: Image<'S> -> J: Image<'S> -> 'T
    when 'S: equality

val releaseNAfter:
  n: int -> f: (Image<'S> list -> 'T list) -> sLst: Image<'S> list -> 'T list
    when 'S: equality

val incRefCountOp: unit -> SlimPipeline.Stage<'a,'a>

val decRefCountOp: unit -> SlimPipeline.Stage<'a,'a>

val (-->) :
  (SlimPipeline.Stage<'a,'b> ->
     SlimPipeline.Stage<'b,'c> -> SlimPipeline.Stage<'a,'c>)

val source: (uint64 -> SlimPipeline.Pipeline<unit,unit>)

val debug: (uint64 -> SlimPipeline.Pipeline<unit,unit>)

val zip:
  (SlimPipeline.Pipeline<'a,'b> ->
     SlimPipeline.Pipeline<'a,'c> -> SlimPipeline.Pipeline<'a,('b * 'c)>)
    when 'b: equality and 'c: equality

val (>=>) :
  pl: SlimPipeline.Pipeline<'a,'b> ->
    stage: Stage<'b,'c> -> SlimPipeline.Pipeline<'a,'c> when 'c: equality

val (>=>>) :
  pl: SlimPipeline.Pipeline<'In,'S> ->
    stage1: Stage<'S,'U> * stage2: Stage<'S,'V> ->
      SlimPipeline.Pipeline<'In,('U * 'V)> when 'U: equality and 'V: equality

val (>>=>) :
  (SlimPipeline.Pipeline<'a,('b * 'c)> ->
     ('b -> 'c -> 'd) -> SlimPipeline.Pipeline<'a,'d>) when 'd: equality

val (>>=>>) :
  (('a * 'b -> 'c * 'd) ->
     SlimPipeline.Pipeline<'e,('a * 'b)> ->
     SlimPipeline.Stage<('a * 'b),('c * 'd)> ->
     SlimPipeline.Pipeline<'e,('c * 'd)>) when 'c: equality and 'd: equality

val sink: pl: SlimPipeline.Pipeline<unit,unit> -> unit

val sinkList: plLst: SlimPipeline.Pipeline<unit,unit> list -> unit

val drainSingle: pl: SlimPipeline.Pipeline<'a,'b> -> 'b

val drainList: pl: SlimPipeline.Pipeline<'a,'b> -> 'b list

val drainLast: pl: SlimPipeline.Pipeline<'a,'b> -> 'b

val tap: (string -> SlimPipeline.Stage<'a,'a>)

val tapIt: (('a -> string) -> SlimPipeline.Stage<'a,'a>)

val ignoreImages: unit -> Stage<Image<'a>,unit> when 'a: equality

val ignoreAll: unit -> (('a -> unit) -> SlimPipeline.Stage<'a,unit>)

val zeroMaker: index: int -> ex: Image<'S> -> Image<'S> when 'S: equality

val liftUnary:
  (string ->
     ('a -> 'b) ->
     SlimPipeline.MemoryNeed ->
     SlimPipeline.NElemsTransformation -> SlimPipeline.Stage<'a,'b>)

val liftUnaryReleaseAfter:
  name: string ->
    f: (Image<'S> -> Image<'T>) ->
    memoryNeed: SlimPipeline.MemoryNeed ->
    nElemsTransformation: SlimPipeline.NElemsTransformation ->
    SlimPipeline.Stage<#Image<'S>,Image<'T>> when 'S: equality and 'T: equality

val liftWindowed:
  (string ->
     uint ->
     uint ->
     (int -> 'a -> 'a) ->
     uint ->
     uint ->
     uint ->
     ('a list -> 'b list) ->
     SlimPipeline.MemoryNeed ->
     SlimPipeline.NElemsTransformation -> SlimPipeline.Stage<'a,'b>)
    when 'a: equality and 'b: equality

val liftWindowedReleaseAfter:
  name: string ->
    window: uint ->
    pad: uint ->
    zeroMaker: (int -> Image<'S> -> Image<'S>) ->
    stride: uint ->
    emitStart: uint ->
    emitCount: uint ->
    f: (Image<'S> list -> Image<'T> list) ->
    memoryNeed: SlimPipeline.MemoryNeed ->
    nElemsTransformation: SlimPipeline.NElemsTransformation ->
    Stage<Image<'S>,Image<'T>> when 'S: equality and 'T: equality

val getBytesPerComponent<'T> : uint64

val write:
  outputDir: string -> suffix: string -> Stage<Image<'T>,unit> when 'T: equality

val show: plt: (Image<'T> -> unit) -> Stage<Image<'T>,unit> when 'T: equality

val plot:
  plt: (float list -> float list -> unit) -> Stage<(float * float) list,unit>

val print: unit -> Stage<'T,unit>

/// Pixel type casting
val cast<'S,'T when 'S: equality and 'T: equality> :
  SlimPipeline.Stage<Image<'S>,Image<'T>> when 'S: equality and 'T: equality

/// Basic arithmetic
val memNeeded<'T> : nTimes: uint64 -> nElems: uint64 -> uint64

val add:
  image: Image<'T> -> SlimPipeline.Stage<#Image<'T>,Image<'T>> when 'T: equality

val inline scalarAddImage:
  i: ^T -> SlimPipeline.Stage<Image<^T>,Image<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val inline imageAddScalar:
  i: ^T -> SlimPipeline.Stage<Image<^T>,Image<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val sub:
  image: Image<'T> -> SlimPipeline.Stage<#Image<'T>,Image<'T>> when 'T: equality

val inline scalarSubImage:
  i: ^T -> SlimPipeline.Stage<Image<^T>,Image<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val inline imageSubScalar:
  i: ^T -> SlimPipeline.Stage<Image<^T>,Image<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val liftRelease2:
  f: (Image<'a> -> Image<'a> -> 'b) -> I: Image<'a> -> J: Image<'a> -> 'b
    when 'a: equality

val mul2: I: Image<'a> -> J: Image<'a> -> Image.Image<'a> when 'a: equality

val mul:
  image: Image<'T> -> SlimPipeline.Stage<#Image<'T>,Image<'T>> when 'T: equality

val inline scalarMulImage:
  i: ^T -> SlimPipeline.Stage<Image<^T>,Image<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val inline imageMulScalar:
  i: ^T -> SlimPipeline.Stage<Image<^T>,Image<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val div:
  image: Image<'T> -> SlimPipeline.Stage<#Image<'T>,Image<'T>> when 'T: equality

val inline scalarDivImage:
  i: ^T -> SlimPipeline.Stage<Image<^T>,Image<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val inline imageDivScalar:
  i: ^T -> SlimPipeline.Stage<Image<^T>,Image<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

/// Simple functions
val abs<'T when 'T: equality> : Stage<Image<'T>,Image<'T>> when 'T: equality

val absFloat: Stage<Image<float>,Image<float>>

val absFloat32: Stage<Image<float32>,Image<float32>>

val absInt: Stage<Image<int>,Image<int>>

val acosFloat: Stage<Image<float>,Image<float>>

val acosFloat32: Stage<Image<float32>,Image<float32>>

val asinFloat: Stage<Image<float>,Image<float>>

val asinFloat32: Stage<Image<float32>,Image<float32>>

val atanFloat: Stage<Image<float>,Image<float>>

val atanFloat32: Stage<Image<float32>,Image<float32>>

val cosFloat: Stage<Image<float>,Image<float>>

val cosFloat32: Stage<Image<float32>,Image<float32>>

val sinFloat: Stage<Image<float>,Image<float>>

val sinFloat32: Stage<Image<float32>,Image<float32>>

val tanFloat: Stage<Image<float>,Image<float>>

val tanFloat32: Stage<Image<float32>,Image<float32>>

val expFloat: Stage<Image<float>,Image<float>>

val expFloat32: Stage<Image<float32>,Image<float32>>

val log10Float: Stage<Image<float>,Image<float>>

val log10Float32: Stage<Image<float32>,Image<float32>>

val logFloat: Stage<Image<float>,Image<float>>

val logFloat32: Stage<Image<float32>,Image<float32>>

val roundFloat: Stage<Image<float>,Image<float>>

val roundFloat32: Stage<Image<float32>,Image<float32>>

val sqrtFloat: Stage<Image<float>,Image<float>>

val sqrtFloat32: Stage<Image<float32>,Image<float32>>

val sqrtInt: Stage<Image<int>,Image<int>>

val squareFloat: Stage<Image<float>,Image<float>>

val squareFloat32: Stage<Image<float32>,Image<float32>>

val squareInt: Stage<Image<int>,Image<int>>

val imageHistogram:
  unit -> SlimPipeline.Stage<Image<'T>,Map<'T,uint64>> when 'T: comparison

val imageHistogramFold:
  unit -> SlimPipeline.Stage<Map<'T,uint64>,Map<'T,uint64>> when 'T: comparison

val histogram:
  unit -> SlimPipeline.Stage<Image<'a>,Map<'a,uint64>> when 'a: comparison

val inline map2pairs<^T,^S
                       when ^T: comparison and
                            ^T: (static member op_Explicit: ^T -> float) and
                            ^S: (static member op_Explicit: ^S -> float)> :
  SlimPipeline.Stage<Map<^T,^S>,(^T * ^S) list>
    when ^T: comparison and ^T: (static member op_Explicit: ^T -> float) and
         ^S: (static member op_Explicit: ^S -> float)

val inline pairs2floats<^T,^S
                          when ^T: (static member op_Explicit: ^T -> float) and
                               ^S: (static member op_Explicit: ^S -> float)> :
  SlimPipeline.Stage<(^T * ^S) list,(float * float) list>
    when ^T: (static member op_Explicit: ^T -> float) and
         ^S: (static member op_Explicit: ^S -> float)

val inline pairs2ints<^T,^S
                        when ^T: (static member op_Explicit: ^T -> int) and
                             ^S: (static member op_Explicit: ^S -> int)> :
  SlimPipeline.Stage<(^T * ^S) list,(int * int) list>
    when ^T: (static member op_Explicit: ^T -> int) and
         ^S: (static member op_Explicit: ^S -> int)

type ImageStats = ImageFunctions.ImageStats

val imageComputeStats:
  unit -> SlimPipeline.Stage<Image<'T>,ImageStats> when 'T: equality

val imageComputeStatsFold: unit -> SlimPipeline.Stage<ImageStats,ImageStats>

val computeStats:
  unit -> SlimPipeline.Stage<Image<'a>,ImageStats> when 'a: equality

val stackFUnstack:
  f: (Image<'T> -> #Image.Image<'b>) ->
    images: Image<'T> list -> Image.Image<'b> list
    when 'T: equality and 'b: equality

val skipNTakeM: n: uint -> m: uint -> lst: 'a list -> 'a list

val stackFUnstackTrim:
  trim: uint32 ->
    f: (Image<'T> -> Image<'S>) ->
    images: Image<'T> list -> Image.Image<'S> list
    when 'T: equality and 'S: equality

val discreteGaussianOp:
  name: string ->
    sigma: float ->
    outputRegionMode: ImageFunctions.OutputRegionMode option ->
    boundaryCondition: ImageFunctions.BoundaryCondition option ->
    winSz: uint option -> Stage<Image<float>,Image<float>>

val discreteGaussian:
  (float ->
     ImageFunctions.OutputRegionMode option ->
     ImageFunctions.BoundaryCondition option ->
     uint option -> Stage<Image<float>,Image<float>>)

val convGauss: sigma: float -> Stage<Image<float>,Image<float>>

val convolveOp:
  name: string ->
    kernel: Image<'T> ->
    outputRegionMode: ImageFunctions.OutputRegionMode option ->
    bc: ImageFunctions.BoundaryCondition option ->
    winSz: uint option -> Stage<Image<'T>,Image<'T>> when 'T: equality

val convolve:
  kernel: Image<'a> ->
    outputRegionMode: ImageFunctions.OutputRegionMode option ->
    boundaryCondition: ImageFunctions.BoundaryCondition option ->
    winSz: uint option -> Stage<Image<'a>,Image<'a>> when 'a: equality

val conv: kernel: Image<'a> -> Stage<Image<'a>,Image<'a>> when 'a: equality

val finiteDiff:
  direction: uint -> order: uint -> Stage<Image<float>,Image<float>>

val private makeMorphOp:
  name: string ->
    radius: uint ->
    winSz: uint option ->
    core: (uint -> Image<'T> -> Image<'T>) -> Stage<Image<'T>,Image<'T>>
    when 'T: equality

val erode: radius: uint -> Stage<Image<uint8>,Image<uint8>>

val dilate: radius: uint -> Stage<Image<uint8>,Image<uint8>>

val opening: radius: uint -> Stage<Image<uint8>,Image<uint8>>

val closing: radius: uint -> Stage<Image<uint8>,Image<uint8>>

/// Full stack operators
val binaryFillHoles: winSz: uint -> Stage<Image<uint8>,Image<uint8>>

val connectedComponents: winSz: uint -> Stage<Image<uint8>,Image<uint64>>

val relabelComponents:
  a: uint -> winSz: uint -> Stage<Image<'a>,Image<'a>> when 'a: equality

val watershed:
  a: float -> winSz: uint -> Stage<Image<'a>,Image<'a>> when 'a: equality

val signedDistanceMap: winSz: uint -> Stage<Image<uint8>,Image<float>>

val otsuThreshold: winSz: uint -> Stage<Image<'a>,Image<'a>> when 'a: equality

val momentsThreshold:
  winSz: uint -> Stage<Image<'a>,Image<'a>> when 'a: equality

val threshold:
  a: float -> b: float -> SlimPipeline.Stage<#Image<'b>,Image<'b>>
    when 'b: equality

val addNormalNoise:
  a: float -> b: float -> SlimPipeline.Stage<#Image<'b>,Image<'b>>
    when 'b: equality

val ImageConstantPad<'T when 'T: equality> :
  padLower: uint list ->
    padUpper: uint list ->
    c: double -> SlimPipeline.Stage<Image<obj>,Image<obj>> when 'T: equality

type FileInfo = ImageFunctions.FileInfo

val getStackDepth: inputDir: string -> suffix: string -> uint

val getStackInfo: inputDir: string -> suffix: string -> FileInfo

val getStackSize: inputDir: string -> suffix: string -> uint * uint * uint

val getStackWidth: inputDir: string -> suffix: string -> uint64

val getStackHeight: inputDir: string -> suffix: string -> uint64

val zero:
  width: uint ->
    height: uint ->
    depth: uint ->
    pl: SlimPipeline.Pipeline<unit,unit> ->
    SlimPipeline.Pipeline<unit,Image<'T>> when 'T: equality

val readFilteredOp:
  name: string ->
    inputDir: string ->
    suffix: string ->
    filter: (string array -> string array) ->
    pl: SlimPipeline.Pipeline<unit,unit> ->
    SlimPipeline.Pipeline<unit,Image<'T>> when 'T: equality

val readFiltered:
  inputDir: string ->
    suffix: string ->
    filter: (string array -> string array) ->
    pl: SlimPipeline.Pipeline<unit,unit> ->
    SlimPipeline.Pipeline<unit,Image<'T>> when 'T: equality

val read:
  inputDir: string ->
    suffix: string ->
    pl: SlimPipeline.Pipeline<unit,unit> ->
    SlimPipeline.Pipeline<unit,Image<'T>> when 'T: equality

val readRandom:
  count: uint ->
    inputDir: string ->
    suffix: string ->
    pl: SlimPipeline.Pipeline<unit,unit> ->
    SlimPipeline.Pipeline<unit,Image<'T>> when 'T: equality

