namespace FSharp




module StackProcessing

type Plan<'S,'T> = SlimPipeline.Plan<'S,'T>

type Profile = SlimPipeline.Profile

type ProfileTransition = SlimPipeline.ProfileTransition

type Image<'S when 'S: equality> = Image.Image<'S>

val getMem: unit -> unit

val incIfImage: x: 'a -> 'a

val incRef: unit -> SlimPipeline.Plan<'a,'a>

val decIfImage: x: 'a -> 'a

val decRef: unit -> SlimPipeline.Plan<'a,'a>

val releaseAfter: f: (Image<'S> -> 'T) -> I: Image<'S> -> 'T when 'S: equality

val releaseAfter2:
  f: (Image<'S> -> Image<'S> -> 'T) -> I: Image<'S> -> J: Image<'S> -> 'T
    when 'S: equality

val (>=>) :
  (SlimPipeline.Pipeline<'a,'b> ->
     SlimPipeline.Plan<'b,'c> -> SlimPipeline.Pipeline<'a,'c>) when 'c: equality

val (-->) :
  (SlimPipeline.Plan<'a,'b> ->
     SlimPipeline.Plan<'b,'c> -> SlimPipeline.Plan<'a,'c>)

val source: (uint64 -> SlimPipeline.Pipeline<unit,unit>)

val debug: (uint64 -> SlimPipeline.Pipeline<unit,unit>)

val zip:
  (SlimPipeline.Pipeline<'a,'b> ->
     SlimPipeline.Pipeline<'a,'c> -> SlimPipeline.Pipeline<'a,('b * 'c)>)
    when 'b: equality and 'c: equality

val promoteStreamingToSliding:
  name: string ->
    winSz: uint ->
    pad: uint ->
    stride: uint ->
    emitStart: uint -> emitCount: uint -> plan: Plan<'T,'S> -> Plan<'T,'S>
    when 'T: equality

val (>=>>) :
  pl: SlimPipeline.Pipeline<'In,'S> ->
    plan1: Plan<'S,'U> * plan2: Plan<'S,'V> ->
      SlimPipeline.Pipeline<'In,('U * 'V)>
    when 'S: equality and 'U: equality and 'V: equality

val (>>=>) :
  (SlimPipeline.Pipeline<'a,('b * 'c)> ->
     ('b -> 'c -> 'd) -> SlimPipeline.Pipeline<'a,'d>) when 'd: equality

val (>>=>>) :
  (('a * 'b -> 'c * 'd) ->
     SlimPipeline.Pipeline<'e,('a * 'b)> ->
     SlimPipeline.Plan<('a * 'b),('c * 'd)> ->
     SlimPipeline.Pipeline<'e,('c * 'd)>) when 'c: equality and 'd: equality

val zeroMaker: index: int -> ex: Image<'S> -> Image<'S> when 'S: equality

val window:
  windowSize: uint ->
    pad: uint -> stride: uint -> SlimPipeline.Plan<Image<'a>,Image<'a> list>
    when 'a: equality

val flatten: unit -> SlimPipeline.Plan<'a list,'a>

val map: f: ('a -> 'b) -> SlimPipeline.Plan<'a,'b>

val sink: pl: SlimPipeline.Pipeline<unit,unit> -> unit

val sinkList: plLst: SlimPipeline.Pipeline<unit,unit> list -> unit

val drainSingle: pl: SlimPipeline.Pipeline<unit,'a> -> 'a

val drainList: pl: SlimPipeline.Pipeline<unit,'a> -> 'a list

val drainLast: pl: SlimPipeline.Pipeline<unit,'a> -> 'a

val tap: (string -> SlimPipeline.Plan<'a,'a>)

val tapIt: (('a -> string) -> SlimPipeline.Plan<'a,'a>)

val ignoreSingles: unit -> Plan<Image<'a>,unit> when 'a: equality

val ignorePairs: unit -> Plan<('a * unit),unit>

val idOp<'T> : (string -> SlimPipeline.Plan<'T,'T>)

val liftUnary:
  name: string ->
    (('a -> 'b) ->
       SlimPipeline.MemoryNeed ->
       SlimPipeline.NElemsTransformation -> SlimPipeline.Plan<'a,'b>)

val liftUnaryReleaseAfter:
  name: string ->
    f: (Image<'S> -> Image<'T>) ->
    memoryNeed: SlimPipeline.MemoryNeed ->
    nElemsTransformation: SlimPipeline.NElemsTransformation ->
    SlimPipeline.Plan<#Image<'S>,Image<'T>> when 'S: equality and 'T: equality

val getBytesPerComponent<'T> : uint64
type System.String with
    
    member icompare: s2: string -> bool

val write:
  outputDir: string -> suffix: string -> Plan<Image<'T>,unit> when 'T: equality

val show: plt: (Image<'T> -> unit) -> Plan<Image<'T>,unit> when 'T: equality

val plot:
  plt: (float list -> float list -> unit) -> Plan<(float * float) list,unit>

val print: unit -> Plan<'T,unit>

/// Pixel type casting
val cast<'S,'T when 'S: equality and 'T: equality> :
  SlimPipeline.Plan<Image<'S>,Image<'T>> when 'S: equality and 'T: equality

/// Basic arithmetic
val memNeeded<'T> : nTimes: uint64 -> nElems: uint64 -> uint64

val add:
  image: Image<'T> -> SlimPipeline.Plan<#Image<'T>,Image<'T>> when 'T: equality

val inline scalarAddImage:
  i: ^T -> SlimPipeline.Plan<Image<^T>,Image<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val inline imageAddScalar:
  i: ^T -> SlimPipeline.Plan<Image<^T>,Image<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val sub:
  image: Image<'T> -> SlimPipeline.Plan<#Image<'T>,Image<'T>> when 'T: equality

val inline scalarSubImage:
  i: ^T -> SlimPipeline.Plan<Image<^T>,Image<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val inline imageSubScalar:
  i: ^T -> SlimPipeline.Plan<Image<^T>,Image<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val liftRelease2:
  f: (Image<'a> -> Image<'a> -> 'b) -> I: Image<'a> -> J: Image<'a> -> 'b
    when 'a: equality

val mul2: I: Image<'a> -> J: Image<'a> -> Image.Image<'a> when 'a: equality

val mul:
  image: Image<'T> -> SlimPipeline.Plan<#Image<'T>,Image<'T>> when 'T: equality

val inline scalarMulImage:
  i: ^T -> SlimPipeline.Plan<Image<^T>,Image<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val inline imageMulScalar:
  i: ^T -> SlimPipeline.Plan<Image<^T>,Image<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val div:
  image: Image<'T> -> SlimPipeline.Plan<#Image<'T>,Image<'T>> when 'T: equality

val inline scalarDivImage:
  i: ^T -> SlimPipeline.Plan<Image<^T>,Image<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val inline imageDivScalar:
  i: ^T -> SlimPipeline.Plan<Image<^T>,Image<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val failTypeMismatch<'T> : name: string -> lst: System.Type list -> unit

/// Simple functions
val private floatNintTypes: System.Type list

val private floatTypes: System.Type list

val abs<'T when 'T: equality> : Plan<Image<'T>,Image<'T>> when 'T: equality

val acos<'T when 'T: equality> : Plan<Image<'T>,Image<'T>> when 'T: equality

val asin<'T when 'T: equality> : Plan<Image<'T>,Image<'T>> when 'T: equality

val atan<'T when 'T: equality> : Plan<Image<'T>,Image<'T>> when 'T: equality

val cos<'T when 'T: equality> : Plan<Image<'T>,Image<'T>> when 'T: equality

val sin<'T when 'T: equality> : Plan<Image<'T>,Image<'T>> when 'T: equality

val tan<'T when 'T: equality> : Plan<Image<'T>,Image<'T>> when 'T: equality

val exp<'T when 'T: equality> : Plan<Image<'T>,Image<'T>> when 'T: equality

val log10<'T when 'T: equality> : Plan<Image<'T>,Image<'T>> when 'T: equality

val log<'T when 'T: equality> : Plan<Image<'T>,Image<'T>> when 'T: equality

val round<'T when 'T: equality> : Plan<Image<'T>,Image<'T>> when 'T: equality

val sqrt<'T when 'T: equality> : Plan<Image<'T>,Image<'T>> when 'T: equality

val square<'T when 'T: equality> : Plan<Image<'T>,Image<'T>> when 'T: equality

val imageHistogram:
  unit -> SlimPipeline.Plan<Image<'T>,Map<'T,uint64>> when 'T: comparison

val imageHistogramFold:
  unit -> SlimPipeline.Plan<Map<'T,uint64>,Map<'T,uint64>> when 'T: comparison

val histogram:
  unit -> SlimPipeline.Plan<Image<'a>,Map<'a,uint64>> when 'a: comparison

val inline map2pairs<^T,^S
                       when ^T: comparison and
                            ^T: (static member op_Explicit: ^T -> float) and
                            ^S: (static member op_Explicit: ^S -> float)> :
  SlimPipeline.Plan<Map<^T,^S>,(^T * ^S) list>
    when ^T: comparison and ^T: (static member op_Explicit: ^T -> float) and
         ^S: (static member op_Explicit: ^S -> float)

val inline pairs2floats<^T,^S
                          when ^T: (static member op_Explicit: ^T -> float) and
                               ^S: (static member op_Explicit: ^S -> float)> :
  SlimPipeline.Plan<(^T * ^S) list,(float * float) list>
    when ^T: (static member op_Explicit: ^T -> float) and
         ^S: (static member op_Explicit: ^S -> float)

val inline pairs2ints<^T,^S
                        when ^T: (static member op_Explicit: ^T -> int) and
                             ^S: (static member op_Explicit: ^S -> int)> :
  SlimPipeline.Plan<(^T * ^S) list,(int * int) list>
    when ^T: (static member op_Explicit: ^T -> int) and
         ^S: (static member op_Explicit: ^S -> int)

type ImageStats = ImageFunctions.ImageStats

val imageComputeStats:
  unit -> SlimPipeline.Plan<Image<'T>,ImageStats> when 'T: equality

val imageComputeStatsFold: unit -> SlimPipeline.Plan<ImageStats,ImageStats>

val computeStats:
  unit -> SlimPipeline.Plan<Image<'a>,ImageStats> when 'a: equality

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

val volFctToLstFctReleaseAfter:
  f: (Image<'S> -> Image<'T>) ->
    pad: uint ->
    stride: uint -> images: Image.Image<'S> list -> Image.Image<'T> list
    when 'S: equality and 'T: equality

val discreteGaussianOp:
  name: string ->
    sigma: float ->
    outputRegionMode: ImageFunctions.OutputRegionMode option ->
    boundaryCondition: ImageFunctions.BoundaryCondition option ->
    winSz: uint option -> Plan<Image<float>,Image<float>>

val discreteGaussian:
  (float ->
     ImageFunctions.OutputRegionMode option ->
     ImageFunctions.BoundaryCondition option ->
     uint option -> Plan<Image<float>,Image<float>>)

val convGauss: sigma: float -> Plan<Image<float>,Image<float>>

val convolveOp:
  name: string ->
    kernel: Image<'T> ->
    outputRegionMode: ImageFunctions.OutputRegionMode option ->
    bc: ImageFunctions.BoundaryCondition option ->
    winSz: uint option -> Plan<Image<'T>,Image<'T>> when 'T: equality

val convolve:
  kernel: Image<'a> ->
    outputRegionMode: ImageFunctions.OutputRegionMode option ->
    boundaryCondition: ImageFunctions.BoundaryCondition option ->
    winSz: uint option -> Plan<Image<'a>,Image<'a>> when 'a: equality

val conv: kernel: Image<'a> -> Plan<Image<'a>,Image<'a>> when 'a: equality

val finiteDiff:
  sigma: float ->
    direction: uint -> order: uint -> Plan<Image<float>,Image<float>>

val private makeMorphOp:
  name: string ->
    radius: uint ->
    winSz: uint option ->
    core: (uint -> Image<'T> -> Image<'T>) -> Plan<Image<'T>,Image<'T>>
    when 'T: equality

val erode: radius: uint -> Plan<Image<uint8>,Image<uint8>>

val dilate: radius: uint -> Plan<Image<uint8>,Image<uint8>>

val opening: radius: uint -> Plan<Image<uint8>,Image<uint8>>

val closing: radius: uint -> Plan<Image<uint8>,Image<uint8>>

/// Full stack operators
val binaryFillHoles:
  winSz: uint -> SlimPipeline.Plan<Image<uint8>,Image.Image<uint8>>

val connectedComponents:
  winSz: uint -> SlimPipeline.Plan<Image<uint8>,Image.Image<uint64>>

val relabelComponents:
  a: uint -> winSz: uint -> SlimPipeline.Plan<Image<'a>,Image.Image<'a>>
    when 'a: equality

val watershed:
  a: float -> winSz: uint -> SlimPipeline.Plan<Image<'a>,Image.Image<'a>>
    when 'a: equality

val signedDistanceMap:
  winSz: uint -> SlimPipeline.Plan<Image<uint8>,Image.Image<float>>

val otsuThreshold:
  winSz: uint -> SlimPipeline.Plan<Image<'a>,Image.Image<uint8>>
    when 'a: equality

val momentsThreshold:
  winSz: uint -> SlimPipeline.Plan<Image<'a>,Image.Image<uint8>>
    when 'a: equality

val threshold:
  a: float -> b: float -> SlimPipeline.Plan<#Image<'b>,Image<uint8>>
    when 'b: equality

val addNormalNoise:
  a: float -> b: float -> SlimPipeline.Plan<#Image<'b>,Image<'b>>
    when 'b: equality

val ImageConstantPad<'T when 'T: equality> :
  padLower: uint list ->
    padUpper: uint list -> c: double -> SlimPipeline.Plan<Image<obj>,Image<obj>>
    when 'T: equality

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

val empty:
  pl: SlimPipeline.Pipeline<unit,unit> -> SlimPipeline.Pipeline<unit,unit>

