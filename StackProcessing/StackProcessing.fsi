namespace FSharp




module StackProcessing

type Stage<'S,'T> = SlimPipeline.Stage<'S,'T>

type Profile = SlimPipeline.Profile

type ProfileTransition = SlimPipeline.ProfileTransition

type Image<'S when 'S: equality> = Image.Image<'S>

val getMem: unit -> unit

val incIfImage: x: 'a -> 'a

val incRef: unit -> SlimPipeline.Stage<'a,'a>

val decIfImage: x: 'a -> 'a

val decRef: unit -> SlimPipeline.Stage<'a,'a>

val releaseAfter: f: (Image<'S> -> 'T) -> I: Image<'S> -> 'T when 'S: equality

val releaseAfter2:
  f: (Image<'S> -> Image<'S> -> 'T) -> I: Image<'S> -> J: Image<'S> -> 'T
    when 'S: equality

val (>=>) :
  (SlimPipeline.Plan<'a,'b> ->
     SlimPipeline.Stage<'b,'c> -> SlimPipeline.Plan<'a,'c>) when 'c: equality

val (-->) :
  (SlimPipeline.Stage<'a,'b> ->
     SlimPipeline.Stage<'b,'c> -> SlimPipeline.Stage<'a,'c>)

val source: (uint64 -> SlimPipeline.Plan<unit,unit>)

val debug: availableMemory: uint64 -> SlimPipeline.Plan<unit,unit>

val zip:
  (SlimPipeline.Plan<'a,'b> ->
     SlimPipeline.Plan<'a,'c> -> SlimPipeline.Plan<'a,('b * 'c)>)
    when 'b: equality and 'c: equality

val promoteStreamingToWindow:
  name: string ->
    winSz: uint ->
    pad: uint ->
    stride: uint ->
    emitStart: uint -> emitCount: uint -> stage: Stage<'T,'S> -> Stage<'T,'S>
    when 'T: equality

val (>=>>) :
  pl: SlimPipeline.Plan<'In,'S> ->
    stage1: Stage<'S,'U> * stage2: Stage<'S,'V> ->
      SlimPipeline.Plan<'In,('U * 'V)>
    when 'S: equality and 'U: equality and 'V: equality

val (>>=>) :
  (SlimPipeline.Plan<'a,('b * 'c)> ->
     ('b -> 'c -> 'd) -> SlimPipeline.Plan<'a,'d>) when 'd: equality

val (>>=>>) :
  (('a * 'b -> 'c * 'd) ->
     SlimPipeline.Plan<'e,('a * 'b)> ->
     SlimPipeline.Stage<('a * 'b),('c * 'd)> -> SlimPipeline.Plan<'e,('c * 'd)>)
    when 'c: equality and 'd: equality

val zeroMaker: index: int -> ex: Image<'S> -> Image<'S> when 'S: equality

val window:
  windowSize: uint ->
    pad: uint -> stride: uint -> SlimPipeline.Stage<Image<'a>,Image<'a> list>
    when 'a: equality

val flatten: unit -> SlimPipeline.Stage<'a list,'a>

val map: f: ('a -> 'b) -> SlimPipeline.Stage<'a,'b>

val sink: pl: SlimPipeline.Plan<unit,unit> -> unit

val sinkList: plLst: SlimPipeline.Plan<unit,unit> list -> unit

val drain: pl: SlimPipeline.Plan<unit,'a> -> 'a

val drainList: pl: SlimPipeline.Plan<unit,'a> -> 'a list

val drainLast: pl: SlimPipeline.Plan<unit,'a> -> 'a

val tap: (string -> SlimPipeline.Stage<'a,'a>)

val tapIt: (('a -> string) -> SlimPipeline.Stage<'a,'a>)

val ignoreSingles: unit -> Stage<'a,unit>

val ignorePairs: unit -> Stage<('a * unit),unit>

val idStage<'T> : (string -> SlimPipeline.Stage<'T,'T>)

val liftUnary:
  name: string ->
    (('a -> 'b) ->
       SlimPipeline.MemoryNeed ->
       SlimPipeline.LengthTransformation -> SlimPipeline.Stage<'a,'b>)

val liftUnaryReleaseAfter:
  name: string ->
    f: (Image<'S> -> Image<'T>) ->
    memoryNeed: SlimPipeline.MemoryNeed ->
    lengthTransformation: SlimPipeline.LengthTransformation ->
    SlimPipeline.Stage<#Image<'S>,Image<'T>> when 'S: equality and 'T: equality

val getBytesPerComponent<'T> : uint64
type System.String with
    
    member icompare: s2: string -> bool

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
val liftRelease2:
  f: (Image<'a> -> Image<'a> -> 'b) -> I: Image<'a> -> J: Image<'a> -> 'b
    when 'a: equality

val memNeeded<'T> : nTimes: uint64 -> nElems: uint64 -> uint64

val add:
  image: Image<'T> -> SlimPipeline.Stage<#Image<'T>,Image<'T>> when 'T: equality

val addPair: I: Image<'a> -> J: Image<'a> -> Image.Image<'a> when 'a: equality

val inline scalarAddImage:
  i: ^T -> SlimPipeline.Stage<Image<^T>,Image<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val inline imageAddScalar:
  i: ^T -> SlimPipeline.Stage<Image<^T>,Image<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val sub:
  image: Image<'T> -> SlimPipeline.Stage<#Image<'T>,Image<'T>> when 'T: equality

val subPair: I: Image<'a> -> J: Image<'a> -> Image.Image<'a> when 'a: equality

val inline scalarSubImage:
  i: ^T -> SlimPipeline.Stage<Image<^T>,Image<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val inline imageSubScalar:
  i: ^T -> SlimPipeline.Stage<Image<^T>,Image<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val mul:
  image: Image<'T> -> SlimPipeline.Stage<#Image<'T>,Image<'T>> when 'T: equality

val mulPair: I: Image<'a> -> J: Image<'a> -> Image.Image<'a> when 'a: equality

val inline scalarMulImage:
  i: ^T -> SlimPipeline.Stage<Image<^T>,Image<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val inline imageMulScalar:
  i: ^T -> SlimPipeline.Stage<Image<^T>,Image<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val div:
  image: Image<'T> -> SlimPipeline.Stage<#Image<'T>,Image<'T>> when 'T: equality

val divPair: I: Image<'a> -> J: Image<'a> -> Image.Image<'a> when 'a: equality

val inline scalarDivImage:
  i: ^T -> SlimPipeline.Stage<Image<^T>,Image<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val inline imageDivScalar:
  i: ^T -> SlimPipeline.Stage<Image<^T>,Image<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val maxOfPair: I: Image<'a> -> J: Image<'a> -> Image.Image<'a> when 'a: equality

val minOfPair: I: Image<'a> -> J: Image<'a> -> Image.Image<'a> when 'a: equality

val getMinMax: I: Image<'a> -> float * float when 'a: equality

val failTypeMismatch<'T> : name: string -> lst: System.Type list -> unit

/// Simple functions
val private floatNintTypes: System.Type list

val private floatTypes: System.Type list

val abs<'T when 'T: equality> : Stage<Image<'T>,Image<'T>> when 'T: equality

val acos<'T when 'T: equality> : Stage<Image<'T>,Image<'T>> when 'T: equality

val asin<'T when 'T: equality> : Stage<Image<'T>,Image<'T>> when 'T: equality

val atan<'T when 'T: equality> : Stage<Image<'T>,Image<'T>> when 'T: equality

val cos<'T when 'T: equality> : Stage<Image<'T>,Image<'T>> when 'T: equality

val sin<'T when 'T: equality> : Stage<Image<'T>,Image<'T>> when 'T: equality

val tan<'T when 'T: equality> : Stage<Image<'T>,Image<'T>> when 'T: equality

val exp<'T when 'T: equality> : Stage<Image<'T>,Image<'T>> when 'T: equality

val log10<'T when 'T: equality> : Stage<Image<'T>,Image<'T>> when 'T: equality

val log<'T when 'T: equality> : Stage<Image<'T>,Image<'T>> when 'T: equality

val round<'T when 'T: equality> : Stage<Image<'T>,Image<'T>> when 'T: equality

val sqrt<'T when 'T: equality> : Stage<Image<'T>,Image<'T>> when 'T: equality

val square<'T when 'T: equality> : Stage<Image<'T>,Image<'T>> when 'T: equality

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
    winSz: uint option -> Stage<Image<float>,Image<float>>

val discreteGaussian:
  (float ->
     ImageFunctions.OutputRegionMode option ->
     ImageFunctions.BoundaryCondition option ->
     uint option -> Stage<Image<float>,Image<float>>)

val convGauss: sigma: float -> Stage<Image<float>,Image<float>>

val createPadding:
  name: 'a -> pad: uint -> Stage<unit,Image<'S>> when 'S: equality

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
  sigma: float ->
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
val binaryFillHoles:
  winSz: uint -> SlimPipeline.Stage<Image<uint8>,Image.Image<uint8>>

val connectedComponents:
  winSz: uint -> SlimPipeline.Stage<Image<uint8>,Image.Image<uint64>>

val relabelComponents:
  a: uint -> winSz: uint -> SlimPipeline.Stage<Image<'a>,Image.Image<'a>>
    when 'a: equality

val watershed:
  a: float -> winSz: uint -> SlimPipeline.Stage<Image<'a>,Image.Image<'a>>
    when 'a: equality

val signedDistanceMap:
  winSz: uint -> SlimPipeline.Stage<Image<uint8>,Image.Image<float>>

val otsuThreshold:
  winSz: uint -> SlimPipeline.Stage<Image<'a>,Image.Image<uint8>>
    when 'a: equality

val momentsThreshold:
  winSz: uint -> SlimPipeline.Stage<Image<'a>,Image.Image<uint8>>
    when 'a: equality

val threshold:
  a: float -> b: float -> SlimPipeline.Stage<#Image<'b>,Image<uint8>>
    when 'b: equality

val addNormalNoise:
  a: float -> b: float -> SlimPipeline.Stage<#Image<'b>,Image<'b>>
    when 'b: equality

val ImageConstantPad<'T when 'T: equality> :
  padLower: uint list ->
    padUpper: uint list ->
    c: double -> SlimPipeline.Stage<Image<obj>,Image<obj>> when 'T: equality

val readFiles: debug: bool -> Stage<string,Image<'T>> when 'T: equality

val readFilePairs:
  debug: bool -> Stage<(string * string),(Image<'T> * Image<'T>)>
    when 'T: equality

type FileInfo = ImageFunctions.FileInfo

val getStackDepth: inputDir: string -> suffix: string -> uint

val getStackInfo: inputDir: string -> suffix: string -> FileInfo

val getStackSize: inputDir: string -> suffix: string -> uint * uint * uint

val getStackWidth: inputDir: string -> suffix: string -> uint64

val getStackHeight: inputDir: string -> suffix: string -> uint64

val srcStage:
  name: string ->
    width: uint ->
    height: uint ->
    depth: uint ->
    mapper: (int -> Image<'T>) -> SlimPipeline.Stage<unit,Image<'T>>
    when 'T: equality

val srcPlan:
  debug: bool ->
    memAvail: uint64 ->
    width: uint ->
    height: uint ->
    depth: uint ->
    stage: Stage<unit,Image<'T>> option -> SlimPipeline.Plan<unit,Image<'T>>
    when 'T: equality

val zero:
  width: uint ->
    height: uint ->
    depth: uint ->
    pl: SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,Image<'T>>
    when 'T: equality

val createByEuler2DTransform:
  img: Image<'T> ->
    depth: uint ->
    transform: (uint -> (float * float * float) * (float * float)) ->
    pl: SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,Image<'T>>
    when 'T: equality

val getFilenames:
  inputDir: string ->
    suffix: string ->
    filter: (string array -> string array) ->
    pl: SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,string>

val readFiltered:
  inputDir: string ->
    suffix: string ->
    filter: (string array -> string array) ->
    pl: SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,Image<'T>>
    when 'T: equality

val read:
  inputDir: string ->
    suffix: string ->
    pl: SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,Image<'T>>
    when 'T: equality

val readRandom:
  count: uint ->
    inputDir: string ->
    suffix: string ->
    pl: SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,Image<'T>>
    when 'T: equality

val empty: pl: SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,unit>

val getConnectedChunkNeighbours:
  inputDir: string ->
    suffix: string ->
    winSz: uint ->
    pl: SlimPipeline.Plan<unit,unit> ->
    SlimPipeline.Plan<unit,(string * string)>

val makeAdjacencyGraph:
  unit ->
    Stage<(Image<uint64> * Image<uint64>),
          (uint * simpleGraph.Graph<uint * uint64>)>

val makeTranslationTable:
  unit ->
    Stage<(uint * simpleGraph.Graph<uint * uint64>),
          (uint * uint64 * uint64) list>

val trd: 'a * 'b * c: 'c -> 'c

val updateConnectedComponents:
  winSz: uint ->
    translationTable: (uint * uint64 * uint64) list ->
    Stage<Image<uint64>,Image<uint64>>

