namespace FSharp




module TinyLinAlg

[<Struct>]
type V3 =
    {
      x: float
      y: float
      z: float
    }

[<Struct>]
type M3 =
    {
      m00: float
      m01: float
      m02: float
      m10: float
      m11: float
      m12: float
      m20: float
      m21: float
      m22: float
    }

val inline v3: x: float -> y: float -> z: float -> V3

val inline add: a: V3 -> b: V3 -> V3

val inline sub: a: V3 -> b: V3 -> V3

val inline scale: s: float -> a: V3 -> V3

val inline mulMV: m: M3 -> v: V3 -> V3

val inline det3: m: M3 -> float

val inv3: m: M3 -> M3

type Affine =
    {
      A: M3
      T: V3
      C: V3
    }

val affinePoint: a: Affine -> p: V3 -> V3


module StackCore

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

val volFctToLstFctReleaseAfter:
  f: (Image<'S> -> Image<'T>) ->
    pad: uint ->
    stride: uint -> images: Image.Image<'S> list -> Image.Image<'T> list
    when 'S: equality and 'T: equality

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

val ignoreSingles: unit -> Stage<'a,unit>

val ignorePairs: unit -> Stage<('a * unit),unit>

val zeroMaker: index: int -> ex: Image<'S> -> Image<'S> when 'S: equality

val window:
  windowSize: uint ->
    pad: uint -> stride: uint -> SlimPipeline.Stage<Image<'a>,Image<'a> list>
    when 'a: equality

val flatten: unit -> SlimPipeline.Stage<'a list,'a>

val map: f: (bool -> 'a -> 'b) -> SlimPipeline.Stage<'a,'b>

val sinkOp: pl: SlimPipeline.Plan<unit,unit> -> unit

val sink: pl: SlimPipeline.Plan<unit,'T> -> unit

val sinkList: plLst: SlimPipeline.Plan<unit,unit> list -> unit

val drain: pl: SlimPipeline.Plan<unit,'a> -> 'a

val drainList: pl: SlimPipeline.Plan<unit,'a> -> 'a list

val drainLast: pl: SlimPipeline.Plan<unit,'a> -> 'a

val tap: (string -> SlimPipeline.Stage<'a,'a>)

val tapIt: (('a -> string) -> SlimPipeline.Stage<'a,'a>)

val idStage<'T> : (string -> SlimPipeline.Stage<'T,'T>)


module StackIO

type FileInfo = ImageFunctions.FileInfo

type ChunkInfo =
    {
      chunks: int list
      size: uint64 list
      topLeftInfo: FileInfo
    }

val getStackDepth: inputDir: string -> suffix: string -> uint

val getStackInfo: inputDir: string -> suffix: string -> FileInfo

val getStackSize: inputDir: string -> suffix: string -> uint * uint * uint

val getStackWidth: inputDir: string -> suffix: string -> uint64

val getStackHeight: inputDir: string -> suffix: string -> uint64

val _getFilenames:
  inputDir: string ->
    suffix: string -> filter: (string array -> string array) -> string array

val getFilenames:
  inputDir: string ->
    suffix: string ->
    filter: (string array -> string array) ->
    pl: SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,string>

val readFiles:
  debug: bool -> StackCore.Stage<string,StackCore.Image<'T>> when 'T: equality

val readFilePairs:
  debug: bool ->
    StackCore.Stage<(string * string),
                    (StackCore.Image<'T> * StackCore.Image<'T>)>
    when 'T: equality

val readFiltered:
  inputDir: string ->
    suffix: string ->
    filter: (string array -> string array) ->
    pl: SlimPipeline.Plan<unit,unit> ->
    SlimPipeline.Plan<unit,StackCore.Image<'T>> when 'T: equality

val read:
  inputDir: string ->
    suffix: string ->
    pl: SlimPipeline.Plan<unit,unit> ->
    SlimPipeline.Plan<unit,StackCore.Image<'T>> when 'T: equality

val readRandom:
  count: uint ->
    inputDir: string ->
    suffix: string ->
    pl: SlimPipeline.Plan<unit,unit> ->
    SlimPipeline.Plan<unit,StackCore.Image<'T>> when 'T: equality

val getChunkInfo: inputDir: string -> suffix: string -> ChunkInfo

val getChunkFilename:
  path: string -> suffix: string -> i: int -> j: int -> k: int -> string

val _readChunk:
  inputDir: string ->
    suffix: string -> i: int -> j: int -> k: int -> Image.Image<'T>
    when 'T: equality

val _readChunkSlice:
  inputDir: string ->
    suffix: string ->
    chunkInfo: ChunkInfo -> udir: uint -> idx: int -> StackCore.Image<'T>
    when 'T: equality

val readChunksAsWindows:
  inputDir: string ->
    suffix: string ->
    pl: SlimPipeline.Plan<unit,unit> ->
    SlimPipeline.Plan<unit,StackCore.Image<'T> list> when 'T: equality

val readChunks:
  inputDir: string ->
    suffix: string ->
    pl: SlimPipeline.Plan<unit,unit> ->
    SlimPipeline.Plan<unit,StackCore.Image<'T>> when 'T: equality

val icompare: s1: string -> s2: string -> bool

val rnd: System.Random

val getUnusedDirectoryName: dir: string -> string

val deleteIfExists: dir: string -> unit

val write:
  outputDir: string ->
    suffix: string -> StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>>
    when 'T: equality

val _writeChunkSlice:
  debug: bool ->
    outputDir: string ->
    suffix: string ->
    width: uint ->
    height: uint -> winSz: uint -> k: int -> stack: StackCore.Image<'T> -> unit
    when 'T: equality

val _writeChunks:
  debug: bool ->
    outputDir: string ->
    suffix: string ->
    width: uint ->
    height: uint -> winSz: uint -> stack: StackCore.Image<'T> -> unit
    when 'T: equality

val writeInChunks:
  outputDir: string ->
    suffix: string ->
    width: uint ->
    height: uint ->
    winSz: uint -> StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>>
    when 'T: equality


module StackImageFunctions

val liftUnary:
  name: string ->
    (('a -> 'b) ->
       SlimPipeline.MemoryNeed ->
       SlimPipeline.LengthTransformation -> SlimPipeline.Stage<'a,'b>)

val liftUnaryReleaseAfter:
  name: string ->
    f: (StackCore.Image<'S> -> StackCore.Image<'T>) ->
    memoryNeed: SlimPipeline.MemoryNeed ->
    lengthTransformation: SlimPipeline.LengthTransformation ->
    SlimPipeline.Stage<#StackCore.Image<'S>,StackCore.Image<'T>>
    when 'S: equality and 'T: equality

val getBytesPerComponent<'T> : uint64
type System.String with
    
    member icompare: s2: string -> bool

/// Pixel type casting
val cast<'S,'T when 'S: equality and 'T: equality> :
  SlimPipeline.Stage<StackCore.Image<'S>,StackCore.Image<'T>>
    when 'S: equality and 'T: equality

/// Basic arithmetic
val liftRelease2:
  f: (StackCore.Image<'a> -> StackCore.Image<'a> -> 'b) ->
    I: StackCore.Image<'a> -> J: StackCore.Image<'a> -> 'b when 'a: equality

val memNeeded<'T> : nTimes: uint64 -> nElems: uint64 -> uint64

val add:
  image: StackCore.Image<'T> ->
    SlimPipeline.Stage<#StackCore.Image<'T>,StackCore.Image<'T>>
    when 'T: equality

val addPair:
  I: StackCore.Image<'a> -> J: StackCore.Image<'a> -> Image.Image<'a>
    when 'a: equality

val inline scalarAddImage:
  i: ^T -> SlimPipeline.Stage<StackCore.Image<^T>,StackCore.Image<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val inline imageAddScalar:
  i: ^T -> SlimPipeline.Stage<StackCore.Image<^T>,StackCore.Image<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val sub:
  image: StackCore.Image<'T> ->
    SlimPipeline.Stage<#StackCore.Image<'T>,StackCore.Image<'T>>
    when 'T: equality

val subPair:
  I: StackCore.Image<'a> -> J: StackCore.Image<'a> -> Image.Image<'a>
    when 'a: equality

val inline scalarSubImage:
  i: ^T -> SlimPipeline.Stage<StackCore.Image<^T>,StackCore.Image<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val inline imageSubScalar:
  i: ^T -> SlimPipeline.Stage<StackCore.Image<^T>,StackCore.Image<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val mul:
  image: StackCore.Image<'T> ->
    SlimPipeline.Stage<#StackCore.Image<'T>,StackCore.Image<'T>>
    when 'T: equality

val mulPair:
  I: StackCore.Image<'a> -> J: StackCore.Image<'a> -> Image.Image<'a>
    when 'a: equality

val inline scalarMulImage:
  i: ^T -> SlimPipeline.Stage<StackCore.Image<^T>,StackCore.Image<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val inline imageMulScalar:
  i: ^T -> SlimPipeline.Stage<StackCore.Image<^T>,StackCore.Image<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val div:
  image: StackCore.Image<'T> ->
    SlimPipeline.Stage<#StackCore.Image<'T>,StackCore.Image<'T>>
    when 'T: equality

val divPair:
  I: StackCore.Image<'a> -> J: StackCore.Image<'a> -> Image.Image<'a>
    when 'a: equality

val inline scalarDivImage:
  i: ^T -> SlimPipeline.Stage<StackCore.Image<^T>,StackCore.Image<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val inline imageDivScalar:
  i: ^T -> SlimPipeline.Stage<StackCore.Image<^T>,StackCore.Image<^T>>
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val maxOfPair:
  I: StackCore.Image<'a> -> J: StackCore.Image<'a> -> Image.Image<'a>
    when 'a: equality

val minOfPair:
  I: StackCore.Image<'a> -> J: StackCore.Image<'a> -> Image.Image<'a>
    when 'a: equality

val getMinMax: I: StackCore.Image<'a> -> float * float when 'a: equality

val failTypeMismatch<'T> : name: string -> lst: System.Type list -> unit

/// Simple functions
val private floatNintTypes: System.Type list

val private floatTypes: System.Type list

val abs<'T when 'T: equality> :
  StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>> when 'T: equality

val acos<'T when 'T: equality> :
  StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>> when 'T: equality

val asin<'T when 'T: equality> :
  StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>> when 'T: equality

val atan<'T when 'T: equality> :
  StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>> when 'T: equality

val cos<'T when 'T: equality> :
  StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>> when 'T: equality

val sin<'T when 'T: equality> :
  StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>> when 'T: equality

val tan<'T when 'T: equality> :
  StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>> when 'T: equality

val exp<'T when 'T: equality> :
  StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>> when 'T: equality

val log10<'T when 'T: equality> :
  StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>> when 'T: equality

val log<'T when 'T: equality> :
  StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>> when 'T: equality

val round<'T when 'T: equality> :
  StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>> when 'T: equality

val sqrt<'T when 'T: equality> :
  StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>> when 'T: equality

val square<'T when 'T: equality> :
  StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>> when 'T: equality

val imageHistogram:
  unit -> SlimPipeline.Stage<StackCore.Image<'T>,Map<'T,uint64>>
    when 'T: comparison

val imageHistogramFold:
  unit -> SlimPipeline.Stage<Map<'T,uint64>,Map<'T,uint64>> when 'T: comparison

val histogram:
  unit -> SlimPipeline.Stage<StackCore.Image<'a>,Map<'a,uint64>>
    when 'a: comparison

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
  unit -> SlimPipeline.Stage<StackCore.Image<'T>,ImageStats> when 'T: equality

val imageComputeStatsFold: unit -> SlimPipeline.Stage<ImageStats,ImageStats>

val computeStats:
  unit -> SlimPipeline.Stage<StackCore.Image<'a>,ImageStats> when 'a: equality

val stackFUnstack:
  f: (StackCore.Image<'T> -> uint) ->
    images: StackCore.Image<'T> list ->
    (Image.Image<'a> -> Image.Image<'a> list) when 'T: equality and 'a: equality

val skipNTakeM: n: uint -> m: uint -> lst: 'a list -> 'a list

val stackFUnstackTrim:
  trim: uint32 ->
    f: (StackCore.Image<'T> -> StackCore.Image<'S>) ->
    images: StackCore.Image<'T> list -> Image.Image<'S> list
    when 'T: equality and 'S: equality

val discreteGaussianOp:
  name: string ->
    sigma: float ->
    outputRegionMode: ImageFunctions.OutputRegionMode option ->
    boundaryCondition: ImageFunctions.BoundaryCondition option ->
    winSz: uint option ->
    StackCore.Stage<StackCore.Image<float>,StackCore.Image<float>>

val discreteGaussian:
  (float ->
     ImageFunctions.OutputRegionMode option ->
     ImageFunctions.BoundaryCondition option ->
     uint option ->
     StackCore.Stage<StackCore.Image<float>,StackCore.Image<float>>)

val convGauss:
  sigma: float -> StackCore.Stage<StackCore.Image<float>,StackCore.Image<float>>

val createPadding:
  name: 'a -> pad: uint -> StackCore.Stage<unit,StackCore.Image<'S>>
    when 'S: equality

val convolveOp:
  name: string ->
    kernel: StackCore.Image<'T> ->
    outputRegionMode: ImageFunctions.OutputRegionMode option ->
    bc: ImageFunctions.BoundaryCondition option ->
    winSz: uint option ->
    StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>> when 'T: equality

val convolve:
  kernel: StackCore.Image<'a> ->
    outputRegionMode: ImageFunctions.OutputRegionMode option ->
    boundaryCondition: ImageFunctions.BoundaryCondition option ->
    winSz: uint option ->
    StackCore.Stage<StackCore.Image<'a>,StackCore.Image<'a>> when 'a: equality

val conv:
  kernel: StackCore.Image<'a> ->
    StackCore.Stage<StackCore.Image<'a>,StackCore.Image<'a>> when 'a: equality

val finiteDiff:
  sigma: float ->
    direction: uint ->
    order: uint ->
    StackCore.Stage<StackCore.Image<float>,StackCore.Image<float>>

val private makeMorphOp:
  name: string ->
    radius: uint ->
    winSz: uint option ->
    core: (uint -> StackCore.Image<'T> -> StackCore.Image<'T>) ->
    StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>> when 'T: equality

val erode:
  radius: uint -> StackCore.Stage<StackCore.Image<uint8>,StackCore.Image<uint8>>

val dilate:
  radius: uint -> StackCore.Stage<StackCore.Image<uint8>,StackCore.Image<uint8>>

val opening:
  radius: uint -> StackCore.Stage<StackCore.Image<uint8>,StackCore.Image<uint8>>

val closing:
  radius: uint -> StackCore.Stage<StackCore.Image<uint8>,StackCore.Image<uint8>>

/// Full stack operators
val binaryFillHoles:
  winSz: uint -> SlimPipeline.Stage<StackCore.Image<uint8>,Image.Image<uint8>>

val connectedComponents:
  winSz: uint -> SlimPipeline.Stage<StackCore.Image<uint8>,Image.Image<uint64>>

val relabelComponents:
  a: uint ->
    winSz: uint -> SlimPipeline.Stage<StackCore.Image<'a>,Image.Image<'a>>
    when 'a: equality

val watershed:
  a: float ->
    winSz: uint -> SlimPipeline.Stage<StackCore.Image<'a>,Image.Image<'a>>
    when 'a: equality

val signedDistanceMap:
  winSz: uint -> SlimPipeline.Stage<StackCore.Image<uint8>,Image.Image<float>>

val otsuThreshold:
  winSz: uint -> SlimPipeline.Stage<StackCore.Image<'a>,Image.Image<uint8>>
    when 'a: equality

val momentsThreshold:
  winSz: uint -> SlimPipeline.Stage<StackCore.Image<'a>,Image.Image<uint8>>
    when 'a: equality

val threshold:
  a: float ->
    b: float -> SlimPipeline.Stage<#StackCore.Image<'b>,StackCore.Image<uint8>>
    when 'b: equality

val addNormalNoise:
  a: float ->
    b: float -> SlimPipeline.Stage<#StackCore.Image<'b>,StackCore.Image<'b>>
    when 'b: equality

val ImageConstantPad<'T when 'T: equality> :
  padLower: uint list ->
    padUpper: uint list ->
    c: double -> SlimPipeline.Stage<StackCore.Image<obj>,StackCore.Image<obj>>
    when 'T: equality

val show:
  plt: (StackCore.Image<'T> -> unit) ->
    StackCore.Stage<StackCore.Image<'T>,unit> when 'T: equality

val plot:
  plt: (float list -> float list -> unit) ->
    StackCore.Stage<(float * float) list,unit>

val print: unit -> StackCore.Stage<'T,unit>

val srcStage:
  name: string ->
    width: uint ->
    height: uint ->
    depth: uint ->
    mapper: (int -> StackCore.Image<'T>) ->
    SlimPipeline.Stage<unit,StackCore.Image<'T>> when 'T: equality

val srcPlan:
  debug: bool ->
    memAvail: uint64 ->
    width: uint ->
    height: uint ->
    depth: uint ->
    stage: StackCore.Stage<unit,StackCore.Image<'T>> option ->
    SlimPipeline.Plan<unit,StackCore.Image<'T>> when 'T: equality

val zero:
  width: uint ->
    height: uint ->
    depth: uint ->
    pl: SlimPipeline.Plan<unit,unit> ->
    SlimPipeline.Plan<unit,StackCore.Image<'T>> when 'T: equality

val createByEuler2DTransform:
  img: StackCore.Image<'T> ->
    depth: uint ->
    transform: (uint -> (float * float * float) * (float * float)) ->
    pl: SlimPipeline.Plan<unit,unit> ->
    SlimPipeline.Plan<unit,StackCore.Image<'T>> when 'T: equality

val empty: pl: SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,unit>

val getConnectedChunkNeighbours:
  inputDir: string ->
    suffix: string ->
    winSz: uint ->
    pl: SlimPipeline.Plan<unit,unit> ->
    SlimPipeline.Plan<unit,(StackCore.Image<uint64> * StackCore.Image<uint64>)>

val makeAdjacencyGraph:
  unit ->
    StackCore.Stage<(StackCore.Image<uint64> * StackCore.Image<uint64>),
                    (uint * simpleGraph.Graph<uint * uint64>)>

val makeTranslationTable:
  unit ->
    StackCore.Stage<(uint * simpleGraph.Graph<uint * uint64>),
                    (uint * uint64 * uint64) list>

val trd: 'a * 'b * c: 'c -> 'c

val updateConnectedComponents:
  winSz: uint ->
    translationTable: (uint * uint64 * uint64) list ->
    StackCore.Stage<StackCore.Image<uint64>,StackCore.Image<uint64>>

val permuteAxes:
  i: uint * j: uint * k: uint ->
    winSz: uint -> StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>>
    when 'T: equality


module ChunkedAffineResampler

type Image<'S when 'S: equality> = Image.Image<'S>

val inline floorToInt: x: float -> int

type ImageGeom =
    {
      W: int
      H: int
      D: int
      Origin: TinyLinAlg.V3
      Spacing: TinyLinAlg.V3
      Direction: TinyLinAlg.M3
    }

val indexToPhysical: g: ImageGeom -> i: int -> j: int -> k: int -> TinyLinAlg.V3

val physicalToContIndex:
  g: ImageGeom -> invDir: TinyLinAlg.M3 -> p: TinyLinAlg.V3 -> TinyLinAlg.V3

val inline clamp: lo: int -> hi: int -> v: int -> int

val inline packKey: cx: int -> cy: int -> cz: int -> int64

type ChunkCache<'T when 'T: equality> =
    System.Collections.Generic.Dictionary<int64,Image<'T>>

module ChunkCache =
    
    val create:
      unit -> System.Collections.Generic.Dictionary<int64,Image<'T>>
        when 'T: equality
    
    val Get:
      inputDir: string ->
        suffix: string ->
        cx: int * cy: int * cz: int -> dict: ChunkCache<'T> -> Image<'T>
        when 'T: equality
    
    val KeepOnly:
      required: System.Collections.Generic.HashSet<int64> ->
        dict: ChunkCache<'T> -> unit when 'T: equality
    
    val Ensure:
      inputDir: string ->
        suffix: string ->
        required: (int * int * int) seq ->
        dict: ChunkCache<'T> -> System.Collections.Generic.HashSet<int64>
        when 'T: equality

val getVoxel:
  inputDir: string ->
    suffix: string ->
    winsz: int ->
    W: int ->
    H: int ->
    D: int ->
    background: 'T ->
    cache: ChunkCache<'T> ->
    x: int -> y: int -> z: int -> dict: ChunkCache<'T> -> 'T when 'T: equality

val trilinearSample:
  inputDir: string ->
    suffix: string ->
    winsz: int ->
    W: int ->
    H: int ->
    D: int ->
    background: 'T ->
    lerp: ('T -> 'T -> float32 -> 'T) ->
    c: TinyLinAlg.V3 -> cache: ChunkCache<'T> -> 'T when 'T: equality

val requiredChunksForSliceTrilinear:
  winsz: int ->
    inG: ImageGeom ->
    outG: ImageGeom ->
    affOutToIn: TinyLinAlg.Affine -> k: int -> (int * int * int) seq

val resampleAffineTrilinearSlices:
  inputDir: string ->
    suffix: string ->
    lerp: ('T -> 'T -> float32 -> 'T) ->
    winsz: int ->
    inG: ImageGeom ->
    outG: ImageGeom ->
    affOutToIn: TinyLinAlg.Affine -> background: 'T -> (int * Image<'T>) seq
    when 'T: equality


module StackProcessing

type Stage<'S,'T> = SlimPipeline.Stage<'S,'T>

type Profile = SlimPipeline.Profile

type ProfileTransition = SlimPipeline.ProfileTransition

type Image<'S when 'S: equality> = Image.Image<'S>

val getMem: (unit -> unit)

val incIfImage: ('a -> 'a)

val incRef: (unit -> SlimPipeline.Stage<'a,'a>)

val decIfImage: ('a -> 'a)

val decRef: (unit -> SlimPipeline.Stage<'a,'a>)

val releaseAfter:
  ((StackCore.Image<'a> -> 'b) -> StackCore.Image<'a> -> 'b) when 'a: equality

val releaseAfter2:
  ((StackCore.Image<'a> -> StackCore.Image<'a> -> 'b) ->
     StackCore.Image<'a> -> StackCore.Image<'a> -> 'b) when 'a: equality

val volFctToLstFctReleaseAfter:
  ((StackCore.Image<'a> -> StackCore.Image<'b>) ->
     uint -> uint -> Image.Image<'a> list -> Image.Image<'b> list)
    when 'a: equality and 'b: equality

val (>=>) :
  (SlimPipeline.Plan<'a,'b> ->
     SlimPipeline.Stage<'b,'c> -> SlimPipeline.Plan<'a,'c>) when 'c: equality

val (-->) :
  (SlimPipeline.Stage<'a,'b> ->
     SlimPipeline.Stage<'b,'c> -> SlimPipeline.Stage<'a,'c>)

val source: (uint64 -> SlimPipeline.Plan<unit,unit>)

val debug: (uint64 -> SlimPipeline.Plan<unit,unit>)

val zip:
  (SlimPipeline.Plan<'a,'b> ->
     SlimPipeline.Plan<'a,'c> -> SlimPipeline.Plan<'a,('b * 'c)>)
    when 'b: equality and 'c: equality

val promoteStreamingToWindow:
  (string ->
     uint ->
     uint ->
     uint -> uint -> uint -> StackCore.Stage<'a,'b> -> StackCore.Stage<'a,'b>)
    when 'a: equality

val (>=>>) :
  (SlimPipeline.Plan<'a,'b> ->
     StackCore.Stage<'b,'c> * StackCore.Stage<'b,'d> ->
       SlimPipeline.Plan<'a,('c * 'd)>)
    when 'b: equality and 'c: equality and 'd: equality

val (>>=>) :
  (SlimPipeline.Plan<'a,('b * 'c)> ->
     ('b -> 'c -> 'd) -> SlimPipeline.Plan<'a,'d>) when 'd: equality

val (>>=>>) :
  (('a * 'b -> 'c * 'd) ->
     SlimPipeline.Plan<'e,('a * 'b)> ->
     SlimPipeline.Stage<('a * 'b),('c * 'd)> -> SlimPipeline.Plan<'e,('c * 'd)>)
    when 'c: equality and 'd: equality

val ignoreSingles: (unit -> StackCore.Stage<'a,unit>)

val ignorePairs: (unit -> StackCore.Stage<('a * unit),unit>)

val zeroMaker:
  (int -> StackCore.Image<'a> -> StackCore.Image<'a>) when 'a: equality

val window:
  (uint ->
     uint ->
     uint -> SlimPipeline.Stage<StackCore.Image<'a>,StackCore.Image<'a> list>)
    when 'a: equality

val flatten: (unit -> SlimPipeline.Stage<'a list,'a>)

val map: ((bool -> 'a -> 'b) -> SlimPipeline.Stage<'a,'b>)

val sinkOp: (SlimPipeline.Plan<unit,unit> -> unit)

val sink: (SlimPipeline.Plan<unit,'a> -> unit)

val sinkList: (SlimPipeline.Plan<unit,unit> list -> unit)

val drain: (SlimPipeline.Plan<unit,'a> -> 'a)

val drainList: (SlimPipeline.Plan<unit,'a> -> 'a list)

val drainLast: (SlimPipeline.Plan<unit,'a> -> 'a)

val tap: (string -> SlimPipeline.Stage<'a,'a>)

val tapIt: (('a -> string) -> SlimPipeline.Stage<'a,'a>)

val idStage<'T> : (string -> SlimPipeline.Stage<'T,'T>)

type FileInfo = ImageFunctions.FileInfo

type ChunkInfo = StackIO.ChunkInfo

val getStackDepth: (string -> string -> uint)

val getStackInfo: (string -> string -> StackIO.FileInfo)

val getStackSize: (string -> string -> uint * uint * uint)

val getStackWidth: (string -> string -> uint64)

val getStackHeight: (string -> string -> uint64)

val getFilenames:
  (string ->
     string ->
     (string array -> string array) ->
     SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,string>)

val readFiles<'T when 'T: equality> :
  (bool -> StackCore.Stage<string,StackCore.Image<'T>>) when 'T: equality

val readFilePairs<'T when 'T: equality> :
  (bool ->
     StackCore.Stage<(string * string),
                     (StackCore.Image<'T> * StackCore.Image<'T>)>)
    when 'T: equality

val readFiltered<'T when 'T: equality> :
  (string ->
     string ->
     (string array -> string array) ->
     SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,StackCore.Image<'T>>)
    when 'T: equality

val read<'T when 'T: equality> :
  (string ->
     string ->
     SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,StackCore.Image<'T>>)
    when 'T: equality

val readRandom<'T when 'T: equality> :
  (uint ->
     string ->
     string ->
     SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,StackCore.Image<'T>>)
    when 'T: equality

val getChunkInfo: (string -> string -> StackIO.ChunkInfo)

val getChunkFilename: (string -> string -> int -> int -> int -> string)

val readChunksAsWindows<'T when 'T: equality> :
  (string ->
     string ->
     SlimPipeline.Plan<unit,unit> ->
     SlimPipeline.Plan<unit,StackCore.Image<'T> list>) when 'T: equality

val readChunks<'T when 'T: equality> :
  (string ->
     string ->
     SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,StackCore.Image<'T>>)
    when 'T: equality

val icompare: (string -> string -> bool)

val deleteIfExists: (string -> unit)

val write:
  (string -> string -> StackCore.Stage<StackCore.Image<'a>,StackCore.Image<'a>>)
    when 'a: equality

val writeInChunks:
  (string ->
     string ->
     uint ->
     uint -> uint -> StackCore.Stage<StackCore.Image<'a>,StackCore.Image<'a>>)
    when 'a: equality

val resampleAffineTrilinearSlices:
  (string ->
     string ->
     ('a -> 'a -> float32 -> 'a) ->
     int ->
     ChunkedAffineResampler.ImageGeom ->
     ChunkedAffineResampler.ImageGeom ->
     TinyLinAlg.Affine -> 'a -> (int * ChunkedAffineResampler.Image<'a>) seq)
    when 'a: equality

type ImageStats = ImageFunctions.ImageStats

val liftUnary:
  (string ->
     ('a -> 'b) ->
     SlimPipeline.MemoryNeed ->
     SlimPipeline.LengthTransformation -> SlimPipeline.Stage<'a,'b>)

val liftUnaryReleaseAfter:
  (string ->
     (StackCore.Image<'a> -> StackCore.Image<'b>) ->
     SlimPipeline.MemoryNeed ->
     SlimPipeline.LengthTransformation ->
     SlimPipeline.Stage<#StackCore.Image<'a>,StackCore.Image<'b>>)
    when 'a: equality and 'b: equality

val getBytesPerComponent<'T> : uint64

val cast<'S,'T when 'S: equality and 'T: equality> :
  SlimPipeline.Stage<StackCore.Image<'S>,StackCore.Image<'T>>
    when 'S: equality and 'T: equality

val liftRelease2:
  ((StackCore.Image<'a> -> StackCore.Image<'a> -> 'b) ->
     StackCore.Image<'a> -> StackCore.Image<'a> -> 'b) when 'a: equality

val memNeeded<'T> : (uint64 -> uint64 -> uint64)

val add:
  (StackCore.Image<'a> ->
     SlimPipeline.Stage<#StackCore.Image<'a>,StackCore.Image<'a>>)
    when 'a: equality

val addPair:
  (StackCore.Image<'a> -> StackCore.Image<'a> -> Image.Image<'a>)
    when 'a: equality

val inline scalarAddImage<^T
                            when ^T: equality and
                                 ^T: (static member op_Explicit: ^T -> float)> :
  (^T -> SlimPipeline.Stage<StackCore.Image<^T>,StackCore.Image<^T>>)
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val inline imageAddScalar<^T
                            when ^T: equality and
                                 ^T: (static member op_Explicit: ^T -> float)> :
  (^T -> SlimPipeline.Stage<StackCore.Image<^T>,StackCore.Image<^T>>)
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val sub:
  (StackCore.Image<'a> ->
     SlimPipeline.Stage<#StackCore.Image<'a>,StackCore.Image<'a>>)
    when 'a: equality

val subPair:
  (StackCore.Image<'a> -> StackCore.Image<'a> -> Image.Image<'a>)
    when 'a: equality

val inline scalarSubImage<^T
                            when ^T: equality and
                                 ^T: (static member op_Explicit: ^T -> float)> :
  (^T -> SlimPipeline.Stage<StackCore.Image<^T>,StackCore.Image<^T>>)
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val inline imageSubScalar<^T
                            when ^T: equality and
                                 ^T: (static member op_Explicit: ^T -> float)> :
  (^T -> SlimPipeline.Stage<StackCore.Image<^T>,StackCore.Image<^T>>)
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val mul:
  (StackCore.Image<'a> ->
     SlimPipeline.Stage<#StackCore.Image<'a>,StackCore.Image<'a>>)
    when 'a: equality

val mulPair:
  (StackCore.Image<'a> -> StackCore.Image<'a> -> Image.Image<'a>)
    when 'a: equality

val inline scalarMulImage<^T
                            when ^T: equality and
                                 ^T: (static member op_Explicit: ^T -> float)> :
  (^T -> SlimPipeline.Stage<StackCore.Image<^T>,StackCore.Image<^T>>)
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val inline imageMulScalar<^T
                            when ^T: equality and
                                 ^T: (static member op_Explicit: ^T -> float)> :
  (^T -> SlimPipeline.Stage<StackCore.Image<^T>,StackCore.Image<^T>>)
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val div:
  (StackCore.Image<'a> ->
     SlimPipeline.Stage<#StackCore.Image<'a>,StackCore.Image<'a>>)
    when 'a: equality

val divPair:
  (StackCore.Image<'a> -> StackCore.Image<'a> -> Image.Image<'a>)
    when 'a: equality

val inline scalarDivImage<^T
                            when ^T: equality and
                                 ^T: (static member op_Explicit: ^T -> float)> :
  (^T -> SlimPipeline.Stage<StackCore.Image<^T>,StackCore.Image<^T>>)
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val inline imageDivScalar<^T
                            when ^T: equality and
                                 ^T: (static member op_Explicit: ^T -> float)> :
  (^T -> SlimPipeline.Stage<StackCore.Image<^T>,StackCore.Image<^T>>)
    when ^T: equality and ^T: (static member op_Explicit: ^T -> float)

val maxOfPair:
  (StackCore.Image<'a> -> StackCore.Image<'a> -> Image.Image<'a>)
    when 'a: equality

val minOfPair:
  (StackCore.Image<'a> -> StackCore.Image<'a> -> Image.Image<'a>)
    when 'a: equality

val getMinMax: (StackCore.Image<'a> -> float * float) when 'a: equality

val failTypeMismatch<'T> : (string -> System.Type list -> unit)

val abs<'T when 'T: equality> :
  StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>> when 'T: equality

val acos<'T when 'T: equality> :
  StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>> when 'T: equality

val asin<'T when 'T: equality> :
  StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>> when 'T: equality

val atan<'T when 'T: equality> :
  StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>> when 'T: equality

val cos<'T when 'T: equality> :
  StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>> when 'T: equality

val sin<'T when 'T: equality> :
  StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>> when 'T: equality

val tan<'T when 'T: equality> :
  StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>> when 'T: equality

val exp<'T when 'T: equality> :
  StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>> when 'T: equality

val log10<'T when 'T: equality> :
  StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>> when 'T: equality

val log<'T when 'T: equality> :
  StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>> when 'T: equality

val round<'T when 'T: equality> :
  StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>> when 'T: equality

val sqrt<'T when 'T: equality> :
  StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>> when 'T: equality

val square<'T when 'T: equality> :
  StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>> when 'T: equality

val imageHistogram:
  (unit -> SlimPipeline.Stage<StackCore.Image<'a>,Map<'a,uint64>>)
    when 'a: comparison

val imageHistogramFold:
  (unit -> SlimPipeline.Stage<Map<'a,uint64>,Map<'a,uint64>>)
    when 'a: comparison

val histogram:
  (unit -> SlimPipeline.Stage<StackCore.Image<'a>,Map<'a,uint64>>)
    when 'a: comparison

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

val imageComputeStats:
  (unit ->
     SlimPipeline.Stage<StackCore.Image<'a>,StackImageFunctions.ImageStats>)
    when 'a: equality

val imageComputeStatsFold:
  (unit ->
     SlimPipeline.Stage<StackImageFunctions.ImageStats,
                        StackImageFunctions.ImageStats>)

val computeStats:
  (unit ->
     SlimPipeline.Stage<StackCore.Image<'a>,StackImageFunctions.ImageStats>)
    when 'a: equality

val stackFUnstack:
  ((StackCore.Image<'a> -> uint) ->
     StackCore.Image<'a> list -> Image.Image<'b> -> Image.Image<'b> list)
    when 'a: equality and 'b: equality

val skipNTakeM: (uint -> uint -> 'a list -> 'a list)

val stackFUnstackTrim:
  (uint32 ->
     (StackCore.Image<'a> -> StackCore.Image<'b>) ->
     StackCore.Image<'a> list -> Image.Image<'b> list)
    when 'a: equality and 'b: equality

val discreteGaussianOp:
  (string ->
     float ->
     ImageFunctions.OutputRegionMode option ->
     ImageFunctions.BoundaryCondition option ->
     uint option ->
     StackCore.Stage<StackCore.Image<float>,StackCore.Image<float>>)

val discreteGaussian:
  (float ->
     ImageFunctions.OutputRegionMode option ->
     ImageFunctions.BoundaryCondition option ->
     uint option ->
     StackCore.Stage<StackCore.Image<float>,StackCore.Image<float>>)

val convGauss:
  (float -> StackCore.Stage<StackCore.Image<float>,StackCore.Image<float>>)

val createPadding:
  ('a -> uint -> StackCore.Stage<unit,StackCore.Image<'b>>) when 'b: equality

val convolveOp:
  (string ->
     StackCore.Image<'a> ->
     ImageFunctions.OutputRegionMode option ->
     ImageFunctions.BoundaryCondition option ->
     uint option -> StackCore.Stage<StackCore.Image<'a>,StackCore.Image<'a>>)
    when 'a: equality

val convolve:
  (StackCore.Image<'a> ->
     ImageFunctions.OutputRegionMode option ->
     ImageFunctions.BoundaryCondition option ->
     uint option -> StackCore.Stage<StackCore.Image<'a>,StackCore.Image<'a>>)
    when 'a: equality

val conv:
  (StackCore.Image<'a> ->
     StackCore.Stage<StackCore.Image<'a>,StackCore.Image<'a>>) when 'a: equality

val finiteDiff:
  (float ->
     uint ->
     uint -> StackCore.Stage<StackCore.Image<float>,StackCore.Image<float>>)

val erode:
  (uint -> StackCore.Stage<StackCore.Image<uint8>,StackCore.Image<uint8>>)

val dilate:
  (uint -> StackCore.Stage<StackCore.Image<uint8>,StackCore.Image<uint8>>)

val opening:
  (uint -> StackCore.Stage<StackCore.Image<uint8>,StackCore.Image<uint8>>)

val closing:
  (uint -> StackCore.Stage<StackCore.Image<uint8>,StackCore.Image<uint8>>)

val binaryFillHoles:
  (uint -> SlimPipeline.Stage<StackCore.Image<uint8>,Image.Image<uint8>>)

val connectedComponents:
  (uint -> SlimPipeline.Stage<StackCore.Image<uint8>,Image.Image<uint64>>)

val relabelComponents:
  (uint -> uint -> SlimPipeline.Stage<StackCore.Image<'a>,Image.Image<'a>>)
    when 'a: equality

val watershed:
  (float -> uint -> SlimPipeline.Stage<StackCore.Image<'a>,Image.Image<'a>>)
    when 'a: equality

val signedDistanceMap:
  (uint -> SlimPipeline.Stage<StackCore.Image<uint8>,Image.Image<float>>)

val otsuThreshold:
  (uint -> SlimPipeline.Stage<StackCore.Image<'a>,Image.Image<uint8>>)
    when 'a: equality

val momentsThreshold:
  (uint -> SlimPipeline.Stage<StackCore.Image<'a>,Image.Image<uint8>>)
    when 'a: equality

val threshold:
  (float ->
     float -> SlimPipeline.Stage<#StackCore.Image<'b>,StackCore.Image<uint8>>)
    when 'b: equality

val addNormalNoise:
  (float ->
     float -> SlimPipeline.Stage<#StackCore.Image<'b>,StackCore.Image<'b>>)
    when 'b: equality

val ImageConstantPad<'T when 'T: equality> :
  (uint list ->
     uint list ->
     double -> SlimPipeline.Stage<StackCore.Image<obj>,StackCore.Image<obj>>)
    when 'T: equality

val show:
  ((StackCore.Image<'a> -> unit) -> StackCore.Stage<StackCore.Image<'a>,unit>)
    when 'a: equality

val plot:
  ((float list -> float list -> unit) ->
     StackCore.Stage<(float * float) list,unit>)

val print: (unit -> StackCore.Stage<'a,unit>)

val srcStage:
  (string ->
     uint ->
     uint ->
     uint ->
     (int -> StackCore.Image<'a>) ->
     SlimPipeline.Stage<unit,StackCore.Image<'a>>) when 'a: equality

val srcPlan:
  (bool ->
     uint64 ->
     uint ->
     uint ->
     uint ->
     StackCore.Stage<unit,StackCore.Image<'a>> option ->
     SlimPipeline.Plan<unit,StackCore.Image<'a>>) when 'a: equality

val zero<'T when 'T: equality> :
  (uint ->
     uint ->
     uint ->
     SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,StackCore.Image<'T>>)
    when 'T: equality

val createByEuler2DTransform<'T when 'T: equality> :
  (StackCore.Image<'T> ->
     uint ->
     (uint -> (float * float * float) * (float * float)) ->
     SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,StackCore.Image<'T>>)
    when 'T: equality

val empty: (SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,unit>)

val getConnectedChunkNeighbours:
  (string ->
     string ->
     uint ->
     SlimPipeline.Plan<unit,unit> ->
     SlimPipeline.Plan<unit,(StackCore.Image<uint64> * StackCore.Image<uint64>)>)

val makeAdjacencyGraph:
  (unit ->
     StackCore.Stage<(StackCore.Image<uint64> * StackCore.Image<uint64>),
                     (uint * simpleGraph.Graph<uint * uint64>)>)

val makeTranslationTable:
  (unit ->
     StackCore.Stage<(uint * simpleGraph.Graph<uint * uint64>),
                     (uint * uint64 * uint64) list>)

val trd: ('a * 'b * 'c -> 'c)

val updateConnectedComponents:
  (uint ->
     (uint * uint64 * uint64) list ->
     StackCore.Stage<StackCore.Image<uint64>,StackCore.Image<uint64>>)

val permuteAxes:
  (uint * uint * uint ->
     uint -> StackCore.Stage<StackCore.Image<'a>,StackCore.Image<'a>>)
    when 'a: equality

