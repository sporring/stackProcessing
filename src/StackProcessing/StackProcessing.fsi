
namespace FSharp




module StackProcessing

type Stage<'S,'T> = SlimPipeline.Stage<'S,'T>

type Profile = SlimPipeline.Profile

type ProfileTransition = SlimPipeline.ProfileTransition

type ResourceOps<'T> = SlimPipeline.ResourceOps<'T>

type StageCostCoefficients = SlimPipeline.StageCostCoefficients

type Window<'T> = SlimPipeline.Window<'T>

type Image<'S when 'S: equality> = Image.Image<'S>

type ImageFacts = Image.ImageFacts

val source: (uint64 -> SlimPipeline.Plan<unit,unit>)

val debug: (uint32 -> uint64 -> SlimPipeline.Plan<unit,unit>)

val commandLineSource:
  (uint64 -> string array -> SlimPipeline.Plan<unit,unit> * string array)

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

val (>=>) :
  (SlimPipeline.Plan<'a,'b> ->
     SlimPipeline.Stage<'b,'c> -> SlimPipeline.Plan<'a,'c>) when 'c: equality

val (-->) :
  (SlimPipeline.Stage<'a,'b> ->
     SlimPipeline.Stage<'b,'c> -> SlimPipeline.Stage<'a,'c>)

val (>=>>) :
  (SlimPipeline.Plan<'a,'b> ->
     StackCore.Stage<'b,'c> * StackCore.Stage<'b,'d> ->
       SlimPipeline.Plan<'a,('c * 'd)>)
    when 'b: equality and 'c: equality and 'd: equality

val (>>=>) :
  (SlimPipeline.Plan<'a,('b * 'c)> ->
     ('b -> 'c -> 'd) -> SlimPipeline.Plan<'a,'d>) when 'd: equality

val (>>=>>) :
  (SlimPipeline.Plan<'a,('b * 'c)> ->
     SlimPipeline.Stage<'b,'d> * SlimPipeline.Stage<'c,'e> ->
       SlimPipeline.Plan<'a,('d * 'e)>) when 'd: equality and 'e: equality

val teeFst:
  (SlimPipeline.Stage<'a,'a> -> SlimPipeline.Stage<('a * 'b),('a * 'b)>)

val teeSnd:
  (SlimPipeline.Stage<'a,'a> -> SlimPipeline.Stage<('b * 'a),('b * 'a)>)

val ignoreSingles: (unit -> StackCore.Stage<'a,unit>)

val ignorePairs: (unit -> StackCore.Stage<('a * unit),unit>)

val zeroMaker:
  (int -> StackCore.Image<'a> -> StackCore.Image<'a>) when 'a: equality

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

val readFilesWithShape<'T when 'T: equality> :
  (bool -> uint -> uint -> StackCore.Stage<string,StackCore.Image<'T>>)
    when 'T: equality

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

val cast<'S,'T when 'S: equality and 'T: equality> :
  SlimPipeline.Stage<StackCore.Image<'S>,StackCore.Image<'T>>
    when 'S: equality and 'T: equality

val add:
  (StackCore.Image<'a> ->
     SlimPipeline.Stage<StackCore.Image<'a>,StackCore.Image<'a>>)
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
     SlimPipeline.Stage<StackCore.Image<'a>,StackCore.Image<'a>>)
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
     SlimPipeline.Stage<StackCore.Image<'a>,StackCore.Image<'a>>)
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
     SlimPipeline.Stage<StackCore.Image<'a>,StackCore.Image<'a>>)
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
  (uint -> SlimPipeline.Stage<StackCore.Image<uint8>,StackCore.Image<uint8>>)

val connectedComponents:
  (uint ->
     SlimPipeline.Stage<StackCore.Image<uint8>,
                        (StackCore.Image<uint64> * uint64)>)

val relabelComponents:
  (uint -> uint -> SlimPipeline.Stage<StackCore.Image<'a>,StackCore.Image<'a>>)
    when 'a: equality

val watershed:
  (float -> uint -> SlimPipeline.Stage<StackCore.Image<'a>,StackCore.Image<'a>>)
    when 'a: equality

val signedDistanceMap:
  (uint -> SlimPipeline.Stage<StackCore.Image<uint8>,StackCore.Image<float>>)

val otsuThreshold:
  (uint -> SlimPipeline.Stage<StackCore.Image<'a>,StackCore.Image<uint8>>)
    when 'a: equality

val momentsThreshold:
  (uint -> SlimPipeline.Stage<StackCore.Image<'a>,StackCore.Image<uint8>>)
    when 'a: equality

val threshold:
  (float ->
     float -> SlimPipeline.Stage<StackCore.Image<'a>,StackCore.Image<uint8>>)
    when 'a: equality

val addNormalNoise:
  (float -> float -> SlimPipeline.Stage<StackCore.Image<'a>,StackCore.Image<'a>>)
    when 'a: equality

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

val writeChunkSlices:
  (string ->
     string -> uint -> StackCore.Stage<StackCore.Image<'a>,StackCore.Image<'a>>)
    when 'a: equality

val makeConnectedComponentTranslationTable:
  (uint ->
     StackCore.Stage<(StackCore.Image<uint64> * uint64),
                     (uint * uint64 * uint64) list>)

val updateConnectedComponents:
  (uint ->
     (uint * uint64 * uint64) list ->
     StackCore.Stage<StackCore.Image<uint64>,StackCore.Image<uint64>>)

val permuteAxes:
  (uint * uint * uint ->
     uint -> StackCore.Stage<StackCore.Image<'a>,StackCore.Image<'a>>)
    when 'a: equality

