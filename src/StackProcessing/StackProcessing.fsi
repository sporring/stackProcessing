
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

type Position3D<'T> = StackPoints.Position3D<'T>

type CoordinatePoint = StackPoints.CoordinatePoint

type PointSetChunk = StackPoints.PointSetChunk

type Affine = TinyLinAlg.Affine

type AffineRegistrationOptions = StackRegistration.AffineRegistrationOptions

type AffineRegistrationResult = StackRegistration.AffineRegistrationResult

type ObjectConnectivity = StackObjects.ObjectConnectivity

type ObjectBounds = StackObjects.ObjectBounds

type StreamedObject = StackObjects.StreamedObject

type Point3D = StackMesh.Point3D

type Triangle = StackMesh.Triangle

type MeshChunk = StackMesh.MeshChunk

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

val readRange<'T when 'T: equality> :
  (string ->
     int ->
     string ->
     string ->
     string ->
     SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,StackCore.Image<'T>>)
    when 'T: equality

val getChunkInfo: (string -> string -> StackIO.ChunkInfo)

val getZarrInfo: (string -> int -> int -> StackIO.ChunkInfo)

val getNexusInfo: (string -> string -> int -> int -> int -> StackIO.ChunkInfo)

val getChunkFilename: (string -> string -> int -> int -> int -> string)

val readSlabStacked<'T when 'T: equality> :
  (string ->
     string ->
     SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,StackCore.Image<'T>>)
    when 'T: equality

val readSlabAsWindows<'T when 'T: equality> :
  (string ->
     string ->
     SlimPipeline.Plan<unit,unit> ->
     SlimPipeline.Plan<unit,StackCore.Image<'T> list>) when 'T: equality

val readSlab<'T when 'T: equality> :
  (string ->
     string ->
     SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,StackCore.Image<'T>>)
    when 'T: equality

val readZarrSlabStacked<'T when 'T: equality> :
  (string ->
     uint ->
     int ->
     int ->
     int ->
     int ->
     int ->
     SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,StackCore.Image<'T>>)
    when 'T: equality

val readZarrSlab<'T when 'T: equality> :
  (string ->
     uint ->
     int ->
     int ->
     int ->
     int ->
     int ->
     SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,StackCore.Image<'T>>)
    when 'T: equality

val readNexusSlabStacked<'T when 'T: equality> :
  (string ->
     string ->
     uint ->
     int ->
     int ->
     int ->
     SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,StackCore.Image<'T>>)
    when 'T: equality

val readNexusSlab<'T when 'T: equality> :
  (string ->
     string ->
     uint ->
     int ->
     int ->
     int ->
     SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,StackCore.Image<'T>>)
    when 'T: equality

val readPointSet:
  (string ->
     SlimPipeline.Plan<unit,unit> ->
     SlimPipeline.Plan<unit,StackPoints.PointSetChunk>)

val deleteIfExists: (string -> unit)

val write:
  (string -> string -> StackCore.Stage<StackCore.Image<'a>,StackCore.Image<'a>>)
    when 'a: equality

val writeZarr:
  (string ->
     string ->
     uint ->
     uint ->
     uint ->
     uint ->
     float ->
     float ->
     float -> int -> StackCore.Stage<StackCore.Image<'a>,StackCore.Image<'a>>)
    when 'a: equality

val writeNexus:
  (string ->
     string ->
     uint ->
     uint ->
     uint ->
     uint ->
     int ->
     int -> int -> StackCore.Stage<StackCore.Image<'a>,StackCore.Image<'a>>)
    when 'a: equality

val writeInSlabs:
  (string ->
     string ->
     uint ->
     uint -> uint -> StackCore.Stage<StackCore.Image<'a>,StackCore.Image<'a>>)
    when 'a: equality

val writePointSet: (string -> StackCore.Stage<StackPoints.PointSetChunk,unit>)

val writeMesh: (string -> string -> StackCore.Stage<StackMesh.MeshChunk,unit>)

val defaultAffineRegistrationOptions:
  StackRegistration.AffineRegistrationOptions

val earthMoversDistance:
  (StackPoints.CoordinatePoint seq -> StackPoints.CoordinatePoint seq -> float)

val transformPointSet:
  (TinyLinAlg.Affine -> StackPoints.PointSetChunk -> StackPoints.PointSetChunk)

val inverseAffine: (TinyLinAlg.Affine -> TinyLinAlg.Affine)

val affineRegistration:
  (StackRegistration.AffineRegistrationOptions ->
     StackPoints.CoordinatePoint seq ->
     StackPoints.CoordinatePoint seq ->
     StackRegistration.AffineRegistrationResult)

val streamConnectedObjects<'T when 'T: equality> :
  (StackObjects.ObjectConnectivity ->
     StackCore.Stage<StackCore.Image<'T>,StackObjects.StreamedObject list>)
    when 'T: equality

val paintObjects:
  (uint32 ->
     uint32 ->
     StackCore.Stage<StackObjects.StreamedObject list,StackCore.Image<uint8>>)

val paintObjectsCropped:
  StackCore.Stage<StackObjects.StreamedObject list,StackCore.Image<uint8>>

val resampleAffineTrilinearSlices:
  (string ->
     string ->
     ('a -> 'a -> float32 -> 'a) ->
     int ->
     StackAffineResampler.ImageGeom ->
     StackAffineResampler.ImageGeom ->
     TinyLinAlg.Affine -> 'a -> (int * StackAffineResampler.Image<'a>) seq)
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

val sqrtWindowed<'T when 'T: equality> :
  (uint -> StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>>)
    when 'T: equality

val square<'T when 'T: equality> :
  StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>> when 'T: equality

val clamp<'T when 'T: equality> :
  (double ->
     double -> SlimPipeline.Stage<StackCore.Image<obj>,StackCore.Image<obj>>)
    when 'T: equality

val shiftScale<'T when 'T: equality> :
  (double ->
     double -> SlimPipeline.Stage<StackCore.Image<obj>,StackCore.Image<obj>>)
    when 'T: equality

val intensityStretch<'T when 'T: equality> :
  (double ->
     double ->
     double ->
     double -> SlimPipeline.Stage<StackCore.Image<obj>,StackCore.Image<obj>>)
    when 'T: equality

val median<'T when 'T: equality> :
  (uint32 ->
     uint32 -> SlimPipeline.Stage<StackCore.Image<obj>,StackCore.Image<obj>>)
    when 'T: equality

val bilateral<'T when 'T: equality> :
  (double ->
     double ->
     uint32 -> SlimPipeline.Stage<StackCore.Image<obj>,StackCore.Image<obj>>)
    when 'T: equality

val gradientMagnitude<'T when 'T: equality> :
  (uint32 -> SlimPipeline.Stage<StackCore.Image<obj>,StackCore.Image<obj>>)
    when 'T: equality

val sobelEdge<'T when 'T: equality> :
  (uint32 -> SlimPipeline.Stage<StackCore.Image<obj>,StackCore.Image<obj>>)
    when 'T: equality

val laplacian<'T when 'T: equality> :
  (uint32 -> SlimPipeline.Stage<StackCore.Image<obj>,StackCore.Image<obj>>)
    when 'T: equality

val grayscaleErode<'T when 'T: equality> :
  (uint32 ->
     uint32 -> SlimPipeline.Stage<StackCore.Image<obj>,StackCore.Image<obj>>)
    when 'T: equality

val grayscaleDilate<'T when 'T: equality> :
  (uint32 ->
     uint32 -> SlimPipeline.Stage<StackCore.Image<obj>,StackCore.Image<obj>>)
    when 'T: equality

val grayscaleOpening<'T when 'T: equality> :
  (uint32 ->
     uint32 -> SlimPipeline.Stage<StackCore.Image<obj>,StackCore.Image<obj>>)
    when 'T: equality

val grayscaleClosing<'T when 'T: equality> :
  (uint32 ->
     uint32 -> SlimPipeline.Stage<StackCore.Image<obj>,StackCore.Image<obj>>)
    when 'T: equality

val whiteTopHat<'T when 'T: equality> :
  (uint32 ->
     uint32 -> SlimPipeline.Stage<StackCore.Image<obj>,StackCore.Image<obj>>)
    when 'T: equality

val blackTopHat<'T when 'T: equality> :
  (uint32 ->
     uint32 -> SlimPipeline.Stage<StackCore.Image<obj>,StackCore.Image<obj>>)
    when 'T: equality

val morphologicalGradient<'T when 'T: equality> :
  (uint32 ->
     uint32 -> SlimPipeline.Stage<StackCore.Image<obj>,StackCore.Image<obj>>)
    when 'T: equality

val binaryContour:
  (bool ->
     uint -> SlimPipeline.Stage<StackCore.Image<uint8>,StackCore.Image<uint8>>)

val binaryMedian:
  (uint32 ->
     uint32 -> SlimPipeline.Stage<StackCore.Image<uint8>,StackCore.Image<uint8>>)

val binaryOpeningByReconstruction:
  (uint32 ->
     bool ->
     uint32 -> SlimPipeline.Stage<StackCore.Image<uint8>,StackCore.Image<uint8>>)

val binaryClosingByReconstruction:
  (uint32 ->
     bool ->
     uint32 -> SlimPipeline.Stage<StackCore.Image<uint8>,StackCore.Image<uint8>>)

val binaryReconstructionByDilation:
  (bool ->
     StackCore.Stage<(StackCore.Image<uint8> * StackCore.Image<uint8>),
                     StackCore.Image<uint8>>)

val binaryReconstructionByErosion:
  (bool ->
     StackCore.Stage<(StackCore.Image<uint8> * StackCore.Image<uint8>),
                     StackCore.Image<uint8>>)

val votingBinaryHoleFilling:
  (uint32 ->
     uint ->
     uint32 -> SlimPipeline.Stage<StackCore.Image<uint8>,StackCore.Image<uint8>>)

val equal<'T when 'T: equality> :
  StackCore.Stage<(StackCore.Image<'T> * StackCore.Image<'T>),
                  StackCore.Image<uint8>> when 'T: equality

val notEqual<'T when 'T: equality> :
  StackCore.Stage<(StackCore.Image<'T> * StackCore.Image<'T>),
                  StackCore.Image<uint8>> when 'T: equality

val greater<'T when 'T: equality> :
  StackCore.Stage<(StackCore.Image<'T> * StackCore.Image<'T>),
                  StackCore.Image<uint8>> when 'T: equality

val greaterEqual<'T when 'T: equality> :
  StackCore.Stage<(StackCore.Image<'T> * StackCore.Image<'T>),
                  StackCore.Image<uint8>> when 'T: equality

val less<'T when 'T: equality> :
  StackCore.Stage<(StackCore.Image<'T> * StackCore.Image<'T>),
                  StackCore.Image<uint8>> when 'T: equality

val lessEqual<'T when 'T: equality> :
  StackCore.Stage<(StackCore.Image<'T> * StackCore.Image<'T>),
                  StackCore.Image<uint8>> when 'T: equality

val andMask:
  StackCore.Stage<(StackCore.Image<uint8> * StackCore.Image<uint8>),
                  StackCore.Image<uint8>>

val orMask:
  StackCore.Stage<(StackCore.Image<uint8> * StackCore.Image<uint8>),
                  StackCore.Image<uint8>>

val xorMask:
  StackCore.Stage<(StackCore.Image<uint8> * StackCore.Image<uint8>),
                  StackCore.Image<uint8>>

val notMask: StackCore.Stage<StackCore.Image<uint8>,StackCore.Image<uint8>>

val mask<'T when 'T: equality> :
  (double ->
     StackCore.Stage<(StackCore.Image<'T> * StackCore.Image<uint8>),
                     StackCore.Image<'T>>) when 'T: equality

val labelContour<'T when 'T: equality> :
  (bool ->
     uint32 -> SlimPipeline.Stage<StackCore.Image<obj>,StackCore.Image<obj>>)
    when 'T: equality

val changeLabel<'T when 'T: equality> :
  (double ->
     double -> SlimPipeline.Stage<StackCore.Image<obj>,StackCore.Image<obj>>)
    when 'T: equality

val marchingCubes<'T when 'T: equality> :
  (float -> StackCore.Stage<StackCore.Image<'T>,StackMesh.MeshChunk>)
    when 'T: equality

val dogKeypoints<'T when 'T: equality> :
  (float ->
     float ->
     uint ->
     float ->
     uint -> StackCore.Stage<StackCore.Image<'T>,StackPoints.PointSetChunk>)
    when 'T: equality

val resize<'T when 'T: equality> :
  (uint ->
     uint ->
     uint ->
     string ->
     SlimPipeline.Plan<unit,StackCore.Image<'T>> ->
     SlimPipeline.Plan<unit,StackCore.Image<'T>>) when 'T: equality

val resample<'T when 'T: equality> :
  (float ->
     float ->
     float ->
     string ->
     SlimPipeline.Plan<unit,StackCore.Image<'T>> ->
     SlimPipeline.Plan<unit,StackCore.Image<'T>>) when 'T: equality

val imageHistogram:
  (unit -> SlimPipeline.Stage<StackCore.Image<'a>,Map<'a,uint64>>)
    when 'a: comparison

val imageHistogramFold:
  (unit -> SlimPipeline.Stage<Map<'a,uint64>,Map<'a,uint64>>)
    when 'a: comparison

val histogram:
  (unit -> SlimPipeline.Stage<StackCore.Image<'a>,Map<'a,uint64>>)
    when 'a: comparison

val quantiles: (float list -> Map<'a,uint64> -> float list) when 'a: comparison

val otsuThresholdFromHistogram<'T when 'T: equality> :
  (uint -> StackCore.Image<'T> list -> float) when 'T: equality

val estimateOtsuThreshold<'T when 'T: equality> :
  (uint -> uint -> string -> string -> uint64 -> float) when 'T: equality

val momentsThresholdFromHistogram<'T when 'T: equality> :
  (uint -> StackCore.Image<'T> list -> float) when 'T: equality

val estimateMomentsThreshold<'T when 'T: equality> :
  (uint -> uint -> string -> string -> uint64 -> float) when 'T: equality

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

val createPadding<'T when 'T: equality> :
  (uint ->
     uint ->
     uint ->
     uint ->
     uint ->
     uint -> double -> StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>>)
    when 'T: equality

val crop<'T when 'T: equality> :
  (uint ->
     uint ->
     uint ->
     uint ->
     uint32 -> uint -> StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>>)
    when 'T: equality

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
  (uint ->
     uint -> SlimPipeline.Stage<StackCore.Image<uint8>,StackCore.Image<float>>)

val otsuThreshold:
  (uint ->
     uint ->
     SlimPipeline.Plan<unit,StackCore.Image<'a>> ->
     SlimPipeline.Plan<unit,StackCore.Image<uint8>>) when 'a: equality

val momentsThreshold:
  (uint ->
     uint ->
     SlimPipeline.Plan<unit,StackCore.Image<'a>> ->
     SlimPipeline.Plan<unit,StackCore.Image<uint8>>) when 'a: equality

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

val writeSlabSlices:
  (string ->
     string -> uint -> StackCore.Stage<StackCore.Image<'a>,StackCore.Image<'a>>)
    when 'a: equality

type ComponentStatistics = StackImageFunctions.ComponentStatistics

type ConnectedComponentTranslationTable =
    StackImageFunctions.ConnectedComponentTranslationTable

val makeConnectedComponentTranslationTable:
  (uint ->
     StackCore.Stage<(StackCore.Image<uint64> * uint64),
                     StackImageFunctions.ConnectedComponentTranslationTable>)

val updateConnectedComponents:
  (uint ->
     StackImageFunctions.ConnectedComponentTranslationTable ->
     StackCore.Stage<StackCore.Image<uint64>,StackCore.Image<uint64>>)

val permuteAxes:
  (uint * uint * uint ->
     uint -> StackCore.Stage<StackCore.Image<'a>,StackCore.Image<'a>>)
    when 'a: equality

