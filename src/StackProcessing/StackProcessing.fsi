
namespace FSharp




module StackProcessing

type Stage<'S,'T> = SlimPipeline.Stage<'S,'T>

type Profile = SlimPipeline.Profile

type ProfileTransition = SlimPipeline.ProfileTransition

type ResourceOps<'T> = SlimPipeline.ResourceOps<'T>

type StageTimeCoefficients = SlimPipeline.StageTimeCoefficients

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

type FileInfo = ImageFunctions.FileInfo

type ChunkInfo = StackIO.ChunkInfo

type Position3D<'T> = StackPoints.Position3D<'T>

type CoordinatePoint = StackPoints.CoordinatePoint

type PointSet = StackPoints.PointSet

type VectorizedMatrix = StackPoints.VectorizedMatrix

type Affine = TinyLinAlg.Affine

type AffineRegistrationOptions = StackRegistration.AffineRegistrationOptions

type AffineRegistrationResult = StackRegistration.AffineRegistrationResult

type ImageSetCoordinateSystem = StackManifest.ImageSetCoordinateSystem

type ImageSetTransform = StackManifest.ImageSetTransform

type ImageSetGrid = StackManifest.ImageSetGrid

type ImageSetItem = StackManifest.ImageSetItem

type ImageSetMember = StackManifest.ImageSetItem

type ImageSetManifest = StackManifest.ImageSetManifest

type StitchPlanItem = StackStitching.StitchPlanItem

type StitchPlan = StackStitching.StitchPlan

type ObjectConnectivity = StackObjects.ObjectConnectivity

type ObjectBounds = StackObjects.ObjectBounds

type StreamedObject = StackObjects.StreamedObject

type ObjectMeasurements = StackObjects.ObjectMeasurements

type ObjectSizeStats = StackObjects.ObjectSizeStats

type BiasPolynomialTerm = StackBias.BiasPolynomialTerm

type BiasPolynomialModel = StackBias.BiasPolynomialModel

type SerialSliceTransform = StackSerialSections.SerialSliceTransform

type SerialSliceManifest = StackSerialSections.SerialSliceManifest

type Point3D = StackMesh.Point3D

type Triangle = StackMesh.Triangle

type TriangleSet = StackMesh.TriangleSet

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
     SlimPipeline.Plan<unit,StackPoints.PointSet>)

val coordinateX:
  (uint32 ->
     uint32 ->
     uint32 ->
     SlimPipeline.Plan<unit,unit> ->
     SlimPipeline.Plan<unit,StackCore.Image<float>>)

val coordinateY:
  (uint32 ->
     uint32 ->
     uint32 ->
     SlimPipeline.Plan<unit,unit> ->
     SlimPipeline.Plan<unit,StackCore.Image<float>>)

val coordinateZ:
  (uint32 ->
     uint32 ->
     uint32 ->
     SlimPipeline.Plan<unit,unit> ->
     SlimPipeline.Plan<unit,StackCore.Image<float>>)

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

val writePointSet:
  (string -> string -> StackCore.Stage<StackPoints.PointSet,unit>)

val writeCSVPointSet: (string -> StackCore.Stage<StackPoints.PointSet,unit>)

val vectorizeMatrix: (float array2d -> StackPoints.VectorizedMatrix)

val unvectorizeMatrix: (StackPoints.VectorizedMatrix -> float array2d)

val pointPairDistances:
  (float ->
     float ->
     float -> StackCore.Stage<StackPoints.PointSet,StackPoints.VectorizedMatrix>)

val writeMatrix:
  (string -> string -> StackCore.Stage<StackPoints.VectorizedMatrix,unit>)

val writeCSVMatrix:
  (string -> StackCore.Stage<StackPoints.VectorizedMatrix,unit>)

val writeCSVHistogram<'T when 'T: comparison> :
  (string -> StackCore.Stage<Map<'T,uint64>,unit>) when 'T: comparison

val selectGroupedValueOutput: (uint -> uint -> StackCore.Stage<'a,'a>)

val writeMesh: (string -> string -> StackCore.Stage<StackMesh.TriangleSet,unit>)

val defaultAffineRegistrationOptions:
  StackRegistration.AffineRegistrationOptions

val earthMoversDistance:
  (StackPoints.CoordinatePoint seq -> StackPoints.CoordinatePoint seq -> float)

val transformPointSet:
  (TinyLinAlg.Affine -> StackPoints.PointSet -> StackPoints.PointSet)

val inverseAffine: (TinyLinAlg.Affine -> TinyLinAlg.Affine)

val affineToMatrix: (TinyLinAlg.Affine -> StackPoints.VectorizedMatrix)

val matrixToAffine: (StackPoints.VectorizedMatrix -> TinyLinAlg.Affine)

val affineRegistration:
  (StackRegistration.AffineRegistrationOptions ->
     StackPoints.CoordinatePoint seq ->
     StackPoints.CoordinatePoint seq ->
     StackRegistration.AffineRegistrationResult)

val affineRegistrationMatrices:
  (StackRegistration.AffineRegistrationOptions ->
     StackCore.Stage<(StackPoints.PointSet * StackPoints.PointSet),
                     StackPoints.VectorizedMatrix>)

val identityImageSetTransform: StackManifest.ImageSetTransform

val imageSetTransformFromMatrix:
  (StackPoints.VectorizedMatrix -> StackManifest.ImageSetTransform)

val imageSetTransformToMatrix:
  (StackManifest.ImageSetTransform -> StackPoints.VectorizedMatrix)

val imageSetTransformFromAffine:
  (TinyLinAlg.Affine -> StackManifest.ImageSetTransform)

val imageSetTransformToAffine:
  (StackManifest.ImageSetTransform -> TinyLinAlg.Affine)

val createImageSetManifest: (string -> string -> StackManifest.ImageSetManifest)

val identityImageSetManifest:
  (string -> string -> StackManifest.ImageSetManifest)

val imageSetGrid: (uint64 list -> StackManifest.ImageSetGrid)

val withImageSetGrid:
  (StackManifest.ImageSetGrid ->
     StackManifest.ImageSetManifest -> StackManifest.ImageSetManifest)

val imageSetGridIndexTransform:
  (StackManifest.ImageSetGrid -> int list -> StackManifest.ImageSetTransform)

val composeImageSetTransforms:
  (StackManifest.ImageSetTransform ->
     StackManifest.ImageSetTransform -> StackManifest.ImageSetTransform)

val updateMovingImageSetItemTransformFromRegistration:
  (string ->
     string ->
     StackManifest.ImageSetTransform ->
     StackManifest.ImageSetManifest -> StackManifest.ImageSetManifest)

val imageSetItem:
  (string ->
     string ->
     string ->
     string ->
     uint64 list ->
     float list ->
     StackManifest.ImageSetTransform ->
     string list -> StackManifest.ImageSetItem)

val scalarImageSetItem:
  (string ->
     string ->
     string ->
     uint64 list ->
     float list ->
     StackManifest.ImageSetTransform ->
     string list -> StackManifest.ImageSetItem)

val gridImageSetItem:
  (string ->
     string ->
     string ->
     uint64 list ->
     float list ->
     int list ->
     StackManifest.ImageSetTransform ->
     string list -> StackManifest.ImageSetItem)

val vectorImageSetItem:
  (string ->
     string ->
     string ->
     uint64 list ->
     float list ->
     StackManifest.ImageSetTransform ->
     string list -> StackManifest.ImageSetItem)

val pointSetManifestItem:
  (string ->
     string ->
     string ->
     uint64 list ->
     float list ->
     StackManifest.ImageSetTransform ->
     string list -> StackManifest.ImageSetItem)

val triangleMeshManifestItem:
  (string ->
     string ->
     string ->
     uint64 list ->
     float list ->
     StackManifest.ImageSetTransform ->
     string list -> StackManifest.ImageSetItem)

val matrixManifestItem:
  (string ->
     string ->
     string ->
     StackManifest.ImageSetTransform ->
     string list -> StackManifest.ImageSetItem)

val imageSetMember:
  (string ->
     string ->
     string ->
     uint64 list ->
     float list -> StackManifest.ImageSetTransform -> StackManifest.ImageSetItem)

val addImageSetItem:
  (StackManifest.ImageSetItem ->
     StackManifest.ImageSetManifest -> StackManifest.ImageSetManifest)

val addImageSetMember:
  (StackManifest.ImageSetItem ->
     StackManifest.ImageSetManifest -> StackManifest.ImageSetManifest)

val replaceImageSetItemTransform:
  (string ->
     StackManifest.ImageSetTransform ->
     StackManifest.ImageSetManifest -> StackManifest.ImageSetManifest)

val replaceImageSetMemberTransform:
  (string ->
     StackManifest.ImageSetTransform ->
     StackManifest.ImageSetManifest -> StackManifest.ImageSetManifest)

val writeImageSetManifest: (string -> 'a -> unit)

val readImageSetManifest: (string -> StackManifest.ImageSetManifest)

val createStitchPlan:
  (StackManifest.ImageSetManifest -> string list -> StackStitching.StitchPlan)

val stitchManifestImages<'T when 'T: equality> :
  (string ->
     string list ->
     float ->
     SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,Image.Image<'T>>)
    when 'T: equality

val streamConnectedObjects<'T when 'T: equality> :
  (StackObjects.ObjectConnectivity ->
     StackCore.Stage<StackCore.Image<'T>,StackObjects.StreamedObject list>)
    when 'T: equality

val removeSmallObjects:
  (uint64 ->
     StackObjects.ObjectConnectivity ->
     StackCore.Stage<StackCore.Image<uint8>,StackCore.Image<uint8>>)

val fillSmallHoles:
  (uint64 ->
     StackObjects.ObjectConnectivity ->
     StackCore.Stage<StackCore.Image<uint8>,StackCore.Image<uint8>>)

val paintObjects:
  (uint32 ->
     uint32 ->
     StackCore.Stage<StackObjects.StreamedObject list,StackCore.Image<uint8>>)

val paintObjectsCropped:
  StackCore.Stage<StackObjects.StreamedObject list,StackCore.Image<uint8>>

val measureObjects:
  StackCore.Stage<StackObjects.StreamedObject list,
                  StackObjects.ObjectMeasurements list>

val objectSizeStats:
  StackCore.Stage<StackObjects.ObjectMeasurements list,
                  StackObjects.ObjectSizeStats>

val objectSizeHistogram:
  (uint64 ->
     StackCore.Stage<StackObjects.ObjectMeasurements list,Map<uint64,uint64>>)

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

val toVectorImage<'T when 'T: equality> :
  Stage<(Image<'T> * Image<'T>),Image<'T list>> when 'T: equality

val vectorElement:
  componentId: uint -> Stage<Image<'T list>,Image<'T>> when 'T: equality

val appendVectorElement:
  Stage<(Image<float list> * Image<float>),Image<float list>>

val vectorMapElements:
  functionName: string -> Stage<Image<float list>,Image<float list>>

val vectorDot: Stage<(Image<float list> * Image<float list>),Image<float>>

val vectorCross3D:
  Stage<(Image<float list> * Image<float list>),Image<float list>>

val vectorAngleTo:
  reference: float list -> Stage<Image<float list>,Image<float>>

val Re: Stage<Image<System.Numerics.Complex>,Image<float>>

val Im: Stage<Image<System.Numerics.Complex>,Image<float>>

val modulus: Stage<Image<System.Numerics.Complex>,Image<float>>

val arg: Stage<Image<System.Numerics.Complex>,Image<float>>

val conjugate:
  Stage<Image<System.Numerics.Complex>,Image<System.Numerics.Complex>>

val toComplex:
  Stage<(Image<float> * Image<float>),Image<System.Numerics.Complex>>

val polarToComplex:
  Stage<(Image<float> * Image<float>),Image<System.Numerics.Complex>>

val FFT:
  chunkX: uint ->
    chunkY: uint ->
    chunkZ: uint -> Stage<Image<'T>,Image<System.Numerics.Complex>>
    when 'T: equality

val invFFT:
  chunkX: uint ->
    chunkY: uint ->
    chunkZ: uint -> Stage<Image<System.Numerics.Complex>,Image<float>>

val shiftFFT:
  chunkX: uint ->
    chunkY: uint ->
    chunkZ: uint ->
    Stage<Image<System.Numerics.Complex>,Image<System.Numerics.Complex>>

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

val clamp:
  lower: double -> upper: double -> Stage<Image<'T>,Image<'T>> when 'T: equality

val shiftScale:
  shift: double -> scale: double -> Stage<Image<'T>,Image<'T>> when 'T: equality

val intensityStretch:
  inputMinimum: double ->
    inputMaximum: double ->
    outputMinimum: double -> outputMaximum: double -> Stage<Image<'T>,Image<'T>>
    when 'T: equality

val smoothWMedian:
  radius: uint32 -> winSz: uint32 -> Stage<Image<'T>,Image<'T>>
    when 'T: equality

val smoothWBilateral:
  domainSigma: double ->
    rangeSigma: double -> winSz: uint32 -> Stage<Image<'T>,Image<'T>>
    when 'T: equality

val gradientMagnitude:
  winSz: uint32 -> Stage<Image<'T>,Image<'T>> when 'T: equality

val gradient:
  order: uint -> winSz: uint option -> Stage<Image<float>,Image<float list>>

val sobelEdge: winSz: uint32 -> Stage<Image<'T>,Image<'T>> when 'T: equality

val laplacian: winSz: uint32 -> Stage<Image<'T>,Image<'T>> when 'T: equality

val grayscaleErode:
  radius: uint32 -> winSz: uint32 -> Stage<Image<'T>,Image<'T>>
    when 'T: equality

val grayscaleDilate:
  radius: uint32 -> winSz: uint32 -> Stage<Image<'T>,Image<'T>>
    when 'T: equality

val grayscaleOpening:
  radius: uint32 -> winSz: uint32 -> Stage<Image<'T>,Image<'T>>
    when 'T: equality

val grayscaleClosing:
  radius: uint32 -> winSz: uint32 -> Stage<Image<'T>,Image<'T>>
    when 'T: equality

val whiteTopHat:
  radius: uint32 -> winSz: uint32 -> Stage<Image<'T>,Image<'T>>
    when 'T: equality

val blackTopHat:
  radius: uint32 -> winSz: uint32 -> Stage<Image<'T>,Image<'T>>
    when 'T: equality

val morphologicalGradient:
  radius: uint32 -> winSz: uint32 -> Stage<Image<'T>,Image<'T>>
    when 'T: equality

val binaryContour:
  (bool ->
     uint -> SlimPipeline.Stage<StackCore.Image<uint8>,StackCore.Image<uint8>>)

val binaryMedian:
  (uint32 ->
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

val maskAnd:
  StackCore.Stage<(StackCore.Image<uint8> * StackCore.Image<uint8>),
                  StackCore.Image<uint8>>

val maskOr:
  StackCore.Stage<(StackCore.Image<uint8> * StackCore.Image<uint8>),
                  StackCore.Image<uint8>>

val maskXor:
  StackCore.Stage<(StackCore.Image<uint8> * StackCore.Image<uint8>),
                  StackCore.Image<uint8>>

val maskNot: StackCore.Stage<StackCore.Image<uint8>,StackCore.Image<uint8>>

val labelContour:
  fullyConnected: bool -> winSz: uint32 -> Stage<Image<'T>,Image<'T>>
    when 'T: equality

val changeLabel:
  fromLabel: double -> toLabel: double -> Stage<Image<'T>,Image<'T>>
    when 'T: equality

val marchingCubes<'T when 'T: equality> :
  (float -> StackCore.Stage<StackCore.Image<'T>,StackMesh.TriangleSet>)
    when 'T: equality

val surfaceArea:
  (float -> float -> float -> StackCore.Stage<StackMesh.TriangleSet,float>)

val dogKeypoints<'T when 'T: equality> :
  (float ->
     float ->
     uint ->
     float -> uint -> StackCore.Stage<StackCore.Image<'T>,StackPoints.PointSet>)
    when 'T: equality

val logBlobKeypoints<'T when 'T: equality> :
  (float ->
     float -> uint -> StackCore.Stage<StackCore.Image<'T>,StackPoints.PointSet>)
    when 'T: equality

val hessianKeypoints<'T when 'T: equality> :
  (float ->
     string ->
     float -> uint -> StackCore.Stage<StackCore.Image<'T>,StackPoints.PointSet>)
    when 'T: equality

val harris3DKeypoints<'T when 'T: equality> :
  (float ->
     float ->
     float ->
     float -> uint -> StackCore.Stage<StackCore.Image<'T>,StackPoints.PointSet>)
    when 'T: equality

val forstner3DKeypoints<'T when 'T: equality> :
  (float ->
     float ->
     float -> uint -> StackCore.Stage<StackCore.Image<'T>,StackPoints.PointSet>)
    when 'T: equality

val phaseCongruencyKeypoints<'T when 'T: equality> :
  (float ->
     float -> uint -> StackCore.Stage<StackCore.Image<'T>,StackPoints.PointSet>)
    when 'T: equality

val siftKeypoints<'T when 'T: equality> :
  (float ->
     float ->
     uint ->
     float -> uint -> StackCore.Stage<StackCore.Image<'T>,StackPoints.PointSet>)
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

val histogram:
  (unit -> SlimPipeline.Stage<StackCore.Image<'a>,Map<'a,uint64>>)
    when 'a: comparison

val sumProjection<'T when 'T: equality> :
  (string -> StackCore.Stage<StackCore.Image<'T>,StackCore.Image<float>>)
    when 'T: equality

val volume:
  (float -> float -> float -> StackCore.Stage<StackCore.Image<uint8>,float>)

val fitBiasModel<'T when 'T: equality> :
  (int ->
     uint32 ->
     StackCore.Stage<StackCore.Image<'T>,StackBias.BiasPolynomialModel>)
    when 'T: equality

val fitBiasModelMasked<'T when 'T: equality> :
  (int ->
     uint32 ->
     StackCore.Stage<(StackCore.Image<'T> * StackCore.Image<uint8>),
                     StackBias.BiasPolynomialModel>) when 'T: equality

val correctBias<'T when 'T: equality> :
  (StackBias.BiasPolynomialModel ->
     StackCore.Stage<StackCore.Image<'T>,StackCore.Image<float>>)
    when 'T: equality

val correctBiasMasked<'T when 'T: equality> :
  (StackBias.BiasPolynomialModel ->
     StackCore.Stage<(StackCore.Image<'T> * StackCore.Image<uint8>),
                     StackCore.Image<float>>) when 'T: equality

val serialIdentityManifest:
  (uint32 -> uint32 -> uint32 -> StackSerialSections.SerialSliceManifest)

val serialPolynomialBiasCorrect<'T when 'T: equality> :
  (int -> StackCore.Stage<StackCore.Image<'T>,StackCore.Image<float>>)
    when 'T: equality

val serialKeypoints2D<'T when 'T: equality> :
  (float -> float -> StackCore.Stage<StackCore.Image<'T>,StackPoints.PointSet>)
    when 'T: equality

val serialKeypointTranslationManifest:
  (uint ->
     uint ->
     StackCore.Stage<StackPoints.PointSet,
                     StackSerialSections.SerialSliceManifest>)

val serialImageTranslationManifest<'T when 'T: equality> :
  (int ->
     StackCore.Stage<StackCore.Image<'T>,StackSerialSections.SerialSliceManifest>)
    when 'T: equality

val serialApplyManifest<'T when 'T: equality> :
  (StackSerialSections.SerialSliceManifest ->
     float -> StackCore.Stage<StackCore.Image<'T>,StackCore.Image<float>>)
    when 'T: equality

val serialApplyManifestInBoundingBox<'T when 'T: equality> :
  (StackSerialSections.SerialSliceManifest ->
     float -> StackCore.Stage<StackCore.Image<'T>,StackCore.Image<float>>)
    when 'T: equality

val quantiles: (float list -> Map<'a,uint64> -> float list) when 'a: comparison

val otsuThresholdFromHistogram: (Map<'a,uint64> -> float) when 'a: comparison

val momentsThresholdFromHistogram: (Map<'a,uint64> -> float) when 'a: comparison

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

val computeStats:
  (unit ->
     SlimPipeline.Stage<StackCore.Image<'a>,StackImageFunctions.ImageStats>)
    when 'a: equality

val smoothWGauss:
  (float ->
     ImageFunctions.OutputRegionMode option ->
     ImageFunctions.BoundaryCondition option ->
     uint option ->
     StackCore.Stage<StackCore.Image<float>,StackCore.Image<float>>)

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
  (uint ->
     uint -> StackCore.Stage<StackCore.Image<float>,StackCore.Image<float>>)

val structureTensor:
  sigma: float -> rho: float -> Stage<Image<float>,Image<float list>>

val PCA: components: uint32 -> Stage<Image<float list>,Image<float list>>

val selectGroupedOutput:
  groupSize: uint -> part: uint -> Stage<Image<'T>,Image<'T>> when 'T: equality

val erode:
  (uint -> StackCore.Stage<StackCore.Image<uint8>,StackCore.Image<uint8>>)

val dilate:
  (uint -> StackCore.Stage<StackCore.Image<uint8>,StackCore.Image<uint8>>)

val opening:
  (uint -> StackCore.Stage<StackCore.Image<uint8>,StackCore.Image<uint8>>)

val closing:
  (uint -> StackCore.Stage<StackCore.Image<uint8>,StackCore.Image<uint8>>)

val connectedComponents:
  (uint ->
     SlimPipeline.Stage<StackCore.Image<uint8>,
                        (StackCore.Image<uint64> * uint64)>)

val relabelComponents:
  (uint -> uint -> SlimPipeline.Stage<StackCore.Image<'a>,StackCore.Image<'a>>)
    when 'a: equality

val signedDistanceBand:
  (uint ->
     uint -> SlimPipeline.Stage<StackCore.Image<uint8>,StackCore.Image<float>>)

val threshold:
  (float ->
     float -> SlimPipeline.Stage<StackCore.Image<'a>,StackCore.Image<uint8>>)
    when 'a: equality

val addNormalNoise:
  (float -> float -> SlimPipeline.Stage<StackCore.Image<'a>,StackCore.Image<'a>>)
    when 'a: equality

val addSaltAndPepperNoise:
  (float -> SlimPipeline.Stage<StackCore.Image<'a>,StackCore.Image<'a>>)
    when 'a: equality

val addShotNoise:
  (float -> SlimPipeline.Stage<StackCore.Image<'a>,StackCore.Image<'a>>)
    when 'a: equality

val addSpeckleNoise:
  (float -> SlimPipeline.Stage<StackCore.Image<'a>,StackCore.Image<'a>>)
    when 'a: equality

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

val normalNoise<'T when 'T: equality> :
  (uint ->
     uint ->
     uint ->
     float ->
     float ->
     SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,StackCore.Image<'T>>)
    when 'T: equality

val saltAndPepperNoise<'T when 'T: equality> :
  (uint ->
     uint ->
     uint ->
     float ->
     SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,StackCore.Image<'T>>)
    when 'T: equality

val shotNoise<'T when 'T: equality> :
  (uint ->
     uint ->
     uint ->
     float ->
     SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,StackCore.Image<'T>>)
    when 'T: equality

val speckleNoise<'T when 'T: equality> :
  (uint ->
     uint ->
     uint ->
     float ->
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

