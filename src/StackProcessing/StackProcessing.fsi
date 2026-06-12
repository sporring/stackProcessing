
namespace FSharp




module StackProcessing

type Stage<'S,'T> = SlimPipeline.Stage<'S,'T>

type Profile = SlimPipeline.Profile

type ProfileTransition = SlimPipeline.ProfileTransition

type ResourceOps<'T> = SlimPipeline.ResourceOps<'T>

type StageTimeCoefficients = SlimPipeline.StageTimeCoefficients

type Window<'T> = SlimPipeline.Window<'T>

type Slab<'T when 'T: equality> = StackCore.Slab<'T>

type ChunkIndex = StackCore.ChunkIndex

type ChunkLayout = StackCore.ChunkLayout

type Chunk<'T when 'T: equality> = StackCore.Chunk<'T>

type VectorChunk<'T when 'T: equality> = StackCore.VectorChunk<'T>

type Image<'S when 'S: equality> = Image.Image<'S>

type ImageFacts = Image.ImageFacts

type Point2D = StackCore.Point2D

type Polygon2D = StackCore.Polygon2D

val optimizerEnabled: (unit -> bool)

val source: (uint64 -> SlimPipeline.Plan<unit,unit>)

val sourceWithOptimizer: (bool -> uint64 -> SlimPipeline.Plan<unit,unit>)

val debug: (uint32 -> bool -> uint64 -> SlimPipeline.Plan<unit,unit>)

val debugDefault: (uint32 -> uint64 -> SlimPipeline.Plan<unit,unit>)

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

val fork:
  (StackCore.Stage<'a,'b> * StackCore.Stage<'a,'c> ->
     StackCore.Stage<'a,('b * 'c)>)

val (-->>) :
  (StackCore.Stage<'a,'b> ->
     StackCore.Stage<'b,'c> * StackCore.Stage<'b,'d> ->
       StackCore.Stage<'a,('c * 'd)>)

val ignoreSingles: (unit -> StackCore.Stage<'a,unit>)

val ignorePairs: (unit -> StackCore.Stage<('a * 'b),unit>)

val zeroMaker:
  (int -> StackCore.Image<'a> -> StackCore.Image<'a>) when 'a: equality

val sink: (SlimPipeline.Plan<unit,'a> -> unit)

val sinkList: (SlimPipeline.Plan<unit,unit> list -> unit)

val drain: (SlimPipeline.Plan<unit,'a> -> 'a)

val drainList: (SlimPipeline.Plan<unit,'a> -> 'a list)

val drainLast: (SlimPipeline.Plan<unit,'a> -> 'a)

val tap: (string -> SlimPipeline.Stage<'a,'a>)

val tapIt: (('a -> string) -> SlimPipeline.Stage<'a,'a>)

val showChartData:
  (string -> #System.IConvertible seq -> #System.IConvertible seq -> unit)

val showChartDataWithLabels:
  (string ->
     string ->
     string ->
     string -> #System.IConvertible seq -> #System.IConvertible seq -> unit)

val showChart:
  (string -> Map<'a,#System.IConvertible> -> unit)
    when 'a: comparison and 'a :> System.IConvertible

val showChartWithLabels:
  (string -> string -> string -> string -> Map<'a,#System.IConvertible> -> unit)
    when 'a: comparison and 'a :> System.IConvertible

val showChartXY:
  (string -> #System.IConvertible seq -> #System.IConvertible seq -> unit)

val showChartXYWithLabels:
  (string ->
     string ->
     string ->
     string -> #System.IConvertible seq -> #System.IConvertible seq -> unit)

val showImage: (Image.Image<'a> -> unit) when 'a: equality

val showImageWithLabels:
  (string -> string -> string -> string -> Image.Image<'a> -> unit)
    when 'a: equality

val showChunk:
  chunk: StackCore.Chunk<'T> -> unit
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val showChunkWithLabels:
  colorMap: string ->
    title: string ->
    xAxis: string -> yAxis: string -> chunk: StackCore.Chunk<'T> -> unit
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

type FileInfo = ImageFunctions.FileInfo

type ChunkInfo = StackIO.ChunkInfo

type Position3D<'T> = StackPoints.Position3D<'T>

type CoordinatePoint = StackPoints.CoordinatePoint

type PointSet = StackPoints.PointSet

type VectorizedMatrix = StackPoints.VectorizedMatrix

type Affine = TinyLinAlg.Affine

type AffineRegistrationOptions = StackRegistration.AffineRegistrationOptions

type AffineRegistrationResult = StackRegistration.AffineRegistrationResult

type RansacResult<'Model,'Item> = StackRansac.RansacResult<'Model,'Item>

type PointMatch2D = StackRansac.PointMatch2D

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

type SerialVolumeGeometry = StackSerialSections.SerialVolumeGeometry

type Point3D = StackMesh.Point3D

type Triangle = StackMesh.Triangle

type TriangleSet = StackMesh.TriangleSet

val getStackDepth: (string -> string -> uint)

val getFileInfo: (string -> ImageFunctions.FileInfo)

val getStackInfo: (string -> string -> StackIO.FileInfo)

val volumeFilePath: (string -> string -> string)

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

val readVolume<'T when 'T: equality> :
  (string ->
     SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,StackCore.Image<'T>>)
    when 'T: equality

val readChunkVolume<'T
                      when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
                           'T :> System.ValueType> :
  (string ->
     SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val readVolumeRandom<'T when 'T: equality> :
  (uint ->
     string ->
     SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,StackCore.Image<'T>>)
    when 'T: equality

val readVolumeRange<'T when 'T: equality> :
  (uint ->
     int ->
     uint ->
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
  (uint ->
     int ->
     uint ->
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

val readChunkSlices<'T
                      when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
                           'T :> System.ValueType> :
  (string ->
     string ->
     SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val readChunkSlicesRandom<'T
                            when 'T: equality and 'T: (new: unit -> 'T) and
                                 'T: struct and 'T :> System.ValueType> :
  (uint ->
     string ->
     string ->
     SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val readChunkSlicesRange<'T
                           when 'T: equality and 'T: (new: unit -> 'T) and
                                'T: struct and 'T :> System.ValueType> :
  (uint ->
     int ->
     uint ->
     string ->
     string ->
     SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkZero<'T
                when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
                     'T :> System.ValueType> :
  (uint ->
     uint ->
     uint ->
     SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkCoordinateX<'T
                       when 'T: equality and 'T: (new: unit -> 'T) and
                            'T: struct and 'T :> System.ValueType> :
  (uint ->
     uint ->
     uint ->
     SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkCoordinateY<'T
                       when 'T: equality and 'T: (new: unit -> 'T) and
                            'T: struct and 'T :> System.ValueType> :
  (uint ->
     uint ->
     uint ->
     SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkCoordinateZ<'T
                       when 'T: equality and 'T: (new: unit -> 'T) and
                            'T: struct and 'T :> System.ValueType> :
  (uint ->
     uint ->
     uint ->
     SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkPolygonMask:
  (uint -> uint -> StackCore.Polygon2D -> StackCore.Chunk<uint8>)

val chunkEuler2DTransformPath:
  (uint ->
     uint -> uint -> string -> uint -> (float * float * float) * (float * float))

val chunkCreateByEuler2DTransformFromChunk<'T
                                             when 'T: equality and
                                                  'T: (new: unit -> 'T) and
                                                  'T: struct and
                                                  'T :> System.ValueType> :
  (uint ->
     (uint -> (float * float * float) * (float * float)) ->
     StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkRepeat<'T
                  when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
                       'T :> System.ValueType> :
  (StackCore.Chunk<'T> ->
     uint ->
     SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkRepeatStage<'T
                       when 'T: equality and 'T: (new: unit -> 'T) and
                            'T: struct and 'T :> System.ValueType> :
  (uint32 -> StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkPad<'T
               when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
                    'T :> System.ValueType> :
  (uint ->
     uint ->
     uint ->
     uint ->
     uint ->
     uint -> 'T -> StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkCrop<'T
                when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
                     'T :> System.ValueType> :
  (uint ->
     uint ->
     uint ->
     uint ->
     uint -> uint -> StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkSqueeze<'T
                   when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
                        'T :> System.ValueType> :
  StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<'T>>
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkConcatenateAlong<'T
                            when 'T: equality and 'T: (new: unit -> 'T) and
                                 'T: struct and 'T :> System.ValueType> :
  (int32 ->
     StackCore.Stage<(StackCore.Chunk<'T> * StackCore.Chunk<'T>),
                     StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkPermuteAxes<'T
                       when 'T: equality and 'T: (new: unit -> 'T) and
                            'T: struct and 'T :> System.ValueType> :
  (int seq -> StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkResample2DNative<'T
                            when 'T: equality and 'T: (new: unit -> 'T) and
                                 'T: struct and 'T :> System.ValueType> :
  (string ->
     uint32 ->
     uint32 ->
     float -> float -> StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkEuler2DTransformNative<'T
                                  when 'T: equality and 'T: (new: unit -> 'T) and
                                       'T: struct and 'T :> System.ValueType> :
  (double * double * double ->
     double * double -> StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkEuler2DRotateNative<'T
                               when 'T: equality and 'T: (new: unit -> 'T) and
                                    'T: struct and 'T :> System.ValueType> :
  (double * double ->
     double -> StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkResize<'T
                  when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
                       'T :> System.ValueType> :
  (uint32 ->
     uint32 ->
     uint32 ->
     string ->
     SlimPipeline.Plan<unit,StackCore.Chunk<'T>> ->
     SlimPipeline.Plan<unit,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkResample<'T
                    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
                         'T :> System.ValueType> :
  (float ->
     float ->
     float ->
     string ->
     SlimPipeline.Plan<unit,StackCore.Chunk<'T>> ->
     SlimPipeline.Plan<unit,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkShow<'T
                when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
                     'T :> System.ValueType> :
  ((StackCore.Chunk<'T> -> unit) -> StackCore.Stage<StackCore.Chunk<'T>,unit>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkSignedDistanceBandNativeParallelCollect:
  (uint32 ->
     uint32 ->
     int -> StackCore.Stage<StackCore.Chunk<uint8>,StackCore.Chunk<float32>>)

val chunkVectorDotFloat32:
  StackCore.Stage<(StackCore.VectorChunk<float32> *
                   StackCore.VectorChunk<float32>),StackCore.Chunk<float32>>

val chunkVectorMagnitudeFloat32:
  StackCore.Stage<StackCore.VectorChunk<float32>,StackCore.Chunk<float32>>

val chunkVector3ToColorFloat32:
  (float32 ->
     float32 ->
     StackCore.Stage<StackCore.VectorChunk<float32>,StackCore.VectorChunk<uint8>>)

val chunkVectorAngleToFloat32:
  (float32 list ->
     StackCore.Stage<StackCore.VectorChunk<float32>,StackCore.Chunk<float32>>)

val chunkConvolveVectorComponentsFloat32NativeParallelCollect:
  (float32 array ->
     float32 array ->
     float32 array ->
     int ->
     StackCore.Stage<StackCore.VectorChunk<float32>,
                     StackCore.VectorChunk<float32>>)

val chunkToVectorImage<'T
                         when 'T: equality and 'T: (new: unit -> 'T) and
                              'T: struct and 'T :> System.ValueType> :
  StackCore.Stage<(StackCore.Chunk<'T> * StackCore.Chunk<'T>),
                  StackCore.VectorChunk<'T>>
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkVectorElement:
  componentId: uint ->
    StackCore.Stage<StackCore.VectorChunk<'T>,StackCore.Chunk<'T>>
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkVectorRange:
  firstComponent: uint ->
    componentCount: uint ->
    StackCore.Stage<StackCore.VectorChunk<'T>,StackCore.VectorChunk<'T>>
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkAppendVectorElement<'T
                               when 'T: equality and 'T: (new: unit -> 'T) and
                                    'T: struct and 'T :> System.ValueType> :
  StackCore.Stage<(StackCore.VectorChunk<'T> * StackCore.Chunk<'T>),
                  StackCore.VectorChunk<'T>>
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkVectorMapElements:
  (string ->
     StackCore.Stage<StackCore.VectorChunk<float>,StackCore.VectorChunk<float>>)

val chunkVector3ToColor:
  (float ->
     float ->
     StackCore.Stage<StackCore.VectorChunk<float>,StackCore.VectorChunk<uint8>>)

val chunkColorToVector3:
  (float ->
     float ->
     StackCore.Stage<StackCore.VectorChunk<uint8>,StackCore.VectorChunk<float>>)

val chunkVectorDot:
  StackCore.Stage<(StackCore.VectorChunk<float> * StackCore.VectorChunk<float>),
                  StackCore.Chunk<float>>

val chunkVectorMagnitude:
  StackCore.Stage<StackCore.VectorChunk<float>,StackCore.Chunk<float>>

val chunkVectorCross3D:
  StackCore.Stage<(StackCore.VectorChunk<float> * StackCore.VectorChunk<float>),
                  StackCore.VectorChunk<float>>

val chunkVectorAngleTo:
  (float list ->
     StackCore.Stage<StackCore.VectorChunk<float>,StackCore.Chunk<float>>)

val chunkPCA:
  (uint32 ->
     StackCore.Stage<StackCore.VectorChunk<float>,StackCore.VectorChunk<float>>)

val chunkPCAFloat32:
  (uint32 ->
     StackCore.Stage<StackCore.VectorChunk<float32>,
                     StackCore.VectorChunk<float32>>)

val chunkStructureTensorNativeParallelCollect:
  (float ->
     int ->
     float ->
     int ->
     int ->
     StackCore.Stage<StackCore.Chunk<float32>,StackCore.VectorChunk<float32>>)

val chunkSelectGroupedVectorOutput:
  (uint ->
     uint ->
     StackCore.Stage<StackCore.VectorChunk<'a>,StackCore.VectorChunk<'a>>)
    when 'a: equality

val chunkToComplex64:
  StackCore.Stage<(StackCore.Chunk<float> * StackCore.Chunk<float>),
                  StackCore.Chunk<float32>>

val chunkPolarToComplex64:
  StackCore.Stage<(StackCore.Chunk<float> * StackCore.Chunk<float>),
                  StackCore.Chunk<float32>>

val chunkComplex64Real:
  StackCore.Stage<StackCore.Chunk<float32>,StackCore.Chunk<float>>

val chunkComplex64Imag:
  StackCore.Stage<StackCore.Chunk<float32>,StackCore.Chunk<float>>

val chunkComplex64Modulus:
  StackCore.Stage<StackCore.Chunk<float32>,StackCore.Chunk<float>>

val chunkComplex64Argument:
  StackCore.Stage<StackCore.Chunk<float32>,StackCore.Chunk<float>>

val chunkComplex64Conjugate:
  StackCore.Stage<StackCore.Chunk<float32>,StackCore.Chunk<float32>>

val chunkFftXYFloat32ToComplex64Interleaved:
  StackCore.Stage<StackCore.Chunk<float32>,StackCore.Chunk<float32>>

val chunkFftXYFloat32ToComplex64InterleavedParallelCollect:
  (int -> StackCore.Stage<StackCore.Chunk<float32>,StackCore.Chunk<float32>>)

val chunkInvFftXYComplex64InterleavedToFloat32:
  StackCore.Stage<StackCore.Chunk<float32>,StackCore.Chunk<float32>>

val chunkInvFftXYComplex64InterleavedToFloat32ParallelCollect:
  (int -> StackCore.Stage<StackCore.Chunk<float32>,StackCore.Chunk<float32>>)

val chunkFftShift3DComplex64Interleaved:
  StackCore.Stage<StackCore.Chunk<float32>,StackCore.Chunk<float32>>

val chunkConnectedComponentsSauf3DUInt8UInt32:
  (unit -> StackCore.Stage<StackCore.Chunk<uint8>,StackCore.Chunk<uint32>>)

val chunkConnectedComponentsSauf3DUInt8UInt32ParallelCollect:
  (int -> int -> StackCore.Stage<StackCore.Chunk<uint8>,StackCore.Chunk<uint32>>)

val chunkConnectedComponentsSauf3DUInt8:
  (unit -> StackCore.Stage<StackCore.Chunk<uint8>,StackCore.Chunk<uint64>>)

val chunkHistogram<'T
                     when 'T: comparison and 'T: (new: unit -> 'T) and
                          'T: struct and 'T :> System.ValueType> :
  (unit -> SlimPipeline.Stage<StackCore.Chunk<'T>,StackCore.Histogram<'T>>)
    when 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkHistogramDense<'T
                          when 'T: comparison and 'T: (new: unit -> 'T) and
                               'T: struct and 'T :> System.ValueType> :
  (unit -> SlimPipeline.Stage<StackCore.Chunk<'T>,StackCore.Histogram<'T>>)
    when 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkHistogramLeftEdges<'T
                              when 'T: equality and 'T: (new: unit -> 'T) and
                                   'T: struct and 'T :> System.ValueType> :
  (float seq ->
     SlimPipeline.Stage<StackCore.Chunk<'T>,StackCore.Histogram<float>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkHistogramFixedBins<'T
                              when 'T: equality and 'T: (new: unit -> 'T) and
                                   'T: struct and 'T :> System.ValueType> :
  (float ->
     float ->
     uint32 ->
     SlimPipeline.Stage<StackCore.Chunk<'T>,StackCore.Histogram<float>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkComputeStats<'T
                        when 'T: equality and 'T: (new: unit -> 'T) and
                             'T: struct and 'T :> System.ValueType> :
  (unit -> StackCore.Stage<StackCore.Chunk<'T>,ImageFunctions.ImageStats>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkHistogramEqualizationDense<'T
                                      when 'T: equality and
                                           'T: (new: unit -> 'T) and 'T: struct and
                                           'T :> System.ValueType> :
  (ChunkCore.ChunkFunctions.DenseHistogram ->
     StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkHistogramEqualizationLeftEdges<'T
                                          when 'T: equality and
                                               'T: (new: unit -> 'T) and
                                               'T: struct and
                                               'T :> System.ValueType> :
  (ChunkCore.ChunkFunctions.LeftEdgeHistogram ->
     StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkHistogramEqualizationSparse<'T
                                       when 'T: comparison and
                                            'T: (new: unit -> 'T) and 'T: struct and
                                            'T :> System.ValueType> :
  (Map<'T,uint64> -> StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<'T>>)
    when 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkHistogramEqualization<'T
                                 when 'T: comparison and 'T: (new: unit -> 'T) and
                                      'T: struct and 'T :> System.ValueType> :
  (obj -> StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<'T>>)
    when 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkQuantiles: (float list -> 'a -> float list)

val chunkVolume:
  (float -> float -> float -> StackCore.Stage<StackCore.Chunk<uint8>,float>)

val chunkSumProjection<'T
                         when 'T: equality and 'T: (new: unit -> 'T) and
                              'T: struct and 'T :> System.ValueType> :
  (string -> StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<float>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkThresholdRange<'T
                          when 'T: equality and 'T: (new: unit -> 'T) and
                               'T: struct and 'T :> System.ValueType> :
  (int -> int -> StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<uint8>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkClamp<'T
                 when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
                      'T :> System.ValueType> :
  (double -> double -> StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkShiftScale<'T
                      when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
                           'T :> System.ValueType> :
  (double -> double -> StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkIntensityWindow<'T
                           when 'T: equality and 'T: (new: unit -> 'T) and
                                'T: struct and 'T :> System.ValueType> :
  (double ->
     double ->
     double ->
     double -> StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkCastToUInt8<'T
                       when 'T: equality and 'T: (new: unit -> 'T) and
                            'T: struct and 'T :> System.ValueType> :
  StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<uint8>>
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkCastToFloat32<'T
                         when 'T: equality and 'T: (new: unit -> 'T) and
                              'T: struct and 'T :> System.ValueType> :
  StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<float32>>
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkCastFromFloat32<'T
                           when 'T: equality and 'T: (new: unit -> 'T) and
                                'T: struct and 'T :> System.ValueType> :
  StackCore.Stage<StackCore.Chunk<float32>,StackCore.Chunk<'T>>
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkCast<'S,'T
                when 'S: equality and 'S: (new: unit -> 'S) and 'S: struct and
                     'S :> System.ValueType and 'T: equality and
                     'T: (new: unit -> 'T) and 'T: struct and
                     'T :> System.ValueType> :
  Stage<Chunk<'S>,Chunk<'T>>
    when 'S: equality and 'S: (new: unit -> 'S) and 'S: struct and
         'S :> System.ValueType and 'T: equality and 'T: (new: unit -> 'T) and
         'T: struct and 'T :> System.ValueType

val inline chunkImageAddScalar:
  value: ^a -> StackCore.Stage<StackCore.Chunk<^b>,StackCore.Chunk<'c>>
    when (^b or ^a) : (static member (+) : ^b * ^a -> 'c) and ^b: equality and
         ^b: (new: unit -> ^b) and ^b: struct and ^b :> System.ValueType and
         'c: equality and 'c: (new: unit -> 'c) and 'c: struct and
         'c :> System.ValueType

val inline chunkImageSubScalar:
  value: ^a -> StackCore.Stage<StackCore.Chunk<^b>,StackCore.Chunk<'c>>
    when (^b or ^a) : (static member (-) : ^b * ^a -> 'c) and ^b: equality and
         ^b: (new: unit -> ^b) and ^b: struct and ^b :> System.ValueType and
         'c: equality and 'c: (new: unit -> 'c) and 'c: struct and
         'c :> System.ValueType

val inline chunkImageMulScalar:
  value: ^a -> StackCore.Stage<StackCore.Chunk<^b>,StackCore.Chunk<'c>>
    when (^b or ^a) : (static member ( * ) : ^b * ^a -> 'c) and ^b: equality and
         ^b: (new: unit -> ^b) and ^b: struct and ^b :> System.ValueType and
         'c: equality and 'c: (new: unit -> 'c) and 'c: struct and
         'c :> System.ValueType

val inline chunkImageDivScalar:
  value: ^a -> StackCore.Stage<StackCore.Chunk<^b>,StackCore.Chunk<'c>>
    when (^b or ^a) : (static member (/) : ^b * ^a -> 'c) and ^b: equality and
         ^b: (new: unit -> ^b) and ^b: struct and ^b :> System.ValueType and
         'c: equality and 'c: (new: unit -> 'c) and 'c: struct and
         'c :> System.ValueType

val inline chunkScalarAddImage:
  value: ^a -> StackCore.Stage<StackCore.Chunk<^b>,StackCore.Chunk<'c>>
    when (^b or ^a) : (static member (+) : ^b * ^a -> 'c) and ^b: equality and
         ^b: (new: unit -> ^b) and ^b: struct and ^b :> System.ValueType and
         'c: equality and 'c: (new: unit -> 'c) and 'c: struct and
         'c :> System.ValueType

val inline chunkScalarSubImage:
  value: ^a -> StackCore.Stage<StackCore.Chunk<^b>,StackCore.Chunk<'c>>
    when (^a or ^b) : (static member (-) : ^a * ^b -> 'c) and ^b: equality and
         ^b: (new: unit -> ^b) and ^b: struct and ^b :> System.ValueType and
         'c: equality and 'c: (new: unit -> 'c) and 'c: struct and
         'c :> System.ValueType

val inline chunkScalarMulImage:
  value: ^a -> StackCore.Stage<StackCore.Chunk<^b>,StackCore.Chunk<'c>>
    when (^b or ^a) : (static member ( * ) : ^b * ^a -> 'c) and ^b: equality and
         ^b: (new: unit -> ^b) and ^b: struct and ^b :> System.ValueType and
         'c: equality and 'c: (new: unit -> 'c) and 'c: struct and
         'c :> System.ValueType

val inline chunkScalarDivImage:
  value: ^a -> StackCore.Stage<StackCore.Chunk<^b>,StackCore.Chunk<'c>>
    when (^a or ^b) : (static member (/) : ^a * ^b -> 'c) and ^b: equality and
         ^b: (new: unit -> ^b) and ^b: struct and ^b :> System.ValueType and
         'c: equality and 'c: (new: unit -> 'c) and 'c: struct and
         'c :> System.ValueType

val inline chunkAddPair<^T
                          when ^T: equality and ^T: (new: unit -> ^T) and
                               ^T: struct and ^T :> System.ValueType and
                               ^T: (static member (+) : ^T * ^T -> ^T)> :
  StackCore.Stage<(StackCore.Chunk<^T> * StackCore.Chunk<^T>),
                  StackCore.Chunk<^T>>
    when ^T: equality and ^T: (new: unit -> ^T) and ^T: struct and
         ^T :> System.ValueType and ^T: (static member (+) : ^T * ^T -> ^T)

val inline chunkSubPair<^T
                          when ^T: equality and ^T: (new: unit -> ^T) and
                               ^T: struct and ^T :> System.ValueType and
                               ^T: (static member (-) : ^T * ^T -> ^T)> :
  StackCore.Stage<(StackCore.Chunk<^T> * StackCore.Chunk<^T>),
                  StackCore.Chunk<^T>>
    when ^T: equality and ^T: (new: unit -> ^T) and ^T: struct and
         ^T :> System.ValueType and ^T: (static member (-) : ^T * ^T -> ^T)

val inline chunkMulPair<^T
                          when ^T: equality and ^T: (new: unit -> ^T) and
                               ^T: struct and ^T :> System.ValueType and
                               ^T: (static member ( * ) : ^T * ^T -> ^T)> :
  StackCore.Stage<(StackCore.Chunk<^T> * StackCore.Chunk<^T>),
                  StackCore.Chunk<^T>>
    when ^T: equality and ^T: (new: unit -> ^T) and ^T: struct and
         ^T :> System.ValueType and ^T: (static member ( * ) : ^T * ^T -> ^T)

val inline chunkDivPair<^T
                          when ^T: equality and ^T: (new: unit -> ^T) and
                               ^T: struct and ^T :> System.ValueType and
                               ^T: (static member (/) : ^T * ^T -> ^T)> :
  StackCore.Stage<(StackCore.Chunk<^T> * StackCore.Chunk<^T>),
                  StackCore.Chunk<^T>>
    when ^T: equality and ^T: (new: unit -> ^T) and ^T: struct and
         ^T :> System.ValueType and ^T: (static member (/) : ^T * ^T -> ^T)

val inline chunkMaxOfPair<'T
                            when 'T: comparison and 'T: (new: unit -> 'T) and
                                 'T: struct and 'T :> System.ValueType> :
  StackCore.Stage<(StackCore.Chunk<'T> * StackCore.Chunk<'T>),
                  StackCore.Chunk<'T>>
    when 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val inline chunkMinOfPair<'T
                            when 'T: comparison and 'T: (new: unit -> 'T) and
                                 'T: struct and 'T :> System.ValueType> :
  StackCore.Stage<(StackCore.Chunk<'T> * StackCore.Chunk<'T>),
                  StackCore.Chunk<'T>>
    when 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkEqual<'T
                 when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
                      'T :> System.ValueType> :
  StackCore.Stage<(StackCore.Chunk<'T> * StackCore.Chunk<'T>),
                  StackCore.Chunk<uint8>>
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkNotEqual<'T
                    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
                         'T :> System.ValueType> :
  StackCore.Stage<(StackCore.Chunk<'T> * StackCore.Chunk<'T>),
                  StackCore.Chunk<uint8>>
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkGreater<'T
                   when 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and
                        'T :> System.ValueType> :
  StackCore.Stage<(StackCore.Chunk<'T> * StackCore.Chunk<'T>),
                  StackCore.Chunk<uint8>>
    when 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkGreaterEqual<'T
                        when 'T: comparison and 'T: (new: unit -> 'T) and
                             'T: struct and 'T :> System.ValueType> :
  StackCore.Stage<(StackCore.Chunk<'T> * StackCore.Chunk<'T>),
                  StackCore.Chunk<uint8>>
    when 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkLess<'T
                when 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and
                     'T :> System.ValueType> :
  StackCore.Stage<(StackCore.Chunk<'T> * StackCore.Chunk<'T>),
                  StackCore.Chunk<uint8>>
    when 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkLessEqual<'T
                     when 'T: comparison and 'T: (new: unit -> 'T) and
                          'T: struct and 'T :> System.ValueType> :
  StackCore.Stage<(StackCore.Chunk<'T> * StackCore.Chunk<'T>),
                  StackCore.Chunk<uint8>>
    when 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkMaskAnd:
  StackCore.Stage<(StackCore.Chunk<uint8> * StackCore.Chunk<uint8>),
                  StackCore.Chunk<uint8>>

val chunkMaskOr:
  StackCore.Stage<(StackCore.Chunk<uint8> * StackCore.Chunk<uint8>),
                  StackCore.Chunk<uint8>>

val chunkMaskXor:
  StackCore.Stage<(StackCore.Chunk<uint8> * StackCore.Chunk<uint8>),
                  StackCore.Chunk<uint8>>

val chunkMaskNot: StackCore.Stage<StackCore.Chunk<uint8>,StackCore.Chunk<uint8>>

val convolveNativeXParallelCollect<'T
                                     when 'T: equality and 'T: (new: unit -> 'T) and
                                          'T: struct and 'T :> System.ValueType> :
  (float32 array ->
     int -> StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val convolveNativeYParallelCollect<'T
                                     when 'T: equality and 'T: (new: unit -> 'T) and
                                          'T: struct and 'T :> System.ValueType> :
  (float32 array ->
     int -> StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val convolveNativeZParallelCollect<'T
                                     when 'T: equality and 'T: (new: unit -> 'T) and
                                          'T: struct and 'T :> System.ValueType> :
  (float32 array ->
     int -> StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val finiteDiffKernel1D: (uint32 -> float32 array)

val finiteDiffNativeXParallelCollect<'T
                                       when 'T: equality and
                                            'T: (new: unit -> 'T) and 'T: struct and
                                            'T :> System.ValueType> :
  (uint32 -> int -> StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val finiteDiffNativeYParallelCollect<'T
                                       when 'T: equality and
                                            'T: (new: unit -> 'T) and 'T: struct and
                                            'T :> System.ValueType> :
  (uint32 -> int -> StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val finiteDiffNativeZParallelCollect<'T
                                       when 'T: equality and
                                            'T: (new: unit -> 'T) and 'T: struct and
                                            'T :> System.ValueType> :
  (uint32 -> int -> StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val separableConvolveNativeParallelCollect<'T
                                             when 'T: equality and
                                                  'T: (new: unit -> 'T) and
                                                  'T: struct and
                                                  'T :> System.ValueType> :
  (float32 array ->
     float32 array ->
     float32 array ->
     int -> StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val boxFilterNativeParallelCollect<'T
                                     when 'T: equality and 'T: (new: unit -> 'T) and
                                          'T: struct and 'T :> System.ValueType> :
  (int -> int -> StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val boxFilterNativeParallelCollectXYZ<'T
                                        when 'T: equality and
                                             'T: (new: unit -> 'T) and
                                             'T: struct and
                                             'T :> System.ValueType> :
  (int ->
     int ->
     int -> int -> StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val gaussianFilterNativeParallelCollect<'T
                                          when 'T: equality and
                                               'T: (new: unit -> 'T) and
                                               'T: struct and
                                               'T :> System.ValueType> :
  (float ->
     int -> int -> StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val gaussianFilterNativeParallelCollectXYZ<'T
                                             when 'T: equality and
                                                  'T: (new: unit -> 'T) and
                                                  'T: struct and
                                                  'T :> System.ValueType> :
  (float ->
     int ->
     float ->
     int ->
     float ->
     int -> int -> StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val gradientVectorNativeParallelCollect:
  (float ->
     int ->
     int ->
     StackCore.Stage<StackCore.Chunk<float32>,StackCore.VectorChunk<float32>>)

val gradientVectorNativeParallelCollectXYZ:
  (float ->
     int ->
     float ->
     int ->
     float ->
     int ->
     int ->
     StackCore.Stage<StackCore.Chunk<float32>,StackCore.VectorChunk<float32>>)

val gradientMagnitudeNativeParallelCollect:
  (float ->
     int ->
     int -> StackCore.Stage<StackCore.Chunk<float32>,StackCore.Chunk<float32>>)

val gradientMagnitudeNativeParallelCollectXYZ:
  (float ->
     int ->
     float ->
     int ->
     float ->
     int ->
     int -> StackCore.Stage<StackCore.Chunk<float32>,StackCore.Chunk<float32>>)

val hessianUpperNativeParallelCollect:
  (float ->
     int ->
     int ->
     StackCore.Stage<StackCore.Chunk<float32>,StackCore.VectorChunk<float32>>)

val hessianUpperNativeParallelCollectXYZ:
  (float ->
     int ->
     float ->
     int ->
     float ->
     int ->
     int ->
     StackCore.Stage<StackCore.Chunk<float32>,StackCore.VectorChunk<float32>>)

val laplacianNativeParallelCollect:
  (float ->
     int ->
     int -> StackCore.Stage<StackCore.Chunk<float32>,StackCore.Chunk<float32>>)

val laplacianNativeParallelCollectXYZ:
  (float ->
     int ->
     float ->
     int ->
     float ->
     int ->
     int -> StackCore.Stage<StackCore.Chunk<float32>,StackCore.Chunk<float32>>)

val sobelMagnitudeNativeParallelCollect:
  (int -> StackCore.Stage<StackCore.Chunk<float32>,StackCore.Chunk<float32>>)

val sobelXNativeParallelCollect<'T
                                  when 'T: equality and 'T: (new: unit -> 'T) and
                                       'T: struct and 'T :> System.ValueType> :
  (int -> StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val sobelYNativeParallelCollect<'T
                                  when 'T: equality and 'T: (new: unit -> 'T) and
                                       'T: struct and 'T :> System.ValueType> :
  (int -> StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val sobelZNativeParallelCollect<'T
                                  when 'T: equality and 'T: (new: unit -> 'T) and
                                       'T: struct and 'T :> System.ValueType> :
  (int -> StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkAddNormalNoise<'T
                          when 'T: equality and 'T: (new: unit -> 'T) and
                               'T: struct and 'T :> System.ValueType> :
  (float -> float -> StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkAddSaltAndPepperNoise<'T
                                 when 'T: equality and 'T: (new: unit -> 'T) and
                                      'T: struct and 'T :> System.ValueType> :
  (float -> StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkAddShotNoise<'T
                        when 'T: equality and 'T: (new: unit -> 'T) and
                             'T: struct and 'T :> System.ValueType> :
  (float -> StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkBinaryDilateZonohedral:
  (uint32 -> StackCore.Stage<StackCore.Chunk<uint8>,StackCore.Chunk<uint8>>)

val chunkBinaryErodeZonohedral:
  (uint32 -> StackCore.Stage<StackCore.Chunk<uint8>,StackCore.Chunk<uint8>>)

val chunkBinaryOpeningZonohedral:
  (uint32 -> StackCore.Stage<StackCore.Chunk<uint8>,StackCore.Chunk<uint8>>)

val chunkBinaryClosingZonohedral:
  (uint32 -> StackCore.Stage<StackCore.Chunk<uint8>,StackCore.Chunk<uint8>>)

val chunkBinaryWhiteTopHatZonohedral:
  (uint32 -> StackCore.Stage<StackCore.Chunk<uint8>,StackCore.Chunk<uint8>>)

val chunkBinaryWhiteTopHatZonohedralParallel:
  (uint32 ->
     int -> StackCore.Stage<StackCore.Chunk<uint8>,StackCore.Chunk<uint8>>)

val chunkBinaryBlackTopHatZonohedral:
  (uint32 -> StackCore.Stage<StackCore.Chunk<uint8>,StackCore.Chunk<uint8>>)

val chunkBinaryBlackTopHatZonohedralParallel:
  (uint32 ->
     int -> StackCore.Stage<StackCore.Chunk<uint8>,StackCore.Chunk<uint8>>)

val chunkBinaryGradientZonohedral:
  (uint32 -> StackCore.Stage<StackCore.Chunk<uint8>,StackCore.Chunk<uint8>>)

val chunkBinaryGradientZonohedralParallel:
  (uint32 ->
     int -> StackCore.Stage<StackCore.Chunk<uint8>,StackCore.Chunk<uint8>>)

val chunkBinaryContourZonohedral:
  (bool -> StackCore.Stage<StackCore.Chunk<uint8>,StackCore.Chunk<uint8>>)

val chunkBinaryContourZonohedralParallel:
  (bool -> int -> StackCore.Stage<StackCore.Chunk<uint8>,StackCore.Chunk<uint8>>)

val readZarrSlabStacked<'T when 'T: equality> :
  (string ->
     int ->
     int ->
     int ->
     int ->
     int ->
     SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,StackCore.Image<'T>>)
    when 'T: equality

val readZarrSlab<'T when 'T: equality> :
  (string ->
     int ->
     int ->
     int ->
     int ->
     int ->
     SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,StackCore.Image<'T>>)
    when 'T: equality

val readZarrRandom<'T when 'T: equality> :
  (uint ->
     string ->
     int ->
     int ->
     int ->
     int ->
     int ->
     SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,StackCore.Image<'T>>)
    when 'T: equality

val readZarrRange<'T when 'T: equality> :
  (uint ->
     int ->
     uint ->
     string ->
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
     int ->
     int ->
     int ->
     SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,StackCore.Image<'T>>)
    when 'T: equality

val readNexusSlab<'T when 'T: equality> :
  (string ->
     string ->
     int ->
     int ->
     int ->
     SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,StackCore.Image<'T>>)
    when 'T: equality

val readNexusRandom<'T when 'T: equality> :
  (uint ->
     string ->
     string ->
     int ->
     int ->
     int ->
     SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,StackCore.Image<'T>>)
    when 'T: equality

val readNexusRange<'T when 'T: equality> :
  (uint ->
     int ->
     uint ->
     string ->
     string ->
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
  width: uint ->
    height: uint ->
    depth: uint32 ->
    (SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,StackCore.Image<'T>>)
    when 'T: equality

val coordinateY:
  width: uint ->
    height: uint ->
    depth: uint32 ->
    (SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,StackCore.Image<'T>>)
    when 'T: equality

val coordinateZ:
  width: uint ->
    height: uint ->
    depth: uint32 ->
    (SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,StackCore.Image<'T>>)
    when 'T: equality

val imageCenter: (uint -> uint -> uint -> TinyLinAlg.V3)

val randomRigidTransformAround:
  (int -> TinyLinAlg.V3 -> float -> TinyLinAlg.Affine)

val randomRigidTransform:
  (int -> uint -> uint -> uint -> float -> TinyLinAlg.Affine)

val deleteIfExists: (string -> unit)

val write:
  (string -> string -> StackCore.Stage<StackCore.Image<'a>,StackCore.Image<'a>>)
    when 'a: equality

val writeThrough:
  (string -> string -> StackCore.Stage<StackCore.Image<'a>,StackCore.Image<'a>>)
    when 'a: equality

val writeChunkSlices<'T when 'T: equality> :
  (string -> string -> StackCore.Stage<StackCore.Chunk<'T>,unit>)
    when 'T: equality

val writeColorChunkSlices:
  (string -> string -> StackCore.Stage<StackCore.VectorChunk<uint8>,unit>)

val writeVolume<'T when 'T: equality> :
  (string -> StackCore.Stage<StackCore.Image<'T>,unit>) when 'T: equality

val writeZarrWithCompression:
  (ZarrNET.Core.ZarrCompression ->
     string ->
     string ->
     uint ->
     uint ->
     uint ->
     uint ->
     float ->
     float ->
     float -> int -> StackCore.Stage<StackCore.Image<'a>,StackCore.Image<'a>>)
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

val writeZarrComplex64InterleavedFloat32:
  (string ->
     string ->
     uint ->
     uint ->
     uint ->
     uint ->
     uint ->
     uint ->
     float ->
     float -> float -> int -> StackCore.Stage<StackCore.Chunk<float32>,unit>)

val fftZComplex64InterleavedZarrTiles:
  (string ->
     string ->
     string ->
     uint ->
     uint ->
     uint ->
     uint ->
     uint -> uint -> uint -> uint -> float -> float -> float -> int -> unit)

val writeZarrSlabWithCompression:
  (ZarrNET.Core.ZarrCompression ->
     string ->
     uint ->
     uint ->
     float ->
     float ->
     float ->
     int ->
     SlimPipeline.Plan<'a,StackCore.Image<'b>> ->
     SlimPipeline.Plan<'a,StackCore.Image<'b>>) when 'b: equality

val writeZarrSlabNamedWithCompression:
  (ZarrNET.Core.ZarrCompression ->
     string ->
     string ->
     uint ->
     uint ->
     float ->
     float ->
     float ->
     int ->
     SlimPipeline.Plan<'a,StackCore.Image<'b>> ->
     SlimPipeline.Plan<'a,StackCore.Image<'b>>) when 'b: equality

val writeZarrSlab:
  (string ->
     uint ->
     uint ->
     float ->
     float ->
     float ->
     int ->
     SlimPipeline.Plan<'a,StackCore.Image<'b>> ->
     SlimPipeline.Plan<'a,StackCore.Image<'b>>) when 'b: equality

val writeZarrSlabNamed:
  (string ->
     string ->
     uint ->
     uint ->
     float ->
     float ->
     float ->
     int ->
     SlimPipeline.Plan<'a,StackCore.Image<'b>> ->
     SlimPipeline.Plan<'a,StackCore.Image<'b>>) when 'b: equality

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

val writeNexusSlab:
  (string ->
     string ->
     uint ->
     uint ->
     int ->
     int ->
     int ->
     SlimPipeline.Plan<'a,StackCore.Image<'b>> ->
     SlimPipeline.Plan<'a,StackCore.Image<'b>>) when 'b: equality

val writeChunks:
  (string ->
     string ->
     uint ->
     uint -> uint -> StackCore.Stage<StackCore.Image<'a>,StackCore.Image<'a>>)
    when 'a: equality

val writeSlabSlices<'T when 'T: equality> :
  (string ->
     string -> uint -> StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>>)
    when 'T: equality

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

val meshFilePath: (string -> string -> string)

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

val ransacFit:
  (int ->
     int ->
     float ->
     int ->
     ('a list -> 'b option) ->
     ('b -> 'a -> float option) ->
     'a seq -> StackRansac.RansacResult<'b,'a> option)

val affine2DFromMatches:
  (StackRansac.PointMatch2D list -> float list list option)

val affine2DRansac:
  (int ->
     float ->
     float -> int -> StackRansac.PointMatch2D seq -> float list list option)

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

val streamConnectedObjectsChunk<'T
                                  when 'T: equality and 'T: (new: unit -> 'T) and
                                       'T: struct and 'T :> System.ValueType> :
  (StackObjects.ObjectConnectivity ->
     StackCore.Stage<StackCore.Chunk<'T>,StackObjects.StreamedObject list>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

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

val paintObjectsChunk:
  (uint32 ->
     uint32 ->
     StackCore.Stage<StackObjects.StreamedObject list,StackCore.Chunk<uint8>>)

val paintObjectsCroppedChunk:
  StackCore.Stage<StackObjects.StreamedObject list,StackCore.Chunk<uint8>>

val measureObjects:
  StackCore.Stage<StackObjects.StreamedObject list,
                  StackObjects.ObjectMeasurements list>

val objectSizes:
  StackCore.Stage<StackObjects.ObjectMeasurements list,uint64 list>

val objectSizeStats:
  StackCore.Stage<StackObjects.ObjectMeasurements list,
                  StackObjects.ObjectSizeStats>

val histogram:
  (uint64 -> StackCore.Stage<uint64 list,StackCore.Histogram<uint64>>)

val chunkRemoveSmallObjects:
  (uint64 ->
     StackObjects.ObjectConnectivity ->
     StackCore.Stage<StackCore.Chunk<uint8>,StackCore.Chunk<uint8>>)

val chunkFillSmallHoles:
  (uint64 ->
     StackObjects.ObjectConnectivity ->
     StackCore.Stage<StackCore.Chunk<uint8>,StackCore.Chunk<uint8>>)

val resampleAffineFromChunks:
  (string ->
     string ->
     ('a -> 'a -> float32 -> 'a) ->
     int ->
     StackAffineResampler.ImageGeom ->
     StackAffineResampler.ImageGeom ->
     TinyLinAlg.Affine -> 'a -> (int * StackAffineResampler.Image<'a>) seq)
    when 'a: equality

val resampleAffineChunk<'T
                          when 'T: equality and 'T: (new: unit -> 'T) and
                               'T: struct and 'T :> System.ValueType> :
  (('T -> 'T -> float32 -> 'T) ->
     StackAffineResampler.ImageGeom ->
     StackAffineResampler.ImageGeom ->
     TinyLinAlg.Affine ->
     'T -> StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val resampleAffine:
  (('a -> 'a -> float32 -> 'a) ->
     int option ->
     StackAffineResampler.ImageGeom ->
     StackAffineResampler.ImageGeom ->
     TinyLinAlg.Affine ->
     'a ->
     StackCore.Stage<StackAffineResampler.Image<'a>,
                     StackAffineResampler.Image<'a>>) when 'a: equality

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

val failTypeMismatch<'T> : (obj -> System.Type list -> unit)

val toVectorImage<'T when 'T: equality> :
  Stage<(Image<'T> * Image<'T>),Image<'T list>> when 'T: equality

val vectorElement:
  componentId: uint -> Stage<Image<'T list>,Image<'T>> when 'T: equality

val vectorRange:
  firstComponent: uint ->
    componentCount: uint -> Stage<Image<'T list>,Image<'T list>>
    when 'T: equality

val vector3ToColor:
  inputMinimum: float ->
    inputMaximum: float -> Stage<Image<float list>,Image<uint8 list>>

val colorToVector3:
  outputMinimum: float ->
    outputMaximum: float -> Stage<Image<uint8 list>,Image<float list>>

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

val FFTFloat32:
  chunkX: uint ->
    chunkY: uint -> chunkZ: uint -> Stage<Image<'T>,Image<Image.ComplexFloat32>>
    when 'T: equality

val invFFT:
  chunkX: uint ->
    chunkY: uint ->
    chunkZ: uint -> Stage<Image<System.Numerics.Complex>,Image<float>>

val invFFTFloat32:
  chunkX: uint ->
    chunkY: uint ->
    chunkZ: uint -> Stage<Image<Image.ComplexFloat32>,Image<float32>>

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
  radius: uint32 -> winSz: uint option -> Stage<Image<'T>,Image<'T>>
    when 'T: equality

val smoothWBilateral:
  domainSigma: double ->
    rangeSigma: double -> winSz: uint option -> Stage<Image<'T>,Image<'T>>
    when 'T: equality

val gradientMagnitude:
  winSz: uint option -> Stage<Image<'T>,Image<'T>> when 'T: equality

val gradient:
  order: uint -> winSz: uint option -> Stage<Image<float>,Image<float list>>

val sobelEdge:
  winSz: uint option -> Stage<Image<'T>,Image<'T>> when 'T: equality

val laplacian:
  winSz: uint option -> Stage<Image<'T>,Image<'T>> when 'T: equality

val grayscaleErode:
  radius: uint32 -> winSz: uint option -> Stage<Image<'T>,Image<'T>>
    when 'T: equality

val grayscaleDilate:
  radius: uint32 -> winSz: uint option -> Stage<Image<'T>,Image<'T>>
    when 'T: equality

val grayscaleOpening:
  radius: uint32 -> winSz: uint option -> Stage<Image<'T>,Image<'T>>
    when 'T: equality

val grayscaleClosing:
  radius: uint32 -> winSz: uint option -> Stage<Image<'T>,Image<'T>>
    when 'T: equality

val whiteTopHat:
  radius: uint32 -> winSz: uint option -> Stage<Image<'T>,Image<'T>>
    when 'T: equality

val blackTopHat:
  radius: uint32 -> winSz: uint option -> Stage<Image<'T>,Image<'T>>
    when 'T: equality

val morphologicalGradient:
  radius: uint32 -> winSz: uint option -> Stage<Image<'T>,Image<'T>>
    when 'T: equality

val binaryContour:
  (bool ->
     uint option ->
     StackCore.Stage<StackCore.Image<uint8>,StackCore.Image<uint8>>)

val binaryMedian:
  (uint32 ->
     uint option ->
     StackCore.Stage<StackCore.Image<uint8>,StackCore.Image<uint8>>)

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
  fullyConnected: bool -> winSz: uint32 option -> Stage<Image<'T>,Image<'T>>
    when 'T: equality

val changeLabel:
  fromLabel: double -> toLabel: double -> Stage<Image<'T>,Image<'T>>
    when 'T: equality

val marchingCubes<'T when 'T: equality> :
  (float -> StackCore.Stage<StackCore.Image<'T>,StackMesh.TriangleSet>)
    when 'T: equality

val marchingCubesChunk<'T
                         when 'T: equality and 'T: (new: unit -> 'T) and
                              'T: struct and 'T :> System.ValueType> :
  (float -> StackCore.Stage<StackCore.Chunk<'T>,StackMesh.TriangleSet>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

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

val chunkDogKeypoints<'T
                        when 'T: equality and 'T: (new: unit -> 'T) and
                             'T: struct and 'T :> System.ValueType> :
  (float ->
     float ->
     uint ->
     float -> uint -> StackCore.Stage<StackCore.Chunk<'T>,StackPoints.PointSet>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkLogBlobKeypoints<'T
                            when 'T: equality and 'T: (new: unit -> 'T) and
                                 'T: struct and 'T :> System.ValueType> :
  (float ->
     float -> uint -> StackCore.Stage<StackCore.Chunk<'T>,StackPoints.PointSet>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkHessianKeypoints<'T
                            when 'T: equality and 'T: (new: unit -> 'T) and
                                 'T: struct and 'T :> System.ValueType> :
  (float ->
     string ->
     float -> uint -> StackCore.Stage<StackCore.Chunk<'T>,StackPoints.PointSet>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkHarris3DKeypoints<'T
                             when 'T: equality and 'T: (new: unit -> 'T) and
                                  'T: struct and 'T :> System.ValueType> :
  (float ->
     float ->
     float ->
     float -> uint -> StackCore.Stage<StackCore.Chunk<'T>,StackPoints.PointSet>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkForstner3DKeypoints<'T
                               when 'T: equality and 'T: (new: unit -> 'T) and
                                    'T: struct and 'T :> System.ValueType> :
  (float ->
     float ->
     float -> uint -> StackCore.Stage<StackCore.Chunk<'T>,StackPoints.PointSet>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkPhaseCongruencyKeypoints<'T
                                    when 'T: equality and 'T: (new: unit -> 'T) and
                                         'T: struct and 'T :> System.ValueType> :
  (float ->
     float -> uint -> StackCore.Stage<StackCore.Chunk<'T>,StackPoints.PointSet>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkSiftKeypoints<'T
                         when 'T: equality and 'T: (new: unit -> 'T) and
                              'T: struct and 'T :> System.ValueType> :
  (float ->
     float ->
     uint ->
     float -> uint -> StackCore.Stage<StackCore.Chunk<'T>,StackPoints.PointSet>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

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

val imHistogram:
  (unit -> SlimPipeline.Stage<StackCore.Image<'a>,StackCore.Histogram<'a>>)
    when 'a: comparison

val imHistogramFixedBins:
  (float ->
     float ->
     uint32 ->
     SlimPipeline.Stage<StackCore.Image<'a>,StackCore.Histogram<float>>)
    when 'a: equality

val histogramEstimate<'T when 'T: comparison> :
  (uint32 ->
     string ->
     float ->
     SlimPipeline.Stage<StackCore.Image<'T>,
                        StackImageFunctions.HistogramEstimate<'T>>)
    when 'T: comparison

val estimateHistogram<'T when 'T: comparison> :
  (uint32 ->
     string ->
     string ->
     uint32 ->
     string ->
     float ->
     SlimPipeline.Plan<unit,unit> ->
     SlimPipeline.Plan<unit,StackImageFunctions.HistogramEstimate<'T>>)
    when 'T: comparison

val histogramEstimateMap<'T when 'T: comparison> :
  SlimPipeline.Stage<StackImageFunctions.HistogramEstimate<'T>,
                     StackCore.Histogram<'T>> when 'T: comparison

val histogramEqualization:
  histogram: StackCore.Histogram<'a> ->
    SlimPipeline.Stage<StackCore.Image<'b>,Image.Image<float>>
    when 'a: comparison and 'b: equality

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

val fitBiasModelChunk<'T
                        when 'T: equality and 'T: (new: unit -> 'T) and
                             'T: struct and 'T :> System.ValueType> :
  (int ->
     uint32 ->
     StackCore.Stage<StackCore.Chunk<'T>,StackBias.BiasPolynomialModel>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val fitBiasModelChunkMasked<'T
                              when 'T: equality and 'T: (new: unit -> 'T) and
                                   'T: struct and 'T :> System.ValueType> :
  (int ->
     uint32 ->
     StackCore.Stage<(StackCore.Chunk<'T> * StackCore.Chunk<uint8>),
                     StackBias.BiasPolynomialModel>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val correctBias<'T when 'T: equality> :
  (StackBias.BiasPolynomialModel ->
     StackCore.Stage<StackCore.Image<'T>,StackCore.Image<float>>)
    when 'T: equality

val correctBiasMasked<'T when 'T: equality> :
  (StackBias.BiasPolynomialModel ->
     StackCore.Stage<(StackCore.Image<'T> * StackCore.Image<uint8>),
                     StackCore.Image<float>>) when 'T: equality

val correctBiasChunk<'T
                       when 'T: equality and 'T: (new: unit -> 'T) and
                            'T: struct and 'T :> System.ValueType> :
  (StackBias.BiasPolynomialModel ->
     StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<float>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val correctBiasChunkMasked<'T
                             when 'T: equality and 'T: (new: unit -> 'T) and
                                  'T: struct and 'T :> System.ValueType> :
  (StackBias.BiasPolynomialModel ->
     StackCore.Stage<(StackCore.Chunk<'T> * StackCore.Chunk<uint8>),
                     StackCore.Chunk<float>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkSerialPolynomialBiasCorrect<'T
                                       when 'T: equality and
                                            'T: (new: unit -> 'T) and 'T: struct and
                                            'T :> System.ValueType> :
  (int -> StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val serialIdentityManifest:
  (uint32 -> uint32 -> int -> StackSerialSections.SerialSliceManifest)

val serialPolynomialBiasCorrect<'T when 'T: equality> :
  (int -> StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>>)
    when 'T: equality

val serialEstTrans<'T when 'T: equality> :
  (int ->
     string ->
     float ->
     float ->
     StackCore.Stage<StackCore.Image<'T>,
                     (StackCore.Image<'T> *
                      StackSerialSections.SerialSliceManifest)>)
    when 'T: equality

val serialEstBoundingBox<'T when 'T: equality> :
  StackCore.Stage<(StackCore.Image<'T> * StackSerialSections.SerialSliceManifest),
                  StackSerialSections.SerialVolumeGeometry> when 'T: equality

val serialApplyTrans<'T when 'T: equality> :
  (obj ->
     StackSerialSections.SerialVolumeGeometry option ->
     StackCore.Stage<(StackCore.Image<'T> *
                      StackSerialSections.SerialSliceManifest),
                     StackCore.Image<'T>>) when 'T: equality

val serialTransImage<'T when 'T: equality> :
  StackCore.Stage<(StackCore.Image<'T> * StackSerialSections.SerialSliceManifest),
                  StackCore.Image<'T>> when 'T: equality

val serialTransManifest<'T when 'T: equality> :
  StackCore.Stage<(StackCore.Image<'T> * StackSerialSections.SerialSliceManifest),
                  StackSerialSections.SerialSliceManifest> when 'T: equality

val serialApplyManifestInBoundingBox<'T when 'T: equality> :
  (StackSerialSections.SerialSliceManifest ->
     obj -> StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>>)
    when 'T: equality

val chunkSerialEstTrans<'T
                          when 'T: equality and 'T: (new: unit -> 'T) and
                               'T: struct and 'T :> System.ValueType> :
  (int ->
     string ->
     float ->
     float ->
     StackCore.Stage<StackCore.Chunk<'T>,
                     (StackCore.Chunk<'T> *
                      StackSerialSections.SerialSliceManifest)>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkSerialEstBoundingBox<'T when 'T: equality> :
  StackCore.Stage<(StackCore.Chunk<'T> * StackSerialSections.SerialSliceManifest),
                  StackSerialSections.SerialVolumeGeometry> when 'T: equality

val chunkSerialApplyTrans<'T
                            when 'T: equality and 'T: (new: unit -> 'T) and
                                 'T: struct and 'T :> System.ValueType> :
  ('T ->
     StackSerialSections.SerialVolumeGeometry option ->
     StackCore.Stage<(StackCore.Chunk<'T> *
                      StackSerialSections.SerialSliceManifest),
                     StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val chunkSerialTransChunk<'T when 'T: equality> :
  StackCore.Stage<(StackCore.Chunk<'T> * StackSerialSections.SerialSliceManifest),
                  StackCore.Chunk<'T>> when 'T: equality

val chunkSerialTransManifest<'T when 'T: equality> :
  StackCore.Stage<(StackCore.Chunk<'T> * StackSerialSections.SerialSliceManifest),
                  StackSerialSections.SerialSliceManifest> when 'T: equality

val chunkSerialApplyManifestInBoundingBox<'T
                                            when 'T: equality and
                                                 'T: (new: unit -> 'T) and
                                                 'T: struct and
                                                 'T :> System.ValueType> :
  (StackSerialSections.SerialSliceManifest ->
     'T -> StackCore.Stage<StackCore.Chunk<'T>,StackCore.Chunk<'T>>)
    when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and
         'T :> System.ValueType

val quantiles:
  (float list -> StackCore.Histogram<'a> -> float list) when 'a: comparison

val otsuThresholdFromHistogram:
  (StackCore.Histogram<'a> -> float) when 'a: comparison

val momentsThresholdFromHistogram:
  (StackCore.Histogram<'a> -> float) when 'a: comparison

val histogramCounts<'T when 'T: comparison> :
  SlimPipeline.Stage<StackCore.Histogram<'T>,Map<'T,uint64>> when 'T: comparison

val inline histogram2pairs<^T
                             when ^T: comparison and
                                  ^T: (static member op_Explicit: ^T -> float)> :
  SlimPipeline.Stage<StackCore.Histogram<^T>,(^T * uint64) list>
    when ^T: comparison and ^T: (static member op_Explicit: ^T -> float)

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
     uint -> uint -> StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>>)
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

val identity<'T> : Stage<'T,'T>

val erode:
  (uint -> StackCore.Stage<StackCore.Image<uint8>,StackCore.Image<uint8>>)

val dilate:
  (uint -> StackCore.Stage<StackCore.Image<uint8>,StackCore.Image<uint8>>)

val dilateZonohedral:
  (uint32 ->
     uint option ->
     SlimPipeline.Stage<StackCore.Image<uint8>,StackCore.Image<uint8>>)

val erodeZonohedral:
  (uint32 ->
     uint option ->
     SlimPipeline.Stage<StackCore.Image<uint8>,StackCore.Image<uint8>>)

val opening:
  (uint -> StackCore.Stage<StackCore.Image<uint8>,StackCore.Image<uint8>>)

val openingZonohedral:
  (uint32 ->
     uint option ->
     SlimPipeline.Stage<StackCore.Image<uint8>,StackCore.Image<uint8>>)

val closing:
  (uint -> StackCore.Stage<StackCore.Image<uint8>,StackCore.Image<uint8>>)

val closingZonohedral:
  (uint32 ->
     uint option ->
     SlimPipeline.Stage<StackCore.Image<uint8>,StackCore.Image<uint8>>)

val connectedComponents:
  (uint32 option ->
     SlimPipeline.Stage<StackCore.Image<uint8>,
                        (StackCore.Image<uint64> * uint64)>)

val connectedComponentsLabels:
  (uint32 option ->
     SlimPipeline.Stage<StackCore.Image<uint8>,StackCore.Image<uint64>>)

val connectedComponentsFullVolumeMemoryBytes: (uint -> uint -> uint -> uint64)

val connectedComponentsFullVolumeFits: (uint64 -> uint -> uint -> uint -> bool)

val relabelComponents:
  (uint ->
     uint32 option ->
     SlimPipeline.Stage<StackCore.Image<'a>,StackCore.Image<'a>>)
    when 'a: equality

val signedDistanceBand:
  (uint ->
     uint -> SlimPipeline.Stage<StackCore.Image<uint8>,StackCore.Image<float>>)

val threshold:
  (float -> float -> StackCore.Stage<StackCore.Image<'a>,StackCore.Image<uint8>>)
    when 'a: equality

val slicesToSlabs:
  chunkDepth: uint ->
    SlimPipeline.Stage<StackCore.Image<'T>,StackCore.Image<'T>>
    when 'T: equality

val windowToSlab<'T when 'T: equality> :
  StackCore.Stage<StackCore.Window<StackCore.Image<'T>>,StackCore.Image<'T>>
    when 'T: equality

val windowToSlabWithRange<'T when 'T: equality> :
  StackCore.Stage<StackCore.Window<StackCore.Image<'T>>,StackCore.Slab<'T>>
    when 'T: equality

val mapSlabWithStage<'S,'T when 'S: equality and 'T: equality> :
  (StackCore.Stage<StackCore.Image<'S>,StackCore.Image<'T>> ->
     StackCore.Stage<StackCore.Slab<'S>,StackCore.Slab<'T>>)
    when 'S: equality and 'T: equality

val slabToWindow<'T when 'T: equality> :
  StackCore.Stage<StackCore.Image<'T>,StackCore.Window<StackCore.Image<'T>>>
    when 'T: equality

val slabWithRangeToWindow<'T when 'T: equality> :
  StackCore.Stage<StackCore.Slab<'T>,StackCore.Window<StackCore.Image<'T>>>
    when 'T: equality

val windowSkipTakeM:
  outputStart: uint ->
    outputCount: uint32 -> StackCore.Stage<StackCore.Window<'a>,'a list>

val slabSkipTakeM<'T when 'T: equality> :
  (uint ->
     uint32 ->
     StackCore.Stage<StackCore.Window<StackCore.Image<'T>>,
                     StackCore.Image<'T> list>) when 'T: equality

val windowedViaSlab<'S,'T when 'S: equality and 'T: equality> :
  (uint ->
     StackCore.Stage<StackCore.Image<'S>,StackCore.Image<'T>> ->
     StackCore.Stage<StackCore.Image<'S>,StackCore.Image<'T>>)
    when 'S: equality and 'T: equality

val windowSlabRoundtrip<'T when 'T: equality> :
  (uint -> StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>>)
    when 'T: equality

val windowedCast<'S,'T when 'S: equality and 'T: equality> :
  (uint -> StackCore.Stage<StackCore.Image<'S>,StackCore.Image<'T>>)
    when 'S: equality and 'T: equality

val windowedThreshold<'T when 'T: equality> :
  (uint ->
     float ->
     float -> StackCore.Stage<StackCore.Image<'T>,StackCore.Image<uint8>>)
    when 'T: equality

val addNormalNoise:
  (float -> float -> StackCore.Stage<StackCore.Image<'a>,StackCore.Image<'a>>)
    when 'a: equality

val addSaltAndPepperNoise:
  (float -> StackCore.Stage<StackCore.Image<'a>,StackCore.Image<'a>>)
    when 'a: equality

val addShotNoise:
  (float -> StackCore.Stage<StackCore.Image<'a>,StackCore.Image<'a>>)
    when 'a: equality

val addSpeckleNoise:
  (float -> StackCore.Stage<StackCore.Image<'a>,StackCore.Image<'a>>)
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

val polygonMask: (uint -> uint -> StackCore.Polygon2D -> StackCore.Image<uint8>)

val repeat<'T when 'T: equality> :
  (StackCore.Image<'T> ->
     uint ->
     SlimPipeline.Plan<unit,unit> -> SlimPipeline.Plan<unit,StackCore.Image<'T>>)
    when 'T: equality

val repeatStage<'T when 'T: equality> :
  (uint -> StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>>)
    when 'T: equality

val euler2DTransformPath:
  (uint ->
     uint -> uint -> string -> uint -> (float * float * float) * (float * float))

val createByEuler2DTransformFromImage<'T when 'T: equality> :
  (uint ->
     (uint -> (float * float * float) * (float * float)) ->
     StackCore.Stage<StackCore.Image<'T>,StackCore.Image<'T>>) when 'T: equality

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

type ComponentStatistics = StackImageFunctions.ComponentStatistics

type ConnectedComponentTranslationTable =
    StackImageFunctions.ConnectedComponentTranslationTable

val makeConnectedComponentTranslationTable:
  (uint32 option ->
     StackCore.Stage<(StackCore.Image<uint64> * uint64),
                     StackImageFunctions.ConnectedComponentTranslationTable>)

val makeConnectedComponentLabelTranslationTable:
  (uint32 option ->
     StackCore.Stage<(StackCore.Image<uint64> * uint64),
                     StackImageFunctions.ConnectedComponentTranslationTable>)

val updateConnectedComponents:
  (uint32 option ->
     StackImageFunctions.ConnectedComponentTranslationTable ->
     StackCore.Stage<StackCore.Image<uint64>,StackCore.Image<uint64>>)

val permuteAxes:
  (uint * uint * uint ->
     uint -> StackCore.Stage<StackCore.Image<'a>,StackCore.Image<'a>>)
    when 'a: equality
