namespace FSharp
module Image
module InternalHelpers =
    val toVectorUInt8: lst: uint8 list -> itk.simple.VectorUInt8
    val toVectorInt8: lst: int8 list -> itk.simple.VectorInt8
    val toVectorUInt16: lst: uint16 list -> itk.simple.VectorUInt16
    val toVectorInt16: lst: int16 list -> itk.simple.VectorInt16
    val toVectorUInt32: lst: uint32 list -> itk.simple.VectorUInt32
    val toVectorInt32: lst: int32 list -> itk.simple.VectorInt32
    val toVectorUInt64: lst: uint64 list -> itk.simple.VectorUInt64
    val toVectorInt64: lst: int64 list -> itk.simple.VectorInt64
    val toVectorFloat32: lst: float32 list -> itk.simple.VectorFloat
    val toVectorFloat64: lst: float list -> itk.simple.VectorDouble
    val fromItkVector: f: ('a -> 'b) -> v: 'a seq -> 'b list
    val fromVectorUInt8: v: itk.simple.VectorUInt8 -> uint8 list
    val fromVectorInt8: v: itk.simple.VectorInt8 -> int8 list
    val fromVectorUInt16: v: itk.simple.VectorUInt16 -> uint16 list
    val fromVectorInt16: v: itk.simple.VectorInt16 -> int16 list
    val fromVectorUInt32: v: itk.simple.VectorUInt32 -> uint list
    val fromVectorInt32: v: itk.simple.VectorInt32 -> int list
    val fromVectorUInt64: v: itk.simple.VectorUInt64 -> uint64 list
    val fromVectorInt64: v: itk.simple.VectorInt64 -> int64 list
    val fromVectorFloat32: v: itk.simple.VectorFloat -> float32 list
    val fromVectorFloat64: v: itk.simple.VectorDouble -> float list
    val fromType<'T> : itk.simple.PixelIDValueEnum
    val ofCastItk<'T> : itkImg: itk.simple.Image -> itk.simple.Image
    val array2dZip: a: 'T array2d -> b: 'U array2d -> ('T * 'U) array2d
    val pixelIdToString: id: itk.simple.PixelIDValueEnum -> string
    val flatIndices: size: uint list -> uint list seq
    val setBoxedPixel:
      sitkImg: itk.simple.Image ->
        t: itk.simple.PixelIDValueEnum ->
        u: itk.simple.VectorUInt32 -> value: obj -> unit
    val getBoxedPixel:
      img: itk.simple.Image ->
        t: itk.simple.PixelIDValueEnum -> u: itk.simple.VectorUInt32 -> obj
    val getBoxedZero:
      t: itk.simple.PixelIDValueEnum -> vSize: uint option -> obj
    val inline mulAdd:
      t: itk.simple.PixelIDValueEnum -> acc: obj -> k: obj -> p: obj -> obj
val getBytesPerComponent: t: System.Type -> uint32
val getBytesPerSItkComponent: t: itk.simple.PixelIDValueEnum -> uint32
val equalOne: v: 'T -> bool
val private syncRoot: obj
val mutable private totalImages: int
val mutable private peakTotalImages: int
val incTotalImages: unit -> unit
val decTotalImages: unit -> unit
val mutable private memUsed: uint32
val mutable private peakMemUsed: uint32
val private incMemUsed: mem: uint32 -> unit
val private decMemUsed: mem: uint32 -> unit
val private printDebugMessage: str: string -> unit
val mutable private debug: bool
[<StructuredFormatDisplay ("{Display}")>]
type Image<'T when 'T: equality> =
    interface System.IComparable
    interface System.IEquatable<Image<'T>>
    new: sz: uint list * ?optionalNumberComponents: uint * ?optionalName: string *
         ?optionalIndex: int * ?optionalQuiet: bool -> Image<'T>
    static member
      (&&&) : f1: Image<'S> * f2: Image<'S> -> Image<'S> when 'S: equality
    static member ( * ) : f1: Image<'T> * f2: Image<'T> -> Image<'T>
    static member (+) : f1: Image<'T> * f2: Image<'T> -> Image<'T>
    static member (-) : f1: Image<'T> * f2: Image<'T> -> Image<'T>
    static member (/) : f1: Image<'T> * f2: Image<'T> -> Image<'T>
    static member
      (^^^) : f1: Image<'S> * f2: Image<'S> -> Image<'S> when 'S: equality
    static member
      (|||) : f1: Image<'S> * f2: Image<'S> -> Image<'S> when 'S: equality
    static member (~~~) : f: Image<'S> -> Image<'S> when 'S: equality
    static member
      Pow: f1: Image<'S> * f2: Image<'S> -> Image<'S> when 'S: equality
    static member eq: f1: Image<'S> * f2: Image<'S> -> bool when 'S: equality
    static member fold: f: ('S -> 'T -> 'S) -> acc0: 'S -> im1: Image<'T> -> 'S
    static member
      foldi: f: (uint list -> 'S -> 'T -> 'S) ->
               acc0: 'S -> im1: Image<'T> -> 'S
    static member getMinMax: img: Image<'T> -> float * float
    static member gt: f1: Image<'S> * f2: Image<'S> -> bool when 'S: equality
    static member gte: f1: Image<'S> * f2: Image<'S> -> bool when 'S: equality
    /// Comparison operators
    static member
      isEqual: f1: Image<'S> * f2: Image<'S> -> Image<'S> when 'S: equality
    static member
      isGreater: f1: Image<'S> * f2: Image<'S> -> Image<'S> when 'S: equality
    static member
      isGreaterEqual: f1: Image<'S> * f2: Image<'S> -> Image<'S>
                        when 'S: equality
    static member
      isLessThan: f1: Image<'S> * f2: Image<'S> -> Image<'S> when 'S: equality
    static member
      isLessThanEqual: f1: Image<'S> * f2: Image<'S> -> Image<'S>
                         when 'S: equality
    static member
      isNotEqual: f1: Image<'S> * f2: Image<'S> -> Image<'S> when 'S: equality
    static member iter: f: ('T -> unit) -> im1: Image<'T> -> unit
    static member iteri: f: (uint list -> 'T -> unit) -> im1: Image<'T> -> unit
    static member lt: f1: Image<'S> * f2: Image<'S> -> bool when 'S: equality
    static member lte: f1: Image<'S> * f2: Image<'S> -> bool when 'S: equality
    static member map: f: ('T -> 'T) -> im1: Image<'T> -> Image<'T>
    static member
      mapi: f: (uint list -> 'T -> 'T) -> im1: Image<'T> -> Image<'T>
    static member maximumImage: f1: Image<'T> -> f2: Image<'T> -> Image<'T>
    static member
      memoryEstimate: width: uint -> height: uint -> noComponent: uint -> uint64
    static member memoryEstimateSItk: sitk: itk.simple.Image -> uint32
    static member minimumImage: f1: Image<'T> -> f2: Image<'T> -> Image<'T>
    static member neq: f1: Image<'S> * f2: Image<'S> -> bool when 'S: equality
    static member ofArray2D: arr: 'T array2d * ?name: string -> Image<'T>
    static member ofArray3D: arr: 'T array3d * ?name: string -> Image<'T>
    static member ofArray4D: arr: 'T array4d * ?name: string -> Image<'T>
    static member
      ofFile: filename: string * ?optionalName: string * ?optionalIndex: int ->
                Image<'T>
    static member
      ofImageList: images: Image<'S> list -> Image<'S list> when 'S: equality
    static member
      ofSimpleITK: itkImg: itk.simple.Image * ?optionalName: string *
                   ?optionalIndex: int -> Image<'T>
    static member setDebug: d: bool -> unit
    static member unzip: im: Image<'T list> -> Image<'T> list
    static member zip: imLst: Image<'T> list -> Image<'T list>
    member CompareTo: other: Image<'T> -> int
    override Equals: obj: obj -> bool
    member Get: coords: uint list -> 'T
    member GetDepth: unit -> uint32
    member GetDimensions: unit -> uint32
    override GetHashCode: unit -> int
    member GetHeight: unit -> uint32
    member GetNumberOfComponentsPerPixel: unit -> uint32
    member GetSize: unit -> uint list
    member GetWidth: unit -> uint32
    member Set: coords: uint list -> value: 'T -> unit
    member private SetImg: itkImg: itk.simple.Image -> unit
    override ToString: unit -> string
    member castTo: unit -> Image<'S> when 'S: equality
    member decRefCount: unit -> unit
    member forAll: p: ('T -> bool) -> bool
    member getNReferences: unit -> int
    member incRefCount: unit -> unit
    member toArray2D: unit -> 'T array2d
    member toArray3D: unit -> 'T array3d
    member toArray4D: unit -> 'T array4d
    member toFile: filename: string * ?optionalFormat: string -> unit
    member toFloat: unit -> Image<float>
    member toFloat32: unit -> Image<float32>
    member toImageList: unit -> Image<'S> list when 'S: equality
    member toInt: unit -> Image<int>
    member toInt16: unit -> Image<int16>
    member toInt64: unit -> Image<int64>
    member toInt8: unit -> Image<int8>
    member toSimpleITK: unit -> itk.simple.Image
    member toUInt: unit -> Image<uint>
    member toUInt16: unit -> Image<uint16>
    member toUInt64: unit -> Image<uint64>
    member toUInt8: unit -> Image<uint8>
    member Display: string
    member Image: itk.simple.Image
    member Item: i0: int * i1: int -> 'T with get
    member Item: i0: int * i1: int -> 'T with set
    member Item: i0: int * i1: int * i2: int -> 'T with get
    member Item: i0: int * i1: int * i2: int -> 'T with set
    member Item: i0: int * i1: int * i2: int * i3: int -> 'T with get
    member Item: i0: int * i1: int * i2: int * i3: int -> 'T with set
    member Name: string
    member index: int with get, set
module ImageFunctions
val inline imageAddScalar:
  f1: Image.Image<^S> -> i: ^S -> Image.Image<^S>
    when ^S: equality and ^S: (static member op_Explicit: ^S -> float)
val inline scalarAddImage:
  i: ^S -> f1: Image.Image<^S> -> Image.Image<^S>
    when ^S: equality and ^S: (static member op_Explicit: ^S -> float)
val inline imageSubScalar:
  f1: Image.Image<^S> -> i: ^S -> Image.Image<^S>
    when ^S: equality and ^S: (static member op_Explicit: ^S -> float)
val inline scalarSubImage:
  i: ^S -> f1: Image.Image<^S> -> Image.Image<^S>
    when ^S: equality and ^S: (static member op_Explicit: ^S -> float)
val inline imageMulScalar:
  f1: Image.Image<^S> -> i: ^S -> Image.Image<^S>
    when ^S: equality and ^S: (static member op_Explicit: ^S -> float)
val inline scalarMulImage:
  i: ^S -> f1: Image.Image<^S> -> Image.Image<^S>
    when ^S: equality and ^S: (static member op_Explicit: ^S -> float)
val inline imageDivScalar:
  f1: Image.Image<^S> -> i: ^S -> Image.Image<^S>
    when ^S: equality and ^S: (static member op_Explicit: ^S -> float)
val inline scalarDivImage:
  i: ^S -> f1: Image.Image<^S> -> Image.Image<^S>
    when ^S: equality and ^S: (static member op_Explicit: ^S -> float)
val inline imagePowScalar:
  f1: Image.Image<^S> * i: ^S -> Image.Image<^S>
    when ^S: equality and ^S: (static member op_Explicit: ^S -> float)
val inline scalarPowImage:
  i: ^S * f1: Image.Image<^S> -> Image.Image<^S>
    when ^S: equality and ^S: (static member op_Explicit: ^S -> float)
val inline sum:
  img: Image.Image<^T> -> ^T
    when ^T: equality and ^T: (static member (+) : ^T * ^T -> ^T) and
         ^T: (static member Zero: ^T)
val inline prod:
  img: Image.Image<^T> -> ^T
    when ^T: equality and ^T: (static member ( * ) : ^T * ^T -> ^T) and
         ^T: (static member One: ^T)
val dump: img: Image.Image<'T> -> string when 'T: equality
val squeeze: img: Image.Image<'T> -> Image.Image<'T> when 'T: equality
val expand: dim: uint -> zero: 'S -> a: 'S list -> 'S list
val concatAlong:
  dim: uint -> a: Image.Image<'T> -> b: Image.Image<'T> -> Image.Image<'T>
    when 'T: equality
val constantPad2D:
  padLower: uint list ->
    padUpper: uint list -> c: double -> img: Image.Image<'T> -> Image.Image<'T>
    when 'T: equality
val inline makeUnaryImageOperatorWith:
  name: string ->
    createFilter: (unit -> 'Filter) ->
    setup: ('Filter -> unit) ->
    invoke: ('Filter -> itk.simple.Image -> itk.simple.Image) ->
    img: Image.Image<'T> -> Image.Image<'S>
    when 'Filter :> System.IDisposable and 'T: equality and 'S: equality
val inline makeUnaryImageOperator:
  name: string ->
    createFilter: (unit -> 'a) ->
    invoke: ('a -> itk.simple.Image -> itk.simple.Image) ->
    (Image.Image<'b> -> Image.Image<'c>)
    when 'a :> System.IDisposable and 'b: equality and 'c: equality
val inline makeBinaryImageOperatorWith:
  name: string ->
    createFilter: (unit -> 'Filter) ->
    setup: ('Filter -> unit) ->
    invoke: ('Filter -> itk.simple.Image -> itk.simple.Image -> itk.simple.Image) ->
    a: Image.Image<'T> -> b: Image.Image<'T> -> Image.Image<'T>
    when 'Filter :> System.IDisposable and 'T: equality
val makeBinaryImageOperator:
  name: string ->
    createFilter: (unit -> 'a) ->
    invoke: ('a -> itk.simple.Image -> itk.simple.Image -> itk.simple.Image) ->
    (Image.Image<'b> -> Image.Image<'b> -> Image.Image<'b>)
    when 'a :> System.IDisposable and 'b: equality
val absImage:
  img: Image.Image<'T> -> Image.Image<'a> when 'T: equality and 'a: equality
val logImage:
  img: Image.Image<'T> -> Image.Image<'a> when 'T: equality and 'a: equality
val log10Image:
  img: Image.Image<'T> -> Image.Image<'a> when 'T: equality and 'a: equality
val expImage:
  img: Image.Image<'T> -> Image.Image<'a> when 'T: equality and 'a: equality
val sqrtImage:
  img: Image.Image<'T> -> Image.Image<'a> when 'T: equality and 'a: equality
val squareImage:
  img: Image.Image<'T> -> Image.Image<'a> when 'T: equality and 'a: equality
val sinImage:
  img: Image.Image<'T> -> Image.Image<'a> when 'T: equality and 'a: equality
val cosImage:
  img: Image.Image<'T> -> Image.Image<'a> when 'T: equality and 'a: equality
val tanImage:
  img: Image.Image<'T> -> Image.Image<'a> when 'T: equality and 'a: equality
val asinImage:
  img: Image.Image<'T> -> Image.Image<'a> when 'T: equality and 'a: equality
val acosImage:
  img: Image.Image<'T> -> Image.Image<'a> when 'T: equality and 'a: equality
val atanImage:
  img: Image.Image<'T> -> Image.Image<'a> when 'T: equality and 'a: equality
val roundImage:
  img: Image.Image<'T> -> Image.Image<'a> when 'T: equality and 'a: equality
val euler2DTransform:
  img: Image.Image<'T> ->
    cx: float * cy: float * a: float -> dx: float * dy: float -> Image.Image<'T>
    when 'T: equality
val euler2DRotate:
  img: Image.Image<'T> -> cx: float * cy: float -> a: float -> Image.Image<'T>
    when 'T: equality
type BoundaryCondition =
    | ZeroPad
    | PerodicPad
    | ZeroFluxNeumannPad
type OutputRegionMode =
    | Valid
    | Same
val internal convolve3:
  img: itk.simple.Image ->
    ker: itk.simple.Image ->
    outputRegionMode: OutputRegionMode option -> itk.simple.Image
val convolve:
  outputRegionMode: OutputRegionMode option ->
    boundaryCondition: BoundaryCondition option ->
    (Image.Image<'T> -> Image.Image<'T> -> Image.Image<'T>) when 'T: equality
val conv:
  img: Image.Image<'T> -> ker: Image.Image<'T> -> Image.Image<'T>
    when 'T: equality
val defaultGaussWindowSize: sigma: float -> uint
/// Gaussian kernel convolution
val gauss:
  dim: uint -> sigma: float -> kernelSize: uint option -> Image.Image<'T>
    when 'T: equality
val private stensil: order: uint32 -> float list
val finiteDiffFilter2D: direction: uint -> order: uint -> Image.Image<float>
val finiteDiffFilter3D:
  sigma: float -> direction: uint -> order: uint -> Image.Image<float>
val finiteDiffFilter4D: direction: uint -> order: uint -> Image.Image<float>
val discreteGaussian:
  dim: uint ->
    sigma: float ->
    kernelSize: uint option ->
    outputRegionMode: OutputRegionMode option ->
    boundaryCondition: BoundaryCondition option ->
    input: Image.Image<'T> -> Image.Image<'T> when 'T: equality
/// Gradient convolution using Derivative filter
val gradientConvolve:
  direction: uint -> order: uint32 -> (Image.Image<'T> -> Image.Image<'T>)
    when 'T: equality
/// Mathematical morphology
/// Binary erosion
val binaryErode: radius: uint -> (Image.Image<uint8> -> Image.Image<uint8>)
/// Binary dilation
val binaryDilate: radius: uint -> (Image.Image<uint8> -> Image.Image<uint8>)
/// Binary opening (erode then dilate)
val binaryOpening: radius: uint -> (Image.Image<uint8> -> Image.Image<uint8>)
/// Binary closing (dilate then erode)
val binaryClosing: radius: uint -> (Image.Image<uint8> -> Image.Image<uint8>)
/// Fill holes in binary regions
val binaryFillHoles: img: Image.Image<uint8> -> Image.Image<uint8>
/// Connected components labeling
val connectedComponents: img: Image.Image<uint8> -> Image.Image<uint64>
/// Relabel components by size, optionally remove small objects
val relabelComponents:
  minObjectSize: uint -> (Image.Image<'T> -> Image.Image<'T>) when 'T: equality
type LabelShapeStatistics =
    {
      Label: int64
      PhysicalSize: float
      Centroid: float list
      BoundingBox: uint32 list
      Elongation: float
      Flatness: float
      FeretDiameter: float
      EquivalentEllipsoidDiameter: float list
      EquivalentSphericalPerimeter: float
      EquivalentSphericalRadius: float
      Indexes: uint32 list
      NumberOfPixels: uint64
      NumberOfPixelsOnBorder: uint64
      OrientedBoundingBoxDirection: float list
      OrientedBoundingBoxOrigin: float list
      OrientedBoundingBoxSize: float list
      OrientedBoundingBoxVertices: float list
      Perimeter: float
      PerimeterOnBorder: float
      PerimeterOnBorderRatio: float
      PrincipalAxes: float list
      PrincipalMoments: float list
      Region: uint32 list
      RLEIndexes: uint32 list
      Roundness: float
    }
/// Compute label shape statistics and return a dictionary of results
val labelShapeStatistics:
  img: Image.Image<'T> -> Map<int64,LabelShapeStatistics> when 'T: equality
/// Compute signed Maurer distance map (positive outside, negative inside)
val signedDistanceMap:
  inside: uint8 ->
    outside: uint8 -> img: Image.Image<uint8> -> Image.Image<float>
/// Morphological watershed (binary or grayscale)
val watershed:
  level: float -> (Image.Image<'T> -> Image.Image<'T>) when 'T: equality
/// Histogram related functions
type ImageStats =
    {
      NumPixels: uint
      Mean: float
      Std: float
      Min: float
      Max: float
      Sum: float
      Var: float
    }
val computeStats: img: Image.Image<'T> -> ImageStats when 'T: equality
val addComputeStats: s1: ImageStats -> s2: ImageStats -> ImageStats
val unique: img: Image.Image<'T> -> 'T list when 'T: comparison
/// Otsu threshold
val otsuThreshold: img: Image.Image<'T> -> Image.Image<uint8> when 'T: equality
/// Otsu multiple thresholds (returns a label map)
val otsuMultiThreshold:
  numThresholds: byte -> img: Image.Image<'T> -> Image.Image<uint8>
    when 'T: equality
/// Moments-based threshold
val momentsThreshold:
  img: Image.Image<'T> -> Image.Image<uint8> when 'T: equality
/// Coordinate fields
val generateCoordinateAxis: axis: int -> size: int list -> Image.Image<uint32>
val histogram: image: Image.Image<'T> -> Map<'T,uint64> when 'T: comparison
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
val addNormalNoise:
  mean: float -> stddev: float -> (Image.Image<'T> -> Image.Image<'T>)
    when 'T: equality
val threshold:
  lower: float -> upper: float -> img: Image.Image<'T> -> Image.Image<uint8>
    when 'T: equality
val toVectorOfImage: images: #itk.simple.Image seq -> itk.simple.VectorOfImage
val stack: images: Image.Image<'T> list -> Image.Image<'T> when 'T: equality
val stackOld: images: Image.Image<'T> list -> Image.Image<'T> when 'T: equality
val extractSub:
  topLeft: uint list ->
    bottomRight: uint list -> img: Image.Image<'T> -> Image.Image<'T>
    when 'T: equality
val extractSlice:
  z: int -> img: Image.Image<'T> -> Image.Image<'T> when 'T: equality
val unstack: vol: Image.Image<'T> -> Image.Image<'T> list when 'T: equality
val unstackSkipNTakeM:
  N: uint -> M: uint -> vol: Image.Image<'T> -> Image.Image<'T> list
    when 'T: equality
type FileInfo =
    {
      dimensions: uint
      size: uint64 list
      componentType: string
      numberOfComponents: uint
    }
val getFileInfo: filename: string -> FileInfo
val toSeqSeq: I: Image.Image<'T> -> float seq seq when 'T: equality
