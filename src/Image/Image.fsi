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
    val toComplexFloat32: lst: float32 list -> System.Numerics.Complex
    val toComplexFloat64: lst: float list -> System.Numerics.Complex
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
    val isComplexPixelId: pid: itk.simple.PixelIDValueEnum -> bool
    val isComplexCompatibleImage: itkImg: itk.simple.Image -> bool
    val isScalarImportSupported<'T> : bool
    val setImportBuffer<'T> :
      importer: itk.simple.ImportImageFilter -> buffer: nativeint -> unit
    val scalarComponentByteSize<'T> : int
    val getConstBuffer<'T> : image: itk.simple.Image -> nativeint
    val copyScalarPixels: image: itk.simple.Image -> pixelCount: int -> 'T array
    val importScalarImage:
      size: uint list -> pixels: 'T array -> itk.simple.Image
    val ofCastItk<'T> : itkImg: itk.simple.Image -> itk.simple.Image
    val private identityDirection: dim: int -> float list
    val canonicalizeSimpleItkImage: image: itk.simple.Image -> itk.simple.Image
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
    val getFloatPixel:
      img: itk.simple.Image -> u: itk.simple.VectorUInt32 -> float
    val setFloatPixel:
      img: itk.simple.Image ->
        u: itk.simple.VectorUInt32 -> value: float -> unit
    val private ensureNativeComplex:
      name: string -> img: itk.simple.Image -> unit
    val extractComplexRealImage: img: itk.simple.Image -> itk.simple.Image
    val extractComplexImagImage: img: itk.simple.Image -> itk.simple.Image
    val inline mulAdd:
      t: itk.simple.PixelIDValueEnum -> acc: obj -> k: obj -> p: obj -> obj
val getBytesPerComponent: t: System.Type -> uint32
val getBytesPerSItkComponent: t: itk.simple.PixelIDValueEnum -> uint32
type ImageFacts =
    {
      Backend: string
      PixelType: string
      ComponentBytes: uint64
      ComponentsPerPixel: uint64
      Size: uint64 list
      VoxelCount: uint64
      MemoryBytes: uint64
    }
module ImageFacts =
    val private product: values: uint64 list -> uint64
    val create:
      backend: string ->
        pixelType: string ->
        componentBytes: uint64 ->
        componentsPerPixel: uint64 -> size: uint64 list -> ImageFacts
    val forType<'T> :
      size: uint list -> componentsPerPixel: uint32 -> ImageFacts
    val ofSimpleITK: sitk: itk.simple.Image -> ImageFacts
    val memoryBytesForType<'T> :
      nVoxels: uint64 -> componentsPerPixel: uint32 -> uint64
    val sliceBytesForType<'T> : width: uint -> height: uint -> uint64
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
val private currentRssBytes: unit -> uint64
val mutable private rssBaselineBytes: uint64
val mutable private peakRssDeltaBytes: uint64
val mutable private debugLevel: uint32
val private resetRssProbe: unit -> unit
val private rssDeltaBytes: unit -> uint64
val private sampleRssDeltaBytes: unit -> uint64 * uint64
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
      fold2: f: ('S -> 'T -> 'T -> 'S) ->
               acc0: 'S -> im1: Image<'T> -> im2: Image<'T> -> 'S
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
    static member memoryEstimate: width: uint -> height: uint -> uint64
    static member memoryEstimateSItk: sitk: itk.simple.Image -> uint32
    static member minimumImage: f1: Image<'T> -> f2: Image<'T> -> Image<'T>
    static member neq: f1: Image<'S> * f2: Image<'S> -> bool when 'S: equality
    static member
      ofArray2D: arr: 'T array2d * ?name: string * ?index: int -> Image<'T>
    static member
      ofArray3D: arr: 'T array3d * ?name: string * ?index: int -> Image<'T>
    static member
      ofArray3DComplex: arr: float array3d * ?name: string ->
                          Image<System.Numerics.Complex>
    static member
      ofArray3DVector: arr: 'S array3d * ?name: string -> Image<'S list>
                         when 'S: equality
    static member
      ofComplexArray2D: arr: System.Numerics.Complex array2d * ?name: string ->
                          Image<System.Numerics.Complex>
    static member
      ofComplexArray3D: arr: System.Numerics.Complex array3d * ?name: string ->
                          Image<System.Numerics.Complex>
    static member
      ofFile: filename: string * ?optionalName: string * ?optionalIndex: int ->
                Image<'T>
    static member
      ofFileComplex: filename: string * ?optionalName: string *
                     ?optionalIndex: int -> Image<System.Numerics.Complex>
    static member
      ofFileVector: filename: string * ?optionalName: string *
                    ?optionalIndex: int -> Image<'S list> when 'S: equality
    static member
      ofImageList: images: Image<'S> list -> Image<'S list> when 'S: equality
    static member
      ofImagePairToComplex: realImg: Image<float> ->
                              imagImg: Image<float> ->
                              Image<System.Numerics.Complex>
    static member
      ofSimpleITK: itkImg: itk.simple.Image * ?optionalName: string *
                   ?optionalIndex: int -> Image<'T>
    static member setDebug: d: bool -> unit
    static member setDebugLevel: level: uint32 -> unit
    static member
      toArray3DVector: img: Image<'S list> -> 'S array3d when 'S: equality
    static member unzip: im: Image<'T list> -> Image<'T> list
    static member zip: imLst: Image<'T> list -> Image<'T list>
    member CompareTo: other: Image<'T> -> int
    override Equals: obj: obj -> bool
    member Get: coords: uint list -> 'T
    member GetDepth: unit -> uint32
    member GetDimensions: unit -> uint32
    member GetFacts: unit -> ImageFacts
    override GetHashCode: unit -> int
    member GetHeight: unit -> uint32
    member GetMemoryBytes: unit -> uint64
    member GetNumberOfComponentsPerPixel: unit -> uint32
    member GetSize: unit -> uint list
    member
      GetSlice: start0: int option * stop0: int option * start1: int option *
                stop1: int option * start2: int option * stop2: int option ->
                  Image<'T>
    member GetWidth: unit -> uint32
    member Set: coords: uint list -> value: 'T -> unit
    member private SetImg: itkImg: itk.simple.Image -> unit
    member
      SetSlice: start0: int option * stop0: int option * start1: int option *
                stop1: int option * start2: int option * stop2: int option ->
                  src: Image<'T> -> unit
    override ToString: unit -> string
    member castTo: unit -> Image<'S> when 'S: equality
    member decRefCount: unit -> unit
    member forAll: p: ('T -> bool) -> bool
    member getNReferences: unit -> int
    member incRefCount: unit -> unit
    member toArray2D: unit -> 'T array2d
    member toArray3D: unit -> 'T array3d
    member toComplex: unit -> Image<System.Numerics.Complex>
    member toComplexArray2D: unit -> System.Numerics.Complex array2d
    member toComplexArray3D: unit -> System.Numerics.Complex array3d
    member toFile: filename: string * ?optionalFormat: string -> unit
    member toFileComplex: filename: string * ?optionalFormat: string -> unit
    member toFileVector: filename: string * ?optionalFormat: string -> unit
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
    member toVectorFloat32: unit -> Image<float32 list>
    member toVectorFloat64: unit -> Image<float list>
    member toVectorInt16: unit -> Image<int16 list>
    member toVectorInt32: unit -> Image<int32 list>
    member toVectorInt64: unit -> Image<int64 list>
    member toVectorInt8: unit -> Image<int8 list>
    member toVectorUInt16: unit -> Image<uint16 list>
    member toVectorUInt32: unit -> Image<uint32 list>
    member toVectorUInt64: unit -> Image<uint64 list>
    member toVectorUInt8: unit -> Image<uint8 list>
    member Display: string
    member Image: itk.simple.Image
    member Item: i0: int * i1: int -> 'T with get
    member Item: i0: int * i1: int -> 'T with set
    member Item: i0: int * i1: int * i2: int -> 'T with get
    member Item: i0: int * i1: int * i2: int -> 'T with set
    member Name: string
    member index: int with get, set
val Re: img: Image<System.Numerics.Complex> -> Image<float>
val Im: img: Image<System.Numerics.Complex> -> Image<float>
val modulus: img: Image<System.Numerics.Complex> -> Image<float>
val arg: img: Image<System.Numerics.Complex> -> Image<float>
val toComplex:
  realImg: Image<float> ->
    imagImg: Image<float> -> Image<System.Numerics.Complex>
val polarToComplex:
  modulusImg: Image<float> ->
    argImg: Image<float> -> Image<System.Numerics.Complex>
val conjugate:
  img: Image<System.Numerics.Complex> -> Image<System.Numerics.Complex>
module ImageFunctions
val inline imageAddScalar:
  img: Image.Image<^S> -> i: ^S -> Image.Image<^S>
    when ^S: equality and ^S: (static member op_Explicit: ^S -> float)
val inline scalarAddImage:
  i: ^S -> img: Image.Image<^S> -> Image.Image<^S>
    when ^S: equality and ^S: (static member op_Explicit: ^S -> float)
val inline imageSubScalar:
  img: Image.Image<^S> -> i: ^S -> Image.Image<^S>
    when ^S: equality and ^S: (static member op_Explicit: ^S -> float)
val inline scalarSubImage:
  i: ^S -> img: Image.Image<^S> -> Image.Image<^S>
    when ^S: equality and ^S: (static member op_Explicit: ^S -> float)
val inline imageMulScalar:
  img: Image.Image<^S> -> i: ^S -> Image.Image<^S>
    when ^S: equality and ^S: (static member op_Explicit: ^S -> float)
val inline scalarMulImage:
  i: ^S -> img: Image.Image<^S> -> Image.Image<^S>
    when ^S: equality and ^S: (static member op_Explicit: ^S -> float)
val inline imageDivScalar:
  img: Image.Image<^S> -> i: ^S -> Image.Image<^S>
    when ^S: equality and ^S: (static member op_Explicit: ^S -> float)
val inline scalarDivImage:
  i: ^S -> img: Image.Image<^S> -> Image.Image<^S>
    when ^S: equality and ^S: (static member op_Explicit: ^S -> float)
val inline imagePowScalar:
  img: Image.Image<^S> * i: ^S -> Image.Image<^S>
    when ^S: equality and ^S: (static member op_Explicit: ^S -> float)
val inline scalarPowImage:
  i: ^S * img: Image.Image<^S> -> Image.Image<^S>
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
val crop2D:
  cropLower: uint list ->
    cropUpper: uint list -> img: Image.Image<'T> -> Image.Image<'T>
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
val clampImage:
  lower: double -> upper: double -> (Image.Image<'T> -> Image.Image<'T>)
    when 'T: equality
val rescaleIntensity:
  outputMinimum: double ->
    outputMaximum: double -> (Image.Image<'T> -> Image.Image<'T>)
    when 'T: equality
val intensityWindow:
  windowMinimum: double ->
    windowMaximum: double ->
    outputMinimum: double ->
    outputMaximum: double -> (Image.Image<'T> -> Image.Image<'T>)
    when 'T: equality
val normalizeImage: img: Image.Image<'T> -> Image.Image<float> when 'T: equality
val shiftScale:
  shift: double -> scale: double -> (Image.Image<'T> -> Image.Image<'T>)
    when 'T: equality
val invertIntensity:
  maximum: double -> (Image.Image<'T> -> Image.Image<'T>) when 'T: equality
val median:
  radius: uint -> (Image.Image<'T> -> Image.Image<'T>) when 'T: equality
val bilateral:
  domainSigma: double ->
    rangeSigma: double -> (Image.Image<'T> -> Image.Image<'T>) when 'T: equality
val gradientMagnitude: img: Image.Image<'T> -> Image.Image<'T> when 'T: equality
val sobelEdge: img: Image.Image<'T> -> Image.Image<'T> when 'T: equality
val laplacian: img: Image.Image<'T> -> Image.Image<'T> when 'T: equality
val equalImage:
  a: Image.Image<'T> -> b: Image.Image<'T> -> Image.Image<uint8>
    when 'T: equality
val notEqualImage:
  a: Image.Image<'T> -> b: Image.Image<'T> -> Image.Image<uint8>
    when 'T: equality
val greaterImage:
  a: Image.Image<'T> -> b: Image.Image<'T> -> Image.Image<uint8>
    when 'T: equality
val greaterEqualImage:
  a: Image.Image<'T> -> b: Image.Image<'T> -> Image.Image<uint8>
    when 'T: equality
val lessImage:
  a: Image.Image<'T> -> b: Image.Image<'T> -> Image.Image<uint8>
    when 'T: equality
val lessEqualImage:
  a: Image.Image<'T> -> b: Image.Image<'T> -> Image.Image<uint8>
    when 'T: equality
val andImage:
  a: Image.Image<uint8> -> b: Image.Image<uint8> -> Image.Image<uint8>
val orImage:
  a: Image.Image<uint8> -> b: Image.Image<uint8> -> Image.Image<uint8>
val xorImage:
  a: Image.Image<uint8> -> b: Image.Image<uint8> -> Image.Image<uint8>
val notImage: img: Image.Image<uint8> -> Image.Image<uint8>
val mask:
  outsideValue: double ->
    img: Image.Image<'T> -> mask: Image.Image<uint8> -> Image.Image<'T>
    when 'T: equality
val euler2DTransform:
  img: Image.Image<'T> ->
    cx: float * cy: float * a: float -> dx: float * dy: float -> Image.Image<'T>
    when 'T: equality
val euler2DRotate:
  img: Image.Image<'T> -> cx: float * cy: float -> a: float -> Image.Image<'T>
    when 'T: equality
val resample2D:
  interpolator: itk.simple.InterpolatorEnum ->
    outputWidth: uint ->
    outputHeight: uint ->
    outputSpacingX: float ->
    outputSpacingY: float -> img: Image.Image<'T> -> Image.Image<'T>
    when 'T: equality
type BoundaryCondition =
    | ZeroPad
    | PerodicPad
    | ZeroFluxNeumannPad
type OutputRegionMode =
    | Valid
    | Same
val private convolutionBoundaryConditionType:
  boundaryCondition: BoundaryCondition option ->
    itk.simple.ConvolutionImageFilter.BoundaryConditionType
val internal convolve3:
  img: itk.simple.Image ->
    ker: itk.simple.Image ->
    outputRegionMode: OutputRegionMode option ->
    boundaryCondition: BoundaryCondition option -> itk.simple.Image
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
val finiteDiffFilter3D: direction: uint -> order: uint -> Image.Image<float>
val gradientVector3D:
  order: uint -> img: Image.Image<float> -> Image.Image<float list>
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
val grayscaleErode:
  radius: uint -> (Image.Image<'T> -> Image.Image<'T>) when 'T: equality
val grayscaleDilate:
  radius: uint -> (Image.Image<'T> -> Image.Image<'T>) when 'T: equality
val grayscaleOpening:
  radius: uint -> (Image.Image<'T> -> Image.Image<'T>) when 'T: equality
val grayscaleClosing:
  radius: uint -> (Image.Image<'T> -> Image.Image<'T>) when 'T: equality
val whiteTopHat:
  radius: uint -> (Image.Image<'T> -> Image.Image<'T>) when 'T: equality
val blackTopHat:
  radius: uint -> (Image.Image<'T> -> Image.Image<'T>) when 'T: equality
val morphologicalGradient:
  radius: uint -> (Image.Image<'T> -> Image.Image<'T>) when 'T: equality
/// Fill holes in binary regions
val binaryFillHoles: img: Image.Image<uint8> -> Image.Image<uint8>
val binaryContour:
  fullyConnected: bool -> (Image.Image<uint8> -> Image.Image<uint8>)
val binaryThinning: img: Image.Image<uint8> -> Image.Image<uint8>
val binaryMedian: radius: uint -> (Image.Image<uint8> -> Image.Image<uint8>)
val binaryOpeningByReconstruction:
  radius: uint ->
    fullyConnected: bool -> (Image.Image<uint8> -> Image.Image<uint8>)
val binaryClosingByReconstruction:
  radius: uint ->
    fullyConnected: bool -> (Image.Image<uint8> -> Image.Image<uint8>)
val binaryReconstructionByDilation:
  fullyConnected: bool ->
    marker: Image.Image<uint8> -> mask: Image.Image<uint8> -> Image.Image<uint8>
val binaryReconstructionByErosion:
  fullyConnected: bool ->
    marker: Image.Image<uint8> -> mask: Image.Image<uint8> -> Image.Image<uint8>
val votingBinaryHoleFilling:
  radius: uint ->
    majorityThreshold: uint -> (Image.Image<uint8> -> Image.Image<uint8>)
type ConnectedComponentsResult =
    {
      Labels: Image.Image<uint64>
      ObjectCount: uint64
    }
/// Connected components labeling
val connectedComponents: img: Image.Image<uint8> -> ConnectedComponentsResult
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
      Roundness: float
    }
/// Compute label shape statistics and return a dictionary of results
val labelShapeStatistics:
  img: Image.Image<'T> -> Map<int64,LabelShapeStatistics> when 'T: equality
type LabelIntensityStatistics =
    {
      Label: int64
      NumberOfPixels: uint64
      PhysicalSize: float
      Mean: float
      Median: float
      Minimum: float
      Maximum: float
      Sum: float
      StandardDeviation: float
      Variance: float
      Skewness: float
      Kurtosis: float
      Centroid: float list
      CenterOfGravity: float list
      BoundingBox: uint list
    }
val labelIntensityStatistics:
  labelImage: Image.Image<'L> ->
    intensityImage: Image.Image<'T> -> Map<int64,LabelIntensityStatistics>
    when 'L: equality and 'T: equality
type LabelOverlapMeasures =
    {
      MeanOverlap: float
      UnionOverlap: float
      JaccardCoefficient: float
      DiceCoefficient: float
      VolumeSimilarity: float
      FalseNegativeError: float
      FalsePositiveError: float
      FalseDiscoveryRate: float
    }
val labelOverlapMeasures:
  source: Image.Image<'T> -> target: Image.Image<'T> -> LabelOverlapMeasures
    when 'T: equality
val labelContour:
  fullyConnected: bool -> (Image.Image<'T> -> Image.Image<'T>) when 'T: equality
val changeLabel:
  fromLabel: double -> toLabel: double -> (Image.Image<'T> -> Image.Image<'T>)
    when 'T: equality
/// Compute signed Maurer distance map (positive outside, negative inside)
val signedDistanceMap:
  inside: uint8 ->
    outside: uint8 -> img: Image.Image<uint8> -> Image.Image<float>
val bandSignedDistanceMap:
  bandRadius: uint -> img: Image.Image<uint8> -> Image.Image<float>
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
val private valuesFromImages:
  bins: uint32 -> images: Image.Image<'T> list -> operation: 'a -> float list
    when 'T: equality
val private binnedHistogram:
  bins: uint32 -> values: float list -> float * float * float * uint64 array
val private orderedHistogramValues:
  histogram: Map<'T,uint64> -> operation: 'a -> (float * uint64) list
    when 'T: comparison
val otsuThresholdFromHistogram:
  histogram: Map<'T,uint64> -> float when 'T: comparison
val private otsuThresholdFromImages:
  bins: uint32 -> images: Image.Image<'a> list -> float when 'a: equality
/// Otsu threshold estimated from a binned histogram of the image values.
val otsuThreshold: img: Image.Image<'T> -> Image.Image<uint8> when 'T: equality
/// Otsu multiple thresholds (returns a label map)
val otsuMultiThreshold:
  numThresholds: byte -> img: Image.Image<'T> -> Image.Image<uint8>
    when 'T: equality
/// Moments-based threshold
val momentsThresholdFromHistogram:
  histogram: Map<'T,uint64> -> float when 'T: comparison
val private momentsThresholdFromImages:
  bins: uint32 -> images: Image.Image<'a> list -> float when 'a: equality
val momentsThreshold:
  img: Image.Image<'T> -> Image.Image<uint8> when 'T: equality
/// Coordinate fields
val generateCoordinateAxis: axis: int -> size: int list -> Image.Image<uint32>
val histogram: img: Image.Image<'T> -> Map<'T,uint64> when 'T: comparison
val histogramFixedBins:
  firstLeftEdge: float ->
    lastLeftEdge: float ->
    bins: uint32 -> img: Image.Image<'T> -> Map<float,uint64> when 'T: equality
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
val quantilesFromHistogram:
  quantiles: float list -> histogram: Map<'T,uint64> -> float list
    when 'T: comparison
val private retainNoNoise:
  img: Image.Image<'T> -> Image.Image<'T> when 'T: equality
val addNormalNoise:
  mean: float -> stddev: float -> (Image.Image<'T> -> Image.Image<'T>)
    when 'T: equality
val addSaltAndPepperNoise:
  probability: float -> (Image.Image<'T> -> Image.Image<'T>) when 'T: equality
val addShotNoise:
  scale: float -> (Image.Image<'T> -> Image.Image<'T>) when 'T: equality
val addSpeckleNoise:
  stddev: float -> (Image.Image<'T> -> Image.Image<'T>) when 'T: equality
val threshold:
  lower: float -> upper: float -> img: Image.Image<'T> -> Image.Image<uint8>
    when 'T: equality
val toVectorImage:
  images: Image.Image<'T> list -> Image.Image<'T list> when 'T: equality
val vectorElement:
  componentId: uint -> img: Image.Image<'T list> -> Image.Image<'T>
    when 'T: equality
val vectorRange:
  firstComponent: uint ->
    componentCount: uint -> img: Image.Image<'T list> -> Image.Image<'T list>
    when 'T: equality
val private requireThreeComponents:
  name: 'a -> img: Image.Image<'T list> -> unit when 'T: equality
val private clampByte: value: float -> byte
val vector3ToColor:
  inputMinimum: float ->
    inputMaximum: float ->
    img: Image.Image<float list> -> Image.Image<uint8 list>
val colorToVector3:
  outputMinimum: float ->
    outputMaximum: float ->
    img: Image.Image<uint8 list> -> Image.Image<float list>
val appendVectorElement:
  vector: Image.Image<float list> ->
    element: Image.Image<float> -> Image.Image<float list>
val mapVectorElements:
  f: (float -> float) -> img: Image.Image<float list> -> Image.Image<float list>
val private ensureMatchingVectorImages:
  name: 'a -> a: Image.Image<float list> -> b: Image.Image<float list> -> unit
val vectorDot:
  a: Image.Image<float list> -> b: Image.Image<float list> -> Image.Image<float>
val vectorCross3D:
  a: Image.Image<float list> ->
    b: Image.Image<float list> -> Image.Image<float list>
val vectorAngleTo:
  reference: float list -> img: Image.Image<float list> -> Image.Image<float>
val structureTensorOuterProduct:
  gradient: Image.Image<float list> -> Image.Image<float list>
val smoothVectorElements3D:
  sigma: float -> img: Image.Image<float list> -> Image.Image<float list>
val structureTensorEigenImages:
  tensor: Image.Image<float list> -> Image.Image<float list> list
val structureTensorEigenMatrix:
  tensor: Image.Image<float list> -> Image.Image<float list>
val stack: images: Image.Image<'T> list -> Image.Image<'T> when 'T: equality
val extractSub:
  topLeft: uint list ->
    bottomRight: uint list -> img: Image.Image<'T> -> Image.Image<'T>
    when 'T: equality
val extractSlice:
  dir: uint -> i: int -> img: Image.Image<'T> -> Image.Image<'T>
    when 'T: equality
val unstack:
  dir: uint -> vol: Image.Image<'T> -> Image.Image<'T> list when 'T: equality
val unstackSkipNTakeM:
  N: uint -> mWish: uint -> vol: Image.Image<'T> -> Image.Image<'T> list
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
val permuteAxes:
  order: uint list -> img: Image.Image<'T> -> Image.Image<'S>
    when 'T: equality and 'S: equality
val FFTXY:
  image: Image.Image<'T> -> Image.Image<System.Numerics.Complex>
    when 'T: equality
val directionalFFT:
  dir: uint -> image: Image.Image<'T> -> Image.Image<System.Numerics.Complex>
    when 'T: equality
val directionalFFTComplex:
  dir: uint ->
    inverse: bool ->
    image: Image.Image<System.Numerics.Complex> ->
    Image.Image<System.Numerics.Complex>
val inverseFFTXY:
  image: Image.Image<System.Numerics.Complex> -> Image.Image<float>
val shiftFFT:
  image: Image.Image<System.Numerics.Complex> ->
    Image.Image<System.Numerics.Complex>
