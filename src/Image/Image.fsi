namespace FSharp
module Image
[<Struct>]
type ComplexFloat32 =
    new: real: float32 * imaginary: float32 -> ComplexFloat32
    val Real: float32
    val Imaginary: float32
    static member Zero: ComplexFloat32
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
    val toComplexFloat32: lst: float32 list -> ComplexFloat32
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
    val inline flatIndex2:
      width: ^a -> x: ^d -> y: ^b -> 'e
        when (^b or ^a) : (static member ( * ) : ^b * ^a -> ^c) and
             (^c or ^d) : (static member (+) : ^c * ^d -> 'e)
    val inline flatIndex3:
      width: ^a -> height: ^e -> x: ^h -> y: ^f -> z: ^d -> 'i
        when (^b or ^a) : (static member ( * ) : ^b * ^a -> ^g) and
             (^c or ^f) : (static member (+) : ^c * ^f -> ^b) and
             (^d or ^e) : (static member ( * ) : ^d * ^e -> ^c) and
             (^g or ^h) : (static member (+) : ^g * ^h -> 'i)
    val inline flatIndex4:
      width: ^a ->
        height: ^i -> depth: ^g -> x: ^l -> y: ^j -> z: ^h -> t: ^f -> 'm
        when (^b or ^a) : (static member ( * ) : ^b * ^a -> ^k) and
             (^c or ^j) : (static member (+) : ^c * ^j -> ^b) and
             (^d or ^i) : (static member ( * ) : ^d * ^i -> ^c) and
             (^e or ^h) : (static member (+) : ^e * ^h -> ^d) and
             (^f or ^g) : (static member ( * ) : ^f * ^g -> ^e) and
             (^k or ^l) : (static member (+) : ^k * ^l -> 'm)
    val importScalarImage:
      size: uint list -> pixels: 'T array -> itk.simple.Image
    val private deepCopyItkImage: itkImg: itk.simple.Image -> itk.simple.Image
    /// <summary>
    /// Creates a shallow SimpleITK image wrapper for an image whose pixel type already matches <typeparamref name="'T" />.
    /// The returned SimpleITK image shares the same pixel container until SimpleITK copy-on-write forces uniqueness.
    /// No cast, deep copy, or disposal of <paramref name="itkImg" /> is performed.
    /// </summary>
    val aliasSimpleITKImage<'T> : itkImg: itk.simple.Image -> itk.simple.Image
    /// <summary>
    /// Creates an independent SimpleITK image with pixel type <typeparamref name="'T" />.
    /// If <paramref name="itkImg" /> already has the requested pixel type, a shallow copy is first made and then
    /// <c>MakeUnique</c> is called to force a deep pixel-buffer copy. If the pixel type differs, SimpleITK's cast filter
    /// is used, which allocates a new output image. The argument is not disposed.
    /// </summary>
    val ofCastITK<'T> : itkImg: itk.simple.Image -> itk.simple.Image
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
    static member private canUseFlatScalar: im: Image<'T> -> bool
    static member
      constant2D: width: uint * height: uint * value: 'T * ?name: string *
                  ?index: int -> Image<'T>
    static member
      coordinateAxis2D: width: uint * height: uint * axis: int * ?name: string *
                        ?index: int -> Image<'T>
    static member eq: f1: Image<'S> * f2: Image<'S> -> bool when 'S: equality
    static member
      private flatCoords: size: uint list -> offset: int -> uint list
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
      ofArray3DVector: arr: 'S array3d * ?name: string * ?index: int ->
                         Image<'S list> when 'S: equality
    static member
      ofArray4D: arr: 'T array4d * ?name: string * ?index: int -> Image<'T>
    static member
      ofComplexArray2D: arr: System.Numerics.Complex array2d * ?name: string *
                        ?index: int -> Image<System.Numerics.Complex>
    static member
      ofComplexArray3D: arr: System.Numerics.Complex array3d * ?name: string *
                        ?index: int -> Image<System.Numerics.Complex>
    static member
      ofComplexFloat32Array2D: arr: ComplexFloat32 array2d * ?name: string *
                               ?index: int -> Image<ComplexFloat32>
    static member
      ofComplexFloat32Array3D: arr: ComplexFloat32 array3d * ?name: string *
                               ?index: int -> Image<ComplexFloat32>
    static member
      ofFile: filename: string * ?optionalName: string * ?optionalIndex: int ->
                Image<'T>
    static member
      ofFileComplex: filename: string * ?optionalName: string *
                     ?optionalIndex: int -> Image<System.Numerics.Complex>
    static member
      ofFileComplexFloat32: filename: string * ?optionalName: string *
                            ?optionalIndex: int -> Image<ComplexFloat32>
    static member
      ofFileVector: filename: string * ?optionalName: string *
                    ?optionalIndex: int -> Image<'S list> when 'S: equality
    static member
      ofFlatArray: size: uint list * pixels: 'T array * ?name: string *
                   ?index: int -> Image<'T>
    static member
      ofImageList: images: Image<'S> list -> Image<'S list> when 'S: equality
    static member
      ofImagePairToComplex: realImg: Image<float> ->
                              imagImg: Image<float> ->
                              Image<System.Numerics.Complex>
    static member
      ofImagePairToComplexFloat32: realImg: Image<float32> ->
                                     imagImg: Image<float32> ->
                                     Image<ComplexFloat32>
    /// <summary>
    /// Creates a safe, independent <c>Image&lt;'T&gt;</c> from a SimpleITK image.
    /// The resulting image does not share its pixel buffer with <paramref name="itkImg" />. Matching pixel types are
    /// deep-copied; non-matching pixel types are converted with SimpleITK's cast filter. Physical metadata is normalized
    /// to StackProcessing defaults. The argument is borrowed and is not disposed.
    /// </summary>
    static member
      ofSimpleITK: itkImg: itk.simple.Image * ?optionalName: string *
                   ?optionalIndex: int -> Image<'T>
    /// <summary>
    /// Creates an aliasing <c>Image&lt;'T&gt;</c> from a SimpleITK image whose pixel type already matches <c>'T</c>.
    /// The returned image uses a shallow SimpleITK copy and may share the same pixel container as
    /// <paramref name="itkImg" /> until SimpleITK copy-on-write forces uniqueness. No cast, deep copy, metadata
    /// canonicalization, or disposal of the argument is performed. This is intended for internal hot paths where
    /// aliasing is acceptable and explicit.
    /// </summary>
    static member
      ofSimpleITKAlias: itkImg: itk.simple.Image * ?optionalName: string *
                        ?optionalIndex: int -> Image<'T>
    /// <summary>
    /// Creates an aliasing <c>Image&lt;'T&gt;</c> by taking over a SimpleITK image whose pixel type already matches <c>'T</c>.
    /// No SimpleITK wrapper copy, deep copy, cast, or metadata canonicalization is performed. The returned image stores
    /// <paramref name="itkImg" /> directly and will dispose it when the image reference count reaches zero. The caller
    /// must not dispose or continue using <paramref name="itkImg" /> after a successful call.
    /// </summary>
    static member
      private ofSimpleITKAliasTransfer: itkImg: itk.simple.Image *
                                        ?optionalName: string *
                                        ?optionalIndex: int -> Image<'T>
    /// <summary>
    /// Creates an <c>Image&lt;'T&gt;</c> from a temporary SimpleITK image and consumes that temporary.
    /// If the pixel type already matches <c>'T</c>, the SimpleITK wrapper is transferred directly into the returned
    /// image with no copy or cast; the returned image will dispose it when its reference count reaches zero. If a cast is
    /// needed, the result is deep-copied through <c>ofSimpleITK</c> and <paramref name="itkImg" /> is disposed before
    /// returning. The caller must not dispose or continue using <paramref name="itkImg" /> after calling this function.
    /// </summary>
    static member
      ofSimpleITKNDispose: itkImg: itk.simple.Image * ?optionalName: string *
                           ?optionalIndex: int -> Image<'T>
    static member
      polygonMask: width: uint * height: uint * polygon: (float * float) list *
                   ?name: string * ?index: int -> Image<uint8>
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
    member copy: ?optionalName: string * ?optionalIndex: int -> Image<'T>
    member decRefCount: unit -> unit
    member forAll: p: ('T -> bool) -> bool
    member getNReferences: unit -> int
    member incRefCount: unit -> unit
    member toArray2D: unit -> 'T array2d
    member toArray3D: unit -> 'T array3d
    member toArray4D: unit -> 'T array4d
    member toComplex: unit -> Image<System.Numerics.Complex>
    member toComplexArray2D: unit -> System.Numerics.Complex array2d
    member toComplexArray3D: unit -> System.Numerics.Complex array3d
    member toComplexFloat32: unit -> Image<ComplexFloat32>
    member toComplexFloat32Array2D: unit -> ComplexFloat32 array2d
    member toComplexFloat32Array3D: unit -> ComplexFloat32 array3d
    member toFile: filename: string * ?optionalFormat: string -> unit
    member toFileComplex: filename: string * ?optionalFormat: string -> unit
    member toFileVector: filename: string * ?optionalFormat: string -> unit
    member toFlatArray: unit -> 'T array
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
    member Item: i0: int * i1: int * i2: int * i3: int -> 'T with get
    member Item: i0: int * i1: int * i2: int * i3: int -> 'T with set
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
module ImageIO
type ImageFileInfo =
    {
      Dimension: int
      Size: uint list
    }
val validatePixelType<'T> : unit -> unit
val imageFileInfo: filename: string -> ImageFileInfo
val readSimpleItkSlice:
  filename: string ->
    dimension: int ->
    width: uint ->
    height: uint ->
    sourceIndex: int -> name: string -> index: int -> Image.Image<'T>
    when 'T: equality
val tiffPixelLayout<'T> :
  unit -> int * BitMiracle.LibTiff.Classic.SampleFormat * int
val supportsDirectTiffRead<'T> : bool
val supportsDirectTiffWrite<'T> : bool
val tiffWriteMode: filename: string -> string
val tiffFieldInt:
  tiff: BitMiracle.LibTiff.Classic.Tiff ->
    tag: BitMiracle.LibTiff.Classic.TiffTag -> fallback: int -> int
val tiffFieldIntDefaulted:
  tiff: BitMiracle.LibTiff.Classic.Tiff ->
    tag: BitMiracle.LibTiff.Classic.TiffTag -> fallback: int -> int
val tiffDirectoryCount: filename: string -> uint32
val tiffBytesPerSample:
  bitsPerSample: int ->
    sampleFormat: BitMiracle.LibTiff.Classic.SampleFormat -> int
val validateTiffSamples: samplesPerPixel: int -> unit
val private setImportImageBufferFromTiffLayout:
  importer: itk.simple.ImportImageFilter ->
    bitsPerSample: int ->
    sampleFormat: BitMiracle.LibTiff.Classic.SampleFormat ->
    buffer: System.IntPtr -> unit
val bytesOfScalarImage2D: image: Image.Image<'T> -> byte array when 'T: equality
val readTiffPage:
  tiff: BitMiracle.LibTiff.Classic.Tiff ->
    width: uint32 ->
    height: uint32 ->
    bitsPerSample: int ->
    sampleFormat: BitMiracle.LibTiff.Classic.SampleFormat ->
    bytesPerSample: int -> index: int -> Image.Image<'T> when 'T: equality
val readTiffSliceFile:
  fileName: string -> sliceIndex: int64 -> Image.Image<'T> when 'T: equality
val writeTiffPage:
  tiff: BitMiracle.LibTiff.Classic.Tiff ->
    image: Image.Image<'T> -> page: int option -> unit when 'T: equality
val writeTiffSliceFile:
  fileName: string -> image: Image.Image<'T> -> unit when 'T: equality
module ImageFunctions
val imageFromTemporarySimpleITK:
  name: string -> index: int -> itkImage: itk.simple.Image -> Image.Image<'T>
    when 'T: equality
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
val smoothingRecursiveGaussian:
  sigma: float -> img: Image.Image<'T> -> Image.Image<'T> when 'T: equality
val laplacianRecursiveGaussian:
  sigma: float -> img: Image.Image<'T> -> Image.Image<'T> when 'T: equality
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
type ResampleInterpolation =
    | NearestNeighbor
    | Linear
module ResampleInterpolation =
    val parse: value: string -> ResampleInterpolation
    val internal toItk:
      _arg1: ResampleInterpolation -> itk.simple.InterpolatorEnum
val euler2DTransform:
  img: Image.Image<'T> ->
    cx: float * cy: float * a: float -> dx: float * dy: float -> Image.Image<'T>
    when 'T: equality
val euler2DRotate:
  img: Image.Image<'T> -> cx: float * cy: float -> a: float -> Image.Image<'T>
    when 'T: equality
val resample2D:
  interpolation: ResampleInterpolation ->
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
val private sphericalOffsets:
  dimensions: uint32 -> radius: uint -> (int * int * int) array
/// CPU implementation of binary dilation with a digital spherical structuring element.
///
/// This native implementation is kept as a direct spherical-footprint reference beside the SimpleITK implementation.
/// It uses the same binary convention as <c>binaryDilate</c>: foreground pixels have value 1
/// and the output contains only 0/1 values. The footprint follows SimpleITK's <c>sitkBall</c>
/// convention so results can be compared directly.
val binaryDilateSphericalNative:
  radius: uint -> img: Image.Image<uint8> -> Image.Image<uint8>
val private zonohedralBestCoefficients: (int * int * int) array
val private zonohedralLineSteps: (int * int * int) array
val zonohedralBestLines: radius: uint32 -> (int * int * int * int) array
val private vhgwDilateLine:
  length: int ->
    count: int ->
    line: uint8 array ->
    prefix: uint8 array ->
    suffix: uint8 array -> lineOutput: uint8 array -> unit
val private lineStarts3D:
  width: int ->
    height: int ->
    depth: int -> dx: int -> dy: int -> dz: int -> (int * int * int) array
val private lineDilate3D:
  width: int ->
    height: int ->
    depth: int ->
    input: uint8 array ->
    dx: int * dy: int * dz: int * length: int -> uint8 array
val private lineErode3D:
  width: int ->
    height: int ->
    depth: int ->
    input: uint8 array ->
    dx: int * dy: int * dz: int * length: int -> uint8 array
val private lineDilate3DRange:
  width: int ->
    height: int ->
    depth: int ->
    inputValidLow: int ->
    inputValidHigh: int ->
    outputLow: int ->
    outputHigh: int ->
    input: uint8 array ->
    dx: int * dy: int * dz: int * length: int -> uint8 array
val private expandZRangeForLine:
  depth: int ->
    outputLow: int ->
    outputHigh: int -> _dx: int * _dy: int * dz: int * length: int -> int * int
val zonohedralZHalo: radius: uint32 -> int
val binaryDilateZonohedralValidSlicesNative:
  radius: uint ->
    outputStart: uint ->
    outputCount: uint ->
    images: Image.Image<uint8> list -> Image.Image<uint8> list
/// Binary dilation using Jensen et al.'s zonohedral best approximation of a spherical structuring element.
///
/// The approximation is represented as a composition of line dilations in the 13 directions used by
/// Gorpho/pygorpho. The native line scans are useful for streaming because valid output slices can be
/// computed from a z-window without materializing the full slab output.
val binaryDilateZonohedralNative:
  radius: uint -> img: Image.Image<uint8> -> Image.Image<uint8>
/// Binary erosion using the same zonohedral approximation as
/// <c>binaryDilateZonohedralNative</c>.
val binaryErodeZonohedralNative:
  radius: uint -> img: Image.Image<uint8> -> Image.Image<uint8>
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
val private imageValues: img: Image.Image<'T> -> 'T seq when 'T: equality
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
val private composeVectorAndRelease:
  components: Image.Image<'T> list -> Image.Image<'T list> when 'T: equality
val private mapScalarComponentsAndCompose:
  f: (Image.Image<float> -> Image.Image<float>) ->
    img: Image.Image<float list> -> Image.Image<float list>
val private scalarMapArray:
  name: string ->
    f: (float -> float) -> img: Image.Image<float> -> Image.Image<float>
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
val private tensorEigenMatrixValues:
  xx: float ->
    xy: float -> xz: float -> yy: float -> yz: float -> zz: float -> float list
val structureTensorEigenMatrix:
  tensor: Image.Image<float list> -> Image.Image<float list>
val stack: images: Image.Image<'T> list -> Image.Image<'T> when 'T: equality
val extractSub:
  topLeft: uint list ->
    bottomRight: uint list -> img: Image.Image<'T> -> Image.Image<'T>
    when 'T: equality
val extractSlice:
  dir: uint -> i: int -> img: Image.Image<'T> -> Image.Image<'a>
    when 'T: equality and 'a: equality
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
  order: uint list -> img: Image.Image<'T> -> Image.Image<'a>
    when 'T: equality and 'a: equality
val FFTXY:
  image: Image.Image<'T> -> Image.Image<System.Numerics.Complex>
    when 'T: equality
val FFTXYFloat32:
  image: Image.Image<'T> -> Image.Image<Image.ComplexFloat32> when 'T: equality
val private dftLine:
  inverse: bool ->
    line: System.Numerics.Complex array -> System.Numerics.Complex array
val private directionalDftComplex2D:
  dir: uint32 ->
    inverse: bool ->
    input: System.Numerics.Complex array2d -> System.Numerics.Complex array2d
val private directionalDftComplex3D:
  dir: uint32 ->
    inverse: bool ->
    input: System.Numerics.Complex array3d -> System.Numerics.Complex array3d
val private toComplex64: value: Image.ComplexFloat32 -> System.Numerics.Complex
val private toComplex32: value: System.Numerics.Complex -> Image.ComplexFloat32
val private directionalDftComplexFloat322D:
  dir: uint32 ->
    inverse: bool ->
    input: Image.ComplexFloat32 array2d -> Image.ComplexFloat32 array2d
val private directionalDftComplexFloat323D:
  dir: uint32 ->
    inverse: bool ->
    input: Image.ComplexFloat32 array3d -> Image.ComplexFloat32 array3d
val directionalFFT:
  dir: uint -> image: Image.Image<'T> -> Image.Image<System.Numerics.Complex>
    when 'T: equality
val directionalFFTFloat32:
  dir: uint -> image: Image.Image<'T> -> Image.Image<Image.ComplexFloat32>
    when 'T: equality
val directionalFFTComplex:
  dir: uint ->
    inverse: bool ->
    image: Image.Image<System.Numerics.Complex> ->
    Image.Image<System.Numerics.Complex>
val directionalFFTComplexFloat32:
  dir: uint ->
    inverse: bool ->
    image: Image.Image<Image.ComplexFloat32> ->
    Image.Image<Image.ComplexFloat32>
val inverseFFTXY:
  image: Image.Image<System.Numerics.Complex> ->
    Image.Image<System.Numerics.Complex>
val inverseFFTXYFloat32:
  image: Image.Image<Image.ComplexFloat32> -> Image.Image<Image.ComplexFloat32>
val realPart: image: Image.Image<System.Numerics.Complex> -> Image.Image<float>
val realPartFloat32:
  image: Image.Image<Image.ComplexFloat32> -> Image.Image<float32>
val inverseFFTXYReal:
  image: Image.Image<System.Numerics.Complex> -> Image.Image<float>
val shiftFFT:
  image: Image.Image<System.Numerics.Complex> ->
    Image.Image<System.Numerics.Complex>
val shiftFFTFloat32:
  image: Image.Image<Image.ComplexFloat32> -> Image.Image<Image.ComplexFloat32>
