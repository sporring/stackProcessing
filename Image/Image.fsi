namespace FSharp
module Image
module internal InternalHelpers =
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
    val GetArray2DFromImage: itkImg: itk.simple.Image -> 'T array2d
    val GetArray3DFromImage: itkImg: itk.simple.Image -> 'T array3d
    val GetArray4DFromImage: itkImg: itk.simple.Image -> 'T array4d
    val Array4Diteri:
      action: (int -> int -> int -> int -> 'T -> unit) ->
        arr: 'T array4d -> unit
    val array2dZip: a: 'T array2d -> b: 'U array2d -> ('T * 'U) array2d
    val pixelIdToString: id: itk.simple.PixelIDValueEnum -> string
[<StructuredFormatDisplay ("{Display}")>]
type Image<'T when 'T: equality> =
    interface System.IComparable<Image<'T>>
    interface System.IEquatable<Image<'T>>
    new: sz: uint list * ?numberComp: uint -> Image<'T>
    static member (%) : i: uint32 * f2: Image<'S> -> Image<'S> when 'S: equality
    static member (%) : i: uint16 * f2: Image<'S> -> Image<'S> when 'S: equality
    static member (%) : i: uint8 * f2: Image<'S> -> Image<'S> when 'S: equality
    static member (%) : f1: Image<'S> * i: uint32 -> Image<'S> when 'S: equality
    static member (%) : f1: Image<'S> * i: uint16 -> Image<'S> when 'S: equality
    static member (%) : f1: Image<'S> * i: uint8 -> Image<'S> when 'S: equality
    static member
      (%) : f1: Image<'S> * f2: Image<'S> -> Image<'S> when 'S: equality
    static member (&&&) : i: int * f2: Image<int> -> Image<int>
    static member (&&&) : f1: Image<int32> * i: int -> Image<int32>
    static member
      (&&&) : f1: Image<'S> * f2: Image<'S> -> Image<'S> when 'S: equality
    static member
      ( * ) : f1: Image<'S> * f2: Image<'S> -> Image<'S> when 'S: equality
    static member
      (+) : f1: Image<'S> * f2: Image<'S> -> Image<'S> when 'S: equality
    static member
      (-) : f1: Image<'S> * f2: Image<'S> -> Image<'S> when 'S: equality
    static member
      (/) : f1: Image<'S> * f2: Image<'S> -> Image<'S> when 'S: equality
    static member (^^^) : i: int * f2: Image<int> -> Image<int>
    static member (^^^) : f1: Image<int> * i: int -> Image<int>
    static member
      (^^^) : f1: Image<'S> * f2: Image<'S> -> Image<'S> when 'S: equality
    static member (|||) : i: int * f2: Image<int> -> Image<int>
    static member (|||) : f1: Image<int> * i: int -> Image<int>
    static member
      (|||) : f1: Image<'S> * f2: Image<'S> -> Image<'S> when 'S: equality
    static member (~~~) : f: Image<'S> -> Image<'S> when 'S: equality
    static member Pow: i: float32 * f2: Image<'S> -> Image<'S> when 'S: equality
    static member Pow: i: uint64 * f2: Image<'S> -> Image<'S> when 'S: equality
    static member Pow: i: int64 * f2: Image<'S> -> Image<'S> when 'S: equality
    static member Pow: i: uint32 * f2: Image<'S> -> Image<'S> when 'S: equality
    static member Pow: i: int32 * f2: Image<'S> -> Image<'S> when 'S: equality
    static member Pow: i: uint16 * f2: Image<'S> -> Image<'S> when 'S: equality
    static member Pow: i: int16 * f2: Image<'S> -> Image<'S> when 'S: equality
    static member Pow: i: uint8 * f2: Image<'S> -> Image<'S> when 'S: equality
    static member Pow: i: int8 * f2: Image<'S> -> Image<'S> when 'S: equality
    static member Pow: i: float * f2: Image<'S> -> Image<'S> when 'S: equality
    static member Pow: f1: Image<'S> * i: float32 -> Image<'S> when 'S: equality
    static member Pow: f1: Image<'S> * i: int64 -> Image<'S> when 'S: equality
    static member Pow: f1: Image<'S> * i: uint64 -> Image<'S> when 'S: equality
    static member Pow: f1: Image<'S> * i: int32 -> Image<'S> when 'S: equality
    static member Pow: f1: Image<'S> * i: uint32 -> Image<'S> when 'S: equality
    static member Pow: f1: Image<'S> * i: int16 -> Image<'S> when 'S: equality
    static member Pow: f1: Image<'S> * i: uint16 -> Image<'S> when 'S: equality
    static member Pow: f1: Image<'S> * i: int8 -> Image<'S> when 'S: equality
    static member Pow: f1: Image<'S> * i: uint8 -> Image<'S> when 'S: equality
    static member Pow: f1: Image<'S> * i: float -> Image<'S> when 'S: equality
    static member
      Pow: f1: Image<'S> * f2: Image<'S> -> Image<'S> when 'S: equality
    static member eq: i: float * f1: Image<float> -> bool
    static member eq: f1: Image<float> * i: float -> bool
    static member eq: i: float32 * f1: Image<float32> -> bool
    static member eq: f1: Image<float32> * i: float32 -> bool
    static member eq: i: uint64 * f1: Image<uint64> -> bool
    static member eq: f1: Image<uint64> * i: uint64 -> bool
    static member eq: i: int64 * f1: Image<int64> -> bool
    static member eq: f1: Image<int64> * i: int64 -> bool
    static member eq: i: uint32 * f1: Image<uint32> -> bool
    static member eq: f1: Image<uint32> * i: uint32 -> bool
    static member eq: i: int32 * f1: Image<int32> -> bool
    static member eq: f1: Image<int32> * i: int32 -> bool
    static member eq: i: uint16 * f1: Image<uint16> -> bool
    static member eq: f1: Image<uint16> * i: uint16 -> bool
    static member eq: i: int16 * f1: Image<int16> -> bool
    static member eq: f1: Image<int16> * i: int16 -> bool
    static member eq: i: uint8 * f1: Image<uint8> -> bool
    static member eq: f1: Image<uint8> * i: uint8 -> bool
    static member eq: i: int8 * f1: Image<int8> -> bool
    static member eq: f1: Image<int8> * i: int8 -> bool
    static member eq: f1: Image<'S> * f2: Image<'S> -> bool when 'S: equality
    static member gt: i: float * f1: Image<float> -> bool
    static member gt: f1: Image<float> * i: float -> bool
    static member gt: i: float32 * f1: Image<float32> -> bool
    static member gt: f1: Image<float32> * i: float32 -> bool
    static member gt: i: uint64 * f1: Image<uint64> -> bool
    static member gt: f1: Image<uint64> * i: uint64 -> bool
    static member gt: i: int64 * f1: Image<int64> -> bool
    static member gt: f1: Image<int64> * i: int64 -> bool
    static member gt: i: uint32 * f1: Image<uint32> -> bool
    static member gt: f1: Image<uint32> * i: uint32 -> bool
    static member gt: i: int32 * f1: Image<int32> -> bool
    static member gt: f1: Image<int32> * i: int32 -> bool
    static member gt: i: uint16 * f1: Image<uint16> -> bool
    static member gt: f1: Image<uint16> * i: uint16 -> bool
    static member gt: i: int16 * f1: Image<int16> -> bool
    static member gt: f1: Image<int16> * i: int16 -> bool
    static member gt: i: uint8 * f1: Image<uint8> -> bool
    static member gt: f1: Image<uint8> * i: uint8 -> bool
    static member gt: i: int8 * f1: Image<int8> -> bool
    static member gt: f1: Image<int8> * i: int8 -> bool
    static member gt: f1: Image<'S> * f2: Image<'S> -> bool when 'S: equality
    static member gte: i: float * f1: Image<float> -> bool
    static member gte: f1: Image<float> * i: float -> bool
    static member gte: i: float32 * f1: Image<float32> -> bool
    static member gte: f1: Image<float32> * i: float32 -> bool
    static member gte: i: uint64 * f1: Image<uint64> -> bool
    static member gte: f1: Image<uint64> * i: uint64 -> bool
    static member gte: i: int64 * f1: Image<int64> -> bool
    static member gte: f1: Image<int64> * i: int64 -> bool
    static member gte: i: uint32 * f1: Image<uint32> -> bool
    static member gte: f1: Image<uint32> * i: uint32 -> bool
    static member gte: i: int32 * f1: Image<int32> -> bool
    static member gte: f1: Image<int32> * i: int32 -> bool
    static member gte: i: uint16 * f1: Image<uint16> -> bool
    static member gte: f1: Image<uint16> * i: uint16 -> bool
    static member gte: i: int16 * f1: Image<int16> -> bool
    static member gte: f1: Image<int16> * i: int16 -> bool
    static member gte: i: uint8 * f1: Image<uint8> -> bool
    static member gte: f1: Image<uint8> * i: uint8 -> bool
    static member gte: i: int8 * f1: Image<int8> -> bool
    static member gte: f1: Image<int8> * i: int8 -> bool
    static member gte: f1: Image<'S> * f2: Image<'S> -> bool when 'S: equality
    static member isEqual: i: float * f1: Image<float> -> Image<float>
    static member isEqual: f1: Image<float> * i: float -> Image<float>
    static member isEqual: i: float32 * f1: Image<float32> -> Image<float32>
    static member isEqual: f1: Image<float32> * i: float32 -> Image<float32>
    static member isEqual: i: uint64 * f1: Image<uint64> -> Image<uint64>
    static member isEqual: f1: Image<uint64> * i: uint64 -> Image<uint64>
    static member isEqual: i: int64 * f1: Image<int64> -> Image<int64>
    static member isEqual: f1: Image<int64> * i: int64 -> Image<int64>
    static member isEqual: i: uint32 * f1: Image<uint32> -> Image<uint32>
    static member isEqual: f1: Image<uint32> * i: uint32 -> Image<uint32>
    static member isEqual: i: int32 * f1: Image<int32> -> Image<int32>
    static member isEqual: f1: Image<int32> * i: int32 -> Image<int32>
    static member isEqual: i: uint16 * f1: Image<uint16> -> Image<uint16>
    static member isEqual: f1: Image<uint16> * i: uint16 -> Image<uint16>
    static member isEqual: i: int16 * f1: Image<int16> -> Image<int16>
    static member isEqual: f1: Image<int16> * i: int16 -> Image<int16>
    static member isEqual: i: uint8 * f1: Image<uint8> -> Image<uint8>
    static member isEqual: f1: Image<uint8> * i: uint8 -> Image<uint8>
    static member isEqual: i: int8 * f1: Image<int8> -> Image<int8>
    static member isEqual: f1: Image<int8> * i: int8 -> Image<int8>
    static member
      isEqual: f1: Image<'S> * f2: Image<'S> -> Image<'S> when 'S: equality
    static member isGreater: i: float * f1: Image<float> -> Image<float>
    static member isGreater: f1: Image<float> * i: float -> Image<float>
    static member isGreater: i: float32 * f1: Image<float32> -> Image<float32>
    static member isGreater: f1: Image<float32> * i: float32 -> Image<float32>
    static member isGreater: i: uint64 * f1: Image<uint64> -> Image<uint64>
    static member isGreater: f1: Image<uint64> * i: uint64 -> Image<uint64>
    static member isGreater: i: int64 * f1: Image<int64> -> Image<int64>
    static member isGreater: f1: Image<int64> * i: int64 -> Image<int64>
    static member isGreater: i: uint32 * f1: Image<uint32> -> Image<uint32>
    static member isGreater: f1: Image<uint32> * i: uint32 -> Image<uint32>
    static member isGreater: i: int32 * f1: Image<int32> -> Image<int32>
    static member isGreater: f1: Image<int32> * i: int32 -> Image<int32>
    static member isGreater: i: uint16 * f1: Image<uint16> -> Image<uint16>
    static member isGreater: f1: Image<uint16> * i: uint16 -> Image<uint16>
    static member isGreater: i: int16 * f1: Image<int16> -> Image<int16>
    static member isGreater: f1: Image<int16> * i: int16 -> Image<int16>
    static member isGreater: i: uint8 * f1: Image<uint8> -> Image<uint8>
    static member isGreater: f1: Image<uint8> * i: uint8 -> Image<uint8>
    static member isGreater: i: int8 * f1: Image<int8> -> Image<int8>
    static member isGreater: f1: Image<int8> * i: int8 -> Image<int8>
    static member
      isGreater: f1: Image<'S> * f2: Image<'S> -> Image<'S> when 'S: equality
    static member isGreaterEqual: i: float * f1: Image<float> -> Image<float>
    static member isGreaterEqual: f1: Image<float> * i: float -> Image<float>
    static member
      isGreaterEqual: i: float32 * f1: Image<float32> -> Image<float32>
    static member
      isGreaterEqual: f1: Image<float32> * i: float32 -> Image<float32>
    static member isGreaterEqual: i: uint64 * f1: Image<uint64> -> Image<uint64>
    static member isGreaterEqual: f1: Image<uint64> * i: uint64 -> Image<uint64>
    static member isGreaterEqual: i: int64 * f1: Image<int64> -> Image<int64>
    static member isGreaterEqual: f1: Image<int64> * i: int64 -> Image<int64>
    static member isGreaterEqual: i: uint32 * f1: Image<uint32> -> Image<uint32>
    static member isGreaterEqual: f1: Image<uint32> * i: uint32 -> Image<uint32>
    static member isGreaterEqual: i: int32 * f1: Image<int32> -> Image<int32>
    static member isGreaterEqual: f1: Image<int32> * i: int32 -> Image<int32>
    static member isGreaterEqual: i: uint16 * f1: Image<uint16> -> Image<uint16>
    static member isGreaterEqual: f1: Image<uint16> * i: uint16 -> Image<uint16>
    static member isGreaterEqual: i: int16 * f1: Image<int16> -> Image<int16>
    static member isGreaterEqual: f1: Image<int16> * i: int16 -> Image<int16>
    static member isGreaterEqual: i: uint8 * f1: Image<uint8> -> Image<uint8>
    static member isGreaterEqual: f1: Image<uint8> * i: uint8 -> Image<uint8>
    static member isGreaterEqual: i: int8 * f1: Image<int8> -> Image<int8>
    static member isGreaterEqual: f1: Image<int8> * i: int8 -> Image<int8>
    static member
      isGreaterEqual: f1: Image<'S> * f2: Image<'S> -> Image<'S>
                        when 'S: equality
    static member isLessThan: i: float * f1: Image<float> -> Image<float>
    static member isLessThan: f1: Image<float> * i: float -> Image<float>
    static member isLessThan: i: float32 * f1: Image<float32> -> Image<float32>
    static member isLessThan: f1: Image<float32> * i: float32 -> Image<float32>
    static member isLessThan: i: uint64 * f1: Image<uint64> -> Image<uint64>
    static member isLessThan: f1: Image<uint64> * i: uint64 -> Image<uint64>
    static member isLessThan: i: int64 * f1: Image<int64> -> Image<int64>
    static member isLessThan: f1: Image<int64> * i: int64 -> Image<int64>
    static member isLessThan: i: uint32 * f1: Image<uint32> -> Image<uint32>
    static member isLessThan: f1: Image<uint32> * i: uint32 -> Image<uint32>
    static member isLessThan: i: int32 * f1: Image<int32> -> Image<int32>
    static member isLessThan: f1: Image<int32> * i: int32 -> Image<int32>
    static member isLessThan: i: uint16 * f1: Image<uint16> -> Image<uint16>
    static member isLessThan: f1: Image<uint16> * i: uint16 -> Image<uint16>
    static member isLessThan: i: int16 * f1: Image<int16> -> Image<int16>
    static member isLessThan: f1: Image<int16> * i: int16 -> Image<int16>
    static member isLessThan: i: uint8 * f1: Image<uint8> -> Image<uint8>
    static member isLessThan: f1: Image<uint8> * i: uint8 -> Image<uint8>
    static member isLessThan: i: int8 * f1: Image<int8> -> Image<int8>
    static member isLessThan: f1: Image<int8> * i: int8 -> Image<int8>
    static member
      isLessThan: f1: Image<'S> * f2: Image<'S> -> Image<'S> when 'S: equality
    static member isLessThanEqual: i: float * f1: Image<float> -> Image<float>
    static member isLessThanEqual: f1: Image<float> * i: float -> Image<float>
    static member
      isLessThanEqual: i: float32 * f1: Image<float32> -> Image<float32>
    static member
      isLessThanEqual: f1: Image<float32> * i: float32 -> Image<float32>
    static member
      isLessThanEqual: i: uint64 * f1: Image<uint64> -> Image<uint64>
    static member
      isLessThanEqual: f1: Image<uint64> * i: uint64 -> Image<uint64>
    static member isLessThanEqual: i: int64 * f1: Image<int64> -> Image<int64>
    static member isLessThanEqual: f1: Image<int64> * i: int64 -> Image<int64>
    static member
      isLessThanEqual: i: uint32 * f1: Image<uint32> -> Image<uint32>
    static member
      isLessThanEqual: f1: Image<uint32> * i: uint32 -> Image<uint32>
    static member isLessThanEqual: i: int32 * f1: Image<int32> -> Image<int32>
    static member isLessThanEqual: f1: Image<int32> * i: int32 -> Image<int32>
    static member
      isLessThanEqual: i: uint16 * f1: Image<uint16> -> Image<uint16>
    static member
      isLessThanEqual: f1: Image<uint16> * i: uint16 -> Image<uint16>
    static member isLessThanEqual: i: int16 * f1: Image<int16> -> Image<int16>
    static member isLessThanEqual: f1: Image<int16> * i: int16 -> Image<int16>
    static member isLessThanEqual: i: uint8 * f1: Image<uint8> -> Image<uint8>
    static member isLessThanEqual: f1: Image<uint8> * i: uint8 -> Image<uint8>
    static member isLessThanEqual: i: int8 * f1: Image<int8> -> Image<int8>
    static member isLessThanEqual: f1: Image<int8> * i: int8 -> Image<int8>
    static member
      isLessThanEqual: f1: Image<'S> * f2: Image<'S> -> Image<'S>
                         when 'S: equality
    static member isNotEqual: i: float * f1: Image<float> -> Image<float>
    static member isNotEqual: f1: Image<float> * i: float -> Image<float>
    static member isNotEqual: i: float32 * f1: Image<float32> -> Image<float32>
    static member isNotEqual: f1: Image<float32> * i: float32 -> Image<float32>
    static member isNotEqual: i: uint64 * f1: Image<uint64> -> Image<uint64>
    static member isNotEqual: f1: Image<uint64> * i: uint64 -> Image<uint64>
    static member isNotEqual: i: int64 * f1: Image<int64> -> Image<int64>
    static member isNotEqual: f1: Image<int64> * i: int64 -> Image<int64>
    static member isNotEqual: i: uint32 * f1: Image<uint32> -> Image<uint32>
    static member isNotEqual: f1: Image<uint32> * i: uint32 -> Image<uint32>
    static member isNotEqual: i: int32 * f1: Image<int32> -> Image<int32>
    static member isNotEqual: f1: Image<int32> * i: int32 -> Image<int32>
    static member isNotEqual: i: uint16 * f1: Image<uint16> -> Image<uint16>
    static member isNotEqual: f1: Image<uint16> * i: uint16 -> Image<uint16>
    static member isNotEqual: i: int16 * f1: Image<int16> -> Image<int16>
    static member isNotEqual: f1: Image<int16> * i: int16 -> Image<int16>
    static member isNotEqual: i: uint8 * f1: Image<uint8> -> Image<uint8>
    static member isNotEqual: f1: Image<uint8> * i: uint8 -> Image<uint8>
    static member isNotEqual: i: int8 * f1: Image<int8> -> Image<int8>
    static member isNotEqual: f1: Image<int8> * i: int8 -> Image<int8>
    static member
      isNotEqual: f1: Image<'S> * f2: Image<'S> -> Image<'S> when 'S: equality
    static member lt: i: float * f1: Image<float> -> bool
    static member lt: f1: Image<float> * i: float -> bool
    static member lt: i: float32 * f1: Image<float32> -> bool
    static member lt: f1: Image<float32> * i: float32 -> bool
    static member lt: i: uint64 * f1: Image<uint64> -> bool
    static member lt: f1: Image<uint64> * i: uint64 -> bool
    static member lt: i: int64 * f1: Image<int64> -> bool
    static member lt: f1: Image<int64> * i: int64 -> bool
    static member lt: i: uint32 * f1: Image<uint32> -> bool
    static member lt: f1: Image<uint32> * i: uint32 -> bool
    static member lt: i: int32 * f1: Image<int32> -> bool
    static member lt: f1: Image<int32> * i: int32 -> bool
    static member lt: i: uint16 * f1: Image<uint16> -> bool
    static member lt: f1: Image<uint16> * i: uint16 -> bool
    static member lt: i: int16 * f1: Image<int16> -> bool
    static member lt: f1: Image<int16> * i: int16 -> bool
    static member lt: i: uint8 * f1: Image<uint8> -> bool
    static member lt: f1: Image<uint8> * i: uint8 -> bool
    static member lt: i: int8 * f1: Image<int8> -> bool
    static member lt: f1: Image<int8> * i: int8 -> bool
    static member lt: f1: Image<'S> * f2: Image<'S> -> bool when 'S: equality
    static member lte: i: float * f1: Image<float> -> bool
    static member lte: f1: Image<float> * i: float -> bool
    static member lte: i: float32 * f1: Image<float32> -> bool
    static member lte: f1: Image<float32> * i: float32 -> bool
    static member lte: i: uint64 * f1: Image<uint64> -> bool
    static member lte: f1: Image<uint64> * i: uint64 -> bool
    static member lte: i: int64 * f1: Image<int64> -> bool
    static member lte: f1: Image<int64> * i: int64 -> bool
    static member lte: i: uint32 * f1: Image<uint32> -> bool
    static member lte: f1: Image<uint32> * i: uint32 -> bool
    static member lte: i: int32 * f1: Image<int32> -> bool
    static member lte: f1: Image<int32> * i: int32 -> bool
    static member lte: i: uint16 * f1: Image<uint16> -> bool
    static member lte: f1: Image<uint16> * i: uint16 -> bool
    static member lte: i: int16 * f1: Image<int16> -> bool
    static member lte: f1: Image<int16> * i: int16 -> bool
    static member lte: i: uint8 * f1: Image<uint8> -> bool
    static member lte: f1: Image<uint8> * i: uint8 -> bool
    static member lte: i: int8 * f1: Image<int8> -> bool
    static member lte: f1: Image<int8> * i: int8 -> bool
    static member lte: f1: Image<'S> * f2: Image<'S> -> bool when 'S: equality
    static member neq: i: float * f1: Image<float> -> bool
    static member neq: f1: Image<float> * i: float -> bool
    static member neq: i: float32 * f1: Image<float32> -> bool
    static member neq: f1: Image<float32> * i: float32 -> bool
    static member neq: i: uint64 * f1: Image<uint64> -> bool
    static member neq: f1: Image<uint64> * i: uint64 -> bool
    static member neq: i: int64 * f1: Image<int64> -> bool
    static member neq: f1: Image<int64> * i: int64 -> bool
    static member neq: i: uint32 * f1: Image<uint32> -> bool
    static member neq: f1: Image<uint32> * i: uint32 -> bool
    static member neq: i: int32 * f1: Image<int32> -> bool
    static member neq: f1: Image<int32> * i: int32 -> bool
    static member neq: i: uint16 * f1: Image<uint16> -> bool
    static member neq: f1: Image<uint16> * i: uint16 -> bool
    static member neq: i: int16 * f1: Image<int16> -> bool
    static member neq: f1: Image<int16> * i: int16 -> bool
    static member neq: i: uint8 * f1: Image<uint8> -> bool
    static member neq: f1: Image<uint8> * i: uint8 -> bool
    static member neq: i: int8 * f1: Image<int8> -> bool
    static member neq: f1: Image<int8> * i: int8 -> bool
    static member neq: f1: Image<'S> * f2: Image<'S> -> bool when 'S: equality
    static member ofArray: arr: 'T array -> Image<'T>
    static member ofArray2D: arr: 'T array2d -> Image<'T>
    static member ofArray3D: arr: 'T array3d -> Image<'T>
    static member ofArray4D: arr: 'T array4d -> Image<'T>
    static member ofFile: filename: string -> Image<'T>
    static member
      ofImageList: images: Image<'S> list -> Image<'S list> when 'S: equality
    static member ofSimpleITK: itkImg: itk.simple.Image -> Image<'T>
    static member sum: img: Image<'T> -> 'T
    override Equals: obj: obj -> bool
    member Get: coords: uint list -> 'T
    member GetDepth: unit -> uint32
    member GetDimensions: unit -> uint32
    override GetHashCode: unit -> int
    member GetHeight: unit -> uint32
    member GetNumberOfComponentsPerPixel: unit -> uint32
    member GetSize: unit -> uint list
    member GetWidth: unit -> uint32
    member Set: coords: uint list * value: 'T -> unit
    member private SetImg: itkImg: itk.simple.Image -> unit
    override ToString: unit -> string
    member cast<'T when 'T: equality> : unit -> Image<'T>
    member castFloatToUInt8: unit -> Image<uint8>
    member castUInt8ToFloat: unit -> Image<float>
    member forAll: unit -> bool
    member memoryEstimate: unit -> uint
    member sum: unit -> 'T
    member toArray2D: unit -> 'T array2d
    member toArray3D: unit -> 'T array3d
    member toArray4D: unit -> 'T array4d
    member toFile: filename: string * ?format: string -> unit
    member toImageList: unit -> Image<'S> list when 'S: equality
    member toSimpleITK: unit -> itk.simple.Image
    member Display: string
    member Image: itk.simple.Image
    member Item: i0: int -> 'T with get
    member Item: i0: int -> 'T with set
    member Item: i0: int * i1: int -> 'T with get
    member Item: i0: int * i1: int -> 'T with set
    member Item: i0: int * i1: int * i2: int -> 'T with get
    member Item: i0: int * i1: int * i2: int -> 'T with set
    member Item: i0: int * i1: int * i2: int * i3: int -> 'T with get
    member Item: i0: int * i1: int * i2: int * i3: int -> 'T with set
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
  createFilter: (unit -> 'Filter) ->
    setup: ('Filter -> unit) ->
    invoke: ('Filter -> itk.simple.Image -> itk.simple.Image) ->
    img: Image.Image<'T> -> Image.Image<'T>
    when 'Filter :> System.IDisposable and 'T: equality
val inline makeUnaryImageOperator:
  createFilter: (unit -> 'a) ->
    invoke: ('a -> itk.simple.Image -> itk.simple.Image) ->
    (Image.Image<'b> -> Image.Image<'b>)
    when 'a :> System.IDisposable and 'b: equality
val inline makeBinaryImageOperatorWith:
  createFilter: (unit -> 'Filter) ->
    setup: ('Filter -> unit) ->
    invoke: ('Filter -> itk.simple.Image -> itk.simple.Image -> itk.simple.Image) ->
    a: Image.Image<'T> -> b: Image.Image<'T> -> Image.Image<'T>
    when 'Filter :> System.IDisposable and 'T: equality
val inline makeBinaryImageOperator:
  createFilter: (unit -> 'a) ->
    invoke: ('a -> itk.simple.Image -> itk.simple.Image -> itk.simple.Image) ->
    (Image.Image<'b> -> Image.Image<'b> -> Image.Image<'b>)
    when 'a :> System.IDisposable and 'b: equality
val inline absImage: img: Image.Image<'T> -> Image.Image<'T> when 'T: equality
val inline logImage: img: Image.Image<'T> -> Image.Image<'T> when 'T: equality
val inline log10Image: img: Image.Image<'T> -> Image.Image<'T> when 'T: equality
val inline expImage: img: Image.Image<'T> -> Image.Image<'T> when 'T: equality
val inline sqrtImage: img: Image.Image<'T> -> Image.Image<'T> when 'T: equality
val inline squareImage:
  img: Image.Image<'T> -> Image.Image<'T> when 'T: equality
val inline sinImage: img: Image.Image<'T> -> Image.Image<'T> when 'T: equality
val inline cosImage: img: Image.Image<'T> -> Image.Image<'T> when 'T: equality
val inline tanImage: img: Image.Image<'T> -> Image.Image<'T> when 'T: equality
val inline asinImage: img: Image.Image<'T> -> Image.Image<'T> when 'T: equality
val inline acosImage: img: Image.Image<'T> -> Image.Image<'T> when 'T: equality
val inline atanImage: img: Image.Image<'T> -> Image.Image<'T> when 'T: equality
val inline roundImage: img: Image.Image<'T> -> Image.Image<'T> when 'T: equality
type BoundaryCondition =
    | ZeroPad
    | PerodicPad
    | ZeroFluxNeumannPad
type OutputRegionMode =
    | Valid
    | Same
val convolve:
  boundaryCondition: BoundaryCondition option ->
    (Image.Image<'T> -> Image.Image<'T> -> Image.Image<'T>) when 'T: equality
val conv:
  img: Image.Image<'T> -> ker: Image.Image<'T> -> Image.Image<'T>
    when 'T: equality
val private stensil: order: uint32 -> float list
val finiteDiffFilter1D: order: uint -> Image.Image<float>
val finiteDiffFilter2D: direction: uint -> order: uint -> Image.Image<float>
val finiteDiffFilter3D: direction: uint -> order: uint -> Image.Image<float>
val finiteDiffFilter4D: direction: uint -> order: uint -> Image.Image<float>
/// Gaussian kernel convolution
val gauss:
  dim: uint -> sigma: float -> kernelSize: uint option -> Image.Image<'T>
    when 'T: equality
val discreteGaussian:
  sigma: float ->
    kernelSize: uint option ->
    boundaryCondition: BoundaryCondition option ->
    (Image.Image<'T> -> Image.Image<'T>) when 'T: equality
val discreteGaussian2D:
  sigma: float ->
    kernelSize: uint option ->
    boundaryCondition: BoundaryCondition option ->
    (Image.Image<'T> -> Image.Image<'T>) when 'T: equality
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
val otsuThreshold: img: Image.Image<'T> -> Image.Image<'T> when 'T: equality
/// Otsu multiple thresholds (returns a label map)
val otsuMultiThreshold:
  numThresholds: byte -> (Image.Image<'T> -> Image.Image<'T>) when 'T: equality
/// Moments-based threshold
val momentsThreshold: img: Image.Image<'T> -> Image.Image<'T> when 'T: equality
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
  lower: float -> upper: float -> (Image.Image<'T> -> Image.Image<'T>)
    when 'T: equality
val stack: images: Image.Image<'T> list -> Image.Image<'T> when 'T: equality
val extractSub:
  topLeft: uint list ->
    bottomRight: uint list -> img: Image.Image<'T> -> Image.Image<'T>
    when 'T: equality
val extractSlice:
  z: uint -> img: Image.Image<'T> -> Image.Image<'T> when 'T: equality
val unstack: vol: Image.Image<'T> -> Image.Image<'T> list when 'T: equality
type FileInfo =
    {
      dimensions: uint
      size: uint64 list
      componentType: string
      numberOfComponents: uint
    }
val getFileInfo: filename: string -> FileInfo
