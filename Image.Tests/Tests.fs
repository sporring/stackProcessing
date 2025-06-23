module Tests

open Expecto
open Image 
open Image.InternalHelpers
open itk.simple

[<Tests>]
let ToVectorTests =
  testList "toVector Tests" [
    testCase "toVectorUInt8" <| fun _ ->
      let value = [ 0uy..9uy ] |> List.randomSample 3
      let result = toVectorUInt8(value)
      let tp = result.GetType()
      Expect.equal tp.Name "VectorUInt8" "Expected VectorUInt8"
      Expect.equal (result |> List.ofSeq) value $"Expected {value}"
          
    testCase "toVectorInt8" <| fun _ ->
      let value = [ 0y..9y ] |> List.randomSample 3
      let result = toVectorInt8(value)
      let tp = result.GetType()
      Expect.equal tp.Name "VectorInt8" "Expected VectorInt8"
      Expect.equal (result |> List.ofSeq) value $"Expected {value}"
    
    testCase "toVectorUInt16" <| fun _ ->
      let value = [ 0us..9us ] |> List.randomSample 3
      let result = toVectorUInt16(value)
      let tp = result.GetType()
      Expect.equal tp.Name "VectorUInt16" "Expected VectorUInt16"
      Expect.equal (result |> List.ofSeq) value $"Expected {value}"
    
    testCase "toVectorInt16" <| fun _ ->
      let value = [ 0s..9s ] |> List.randomSample 3
      let result = toVectorInt16(value)
      let tp = result.GetType()
      Expect.equal tp.Name "VectorInt16" "Expected VectorInt16"
      Expect.equal (result |> List.ofSeq) value $"Expected {value}"
    
    testCase "toVectorUInt32" <| fun _ ->
      let value = [ 0u..9u ] |> List.randomSample 3
      let result = toVectorUInt32(value)
      let tp = result.GetType()
      Expect.equal tp.Name "VectorUInt32" "Expected VectorUInt32"
      Expect.equal (result |> List.ofSeq) value $"Expected {value}"
    
    testCase "toVectorInt32" <| fun _ ->
      let value = [ 0..9 ] |> List.randomSample 3
      let result = toVectorInt32(value)
      let tp = result.GetType()
      Expect.equal tp.Name "VectorInt32" "Expected VectorInt32"
      Expect.equal (result |> List.ofSeq) value $"Expected {value}"
    
    testCase "toVectorUInt64" <| fun _ ->
      let value = [ 0uL..9uL ] |> List.randomSample 3
      let result = toVectorUInt64(value)
      let tp = result.GetType()
      Expect.equal tp.Name "VectorUInt64" "Expected VectorUInt64"
      Expect.equal (result |> List.ofSeq) value $"Expected {value}"
    
    testCase "toVectorInt64" <| fun _ ->
      let value = [ 0L..9L ] |> List.randomSample 3
      let result = toVectorInt64(value)
      let tp = result.GetType()
      Expect.equal tp.Name "VectorInt64" "Expected VectorInt64"
      Expect.equal (result |> List.ofSeq) value $"Expected {value}"
    
    testCase "toVectorFloat32" <| fun _ ->
      let value = List.init 10 (fun i -> float32 i) |> List.randomSample 3
      let result = toVectorFloat32(value)
      let tp = result.GetType()
      Expect.equal tp.Name "VectorFloat" "Expected VectorFloat"
      Expect.equal (result |> List.ofSeq) value $"Expected {value}"
    
    testCase "toVectorFloat64" <| fun _ ->
      let value = List.init 10 (fun i -> float i) |> List.randomSample 3
      let result = toVectorFloat64(value)
      let tp = result.GetType()
      Expect.equal tp.Name "VectorDouble" "Expected VectorDouble"
      Expect.equal (result |> List.ofSeq) value $"Expected {value}"
  ]

[<Tests>]
let FromVectorTests =
  testList "FromVector Tests" [
    testCase "fromVectorUInt8" <| fun _ ->
      let value = [ 0uy..9uy ] |> List.randomSample 3
      let result = value |> toVectorUInt8 |> fromVectorUInt8
      Expect.equal result value $"Got {result} expected{value}"
          
    testCase "fromVectorInt8" <| fun _ ->
      let value = [ 0y..9y ] |> List.randomSample 3
      let result = value |> toVectorInt8 |> fromVectorInt8
      Expect.equal result value $"Got {result} expected{value}"
    
    testCase "fromVectorUInt16" <| fun _ ->
      let value = [ 0us..9us ] |> List.randomSample 3
      let result = value |> toVectorUInt16 |> fromVectorUInt16
      Expect.equal result value $"Got {result} expected{value}"
    
    testCase "fromVectorInt16" <| fun _ ->
      let value = [ 0s..9s ] |> List.randomSample 3
      let result = value |> toVectorInt16 |> fromVectorInt16
      Expect.equal result value $"Got {result} expected{value}"
    
    testCase "fromVectorUInt32" <| fun _ ->
      let value = [ 0u..9u ] |> List.randomSample 3
      let result = value |> toVectorUInt32 |> fromVectorUInt32
      Expect.equal result value $"Got {result} expected{value}"
    
    testCase "fromVectorInt32" <| fun _ ->
      let value = [ 0..9 ] |> List.randomSample 3
      let result = value |> toVectorInt32 |> fromVectorInt32
      Expect.equal result value $"Got {result} expected{value}"
    
    testCase "fromVectorUInt64" <| fun _ ->
      let value = [ 0uL..9uL ] |> List.randomSample 3
      let result = value |> toVectorUInt64 |> fromVectorUInt64
      Expect.equal result value $"Got {result} expected{value}"
    
    testCase "fromVectorInt64" <| fun _ ->
      let value = [ 0L..9L ] |> List.randomSample 3
      let result = value |> toVectorInt64 |> fromVectorInt64
      Expect.equal result value $"Got {result} expected{value}"
    
    testCase "fromVectorFloat32" <| fun _ ->
      let value = List.init 10 (fun i -> float32 i) |> List.randomSample 3
      let result = value |> toVectorFloat32 |> fromVectorFloat32
      Expect.equal result value $"Got {result} expected{value}"
    
    testCase "fromVectorFloat64" <| fun _ ->
      let value = List.init 10 (fun i -> float i) |> List.randomSample 3
      let result = value |> toVectorFloat64 |> fromVectorFloat64
      Expect.equal result value $"Got {result} expected{value}"
  ]

[<Tests>]
let FromTypeTests =
  testList "FromType Tests" [
    testCase "uint8" <| fun _ ->
      let result = fromType<uint8>
      Expect.equal result itk.simple.PixelIDValueEnum.sitkUInt8 $"Expected {itk.simple.PixelIDValueEnum.sitkUInt8}"

    testCase "int8" <| fun _ ->
      let result = fromType<int8>
      Expect.equal result itk.simple.PixelIDValueEnum.sitkInt8 $"Expected {itk.simple.PixelIDValueEnum.sitkInt8}"

    testCase "uint16" <| fun _ ->
      let result = fromType<uint16>
      Expect.equal result itk.simple.PixelIDValueEnum.sitkUInt16 $"Expected {itk.simple.PixelIDValueEnum.sitkUInt16}"

    testCase "int16" <| fun _ ->
      let result = fromType<int16>
      Expect.equal result itk.simple.PixelIDValueEnum.sitkInt16 $"Expected {itk.simple.PixelIDValueEnum.sitkInt16}"

    testCase "uint32" <| fun _ ->
      let result = fromType<uint32>
      Expect.equal result itk.simple.PixelIDValueEnum.sitkUInt32 $"Expected {itk.simple.PixelIDValueEnum.sitkUInt32}"

    testCase "int32" <| fun _ ->
      let result = fromType<int32>
      Expect.equal result itk.simple.PixelIDValueEnum.sitkInt32 $"Expected {itk.simple.PixelIDValueEnum.sitkInt32}"

    testCase "uint64" <| fun _ ->
      let result = fromType<uint64>
      Expect.equal result itk.simple.PixelIDValueEnum.sitkUInt64 $"Expected {itk.simple.PixelIDValueEnum.sitkUInt64}"

    testCase "int64" <| fun _ ->
      let result = fromType<int64>
      Expect.equal result itk.simple.PixelIDValueEnum.sitkInt64 $"Expected {itk.simple.PixelIDValueEnum.sitkInt64}"

    testCase "float32" <| fun _ ->
      let result = fromType<float32>
      Expect.equal result itk.simple.PixelIDValueEnum.sitkFloat32 $"Expected {itk.simple.PixelIDValueEnum.sitkFloat32}"

    testCase "float" <| fun _ ->
      let result = fromType<float>
      Expect.equal result itk.simple.PixelIDValueEnum.sitkFloat64 $"Expected {itk.simple.PixelIDValueEnum.sitkFloat64}"

    testCase "System.Numerics.Complex" <| fun _ ->
      let result = fromType<System.Numerics.Complex>
      Expect.equal result itk.simple.PixelIDValueEnum.sitkVectorFloat64 $"Expected {itk.simple.PixelIDValueEnum.sitkVectorFloat64}"

    testCase "uint8 list" <| fun _ ->
      let result = fromType<uint8 list>
      Expect.equal result itk.simple.PixelIDValueEnum.sitkVectorUInt8 $"Expected {itk.simple.PixelIDValueEnum.sitkVectorUInt8}"

    testCase "int8 list" <| fun _ ->
      let result = fromType<int8 list>
      Expect.equal result itk.simple.PixelIDValueEnum.sitkVectorInt8 $"Expected {itk.simple.PixelIDValueEnum.sitkVectorInt8}"

    testCase "uint16 list" <| fun _ ->
      let result = fromType<uint16 list>
      Expect.equal result itk.simple.PixelIDValueEnum.sitkVectorUInt16 $"Expected {itk.simple.PixelIDValueEnum.sitkVectorUInt16}"

    testCase "int16 list" <| fun _ ->
      let result = fromType<int16 list>
      Expect.equal result itk.simple.PixelIDValueEnum.sitkVectorInt16 $"Expected {itk.simple.PixelIDValueEnum.sitkVectorInt16}"

    testCase "uint32 list" <| fun _ ->
      let result = fromType<uint32 list>
      Expect.equal result itk.simple.PixelIDValueEnum.sitkVectorUInt32 $"Expected {itk.simple.PixelIDValueEnum.sitkVectorUInt32}"

    testCase "int32 list" <| fun _ ->
      let result = fromType<int32 list>
      Expect.equal result itk.simple.PixelIDValueEnum.sitkVectorInt32 $"Expected {itk.simple.PixelIDValueEnum.sitkVectorInt32}"

    testCase "uint64 list" <| fun _ ->
      let result = fromType<uint64 list>
      Expect.equal result itk.simple.PixelIDValueEnum.sitkVectorUInt64 $"Expected {itk.simple.PixelIDValueEnum.sitkVectorUInt64}"

    testCase "int64 list" <| fun _ ->
      let result = fromType<int64 list>
      Expect.equal result itk.simple.PixelIDValueEnum.sitkVectorInt64 $"Expected {itk.simple.PixelIDValueEnum.sitkVectorInt64}"

    testCase "float32 list" <| fun _ ->
      let result = fromType<float32 list>
      Expect.equal result itk.simple.PixelIDValueEnum.sitkVectorFloat32 $"Expected {itk.simple.PixelIDValueEnum.sitkVectorFloat32}"

    testCase "float list" <| fun _ ->
      let result = fromType<float list>
      Expect.equal result itk.simple.PixelIDValueEnum.sitkVectorFloat64 $"Expected {itk.simple.PixelIDValueEnum.sitkVectorFloat64}"
  ]

[<Tests>]
let ofCastItkTests =
  let sz = [3u;2u] |> toVectorUInt32
  let imgInt32 = new itk.simple.Image(sz, fromType<int32>)
  let imgFloat64 = new itk.simple.Image(sz, fromType<float>)
  testList "FromofCastItk Tests" [
    testCase "int32->uint8" <| fun _ ->
      let cast = ofCastItk<uint8> imgInt32
      let result = cast.GetPixelIDTypeAsString()
      let expected = "8-bit unsigned integer"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "int32->int8" <| fun _ ->
      let cast = ofCastItk<int8> imgInt32
      let result = cast.GetPixelIDTypeAsString()
      let expected = "8-bit signed integer"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "int32->uint16" <| fun _ ->
      let cast = ofCastItk<uint16> imgInt32
      let result = cast.GetPixelIDTypeAsString()
      let expected = "16-bit unsigned integer"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "int32->int16" <| fun _ ->
      let cast = ofCastItk<int16> imgInt32
      let result = cast.GetPixelIDTypeAsString()
      let expected = "16-bit signed integer"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "int32->uint32" <| fun _ ->
      let cast = ofCastItk<uint32> imgInt32
      let result = cast.GetPixelIDTypeAsString()
      let expected = "32-bit unsigned integer"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "int32->int32" <| fun _ ->
      let cast = ofCastItk<int32> imgInt32
      let result = cast.GetPixelIDTypeAsString()
      let expected = "32-bit signed integer"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "int32->uint64" <| fun _ ->
      let cast = ofCastItk<uint64> imgInt32
      let result = cast.GetPixelIDTypeAsString()
      let expected = "64-bit unsigned integer"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "int32->int64" <| fun _ ->
      let cast = ofCastItk<int64> imgInt32
      let result = cast.GetPixelIDTypeAsString()
      let expected = "64-bit signed integer"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "int32->float32" <| fun _ ->
      let cast = ofCastItk<float32> imgInt32
      let result = cast.GetPixelIDTypeAsString()
      let expected = "32-bit float"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "int32->float64" <| fun _ ->
      let cast = ofCastItk<float> imgInt32
      let result = cast.GetPixelIDTypeAsString()
      let expected = "64-bit float"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "int32->System.Numerics.Complex" <| fun _ ->
      let cast = ofCastItk<System.Numerics.Complex> imgInt32
      let result = cast.GetPixelIDTypeAsString()
      let expected = "vector of 64-bit float"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "int32->uint8 list" <| fun _ ->
      let cast = ofCastItk<uint8 list> imgInt32
      let result = cast.GetPixelIDTypeAsString()
      let expected = "vector of 8-bit unsigned integer"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "int32->int8 list" <| fun _ ->
      let cast = ofCastItk<int8 list> imgInt32
      let result = cast.GetPixelIDTypeAsString()
      let expected = "vector of 8-bit signed integer"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "int32->uint16 list" <| fun _ ->
      let cast = ofCastItk<uint16 list> imgInt32
      let result = cast.GetPixelIDTypeAsString()
      let expected = "vector of 16-bit unsigned integer"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "int32->int16 list" <| fun _ ->
      let cast = ofCastItk<int16 list> imgInt32
      let result = cast.GetPixelIDTypeAsString()
      let expected = "vector of 16-bit signed integer"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "int32->uint32 list" <| fun _ ->
      let cast = ofCastItk<uint32 list> imgInt32
      let result = cast.GetPixelIDTypeAsString()
      let expected = "vector of 32-bit unsigned integer"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "int32->int32 list" <| fun _ ->
      let cast = ofCastItk<int32 list> imgInt32
      let result = cast.GetPixelIDTypeAsString()
      let expected = "vector of 32-bit signed integer"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "int32->uint64 list" <| fun _ ->
      let cast = ofCastItk<uint64 list> imgInt32
      let result = cast.GetPixelIDTypeAsString()
      let expected = "vector of 64-bit unsigned integer"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "int32->int64 list" <| fun _ ->
      let cast = ofCastItk<int64 list> imgInt32
      let result = cast.GetPixelIDTypeAsString()
      let expected = "vector of 64-bit signed integer"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "int32->float32 list" <| fun _ ->
      let cast = ofCastItk<float32 list> imgInt32
      let result = cast.GetPixelIDTypeAsString()
      let expected = "vector of 32-bit float"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "int32->float64 list" <| fun _ ->
      let cast = ofCastItk<float list> imgInt32
      let result = cast.GetPixelIDTypeAsString()
      let expected = "vector of 64-bit float"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "float->uint8" <| fun _ ->
      let cast = ofCastItk<uint8> imgFloat64
      let result = cast.GetPixelIDTypeAsString()
      let expected = "8-bit unsigned integer"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "float->int8" <| fun _ ->
      let cast = ofCastItk<int8> imgFloat64
      let result = cast.GetPixelIDTypeAsString()
      let expected = "8-bit signed integer"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "float->uint16" <| fun _ ->
      let cast = ofCastItk<uint16> imgFloat64
      let result = cast.GetPixelIDTypeAsString()
      let expected = "16-bit unsigned integer"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "float->int16" <| fun _ ->
      let cast = ofCastItk<int16> imgFloat64
      let result = cast.GetPixelIDTypeAsString()
      let expected = "16-bit signed integer"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "float->uint32" <| fun _ ->
      let cast = ofCastItk<uint32> imgFloat64
      let result = cast.GetPixelIDTypeAsString()
      let expected = "32-bit unsigned integer"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "float->int32" <| fun _ ->
      let cast = ofCastItk<int32> imgFloat64
      let result = cast.GetPixelIDTypeAsString()
      let expected = "32-bit signed integer"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "float->uint64" <| fun _ ->
      let cast = ofCastItk<uint64> imgFloat64
      let result = cast.GetPixelIDTypeAsString()
      let expected = "64-bit unsigned integer"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "float->int64" <| fun _ ->
      let cast = ofCastItk<int64> imgFloat64
      let result = cast.GetPixelIDTypeAsString()
      let expected = "64-bit signed integer"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "float->float32" <| fun _ ->
      let cast = ofCastItk<float32> imgFloat64
      let result = cast.GetPixelIDTypeAsString()
      let expected = "32-bit float"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "float->float64" <| fun _ ->
      let cast = ofCastItk<float> imgFloat64
      let result = cast.GetPixelIDTypeAsString()
      let expected = "64-bit float"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "float->System.Numerics.Complex" <| fun _ ->
      let cast = ofCastItk<System.Numerics.Complex> imgFloat64
      let result = cast.GetPixelIDTypeAsString()
      let expected = "vector of 64-bit float"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "float->uint8 list" <| fun _ ->
      let cast = ofCastItk<uint8 list> imgFloat64
      let result = cast.GetPixelIDTypeAsString()
      let expected = "vector of 8-bit unsigned integer"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "float->int8 list" <| fun _ ->
      let cast = ofCastItk<int8 list> imgFloat64
      let result = cast.GetPixelIDTypeAsString()
      let expected = "vector of 8-bit signed integer"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "float->uint16 list" <| fun _ ->
      let cast = ofCastItk<uint16 list> imgFloat64
      let result = cast.GetPixelIDTypeAsString()
      let expected = "vector of 16-bit unsigned integer"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "float->int16 list" <| fun _ ->
      let cast = ofCastItk<int16 list> imgFloat64
      let result = cast.GetPixelIDTypeAsString()
      let expected = "vector of 16-bit signed integer"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "float->uint32 list" <| fun _ ->
      let cast = ofCastItk<uint32 list> imgFloat64
      let result = cast.GetPixelIDTypeAsString()
      let expected = "vector of 32-bit unsigned integer"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "float->int32 list" <| fun _ ->
      let cast = ofCastItk<int32 list> imgFloat64
      let result = cast.GetPixelIDTypeAsString()
      let expected = "vector of 32-bit signed integer"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "float->uint64 list" <| fun _ ->
      let cast = ofCastItk<uint64 list> imgFloat64
      let result = cast.GetPixelIDTypeAsString()
      let expected = "vector of 64-bit unsigned integer"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "float->int64 list" <| fun _ ->
      let cast = ofCastItk<int64 list> imgFloat64
      let result = cast.GetPixelIDTypeAsString()
      let expected = "vector of 64-bit signed integer"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "float->float32 list" <| fun _ ->
      let cast = ofCastItk<float32 list> imgFloat64
      let result = cast.GetPixelIDTypeAsString()
      let expected = "vector of 32-bit float"
      Expect.equal result expected $"Got {result} expected {expected}"

    testCase "float->float64 list" <| fun _ ->
      let cast = ofCastItk<float list> imgFloat64
      let result = cast.GetPixelIDTypeAsString()
      let expected = "vector of 64-bit float"
      Expect.equal result expected $"Got {result} expected {expected}"
  ]

[<Tests>]
let creationTests =
  testList "creation Tests" [
    testCase "constructor scalar 2d image" <| fun _ ->
      let w,h,d = 3u,10u,1u;
      let img = Image<int8>([w;h])
      let sz = img.GetSize()
      let dpt = img.GetDepth() // this returns 0. How can that make sense?
      let dim = img.GetDimension()
      let height = img.GetHeight()
      let width = img.GetWidth()
      let comp = img.GetNumberOfComponentsPerPixel()
      Expect.equal sz [w;h] $"Got {sz} expected [{w};{h}]"
      Expect.equal dpt d $"Got {dpt} expected {d}"
      Expect.equal dim 2u $"Got {dim} expected 2u"
      Expect.equal height h $"Got {height} expected {h}"
      Expect.equal width w $"Got {width} expected {w}"
      Expect.equal comp 1u $"Got {comp} expected 1u"

    testCase "constructor scalar 2d image with empty 3rd dim" <| fun _ ->
      let w,h,d = 3u,10u,1u;
      let img = Image<int8>([w;h;d])
      let sz = img.GetSize()
      let dpt = img.GetDepth() // this returns 0. How can that make sense?
      let dim = img.GetDimension()
      let height = img.GetHeight()
      let width = img.GetWidth()
      let comp = img.GetNumberOfComponentsPerPixel()
      Expect.equal sz [w;h;d] $"Got {sz} expected [{w};{h};{d}]"
      Expect.equal dpt d $"Got {dpt} expected {d}"
      Expect.equal dim 3u $"Got {dim} expected 3u"
      Expect.equal height h $"Got {height} expected {h}"
      Expect.equal width w $"Got {width} expected {w}"
      Expect.equal comp 1u $"Got {comp} expected 1u"

    testCase "constructor scalar 3d image" <| fun _ ->
      let w,h,d = 3u,10u,5u;
      let img = Image<int8>([w;h;d])
      let sz = img.GetSize()
      let dpt = img.GetDepth() // this returns 0. How can that make sense?
      let dim = img.GetDimension()
      let height = img.GetHeight()
      let width = img.GetWidth()
      let comp = img.GetNumberOfComponentsPerPixel()
      Expect.equal sz [w;h;d] $"Got {sz} expected [{w};{h};{d}]"
      Expect.equal dpt d $"Got {dpt} expected {d}"
      Expect.equal dim 3u $"Got {dim} expected 3u"
      Expect.equal height h $"Got {height} expected {h}"
      Expect.equal width w $"Got {width} expected {w}"
      Expect.equal comp 1u $"Got {comp} expected 1u"

    testCase "constructor n-vector 2d image" <| fun _ ->
      let w,h,d,n = 3u,10u,1u,3u;
      let img = Image<float list>([w;h],n)
      let sz = img.GetSize()
      let dpt = img.GetDepth() // this returns 0. How can that make sense?
      let dim = img.GetDimension()
      let height = img.GetHeight()
      let width = img.GetWidth()
      let comp = img.GetNumberOfComponentsPerPixel()
      Expect.equal sz [w;h] $"Got {sz} expected [{w};{h}]"
      Expect.equal dpt d $"Got {dpt} expected {d}"
      Expect.equal dim 2u $"Got {dim} expected 3u"
      Expect.equal height h $"Got {height} expected {h}"
      Expect.equal width w $"Got {width} expected {w}"
      Expect.equal comp n $"Got {comp} expected {n}"
  ]


[<Tests>]
let imageCoreTests =
  testList "Image Core Functionality" [

    testCase "ofArray2D / toArray2D match" <| fun _ ->
      let arr = array2D [ [ 42.0f; 84.0f ] ]
      let img = Image<float32>.ofArray2D arr
      let back = img.toArray2D()
      Expect.equal back arr "Array should roundtrip via image"

    testCase "ToString includes size info" <| fun _ ->
      let arr = array2D [ [ 1.0f; 2.0f ]; [ 3.0f; 4.0f ] ]
      let img = Image<float32>.ofArray2D arr
      let s = img.ToString()
      Expect.stringContains s "2x2" "ToString should mention image size"

    testCase "ofSimpleITK/toSimpleITK roundtrip" <| fun _ ->
      let arr = array2D [ [ 5.0f; 6.0f ]; [ 7.0f; 8.0f ] ]
      let orig = Image<float32>.ofArray2D arr
      let roundtrip = Image<float32>.ofSimpleITK(orig.toSimpleITK())
      Expect.equal (roundtrip.toArray2D()) arr "Expected roundtrip to preserve array"

    testCase "castTo converts image type" <| fun _ ->
      let arr = array2D [ [ 10.0f; 20.0f ]; [ 30.0f; 40.0f ] ]
      let imgF = Image<float32>.ofArray2D arr
      let imgB = imgF.castTo<uint8>()
      Expect.equal (imgB.toArray2D()) (array2D [ [10uy; 20uy]; [30uy; 40uy] ]) "Expected image cast to byte"

    testCase "ofFile / toFile roundtrip" <| fun _ ->
      let arr = array2D [ [ 1.0f; 2.0f ]; [ 3.0f; 4.0f ] ]
      let img = Image<float32>.ofArray2D arr
      let tmp = System.IO.Path.GetTempFileName() + ".tiff"
      img.toFile tmp
      let reloaded = Image<float32>.ofFile tmp
      Expect.equal (reloaded.toArray2D()) arr $"Expected file I/O roundtrip {tmp}"
  ]

let array2dZip (a: 'T[,]) (b: 'U[,]) : ('T * 'U)[,] =
    let wA, hA = a.GetLength(0), a.GetLength(1)
    let wB, hB = b.GetLength(0), b.GetLength(1)
    if wA <> wB || hA <> hB then
        invalidArg "b" $"Array dimensions must match: {wA}x{hA} vs {wB}x{hB}"
    Array2D.init wA hA (fun x y -> a.[x, y], b.[x, y])

[<Tests>]
let imageOperatorTests =
  testList "Image Operator Overloads" [
    testCase "image + image" <| fun _ ->
      let a = Image<float>.ofArray2D (array2D [ [1.0; 2.0]; [3.0; 4.0] ])
      let b = Image<float>.ofArray2D (array2D [ [4.0; 3.0]; [2.0; 1.0] ])
      let c = a + b
      let expected = array2D [ [5.0; 5.0]; [5.0; 5.0] ]
      array2dZip (c.toArray2D()) expected
      |> Array2D.iter (fun (a,b)-> Expect.isLessThan (abs (a-b)) (1e-6) "image + image")

    testCase "image - image" <| fun _ ->
      let a = Image<float>.ofArray2D (array2D [ [5.0; 5.0]; [5.0; 5.0] ])
      let b = Image<float>.ofArray2D (array2D [ [1.0; 2.0]; [3.0; 4.0] ])
      let c = a - b
      let expected = array2D [ [4.0; 3.0]; [2.0; 1.0] ]
      array2dZip (c.toArray2D()) expected
      |> Array2D.iter (fun (a,b)-> Expect.isLessThan (abs (a-b)) (1e-6) "image - image")

    testCase "image * image" <| fun _ ->
      let a = Image<float>.ofArray2D (array2D [ [5.0; 5.0]; [5.0; 5.0] ])
      let b = Image<float>.ofArray2D (array2D [ [1.0; 2.0]; [3.0; 4.0] ])
      let c = a * b
      let expected = array2D [ [5.0; 10.0]; [15.0; 20.0] ]
      array2dZip (c.toArray2D()) expected
      |> Array2D.iter (fun (a,b)-> Expect.isLessThan (abs (a-b)) (1e-6) "image * image")

    testCase "image / image" <| fun _ ->
      let a = Image<float>.ofArray2D (array2D [ [5.0; 5.0]; [5.0; 5.0] ])
      let b = Image<float>.ofArray2D (array2D [ [1.0; 2.0]; [3.0; 4.0] ])
      let c = a / b
      let expected = array2D [ [5.0/1.0; 5.0/2.0]; [5.0/3.0; 5.0/4.0] ]
      array2dZip (c.toArray2D()) expected
      |> Array2D.iter (fun (a,b)-> Expect.isLessThan (abs (a-b)) (1e-6) "image / image")

    testCase "image + scalar" <| fun _ ->
      let a = Image<float>.ofArray2D (array2D [ [1.0; 2.0]; [3.0; 4.0] ])
      let b = 2.0
      let c = a + b
      let expected = array2D [ [3.0; 4.0]; [5.0; 6.0] ]
      array2dZip (c.toArray2D()) expected
      |> Array2D.iter (fun (a,b)-> Expect.isLessThan (abs (a-b)) (1e-6) "image + image")

    testCase "image - scalar" <| fun _ ->
      let a = Image<float>.ofArray2D (array2D [ [5.0; 4.0]; [3.0; 2.0] ])
      let b = 2.0
      let c = a - b
      let expected = array2D [ [3.0; 2.0]; [1.0; 0.0] ]
      array2dZip (c.toArray2D()) expected
      |> Array2D.iter (fun (a,b)-> Expect.isLessThan (abs (a-b)) (1e-6) "image - image")

    testCase "image * scalar" <| fun _ ->
      let a = Image<float>.ofArray2D (array2D [ [1.0; 2.0]; [3.0; 4.0] ])
      let b = 2.0
      let c = a * b
      let expected = array2D [ [2.0; 4.0]; [6.0; 8.0] ]
      array2dZip (c.toArray2D()) expected
      |> Array2D.iter (fun (a,b)-> Expect.isLessThan (abs (a-b)) (1e-6) "image * image")

    testCase "image / scalar" <| fun _ ->
      let a = Image<float>.ofArray2D (array2D [ [1.0; 2.0]; [3.0; 4.0] ])
      let b = 2.0
      let c = a / b
      let expected = array2D [ [1.0/2.0; 2.0/2.0]; [3.0/2.0; 4.0/2.0] ]
      array2dZip (c.toArray2D()) expected
      |> Array2D.iter (fun (a,b)-> Expect.isLessThan (abs (a-b)) (1e-6) "image / image")

    testCase "scalar + image" <| fun _ ->
      let a = Image<float>.ofArray2D (array2D [ [1.0; 2.0]; [3.0; 4.0] ])
      let b = 2.0
      let c = b + a
      let expected = array2D [ [3.0; 4.0]; [5.0; 6.0] ]
      array2dZip (c.toArray2D()) expected
      |> Array2D.iter (fun (a,b)-> Expect.isLessThan (abs (a-b)) (1e-6) "image + image")

    testCase "scalar - image" <| fun _ ->
      let a = Image<float>.ofArray2D (array2D [ [5.0; 4.0]; [3.0; 2.0] ])
      let b = 2.0
      let c = b - a
      let expected = array2D [ [-3.0; -2.0]; [-1.0; -0.0] ]
      array2dZip (c.toArray2D()) expected
      |> Array2D.iter (fun (a,b)-> Expect.isLessThan (abs (a-b)) (1e-6) "image - image")

    testCase "scalar * image" <| fun _ ->
      let a = Image<float>.ofArray2D (array2D [ [1.0; 2.0]; [3.0; 4.0] ])
      let b = 2.0
      let c = b * a
      let expected = array2D [ [2.0; 4.0]; [6.0; 8.0] ]
      array2dZip (c.toArray2D()) expected
      |> Array2D.iter (fun (a,b)-> Expect.isLessThan (abs (a-b)) (1e-6) "image * image")

    testCase "scalar / image" <| fun _ ->
      let a = Image<float>.ofArray2D (array2D [ [1.0; 2.0]; [3.0; 4.0] ])
      let b = 2.0
      let c = b / a
      let expected = array2D [ [2.0; 1.0]; [2.0/3.0; 1.0/2.0] ]
      array2dZip (c.toArray2D()) expected
      |> Array2D.iter (fun (a,b)-> Expect.isLessThan (abs (a-b)) (1e-6) "image / image")
  ]

[<Tests>]
let vectorImageCompositionTests =
  testList "Vector image composition and decomposition" [

    testCase "Compose 2 images and split back" <| fun _ ->
      let a = Image<float32>.ofArray2D (array2D [ [ 1.0f; 2.0f ]; [ 3.0f; 4.0f ] ])
      let b = Image<float32>.ofArray2D (array2D [ [ 5.0f; 6.0f ]; [ 7.0f; 8.0f ] ])
      let composed = Image<float32 list>.ofImageList [ a; b ]
      let split = composed.toImageList()

      Expect.equal (split.Length) 2 "Should split into 2 images"
      Expect.equal (split[0].toArray2D()) (a.toArray2D()) "First image should match original"
      Expect.equal (split[1].toArray2D()) (b.toArray2D()) "Second image should match original"

    testCase "Compose 3 images and split back" <| fun _ ->
      let a = Image<uint8>.ofArray2D (array2D [ [ 10uy; 20uy ]; [ 30uy; 40uy ] ])
      let b = Image<uint8>.ofArray2D (array2D [ [ 1uy; 2uy ]; [ 3uy; 4uy ] ])
      let c = Image<uint8>.ofArray2D (array2D [ [ 9uy; 8uy ]; [ 7uy; 6uy ] ])
      let composed = Image<uint8 list>.ofImageList [ a; b; c ]
      let split = composed.toImageList()

      Expect.equal split.Length 3 "Should split into 3 images"
      Expect.equal (split.[0].toArray2D()) (a.toArray2D()) "Image 0 matches"
      Expect.equal (split.[1].toArray2D()) (b.toArray2D()) "Image 1 matches"
      Expect.equal (split.[2].toArray2D()) (c.toArray2D()) "Image 2 matches"

    testCase "Empty image list throws" <| fun _ ->
      Expect.throws (fun () -> Image<float list>.ofImageList [] |> ignore) "Empty list should throw"

    testCase "Too many images throws" <| fun _ ->
      let imgs = List.replicate 11 (Image<float>.ofArray2D (array2D [ [ 1.0 ] ]))
      Expect.throws (fun () -> Image<float list>.ofImageList imgs |> ignore) "Too many images should throw"
  ]



[<EntryPoint>]
let main argv =
  runTestsWithArgs defaultConfig argv (testList "All Tests" [
    ToVectorTests 
    FromVectorTests 
    FromTypeTests 
    ofCastItkTests 
    creationTests 
    imageCoreTests
    imageOperatorTests
    vectorImageCompositionTests
  ])