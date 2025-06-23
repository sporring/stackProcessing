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

    testCase "cast converts image type" <| fun _ ->
      let arr = array2D [ [ 10.0f; 20.0f ]; [ 30.0f; 40.0f ] ]
      let imgF = Image<float32>.ofArray2D arr
      let imgB = imgF.cast<uint8>()
      Expect.equal (imgB.toArray2D()) (array2D [ [10uy; 20uy]; [30uy; 40uy] ]) "Expected image cast to byte"

    testCase "ofFile / toFile roundtrip" <| fun _ ->
      let arr = array2D [ [ 1.0f; 2.0f ]; [ 3.0f; 4.0f ] ]
      let img = Image<float32>.ofArray2D arr
      let tmp = System.IO.Path.GetTempFileName() + ".tiff"
      img.toFile tmp
      let reloaded = Image<float32>.ofFile tmp
      Expect.equal (reloaded.toArray2D()) arr $"Expected file I/O roundtrip {tmp}"
  ]

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

[<Tests>]
let IsEqualTests =
  testList "Image isEqual/op_Equality Tests" [
    // int8
    testCase "int8 image = image" <| fun _ ->
        let arr = array2D [ [ 1y; 2y ]; [ 3y; 4y ] ]
        let img1 = Image<int8>.ofArray2D arr 
        let img2 = Image<int8>.ofArray2D arr 
        Expect.isTrue (img1 = img2) "Expected images to be equal"

    testCase "int8 image = scalar" <| fun _ ->
        let arr = array2D [ [ 2y; 2y ]; [ 2y; 2y ] ]
        let img = Image<int8>.ofArray2D arr 
        Expect.isTrue (Image<int8>.eq(img,2y)) "Expected all pixels to match scalar"

    testCase "scalar = int8 image" <| fun _ ->
        let arr = array2D [ [ 3y; 3y ]; [ 3y; 3y ] ]
        let img = Image<int8>.ofArray2D arr 
        Expect.isTrue (Image<int8>.eq(3y, img)) "Expected scalar to equal all pixels"

    // uint8
    testCase "uint8 image = image" <| fun _ ->
        let arr = array2D [ [ 1uy; 2uy ]; [ 3uy; 4uy ] ]
        let img1 = Image<byte>.ofArray2D arr 
        let img2 = Image<byte>.ofArray2D arr 
        Expect.isTrue (img1 = img2) "Expected images to be equal"

    testCase "uint8 image = scalar" <| fun _ ->
        let arr = array2D [ [ 2uy; 2uy ]; [ 2uy; 2uy ] ]
        let img = Image<byte>.ofArray2D arr 
        Expect.isTrue (Image<byte>.eq(img, 2uy)) "Expected all pixels to match scalar"

    testCase "scalar = uint8 image" <| fun _ ->
        let arr = array2D [ [ 3uy; 3uy ]; [ 3uy; 3uy ] ]
        let img = Image<byte>.ofArray2D arr 
        Expect.isTrue (Image<byte>.eq(3uy, img)) "Expected scalar to equal all pixels"

    // int16
    testCase "int16 image = image" <| fun _ ->
        let arr = array2D [ [ 1s; 2s ]; [ 3s; 4s ] ]
        let img1 = Image<int16>.ofArray2D arr
        let img2 = Image<int16>.ofArray2D arr 
        Expect.isTrue (img1 = img2) "Expected images to be equal"

    testCase "int16 image = scalar" <| fun _ ->
        let arr = array2D [ [ 2s; 2s ]; [ 2s; 2s ] ]
        let img = Image<int16>.ofArray2D arr 
        Expect.isTrue (Image<int16>.eq(img, 2s)) "Expected all pixels to match scalar"

    testCase "scalar = int16 image" <| fun _ ->
        let arr = array2D [ [ 3s; 3s ]; [ 3s; 3s ] ]
        let img = Image<int16>.ofArray2D arr 
        Expect.isTrue (Image<int16>.eq(3s, img)) "Expected scalar to equal all pixels"

    // uint16
    testCase "uint16 image = image" <| fun _ ->
        let arr = array2D [ [ 1us; 2us ]; [ 3us; 4us ] ]
        let img1 = Image<uint16>.ofArray2D arr 
        let img2 = Image<uint16>.ofArray2D arr 
        Expect.isTrue (img1 = img2) "Expected images to be equal"

    testCase "uint16 image = scalar" <| fun _ ->
        let arr = array2D [ [ 2us; 2us ]; [ 2us; 2us ] ]
        let img = Image<uint16>.ofArray2D arr 
        Expect.isTrue (Image<uint16>.eq(img, 2us)) "Expected all pixels to match scalar"

    testCase "scalar = uint16 image" <| fun _ ->
        let arr = array2D [ [ 3us; 3us ]; [ 3us; 3us ] ]
        let img = Image<uint16>.ofArray2D arr 
        Expect.isTrue (Image<uint16>.eq(3us, img)) "Expected scalar to equal all pixels"

    // int32
    testCase "int32 image = image" <| fun _ ->
        let arr = array2D [ [ 1; 2 ]; [ 3; 4 ] ]
        let img1 = Image<int>.ofArray2D arr 
        let img2 = Image<int>.ofArray2D arr 
        Expect.isTrue (img1 = img2) "Expected images to be equal"

    testCase "int32 image = scalar" <| fun _ ->
        let arr = array2D [ [ 2; 2 ]; [ 2; 2 ] ]
        let img = Image<int>.ofArray2D arr 
        Expect.isTrue (Image<int>.eq(img, 2)) "Expected all pixels to match scalar"

    testCase "scalar = int32 image" <| fun _ ->
        let arr = array2D [ [ 3; 3 ]; [ 3; 3 ] ]
        let img = Image<int>.ofArray2D arr 
        Expect.isTrue (Image<int>.eq(3, img)) "Expected scalar to equal all pixels"

    // uint32
    testCase "uint32 image = image" <| fun _ ->
        let arr = array2D [ [ 1u; 2u ]; [ 3u; 4u ] ]
        let img1 = Image<uint32>.ofArray2D arr 
        let img2 = Image<uint32>.ofArray2D arr 
        Expect.isTrue (img1 = img2) "Expected images to be equal"

    testCase "uint32 image = scalar" <| fun _ ->
        let arr = array2D [ [ 2u; 2u ]; [ 2u; 2u ] ]
        let img = Image<uint32>.ofArray2D arr 
        Expect.isTrue (Image<uint32>.eq(img, 2u)) "Expected all pixels to match scalar"

    testCase "scalar = uint32 image" <| fun _ ->
        let arr = array2D [ [ 3u; 3u ]; [ 3u; 3u ] ]
        let img = Image<uint32>.ofArray2D arr 
        Expect.isTrue (Image<uint32>.eq(3u, img)) "Expected scalar to equal all pixels"

    // int64
    testCase "int64 image = image" <| fun _ ->
        let arr = array2D [ [ 1L; 2L ]; [ 3L; 4L ] ]
        let img1 = Image<int64>.ofArray2D arr 
        let img2 = Image<int64>.ofArray2D arr 
        Expect.isTrue (img1 = img2) "Expected images to be equal"

    testCase "int64 image = scalar" <| fun _ ->
        let arr = array2D [ [ 2L; 2L ]; [ 2L; 2L ] ]
        let img = Image<int64>.ofArray2D arr 
        Expect.isTrue (Image<int64>.eq(img, 2L)) "Expected all pixels to match scalar"

    testCase "scalar = int64 image" <| fun _ ->
        let arr = array2D [ [ 3L; 3L ]; [ 3L; 3L ] ]
        let img = Image<int64>.ofArray2D arr 
        Expect.isTrue (Image<int64>.eq(3L, img)) "Expected scalar to equal all pixels"

    // uint64
    testCase "uint64 image = image" <| fun _ ->
        let arr = array2D [ [ 1UL; 2UL ]; [ 3UL; 4UL ] ]
        let img1 = Image<uint64>.ofArray2D arr 
        let img2 = Image<uint64>.ofArray2D arr 
        Expect.isTrue (img1 = img2) "Expected images to be equal"

    testCase "uint64 image = scalar" <| fun _ ->
        let arr = array2D [ [ 2UL; 2UL ]; [ 2UL; 2UL ] ]
        let img = Image<uint64>.ofArray2D arr 
        Expect.isTrue (Image<uint64>.eq(img, 2UL)) "Expected all pixels to match scalar"

    testCase "scalar = uint64 image" <| fun _ ->
        let arr = array2D [ [ 3UL; 3UL ]; [ 3UL; 3UL ] ]
        let img = Image<uint64>.ofArray2D arr 
        Expect.isTrue (Image<uint64>.eq(3UL, img)) "Expected scalar to equal all pixels"

    // float32
    testCase "float32 image = image" <| fun _ ->
        let arr = array2D [ [ 1.0f; 2.0f ]; [ 3.0f; 4.0f ] ]
        let img1 = Image<float32>.ofArray2D arr 
        let img2 = Image<float32>.ofArray2D arr 
        Expect.isTrue (img1 = img2) "Expected images to be equal"

    testCase "float32 image = scalar" <| fun _ ->
        let arr = array2D [ [ 2.0f; 2.0f ]; [ 2.0f; 2.0f ] ]
        let img = Image<float32>.ofArray2D arr 
        Expect.isTrue (Image<float32>.eq(img, 2.0f)) "Expected all pixels to match scalar"

    testCase "scalar = float32 image" <| fun _ ->
        let arr = array2D [ [ 3.0f; 3.0f ]; [ 3.0f; 3.0f ] ]
        let img = Image<float32>.ofArray2D arr 
        Expect.isTrue (Image<float32>.eq(3.0f, img)) "Expected scalar to equal all pixels"

    // float
    testCase "float image = image" <| fun _ ->
        let arr = array2D [ [ 1.0; 2.0 ]; [ 3.0; 4.0 ] ]
        let img1 = Image<float>.ofArray2D arr 
        let img2 = Image<float>.ofArray2D arr 
        Expect.isTrue (img1 = img2) "Expected images to be equal"

    testCase "float image = scalar" <| fun _ ->
        let arr = array2D [ [ 2.0; 2.0 ]; [ 2.0; 2.0 ] ]
        let img = Image<float>.ofArray2D arr 
        Expect.isTrue (Image<float>.eq(img, 2.0)) "Expected all pixels to match scalar"

    testCase "scalar = float image" <| fun _ ->
        let arr = array2D [ [ 3.0; 3.0 ]; [ 3.0; 3.0 ] ]
        let img = Image<float>.ofArray2D arr 
        Expect.isTrue (Image<float>.eq(3.0, img)) "Expected scalar to equal all pixels"
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
    IsEqualTests
  ])