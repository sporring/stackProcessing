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
      let arr = array2D [| [| 42.0f; 84.0f |] |]
      let img = Image<float32>.ofArray2D arr
      let back = img.toArray2D()
      Expect.equal back arr "Array should roundtrip via image"

    testCase "ToString includes size info" <| fun _ ->
      let arr = array2D [| [| 1.0f; 2.0f |]; [| 3.0f; 4.0f |] |]
      let img = Image<float32>.ofArray2D arr
      let s = img.ToString()
      Expect.stringContains s "2x2" "ToString should mention image size"

    testCase "ofSimpleITK/toSimpleITK roundtrip" <| fun _ ->
      let arr = array2D [| [| 5.0f; 6.0f |]; [| 7.0f; 8.0f |] |]
      let orig = Image<float32>.ofArray2D arr
      let roundtrip = Image<float32>.ofSimpleITK(orig.toSimpleITK())
      Expect.equal (roundtrip.toArray2D()) arr "Expected roundtrip to preserve array"

    testCase "cast converts image type" <| fun _ ->
      let arr = array2D [| [| 10.0f; 20.0f |]; [| 30.0f; 40.0f |] |]
      let imgF = Image<float32>.ofArray2D arr
      let imgB = imgF.cast<uint8>()
      Expect.equal (imgB.toArray2D()) (array2D [| [| 10uy; 20uy |]; [| 30uy; 40uy |] |]) "Expected image cast to byte"

    testCase "ofFile / toFile roundtrip" <| fun _ ->
      let arr = array2D [| [| 1.0f; 2.0f |]; [| 3.0f; 4.0f |] |]
      let img = Image<float32>.ofArray2D arr
      let tmp = System.IO.Path.GetTempFileName() + ".tiff"
      img.toFile tmp
      let reloaded = Image<float32>.ofFile tmp
      Expect.equal (reloaded.toArray2D()) arr $"Expected file I/O roundtrip {tmp}"
  ]

let floatArray2DFloatClose arr1 arr2 tol str = 
  array2dZip arr1 arr2 
  |> Array2D.iter (fun (a,b)-> Expect.isLessThan (abs (a-b)) tol str)

[<Tests>]
let imageOperatorTests =
  testList "Image Operator Overloads" [
    testCase "image + image" <| fun _ ->
      let a = Image<float>.ofArray2D (array2D [| [| 1.0; 2.0 |]; [| 3.0; 4.0 |] |])
      let b = Image<float>.ofArray2D (array2D [| [| 4.0; 3.0 |]; [| 2.0; 1.0 |] |])
      let c = a + b
      let expected = array2D [| [| 5.0; 5.0 |]; [| 5.0; 5.0 |] |]
      floatArray2DFloatClose (c.toArray2D()) expected 1e-6 "image + image"

    testCase "image - image" <| fun _ ->
      let a = Image<float>.ofArray2D (array2D [| [| 5.0; 5.0 |]; [| 5.0; 5.0 |] |])
      let b = Image<float>.ofArray2D (array2D [| [| 1.0; 2.0 |]; [| 3.0; 4.0 |] |])
      let c = a - b
      let expected = array2D [| [| 4.0; 3.0 |]; [| 2.0; 1.0 |] |]
      floatArray2DFloatClose (c.toArray2D()) expected 1e-6 "image - image"

    testCase "image * image" <| fun _ ->
      let a = Image<float>.ofArray2D (array2D [| [| 5.0; 5.0 |]; [| 5.0; 5.0 |] |])
      let b = Image<float>.ofArray2D (array2D [| [| 1.0; 2.0 |]; [| 3.0; 4.0 |] |])
      let c = a * b
      let expected = array2D [| [| 5.0; 10.0 |]; [| 15.0; 20.0 |] |]
      floatArray2DFloatClose (c.toArray2D()) expected 1e-6 "image * image"

    testCase "image / image" <| fun _ ->
      let a = Image<float>.ofArray2D (array2D [| [| 5.0; 5.0 |]; [| 5.0; 5.0 |] |])
      let b = Image<float>.ofArray2D (array2D [| [| 1.0; 2.0 |]; [| 3.0; 4.0 |] |])
      let c = a / b
      let expected = array2D [| [| 5.0/1.0; 5.0/2.0 |]; [| 5.0/3.0; 5.0/4.0 |] |]
      floatArray2DFloatClose (c.toArray2D()) expected 1e-6 "image / image"

    testCase "image + scalar" <| fun _ ->
      let a = Image<float>.ofArray2D (array2D [| [| 1.0; 2.0 |]; [| 3.0; 4.0 |] |])
      let b = 2.0
      let c = a + b
      let expected = array2D [| [| 3.0; 4.0 |]; [| 5.0; 6.0 |] |]
      floatArray2DFloatClose (c.toArray2D()) expected 1e-6 "image + image"

    testCase "image - scalar" <| fun _ ->
      let a = Image<float>.ofArray2D (array2D [| [| 5.0; 4.0 |]; [| 3.0; 2.0 |] |])
      let b = 2.0
      let c = a - b
      let expected = array2D [| [| 3.0; 2.0 |]; [| 1.0; 0.0 |] |]
      floatArray2DFloatClose (c.toArray2D()) expected 1e-6 "image - image"

    testCase "image * scalar" <| fun _ ->
      let a = Image<float>.ofArray2D (array2D [| [| 1.0; 2.0 |]; [| 3.0; 4.0 |] |])
      let b = 2.0
      let c = a * b
      let expected = array2D [| [| 2.0; 4.0 |]; [| 6.0; 8.0 |] |]
      floatArray2DFloatClose (c.toArray2D()) expected 1e-6 "image * image"

    testCase "image / scalar" <| fun _ ->
      let a = Image<float>.ofArray2D (array2D [| [| 1.0; 2.0 |]; [| 3.0; 4.0 |] |])
      let b = 2.0
      let c = a / b
      let expected = array2D [| [| 1.0/2.0; 2.0/2.0 |]; [| 3.0/2.0; 4.0/2.0 |] |]
      floatArray2DFloatClose (c.toArray2D()) expected 1e-6 "image / image"

    testCase "scalar + image" <| fun _ ->
      let a = Image<float>.ofArray2D (array2D [| [| 1.0; 2.0 |]; [| 3.0; 4.0 |] |])
      let b = 2.0
      let c = b + a
      let expected = array2D [| [| 3.0; 4.0 |]; [| 5.0; 6.0 |] |]
      floatArray2DFloatClose (c.toArray2D()) expected 1e-6 "image + image"

    testCase "scalar - image" <| fun _ ->
      let a = Image<float>.ofArray2D (array2D [| [| 5.0; 4.0 |]; [| 3.0; 2.0 |] |])
      let b = 2.0
      let c = b - a
      let expected = array2D [| [| -3.0; -2.0 |]; [| -1.0; -0.0 |] |]
      floatArray2DFloatClose (c.toArray2D()) expected 1e-6 "image - image"

    testCase "scalar * image" <| fun _ ->
      let a = Image<float>.ofArray2D (array2D [| [| 1.0; 2.0 |]; [| 3.0; 4.0 |] |])
      let b = 2.0
      let c = b * a
      let expected = array2D [| [| 2.0; 4.0 |]; [| 6.0; 8.0 |] |]
      floatArray2DFloatClose (c.toArray2D()) expected 1e-6 "image * image"

    testCase "scalar / image" <| fun _ ->
      let a = Image<float>.ofArray2D (array2D [| [| 1.0; 2.0 |]; [| 3.0; 4.0 |] |])
      let b = 2.0
      let c = b / a
      let expected = array2D [| [| 2.0; 1.0 |]; [| 2.0/3.0; 1.0/2.0 |] |]
      floatArray2DFloatClose (c.toArray2D()) expected 1e-6 "image / image"
  ]

[<Tests>]
let vectorImageCompositionTests =
  testList "Vector image composition and decomposition" [

    testCase "Compose 2 images and split back" <| fun _ ->
      let a = Image<float32>.ofArray2D (array2D [| [| 1.0f; 2.0f |]; [| 3.0f; 4.0f |] |])
      let b = Image<float32>.ofArray2D (array2D [| [| 5.0f; 6.0f |]; [| 7.0f; 8.0f |] |])
      let composed = Image<float32 list>.ofImageList [ a; b ]
      let split = composed.toImageList()

      Expect.equal (split.Length) 2 "Should split into 2 images"
      Expect.equal (split[0].toArray2D()) (a.toArray2D()) "First image should match original"
      Expect.equal (split[1].toArray2D()) (b.toArray2D()) "Second image should match original"

    testCase "Compose 3 images and split back" <| fun _ ->
      let a = Image<uint8>.ofArray2D (array2D [| [| 10uy; 20uy |]; [| 30uy; 40uy |] |])
      let b = Image<uint8>.ofArray2D (array2D [| [| 1uy; 2uy |]; [| 3uy; 4uy |] |])
      let c = Image<uint8>.ofArray2D (array2D [| [| 9uy; 8uy |]; [| 7uy; 6uy |] |])
      let composed = Image<uint8 list>.ofImageList [ a; b; c ]
      let split = composed.toImageList()

      Expect.equal split.Length 3 "Should split into 3 images"
      Expect.equal (split[0].toArray2D()) (a.toArray2D()) "Image 0 matches"
      Expect.equal (split[1].toArray2D()) (b.toArray2D()) "Image 1 matches"
      Expect.equal (split[2].toArray2D()) (c.toArray2D()) "Image 2 matches"

    testCase "Empty image list throws" <| fun _ ->
      Expect.throws (fun () -> Image<float list>.ofImageList [] |> ignore) "Empty list should throw"

    testCase "Too many images throws" <| fun _ ->
      let imgs = List.replicate 11 (Image<float>.ofArray2D (array2D [| [| 1.0 |] |]))
      Expect.throws (fun () -> Image<float list>.ofImageList imgs |> ignore) "Too many images should throw"
  ]

[<Tests>]
let IsEqualTests =
  testList "Image isEqual/op_Equality Tests" [
    // int8
    testCase "int8 image = image" <| fun _ ->
        let arr = array2D [| [| 1y; 2y |]; [| 3y; 4y |] |]
        let img1 = Image<int8>.ofArray2D arr 
        let img2 = Image<int8>.ofArray2D arr 
        Expect.isTrue (img1 = img2) "Expected images to be equal"

    testCase "int8 image = scalar" <| fun _ ->
        let arr = array2D [| [| 2y; 2y |]; [| 2y; 2y |] |]
        let img = Image<int8>.ofArray2D arr 
        Expect.isTrue (Image<int8>.eq(img,2y)) "Expected all pixels to match scalar"

    testCase "scalar = int8 image" <| fun _ ->
        let arr = array2D [| [| 3y; 3y |]; [| 3y; 3y |] |]
        let img = Image<int8>.ofArray2D arr 
        Expect.isTrue (Image<int8>.eq(3y, img)) "Expected scalar to equal all pixels"

    // uint8
    testCase "uint8 image = image" <| fun _ ->
        let arr = array2D [| [| 1uy; 2uy |]; [| 3uy; 4uy |] |]
        let img1 = Image<byte>.ofArray2D arr 
        let img2 = Image<byte>.ofArray2D arr 
        Expect.isTrue (img1 = img2) "Expected images to be equal"

    testCase "uint8 image = scalar" <| fun _ ->
        let arr = array2D [| [| 2uy; 2uy |]; [| 2uy; 2uy |] |]
        let img = Image<byte>.ofArray2D arr 
        Expect.isTrue (Image<byte>.eq(img, 2uy)) "Expected all pixels to match scalar"

    testCase "scalar = uint8 image" <| fun _ ->
        let arr = array2D [| [| 3uy; 3uy |]; [| 3uy; 3uy |] |]
        let img = Image<byte>.ofArray2D arr 
        Expect.isTrue (Image<byte>.eq(3uy, img)) "Expected scalar to equal all pixels"

    // int16
    testCase "int16 image = image" <| fun _ ->
        let arr = array2D [| [| 1s; 2s |]; [| 3s; 4s |] |]
        let img1 = Image<int16>.ofArray2D arr
        let img2 = Image<int16>.ofArray2D arr 
        Expect.isTrue (img1 = img2) "Expected images to be equal"

    testCase "int16 image = scalar" <| fun _ ->
        let arr = array2D [| [| 2s; 2s |]; [| 2s; 2s |] |]
        let img = Image<int16>.ofArray2D arr 
        Expect.isTrue (Image<int16>.eq(img, 2s)) "Expected all pixels to match scalar"

    testCase "scalar = int16 image" <| fun _ ->
        let arr = array2D [| [| 3s; 3s |]; [| 3s; 3s |] |]
        let img = Image<int16>.ofArray2D arr 
        Expect.isTrue (Image<int16>.eq(3s, img)) "Expected scalar to equal all pixels"

    // uint16
    testCase "uint16 image = image" <| fun _ ->
        let arr = array2D [| [| 1us; 2us |]; [| 3us; 4us |] |]
        let img1 = Image<uint16>.ofArray2D arr 
        let img2 = Image<uint16>.ofArray2D arr 
        Expect.isTrue (img1 = img2) "Expected images to be equal"

    testCase "uint16 image = scalar" <| fun _ ->
        let arr = array2D [| [| 2us; 2us |]; [| 2us; 2us |] |]
        let img = Image<uint16>.ofArray2D arr 
        Expect.isTrue (Image<uint16>.eq(img, 2us)) "Expected all pixels to match scalar"

    testCase "scalar = uint16 image" <| fun _ ->
        let arr = array2D [| [| 3us; 3us |]; [| 3us; 3us |] |]
        let img = Image<uint16>.ofArray2D arr 
        Expect.isTrue (Image<uint16>.eq(3us, img)) "Expected scalar to equal all pixels"

    // int32
    testCase "int32 image = image" <| fun _ ->
        let arr = array2D [| [| 1; 2 |]; [| 3; 4 |] |]
        let img1 = Image<int>.ofArray2D arr 
        let img2 = Image<int>.ofArray2D arr 
        Expect.isTrue (img1 = img2) "Expected images to be equal"

    testCase "int32 image = scalar" <| fun _ ->
        let arr = array2D [| [| 2; 2 |]; [| 2; 2 |] |]
        let img = Image<int>.ofArray2D arr 
        Expect.isTrue (Image<int>.eq(img, 2)) "Expected all pixels to match scalar"

    testCase "scalar = int32 image" <| fun _ ->
        let arr = array2D [| [| 3; 3 |]; [| 3; 3 |] |]
        let img = Image<int>.ofArray2D arr 
        Expect.isTrue (Image<int>.eq(3, img)) "Expected scalar to equal all pixels"

    // uint32
    testCase "uint32 image = image" <| fun _ ->
        let arr = array2D [| [| 1u; 2u |]; [| 3u; 4u |] |]
        let img1 = Image<uint32>.ofArray2D arr 
        let img2 = Image<uint32>.ofArray2D arr 
        Expect.isTrue (img1 = img2) "Expected images to be equal"

    testCase "uint32 image = scalar" <| fun _ ->
        let arr = array2D [| [| 2u; 2u |]; [| 2u; 2u |] |]
        let img = Image<uint32>.ofArray2D arr 
        Expect.isTrue (Image<uint32>.eq(img, 2u)) "Expected all pixels to match scalar"

    testCase "scalar = uint32 image" <| fun _ ->
        let arr = array2D [| [| 3u; 3u |]; [| 3u; 3u |] |]
        let img = Image<uint32>.ofArray2D arr 
        Expect.isTrue (Image<uint32>.eq(3u, img)) "Expected scalar to equal all pixels"

    // int64
    testCase "int64 image = image" <| fun _ ->
        let arr = array2D [| [| 1L; 2L |]; [| 3L; 4L |] |]
        let img1 = Image<int64>.ofArray2D arr 
        let img2 = Image<int64>.ofArray2D arr 
        Expect.isTrue (img1 = img2) "Expected images to be equal"

    testCase "int64 image = scalar" <| fun _ ->
        let arr = array2D [| [| 2L; 2L |]; [| 2L; 2L |] |]
        let img = Image<int64>.ofArray2D arr 
        Expect.isTrue (Image<int64>.eq(img, 2L)) "Expected all pixels to match scalar"

    testCase "scalar = int64 image" <| fun _ ->
        let arr = array2D [| [| 3L; 3L |]; [| 3L; 3L |] |]
        let img = Image<int64>.ofArray2D arr 
        Expect.isTrue (Image<int64>.eq(3L, img)) "Expected scalar to equal all pixels"

    // uint64
    testCase "uint64 image = image" <| fun _ ->
        let arr = array2D [| [| 1UL; 2UL |]; [| 3UL; 4UL |] |]
        let img1 = Image<uint64>.ofArray2D arr 
        let img2 = Image<uint64>.ofArray2D arr 
        Expect.isTrue (img1 = img2) "Expected images to be equal"

    testCase "uint64 image = scalar" <| fun _ ->
        let arr = array2D [| [| 2UL; 2UL |]; [| 2UL; 2UL |] |]
        let img = Image<uint64>.ofArray2D arr 
        Expect.isTrue (Image<uint64>.eq(img, 2UL)) "Expected all pixels to match scalar"

    testCase "scalar = uint64 image" <| fun _ ->
        let arr = array2D [| [| 3UL; 3UL |]; [| 3UL; 3UL |] |]
        let img = Image<uint64>.ofArray2D arr 
        Expect.isTrue (Image<uint64>.eq(3UL, img)) "Expected scalar to equal all pixels"

    // float32
    testCase "float32 image = image" <| fun _ ->
        let arr = array2D [| [| 1.0f; 2.0f |]; [| 3.0f; 4.0f |] |]
        let img1 = Image<float32>.ofArray2D arr 
        let img2 = Image<float32>.ofArray2D arr 
        Expect.isTrue (img1 = img2) "Expected images to be equal"

    testCase "float32 image = scalar" <| fun _ ->
        let arr = array2D [| [| 2.0f; 2.0f |]; [| 2.0f; 2.0f |] |]
        let img = Image<float32>.ofArray2D arr 
        Expect.isTrue (Image<float32>.eq(img, 2.0f)) "Expected all pixels to match scalar"

    testCase "scalar = float32 image" <| fun _ ->
        let arr = array2D [| [| 3.0f; 3.0f |]; [| 3.0f; 3.0f |] |]
        let img = Image<float32>.ofArray2D arr 
        Expect.isTrue (Image<float32>.eq(3.0f, img)) "Expected scalar to equal all pixels"

    // float
    testCase "float image = image" <| fun _ ->
        let arr = array2D [| [| 1.0; 2.0 |]; [| 3.0; 4.0 |] |]
        let img1 = Image<float>.ofArray2D arr 
        let img2 = Image<float>.ofArray2D arr 
        Expect.isTrue (img1 = img2) "Expected images to be equal"

    testCase "float image = scalar" <| fun _ ->
        let arr = array2D [| [| 2.0; 2.0 |]; [| 2.0; 2.0 |] |]
        let img = Image<float>.ofArray2D arr 
        Expect.isTrue (Image<float>.eq(img, 2.0)) "Expected all pixels to match scalar"

    testCase "scalar = float image" <| fun _ ->
        let arr = array2D [| [| 3.0; 3.0 |]; [| 3.0; 3.0 |] |]
        let img = Image<float>.ofArray2D arr 
        Expect.isTrue (Image<float>.eq(3.0, img)) "Expected scalar to equal all pixels"
  ]

[<Tests>]
let IsNotEqualTests =
  testList "Image isNotEqual/neq Tests" [

    // int8
    testCase "int8 identical images not unequal" <| fun _ ->
        let arr = array2D [| [| 1y; 2y |]; [| 3y; 4y |] |]
        let img1 = Image<int8>.ofArray2D arr
        let img2 = Image<int8>.ofArray2D arr
        Expect.isFalse (Image<int8>.neq(img1, img2)) "Expected identical images to NOT be not equal"

    testCase "int8 image <> scalar (different)" <| fun _ ->
        let arr = array2D [| [| 2y; 2y |]; [| 2y; 2y |] |]
        let img = Image<int8>.ofArray2D arr
        Expect.isTrue (Image<int8>.neq(img, 3y)) "Expected image to differ from scalar"

    testCase "int8 scalar <> image (different)" <| fun _ ->
        let arr = array2D [| [| 3y; 3y |]; [| 3y; 3y |] |]
        let img = Image<int8>.ofArray2D arr
        Expect.isTrue (Image<int8>.neq(2y, img)) "Expected scalar to differ from image"

    // uint8
    testCase "uint8 identical images not unequal" <| fun _ ->
        let arr = array2D [| [| 1uy; 2uy |]; [| 3uy; 4uy |] |]
        let img1 = Image<uint8>.ofArray2D arr
        let img2 = Image<uint8>.ofArray2D arr
        Expect.isFalse (Image<uint8>.neq(img1, img2)) "Expected identical images to NOT be not equal"

    testCase "uint8 image <> scalar (different)" <| fun _ ->
        let arr = array2D [| [| 2uy; 2uy |]; [| 2uy; 2uy |] |]
        let img = Image<uint8>.ofArray2D arr
        Expect.isTrue (Image<uint8>.neq(img, 3uy)) "Expected image to differ from scalar"

    testCase "uint8 scalar <> image (different)" <| fun _ ->
        let arr = array2D [| [| 3uy; 3uy |]; [| 3uy; 3uy |] |]
        let img = Image<uint8>.ofArray2D arr
        Expect.isTrue (Image<uint8>.neq(2uy, img)) "Expected scalar to differ from image"

    // int16
    testCase "int16 identical images not unequal" <| fun _ ->
        let arr = array2D [| [| 1s; 2s |]; [| 3s; 4s |] |]
        let img1 = Image<int16>.ofArray2D arr
        let img2 = Image<int16>.ofArray2D arr
        Expect.isFalse (Image<int16>.neq(img1, img2)) "Expected identical images to NOT be not equal"

    testCase "int16 image <> scalar (different)" <| fun _ ->
        let arr = array2D [| [| 2s; 2s |]; [| 2s; 2s |] |]
        let img = Image<int16>.ofArray2D arr
        Expect.isTrue (Image<int16>.neq(img, 3s)) "Expected image to differ from scalar"

    testCase "int16 scalar <> image (different)" <| fun _ ->
        let arr = array2D [| [| 3s; 3s |]; [| 3s; 3s |] |]
        let img = Image<int16>.ofArray2D arr
        Expect.isTrue (Image<int16>.neq(2s, img)) "Expected scalar to differ from image"

    // int32
    testCase "int32 identical images not unequal" <| fun _ ->
        let arr = array2D [| [| 1; 2 |]; [| 3; 4 |] |]
        let img1 = Image<int>.ofArray2D arr
        let img2 = Image<int>.ofArray2D arr
        Expect.isFalse (Image<int>.neq(img1, img2)) "Expected identical images to NOT be not equal"

    testCase "int32 image <> scalar (different)" <| fun _ ->
        let arr = array2D [| [| 2; 2 |]; [| 2; 2 |] |]
        let img = Image<int>.ofArray2D arr
        Expect.isTrue (Image<int>.neq(img, 3)) "Expected image to differ from scalar"

    testCase "int32 scalar <> image (different)" <| fun _ ->
        let arr = array2D [| [| 3; 3 |]; [| 3; 3 |] |]
        let img = Image<int>.ofArray2D arr
        Expect.isTrue (Image<int>.neq(2, img)) "Expected scalar to differ from image"

    // float
    testCase "float identical images not unequal" <| fun _ ->
        let arr = array2D [| [| 1.0; 2.0 |]; [| 3.0; 4.0 |] |]
        let img1 = Image<float>.ofArray2D arr
        let img2 = Image<float>.ofArray2D arr
        Expect.isFalse (Image<float>.neq(img1, img2)) "Expected identical float images to NOT be not equal"

    testCase "float image <> scalar (different)" <| fun _ ->
        let arr = array2D [| [| 2.0; 2.0 |]; [| 2.0; 2.0 |] |]
        let img = Image<float>.ofArray2D arr
        Expect.isTrue (Image<float>.neq(img, 3.0)) "Expected float image to differ from scalar"

    testCase "float scalar <> image (different)" <| fun _ ->
        let arr = array2D [| [| 3.0; 3.0 |]; [| 3.0; 3.0 |] |]
        let img = Image<float>.ofArray2D arr
        Expect.isTrue (Image<float>.neq(2.0, img)) "Expected float scalar to differ from image"
  ]

[<Tests>]
let LessThanTests =
  testList "Image lessThan/lt Tests" [

    // int8
    testCase "int8 image < image" <| fun _ ->
        let img1 = Image<int8>.ofArray2D (array2D [| [| 1y; 2y |]; [| 3y; 4y |] |])
        let img2 = Image<int8>.ofArray2D (array2D [| [| 2y; 3y |]; [| 4y; 5y |] |])
        Expect.isTrue (Image<int8>.lt(img1, img2)) "Expected img1 < img2"

    testCase "int8 image < scalar" <| fun _ ->
        let img = Image<int8>.ofArray2D (array2D [| [| 1y; 1y |]; [| 1y; 1y |] |])
        Expect.isTrue (Image<int8>.lt(img, 2y)) "Expected image < scalar"

    testCase "int8 scalar < image" <| fun _ ->
        let img = Image<int8>.ofArray2D (array2D [| [| 5y; 5y |]; [| 5y; 5y |] |])
        Expect.isTrue (Image<int8>.lt(4y, img)) "Expected scalar < image"

    testCase "int8 image < image (not true)" <| fun _ ->
        let img1 = Image<int8>.ofArray2D (array2D [| [| 5y; 6y |]; [| 7y; 8y |] |])
        let img2 = Image<int8>.ofArray2D (array2D [| [| 2y; 3y |]; [| 4y; 5y |] |])
        Expect.isFalse (Image<int8>.lt(img1, img2)) "Expected img1 NOT < img2"

    // uint8
    testCase "uint8 image < scalar" <| fun _ ->
        let img = Image<uint8>.ofArray2D (array2D [| [| 1uy; 1uy |]; [| 1uy; 1uy |] |])
        Expect.isTrue (Image<uint8>.lt(img, 2uy)) "Expected uint8 image < scalar"

    testCase "uint8 scalar < image" <| fun _ ->
        let img = Image<uint8>.ofArray2D (array2D [| [| 3uy; 3uy |]; [| 3uy; 3uy |] |])
        Expect.isTrue (Image<uint8>.lt(2uy, img)) "Expected scalar < image"

    // int32
    testCase "int32 image < scalar" <| fun _ ->
        let img = Image<int>.ofArray2D (array2D [| [| 10; 10 |]; [| 10; 10 |] |])
        Expect.isTrue (Image<int>.lt(img, 11)) "Expected image < scalar"

    testCase "int32 scalar < image" <| fun _ ->
        let img = Image<int>.ofArray2D (array2D [| [| 10; 10 |]; [| 10; 10 |] |])
        Expect.isTrue (Image<int>.lt(9, img)) "Expected scalar < image"

    testCase "int32 image < image (equal, so false)" <| fun _ ->
        let arr = array2D [| [| 3; 3 |]; [| 3; 3 |] |]
        let img1 = Image<int>.ofArray2D arr
        let img2 = Image<int>.ofArray2D arr
        Expect.isFalse (Image<int>.lt(img1, img2)) "Expected img1 < img2 to be false for identical arrays"

    // float
    testCase "float image < scalar" <| fun _ ->
        let img = Image<float>.ofArray2D (array2D [| [| 1.0; 1.0 |]; [| 1.0; 1.0 |] |])
        Expect.isTrue (Image<float>.lt(img, 2.0)) "Expected float image < scalar"

    testCase "float scalar < image" <| fun _ ->
        let img = Image<float>.ofArray2D (array2D [| [| 5.0; 5.0 |]; [| 5.0; 5.0 |] |])
        Expect.isTrue (Image<float>.lt(3.5, img)) "Expected scalar < float image"
  ]

[<Tests>]
let LessEqualTests =
  testList "Image isLessEqual/lte Tests" [

    // int8
    testCase "int8 image <= image" <| fun _ ->
        let img1 = Image<int8>.ofArray2D (array2D [| [| 1y; 2y |]; [| 3y; 4y |] |])
        let img2 = Image<int8>.ofArray2D (array2D [| [| 1y; 2y |]; [| 3y; 4y |] |])
        Expect.isTrue (Image<int8>.lte(img1, img2)) "Expected img1 <= img2 (equal values)"

    testCase "int8 image <= scalar (equal)" <| fun _ ->
        let img = Image<int8>.ofArray2D (array2D [| [| 2y; 2y |]; [| 2y; 2y |] |])
        Expect.isTrue (Image<int8>.lte(img, 2y)) "Expected image <= scalar (equal values)"

    testCase "int8 scalar <= image (less)" <| fun _ ->
        let img = Image<int8>.ofArray2D (array2D [| [| 5y; 5y |]; [| 5y; 5y |] |])
        Expect.isTrue (Image<int8>.lte(4y, img)) "Expected scalar <= image"

    testCase "int8 image <= scalar (false)" <| fun _ ->
        let img = Image<int8>.ofArray2D (array2D [| [| 3y; 3y |]; [| 3y; 3y |] |])
        Expect.isFalse (Image<int8>.lte(img, 2y)) "Expected image NOT <= scalar"

    // uint8
    testCase "uint8 image <= scalar" <| fun _ ->
        let img = Image<uint8>.ofArray2D (array2D [| [| 1uy; 2uy |]; [| 3uy; 4uy |] |])
        Expect.isTrue (Image<uint8>.lte(img, 5uy)) "Expected image <= scalar"

    testCase "uint8 scalar <= image" <| fun _ ->
        let img = Image<uint8>.ofArray2D (array2D [| [| 3uy; 4uy |]; [| 5uy; 6uy |] |])
        Expect.isTrue (Image<uint8>.lte(2uy, img)) "Expected scalar <= image"

    // int32
    testCase "int32 image <= image (true)" <| fun _ ->
        let img1 = Image<int32>.ofArray2D (array2D [| [| 1; 2 |]; [| 3; 4 |] |])
        let img2 = Image<int32>.ofArray2D (array2D [| [| 1; 2 |]; [| 3; 4 |] |])
        Expect.isTrue (Image<int32>.lte(img1, img2)) "Expected img1 <= img2"

    testCase "int32 image <= scalar (false)" <| fun _ ->
        let img = Image<int32>.ofArray2D (array2D [| [| 5; 6 |]; [| 7; 8 |] |])
        Expect.isFalse (Image<int32>.lte(img, 4)) "Expected image NOT <= scalar"

    testCase "int32 scalar <= image (true)" <| fun _ ->
        let img = Image<int32>.ofArray2D (array2D [| [| 10; 10 |]; [| 10; 10 |] |])
        Expect.isTrue (Image<int32>.lte(10, img)) "Expected scalar <= image"

    // float
    testCase "float image <= scalar (equal)" <| fun _ ->
        let img = Image<float>.ofArray2D (array2D [| [| 2.5; 2.5 |]; [| 2.5; 2.5 |] |])
        Expect.isTrue (Image<float>.lte(img, 2.5)) "Expected float image <= scalar"

    testCase "float scalar <= image (true)" <| fun _ ->
        let img = Image<float>.ofArray2D (array2D [| [| 3.0; 3.0 |]; [| 3.0; 3.0 |] |])
        Expect.isTrue (Image<float>.lte(2.9, img)) "Expected scalar <= float image"

    testCase "float image <= image (false)" <| fun _ ->
        let img1 = Image<float>.ofArray2D (array2D [| [| 5.0; 6.0 |]; [| 7.0; 8.0 |] |])
        let img2 = Image<float>.ofArray2D (array2D [| [| 4.0; 5.0 |]; [| 6.0; 7.0 |] |])
        Expect.isFalse (Image<float>.lte(img1, img2)) "Expected img1 NOT <= img2"
  ]

[<Tests>]
let GreaterThanTests =
  testList "Image isGreater/gt Tests" [

    // int8
    testCase "int8 image > image" <| fun _ ->
        let img1 = Image<int8>.ofArray2D (array2D [| [| 5y; 6y |]; [| 7y; 8y |] |])
        let img2 = Image<int8>.ofArray2D (array2D [| [| 1y; 2y |]; [| 3y; 4y |] |])
        Expect.isTrue (Image<int8>.gt(img1, img2)) "Expected img1 > img2"

    testCase "int8 image > scalar" <| fun _ ->
        let img = Image<int8>.ofArray2D (array2D [| [| 5y; 5y |]; [| 5y; 5y |] |])
        Expect.isTrue (Image<int8>.gt(img, 4y)) "Expected image > scalar"

    testCase "int8 scalar > image" <| fun _ ->
        let img = Image<int8>.ofArray2D (array2D [| [| 3y; 3y |]; [| 3y; 3y |] |])
        Expect.isTrue (Image<int8>.gt(4y, img)) "Expected scalar > image"

    testCase "int8 image > scalar (false)" <| fun _ ->
        let img = Image<int8>.ofArray2D (array2D [| [| 2y; 2y |]; [| 2y; 2y |] |])
        Expect.isFalse (Image<int8>.gt(img, 3y)) "Expected image NOT > scalar"

    // uint8
    testCase "uint8 image > scalar" <| fun _ ->
        let img = Image<uint8>.ofArray2D (array2D [| [| 10uy; 10uy |]; [| 10uy; 10uy |] |])
        Expect.isTrue (Image<uint8>.gt(img, 9uy)) "Expected uint8 image > scalar"

    testCase "uint8 scalar > image" <| fun _ ->
        let img = Image<uint8>.ofArray2D (array2D [| [| 1uy; 1uy |]; [| 1uy; 1uy |] |])
        Expect.isTrue (Image<uint8>.gt(2uy, img)) "Expected scalar > image"

    // int32
    testCase "int32 image > image (false)" <| fun _ ->
        let img1 = Image<int>.ofArray2D (array2D [| [| 1; 2 |]; [| 3; 4 |] |])
        let img2 = Image<int>.ofArray2D (array2D [| [| 5; 6 |]; [| 7; 8 |] |])
        Expect.isFalse (Image<int>.gt(img1, img2)) "Expected img1 NOT > img2"

    testCase "int32 scalar > image (true)" <| fun _ ->
        let img = Image<int>.ofArray2D (array2D [| [| 10; 10 |]; [| 10; 10 |] |])
        Expect.isTrue (Image<int>.gt(11, img)) "Expected scalar > image"

    // float
    testCase "float image > scalar" <| fun _ ->
        let img = Image<float>.ofArray2D (array2D [| [| 2.5; 2.5 |]; [| 2.5; 2.5 |] |])
        Expect.isTrue (Image<float>.gt(img, 2.0)) "Expected float image > scalar"

    testCase "float scalar > image" <| fun _ ->
        let img = Image<float>.ofArray2D (array2D [| [| 3.5; 3.5 |]; [| 3.5; 3.5 |] |])
        Expect.isTrue (Image<float>.gt(4.0, img)) "Expected scalar > float image"

    testCase "float image > image (false)" <| fun _ ->
        let arr = array2D [| [| 2.0; 2.0 |]; [| 2.0; 2.0 |] |]
        let img1 = Image<float>.ofArray2D arr
        let img2 = Image<float>.ofArray2D arr
        Expect.isFalse (Image<float>.gt(img1, img2)) "Expected identical float images NOT to be greater than"
  ]

[<Tests>]
let GreaterEqualTests =
  testList "Image isGreaterEqual/gte Tests" [

    // int8
    testCase "int8 image >= image (equal)" <| fun _ ->
        let img = Image<int8>.ofArray2D (array2D [| [| 1y; 2y |]; [| 3y; 4y |] |])
        Expect.isTrue (Image<int8>.gte(img, img)) "Expected image >= itself"

    testCase "int8 image >= scalar" <| fun _ ->
        let img = Image<int8>.ofArray2D (array2D [| [| 5y; 5y |]; [| 5y; 5y |] |])
        Expect.isTrue (Image<int8>.gte(img, 4y)) "Expected image >= scalar"

    testCase "int8 scalar >= image" <| fun _ ->
        let img = Image<int8>.ofArray2D (array2D [| [| 1y; 2y |]; [| 3y; 4y |] |])
        Expect.isTrue (Image<int8>.gte(4y, img)) "Expected scalar >= image"

    testCase "int8 image >= scalar (false)" <| fun _ ->
        let img = Image<int8>.ofArray2D (array2D [| [| 1y; 1y |]; [| 1y; 1y |] |])
        Expect.isFalse (Image<int8>.gte(img, 2y)) "Expected image NOT >= scalar"

    // uint8
    testCase "uint8 scalar >= image" <| fun _ ->
        let img = Image<uint8>.ofArray2D (array2D [| [| 3uy; 3uy |]; [| 3uy; 3uy |] |])
        Expect.isTrue (Image<uint8>.gte(3uy, img)) "Expected scalar >= image"

    testCase "uint8 image >= scalar" <| fun _ ->
        let img = Image<uint8>.ofArray2D (array2D [| [| 4uy; 5uy |]; [| 6uy; 7uy |] |])
        Expect.isTrue (Image<uint8>.gte(img, 4uy)) "Expected image >= scalar"

    // int32
    testCase "int32 image >= image (false)" <| fun _ ->
        let img1 = Image<int>.ofArray2D (array2D [| [| 1; 2 |]; [| 3; 4 |] |])
        let img2 = Image<int>.ofArray2D (array2D [| [| 5; 6 |]; [| 7; 8 |] |])
        Expect.isFalse (Image<int>.gte(img1, img2)) "Expected img1 NOT >= img2"

    testCase "int32 scalar >= image (true)" <| fun _ ->
        let img = Image<int>.ofArray2D (array2D [| [| 10; 10 |]; [| 10; 10 |] |])
        Expect.isTrue (Image<int>.gte(10, img)) "Expected scalar >= image"

    // float
    testCase "float image >= scalar" <| fun _ ->
        let img = Image<float>.ofArray2D (array2D [| [| 2.5; 2.5 |]; [| 2.5; 2.5 |] |])
        Expect.isTrue (Image<float>.gte(img, 2.5)) "Expected float image >= scalar"

    testCase "float scalar >= image" <| fun _ ->
        let img = Image<float>.ofArray2D (array2D [| [| 2.5; 2.5 |]; [| 2.5; 2.5 |] |])
        Expect.isTrue (Image<float>.gte(2.5, img)) "Expected float scalar >= image"

    testCase "float image >= image (false)" <| fun _ ->
        let img1 = Image<float>.ofArray2D (array2D [| [| 1.0; 2.0 |]; [| 3.0; 4.0 |] |])
        let img2 = Image<float>.ofArray2D (array2D [| [| 5.0; 6.0 |]; [| 7.0; 8.0 |] |])
        Expect.isFalse (Image<float>.gte(img1, img2)) "Expected img1 NOT >= img2"
  ]

[<Tests>]
let ModulusTests =
  testList "Image modulus (%) Tests" [

    // Image % Image
    testCase "Image % Image (uint8)" <| fun _ ->
        let img1 = Image<uint8>.ofArray2D (array2D [ [ 5uy; 10uy ]; [ 15uy; 20uy ] ])
        let img2 = Image<uint8>.ofArray2D (array2D [ [ 2uy; 3uy ]; [ 4uy; 5uy ] ])
        let result = Image<uint8>.op_Modulus(img1, img2)
        let expected = (array2D [ [ 1uy; 1uy ]; [ 3uy; 0uy ] ])
        Expect.equal (result.toArray2D()) expected "Expected element-wise modulus result"

    // Image % scalar
    testCase "Image % scalar (uint8)" <| fun _ ->
        let img = Image<uint8>.ofArray2D (array2D [ [ 5uy; 10uy ]; [ 15uy; 20uy ] ])
        let result = Image<uint8>.op_Modulus(img, 6uy)
        let expected = (array2D [ [ 5uy; 4uy ]; [ 3uy; 2uy ] ])
        Expect.equal (result.toArray2D()) expected "Expected image % scalar result"

    // Scalar % Image
    testCase "Scalar % Image (uint8)" <| fun _ ->
        let img = Image<uint8>.ofArray2D (array2D [ [ 2uy; 3uy ]; [ 4uy; 5uy ] ])
        let result = Image<uint8>.op_Modulus(10uy, img)
        let expected = (array2D [ [ 0uy; 1uy ]; [ 2uy; 0uy ] ])
        Expect.equal (result.toArray2D()) expected "Expected scalar % image result"

    // Image % scalar (uint16)
    testCase "Image % scalar (uint16)" <| fun _ ->
        let img = Image<uint16>.ofArray2D (array2D [ [ 100us; 200us ]; [ 300us; 400us ] ])
        let result = Image<uint16>.op_Modulus(img, 128us)
        let expected = (array2D [ [ 100us; 72us ]; [ 44us; 16us ] ])
        Expect.equal (result.toArray2D()) expected "Expected uint16 image % scalar result"

    // Scalar % Image (uint32)
    testCase "Scalar % Image (uint32)" <| fun _ ->
        let img = Image<uint32>.ofArray2D (array2D [ [ 3u; 4u ]; [ 5u; 6u ] ])
        let result = Image<uint32>.op_Modulus(10u, img)
        let expected = (array2D [ [ 1u; 2u ]; [ 0u; 4u ] ])
        Expect.equal (result.toArray2D()) expected "Expected uint32 scalar % image result"
  ]

[<Tests>]
let PowTests =
  testList "Image Pow Tests" [

    // Image ^ Image
    testCase "Image ^ Image (uint8)" <| fun _ ->
        let baseImg = Image<uint8>.ofArray2D (array2D [ [ 2uy; 3uy ]; [ 4uy; 5uy ] ])
        let expImg  = Image<uint8>.ofArray2D (array2D [ [ 3uy; 2uy ]; [ 1uy; 0uy ] ])
        let result  = Image<uint8>.Pow(baseImg, expImg)
        let expected = array2D [ [ 8uy; 9uy ]; [ 4uy; 1uy ] ]
        Expect.equal (result.toArray2D()) expected "Expected image ^ image result"

    // Image ^ scalar
    testCase "Image ^ scalar (uint8 ^ int32)" <| fun _ ->
        let baseImg = Image<uint8>.ofArray2D (array2D [ [ 2uy; 3uy ]; [ 4uy; 5uy ] ])
        let result = Image<uint8>.Pow(baseImg, 2)
        let expected = array2D [ [ 4uy; 9uy ]; [ 16uy; 25uy ] ]
        Expect.equal (result.toArray2D()) expected "Expected image ^ scalar result"

    // Scalar ^ Image
    testCase "Scalar ^ Image (int32 ^ uint8)" <| fun _ ->
        let expImg = Image<uint8>.ofArray2D (array2D [ [ 1uy; 2uy ]; [ 3uy; 4uy ] ])
        let result = Image<uint8>.Pow(2, expImg)
        let expected = array2D [ [ 2uy; 4uy ]; [ 8uy; 16uy ] ]
        Expect.equal (result.toArray2D()) expected "Expected scalar ^ image result"

    // Image ^ scalar (float32 input, float result)
    testCase "Image ^ scalar (float32)" <| fun _ ->
        let baseImg = Image<float32>.ofArray2D (array2D [ [ 9.0f; 16.0f ]; [ 25.0f; 36.0f ] ])
        let result = Image<float32>.Pow(baseImg, 0.5f)
        let expected = array2D [ [ 3.0f; 4.0f ]; [ 5.0f; 6.0f ] ]
        Expect.equal (result.toArray2D()) expected "Expected sqrt of image elements"

    // Scalar ^ Image (float)
    testCase "Scalar ^ Image (float base)" <| fun _ ->
        let expImg = Image<float>.ofArray2D (array2D [ [ 0.0; 1.0 ]; [ 2.0; 3.0 ] ])
        let result = Image<float>.Pow(10.0, expImg)
        let expected = array2D [ [ 1.0; 10.0 ]; [ 100.0; 1000.0 ] ]
        Expect.equal (result.toArray2D()) expected "Expected 10^img result"
  ]

[<Tests>]
let BitwiseTests =
  testList "Image Bitwise Operator Tests" [

    // AND image & image
    testCase "Bitwise AND: image &&& image" <| fun _ ->
        let img1 = Image<int>.ofArray2D (array2D [ [ 0b1100; 0b1010 ]; [ 0b1111; 0b0000 ] ])
        let img2 = Image<int>.ofArray2D (array2D [ [ 0b1010; 0b1100 ]; [ 0b1111; 0b1111 ] ])
        let result = Image<int>.op_BitwiseAnd(img1, img2)
        let expected = array2D [ [ 0b1000; 0b1000 ]; [ 0b1111; 0b0000 ] ]
        Expect.equal (result.toArray2D()) expected "Expected bitwise AND of two images"

    // AND image & scalar
    testCase "Bitwise AND: image &&& scalar" <| fun _ ->
        let img = Image<int>.ofArray2D (array2D [ [ 0b1111; 0b1010 ]; [ 0b1100; 0b0110 ] ])
        let result = Image<int>.op_BitwiseAnd(img, 0b1100)
        let expected = array2D [ [ 0b1100; 0b1000 ]; [ 0b1100; 0b0100 ] ]
        Expect.equal (result.toArray2D()) expected "Expected bitwise AND of image and scalar"

    // AND scalar & image
    testCase "Bitwise AND: scalar &&& image" <| fun _ ->
        let img = Image<int>.ofArray2D (array2D [ [ 0b0110; 0b1100 ]; [ 0b1010; 0b1111 ] ])
        let result = Image<int>.op_BitwiseAnd(0b1110, img)
        let expected = array2D [ [ 0b0110; 0b1100 ]; [ 0b1010; 0b1110 ] ]
        Expect.equal (result.toArray2D()) expected "Expected bitwise AND of scalar and image"

    // XOR image ^^^ image
    testCase "Bitwise XOR: image ^^^ image" <| fun _ ->
        let img1 = Image<int>.ofArray2D (array2D [ [ 0b1010; 0b1111 ]; [ 0b0000; 0b0101 ] ])
        let img2 = Image<int>.ofArray2D (array2D [ [ 0b0101; 0b1010 ]; [ 0b1111; 0b0011 ] ])
        let result = Image<int>.op_ExclusiveOr(img1, img2)
        let expected = array2D [ [ 0b1111; 0b0101 ]; [ 0b1111; 0b0110 ] ]
        Expect.equal (result.toArray2D()) expected "Expected bitwise XOR of two images"

    // XOR image ^^^ scalar
    testCase "Bitwise XOR: image ^^^ scalar" <| fun _ ->
        let img = Image<int>.ofArray2D (array2D [ [ 0b1100; 0b1010 ]; [ 0b1111; 0b0000 ] ])
        let result = Image<int>.op_ExclusiveOr(img, 0b1111)
        let expected = array2D [ [ 0b0011; 0b0101 ]; [ 0b0000; 0b1111 ] ]
        Expect.equal (result.toArray2D()) expected "Expected bitwise XOR of image and scalar"

    // XOR scalar ^^^ image
    testCase "Bitwise XOR: scalar ^^^ image" <| fun _ ->
        let img = Image<int>.ofArray2D (array2D [ [ 0b0000; 0b1111 ]; [ 0b1010; 0b1100 ] ])
        let result = Image<int>.op_ExclusiveOr(0b1111, img)
        let expected = array2D [ [ 0b1111; 0b0000 ]; [ 0b0101; 0b0011 ] ]
        Expect.equal (result.toArray2D()) expected "Expected bitwise XOR of scalar and image"

    // OR image ||| image
    testCase "Bitwise OR: image ||| image" <| fun _ ->
        let img1 = Image<int>.ofArray2D (array2D [ [ 0b0000; 0b1100 ]; [ 0b1010; 0b1001 ] ])
        let img2 = Image<int>.ofArray2D (array2D [ [ 0b1111; 0b0011 ]; [ 0b0101; 0b0110 ] ])
        let result = Image<int>.op_BitwiseOr(img1, img2)
        let expected = array2D [ [ 0b1111; 0b1111 ]; [ 0b1111; 0b1111 ] ]
        Expect.equal (result.toArray2D()) expected "Expected bitwise OR of two images"

    // OR image ||| scalar
    testCase "Bitwise OR: image ||| scalar" <| fun _ ->
        let img = Image<int>.ofArray2D (array2D [ [ 0b0000; 0b0101 ]; [ 0b1010; 0b1100 ] ])
        let result = Image<int>.op_BitwiseOr(img, 0b0011)
        let expected = array2D [ [ 0b0011; 0b0111 ]; [ 0b1011; 0b1111 ] ]
        Expect.equal (result.toArray2D()) expected "Expected bitwise OR of image and scalar"

    // OR scalar ||| image
    testCase "Bitwise OR: scalar ||| image" <| fun _ ->
        let img = Image<int>.ofArray2D (array2D [ [ 0b1000; 0b0010 ]; [ 0b0100; 0b0000 ] ])
        let result = Image<int>.op_BitwiseOr(0b0001, img)
        let expected = array2D [ [ 0b1001; 0b0011 ]; [ 0b0101; 0b0001 ] ]
        Expect.equal (result.toArray2D()) expected "Expected bitwise OR of scalar and image"

    // NOT ~~~image
    testCase "Bitwise NOT: ~~~image (invert intensity)" <| fun _ ->
        let img = Image<int>.ofArray2D (array2D [ [ 0; 100 ]; [ 200; 255 ] ])
        let result = Image<int>.op_LogicalNot(img)
        // Assumes ITK uses `InvertIntensityImageFilter()` with max = 255
        let expected = array2D [ [ 255; 155 ]; [ 55; 0 ] ]
        Expect.equal (result.toArray2D()) expected "Expected inverted image values"
  ]

[<Tests>]
let IndexingTests =
  testList "Image Indexing Tests" [

    // 2D Get + Set (int32)
    testCase "2D get/set single pixel (int32)" <| fun _ ->
        let img = Image<int32>.ofArray2D (array2D [ [ 0; 0 ]; [ 0; 0 ] ])
        img[0, 1] <- 42
        let v = img[0, 1]
        Expect.equal v 42 "Expected pixel at (0,1) to be 42"

    testCase "2D get/set using Get/Set (uint8)" <| fun _ ->
        let img = Image<uint8>.ofArray2D (array2D [ [ 1uy; 2uy ]; [ 3uy; 4uy ] ])
        img.Set([0u; 0u], 99uy)
        let v = img.Get([0u; 0u])
        Expect.equal v 99uy "Expected pixel at [0;0] to be 99"

    // 2D Get original pixel (float)
    testCase "2D get original pixel (float)" <| fun _ ->
        let img = Image<float>.ofArray2D (array2D [ [ 1.5; 2.5 ]; [ 3.5; 4.5 ] ])
        let v = img[1, 0]
        Expect.equal v 3.5 "Expected pixel at (1,0) to be 3.5"

    // 3D Get/Set (int16)
    testCase "3D get/set single voxel (int16)" <| fun _ ->
        let data = Array3D.zeroCreate<int16> 2 2 2
        data[1, 1, 1] <- 123s
        let img = Image<int16>.ofArray3D data
        let v = img[1, 1, 1]
        Expect.equal v 123s "Expected voxel at (1,1,1) to be 123"

    testCase "3D set using Set, get using indexer (int16)" <| fun _ ->
        let data = Array3D.zeroCreate<int16> 2 2 2
        let img = Image<int16>.ofArray3D data
        img.Set([1u; 0u; 1u], 77s)
        let v = img[1, 0, 1]
        Expect.equal v 77s "Expected voxel at (1,0,1) to be 77"
  ]

[<Tests>]
let UnaryFunctionTests =
  testList "ImageFunctions unary image operator tests" [

    // abs
    testCase "abs of negative and positive values" <| fun _ ->
        let img = Image<float>.ofArray2D (array2D [ [ -1.0; 0.0 ]; [ 1.5; -2.5 ] ])
        let result = ImageFunctions.abs img
        let expected = array2D [ [ 1.0; 0.0 ]; [ 1.5; 2.5 ] ]
        Expect.equal (result.toArray2D()) expected "Expected absolute values"

    // log
    testCase "log of values > 0" <| fun _ ->
        let img = Image<float>.ofArray2D (array2D [ [ 1.0; System.Math.E ]; [ 10.0; 100.0 ] ])
        let result = ImageFunctions.log img
        let expected = array2D [ [ 0.0; 1.0 ]; [ System.Math.Log 10.0; System.Math.Log 100.0 ] ]
        floatArray2DFloatClose (result.toArray2D()) expected 1e-10 "Expected natural log values"

    // log10
    testCase "log10 of powers of ten" <| fun _ ->
        let img = Image<float>.ofArray2D (array2D [ [ 1.0; 10.0 ]; [ 100.0; 1000.0 ] ])
        let result = ImageFunctions.log10 img
        let expected = array2D [ [ 0.0; 1.0 ]; [ 2.0; 3.0 ] ]
        floatArray2DFloatClose (result.toArray2D()) expected 1e-10 "Expected base-10 log values"

    // exp
    testCase "exp of small numbers" <| fun _ ->
        let img = Image<float>.ofArray2D (array2D [ [ 0.0; 1.0 ]; [ 2.0; 3.0 ] ])
        let result = ImageFunctions.exp img
        let expected = array2D [ [ 1.0; System.Math.Exp 1.0 ]; [ System.Math.Exp 2.0; System.Math.Exp 3.0 ] ]
        floatArray2DFloatClose (result.toArray2D()) expected 1e-10 "Expected exponentials"

    // sqrt
    testCase "sqrt of perfect squares" <| fun _ ->
        let img = Image<float>.ofArray2D (array2D [ [ 0.0; 1.0 ]; [ 4.0; 9.0 ] ])
        let result = ImageFunctions.sqrt img
        let expected = array2D [ [ 0.0; 1.0 ]; [ 2.0; 3.0 ] ]
        floatArray2DFloatClose (result.toArray2D()) expected 1e-10 "Expected square roots"

    // square
    testCase "square values" <| fun _ ->
        let img = Image<float>.ofArray2D (array2D [ [ 2.0; -3.0 ]; [ 4.0; -5.0 ] ])
        let result = ImageFunctions.square img
        let expected = array2D [ [ 4.0; 9.0 ]; [ 16.0; 25.0 ] ]
        floatArray2DFloatClose (result.toArray2D()) expected 1e-10 "Expected squares"

    // sin
    testCase "sin of common angles" <| fun _ ->
        let img = Image<float>.ofArray2D (array2D [ [ 0.0; System.Math.PI / 2.0 ] ])
        let result = ImageFunctions.sin img
        let expected = array2D [ [ 0.0; 1.0 ] ]
        floatArray2DFloatClose (result.toArray2D()) expected 1e-10 "Expected sin values"

    // cos
    testCase "cos of common angles" <| fun _ ->
        let img = Image<float>.ofArray2D (array2D [ [ 0.0; System.Math.PI ] ])
        let result = ImageFunctions.cos img
        let expected = array2D [ [ 1.0; -1.0 ] ]
        floatArray2DFloatClose (result.toArray2D()) expected 1e-10 "Expected cos values"

    // tan
    testCase "tan of small angles" <| fun _ ->
        let img = Image<float>.ofArray2D (array2D [ [ 0.0; System.Math.PI / 4.0 ] ])
        let result = ImageFunctions.tan img
        let expected = array2D [ [ 0.0; 1.0 ] ]
        floatArray2DFloatClose (result.toArray2D()) expected 1e-10 "Expected tan values"

    // asin
    testCase "asin of valid inputs" <| fun _ ->
        let img = Image<float>.ofArray2D (array2D [ [ 0.0; 1.0 ] ])
        let result = ImageFunctions.asin img
        let expected = array2D [ [ 0.0; System.Math.PI / 2.0 ] ]
        floatArray2DFloatClose (result.toArray2D()) expected 1e-10 "Expected asin values"

    // acos
    testCase "acos of valid inputs" <| fun _ ->
        let img = Image<float>.ofArray2D (array2D [ [ 1.0; 0.0 ] ])
        let result = ImageFunctions.acos img
        let expected = array2D [ [ 0.0; System.Math.PI / 2.0 ] ]
        floatArray2DFloatClose (result.toArray2D()) expected 1e-10 "Expected acos values"

    // atan
    testCase "atan of known values" <| fun _ ->
        let img = Image<float>.ofArray2D (array2D [ [ 0.0; 1.0 ] ])
        let result = ImageFunctions.atan img
        let expected = array2D [ [ 0.0; System.Math.PI / 4.0 ] ]
        floatArray2DFloatClose (result.toArray2D()) expected 1e-10 "Expected atan values"

    // round
    testCase "round to nearest int" <| fun _ ->
        let img = Image<float>.ofArray2D (array2D [ [ 1.1; 2.5 ]; [ 3.9; 4.0 ] ])
        let result = ImageFunctions.round img
        let expected = array2D [ [ 1.0; 3.0 ]; [ 4.0; 4.0 ] ]
        floatArray2DFloatClose (result.toArray2D()) expected 1e-10 "Expected rounded values"
  ]

[<Tests>]
let ImageProcessingTests =
  testList "ImageFunctions advanced image operations" [

    // squeeze
    testCase "squeeze lowers dimensions" <| fun _ ->
        let img = Image<int>([10u;1u;12u])
        let result = ImageFunctions.squeeze img
        Expect.equal (result.GetDimension()) 2u "Expected reduced dimensionality"

    testCase "squeeze removes singleton dimensions" <| fun _ ->
        let img = Image<int>([10u;1u;12u])
        img[0,0,0] <- 1
        img[9,0,11] <- 2
        let sq = ImageFunctions.squeeze img
        let result = sq.toArray2D()
        Expect.equal ([result[0,0];result[9,11]]) [1;2] "Expected value"

    // concatAlong
    testCase "concatAlong concatenates along dim 0" <| fun _ ->
        let a = Image<float>.ofArray2D (array2D [ [ 1.0; 2.0 ] ])
        let b = Image<float>.ofArray2D (array2D [ [ 3.0; 4.0 ] ])
        let result = ImageFunctions.concatAlong 0u a b
        let expected = array2D [ [ 1.0; 2.0 ]; [ 3.0; 4.0 ] ]
        floatArray2DFloatClose (result.toArray2D()) expected 1e-10 "Expected vertical concatenation"

    // conv
    testCase "conv with identity kernel" <| fun _ ->
        let img = Image<float>.ofArray2D (array2D [ [ 1.0; 2.0 ]; [ 3.0; 4.0 ] ])
        let arr = Array2D.init 3 3 (fun m n -> if m=1 && n=1 then 1.0 else 0.0)
        let ker = Image<float>.ofArray2D arr
        let result = ImageFunctions.conv img ker
        let expected = img.toArray2D()
        floatArray2DFloatClose (result.toArray2D()) expected 1e-10 "Expected convolution with identity kernel"

    // discreteGaussian
    testCase "discreteGaussian smooths image" <| fun _ ->
        let arr = Array2D.init 5 5 (fun m n -> if m=2 && n=2 then 1.0 else 0.0)
        let img = Image<float>.ofArray2D arr
        let blurred = ImageFunctions.discreteGaussian 1.0 img
        Expect.isTrue (
          blurred[2,2]< 1.0
          && blurred[2,2] > 0.0 
          && blurred[2,1] < blurred[2,2]
          && blurred[2,3] < blurred[2,2] 
          && blurred[1,2] < blurred[2,2] 
          && blurred[3,2] < blurred[2,2]
          ) "Expected smoothing at center"

    // recursiveGaussian
    testCase "recursiveGaussian blurs in x direction" <| fun _ ->
        let arr = Array2D.init 5 5 (fun m n -> if m=2 && n=2 then 1.0 else 0.0)
        let img = Image<float>.ofArray2D arr
        let blurred = ImageFunctions.recursiveGaussian 1.0 0u img
        Expect.isTrue (
          blurred[2,2]< 1.0
          && blurred[2,2] > 0.0 
          && blurred[2,1] = 0.0
          && blurred[2,3] = 0.0
          && blurred[1,2] < blurred[2,2] 
          && blurred[3,2] < blurred[2,2]
          ) "Expected smoothing in x-direction"

    // laplacianConvolve
    testCase "laplacianConvolve highlights edges" <| fun _ ->
        let arr = Array2D.init 5 5 (fun m n -> if m=2 && n=2 then 1.0 else 0.0)
        let img = Image<float>.ofArray2D arr
        let lap = ImageFunctions.laplacianConvolve 1.0 img
        Expect.isTrue (
          lap[2,2] < 0.0
          && lap[2,1] > lap[2,2]
          && lap[2,3] > lap[2,2]
          && lap[1,2] > lap[2,2] 
          && lap[3,2] > lap[2,2]
          ) "Expected negative Laplacian at peak"

    // gradientConvolve
    testCase "gradientConvolve estimates gradient in x" <| fun _ ->
        let arr = Array2D.init 5 5 (fun m n -> if m=2 && n=2 then 1.0 else 0.0)
        let img = Image<float>.ofArray2D arr
        let grad = ImageFunctions.gradientConvolve 0u 1u img
        Expect.isTrue (
          grad[1,2] > 0.0
          && grad[3,2] < 0.0 
          ) "Expected positive gradient"
  ]

[<Tests>]
let MorphologyTests =
  testList "ImageFunctions morphology tests" [

    testCase "binaryErode reduces foreground area" <| fun _ ->
      let img = Image<int>.ofArray2D (array2D [[0;1;1;0]])
      let eroded = ImageFunctions.binaryErode 1u 1.0 img
      Expect.isTrue(eroded[0,1] < 1 && eroded[0,2] < 1) "Foreground should shrink"

    testCase "binaryDilate expands foreground area" <| fun _ ->
      let img = Image<int>.ofArray2D (array2D [[0;1;0]])
      let dilated = ImageFunctions.binaryDilate 1u 1.0 img
      Expect.isTrue(dilated[0,0] > 0 && dilated[0,2] > 0) "Foreground should grow"

    testCase "binaryOpening removes small regions" <| fun _ ->
      let img = Image<int>.ofArray2D (array2D [[0;1;0;1;0]])
      let opened = ImageFunctions.binaryOpening 1u 1.0 img
      Expect.isTrue(opened[0,1] < 1 && opened[0,3] < 1) "Isolated pixels should be removed"

    testCase "binaryClosing fills small gaps" <| fun _ ->
      let img = Image<int>.ofArray2D (array2D [[1;0;1]])
      let closed = ImageFunctions.binaryClosing 1u 1.0 img
      Expect.isTrue(closed[0,1] > 0) "Gap should be filled"

    testCase "binaryFillHoles fills a hole" <| fun _ ->
      let img = Image<int>.ofArray2D (array2D [[1;1;1]; [1;0;1]; [1;1;1]])
      let filled = ImageFunctions.binaryFillHoles 1.0 img
      Expect.isTrue(filled[1,1] > 0) "Hole should be filled"

    testCase "connectedComponents labels two blobs" <| fun _ ->
      let img = Image<int>.ofArray2D (array2D [[0;1;1;0;1]])
      let cc = ImageFunctions.connectedComponents img
      Expect.isTrue(cc[0,0] <> cc[0,2] && cc[0,3] = 0) "Should label blobs separately"

    testCase "relabelComponents removes small component" <| fun _ ->
      let img = Image<int>.ofArray2D (array2D [[1;0;1]])
      let cc = ImageFunctions.connectedComponents img
      let relabeled = ImageFunctions.relabelComponents 2u cc
      Expect.isTrue(relabeled[0,0] = 0 || relabeled[0,2] = 0) "Small blobs should be removed"

    testCase "signedDistanceMap outputs correct sign" <| fun _ ->
      let img = Image<uint8>.ofArray2D (array2D [[0uy;0uy;1uy;1uy; 1uy;0uy;0uy]])
      let dmap = ImageFunctions.signedDistanceMap 1uy 0uy img
      Expect.isTrue(dmap[0,0] > 0 && dmap[0,3] < 0) "Foreground should be negative"

    testCase "watershed separates peaks" <| fun _ ->
      let img = Image<int>.ofArray2D (array2D [[0;1;0;1;0]])
      let ws = ImageFunctions.watershed 0.0 img
      Expect.isTrue(ws[0,1] <> ws[0,3]) "Separate peaks should get separate labels"
  ]

[<Tests>]
let labelShapeStatisticsTests =
  testList "Label shape statistics" [

    test "Correctly computes stats from labeled regions" {
        // Create a test label image (2 labels)
        let arr = array2D [
            [0uy; 0uy; 1uy; 1uy; 0uy]
            [0uy; 2uy; 2uy; 0uy; 0uy]
            [0uy; 0uy; 0uy; 0uy; 0uy]
        ]
        let img = Image<uint8>.ofArray2D arr

        // Run the high-level labelShapeStatistics function
        let stats = ImageFunctions.labelShapeStatistics img

        // Expect two labels
        Expect.equal stats.Count 2 "Should detect two labels"

        // Label 1 assertions
        let s1 = stats[1L]
        Expect.equal s1.NumberOfPixels 2UL "Label 1 pixel count"
        Expect.floatClose Accuracy.low (List.item 0 s1.Centroid) 0.0 "Label 1 centroid Y"
        Expect.floatClose Accuracy.low (List.item 1 s1.Centroid) 2.5 "Label 1 centroid X"

        // Label 2 assertions
        let s2 = stats[2L]
        Expect.equal s2.NumberOfPixels 2UL "Label 2 pixel count"
        Expect.floatClose Accuracy.low (List.item 0 s2.Centroid) 1.0 "Label 2 centroid Y"
        Expect.floatClose Accuracy.low (List.item 1 s2.Centroid) 1.5 "Label 2 centroid X"
    }
  ]

open Expecto
open ImageFunctions  // Replace with your actual module name

[<Tests>]
let imageStatsAndThresholdTests =
  testList "Image statistics and thresholding" [

    test "Compute image statistics correctly" {
        let data = array2D [
            [1.0; 2.0; 3.0]
            [4.0; 5.0; 6.0]
        ]
        let img = Image<float>.ofArray2D data
        let stats = ImageFunctions.computeStats img

        Expect.floatClose Accuracy.low stats.Mean 3.5 "Mean"
        Expect.floatClose Accuracy.low stats.StdDev 1.870828693 "Standard deviation"
        Expect.floatClose Accuracy.low stats.Minimum 1.0 "Minimum"
        Expect.floatClose Accuracy.low stats.Maximum 6.0 "Maximum"
        Expect.floatClose Accuracy.low stats.Sum 21.0 "Sum"
        Expect.floatClose Accuracy.low stats.Variance 3.5 "Variance"
    }

    test "Otsu threshold separates binary regions" {
        let arr = array2D [
            [0uy; 0uy; 0uy; 255uy; 255uy; 255uy]
        ]
        let img = Image<uint8>.ofArray2D arr
        let thresh = ImageFunctions.otsuThreshold img

        // Check binary values: should be 0 and 1
        let unique = ImageFunctions.unique thresh
        Expect.containsAll unique [0uy; 1uy] "Otsu thresholding should produce binary image"
    }

    test "Otsu multiple thresholding produces multiple classes" {
        let arr = array2D [
            [10uy; 20uy; 30uy; 120uy; 130uy; 200uy]
        ]
        let img = Image<uint8>.ofArray2D arr
        let thresh = ImageFunctions.otsuMultiThreshold 2uy img

        // Should contain three unique classes: 0, 1, 2
        let unique = ImageFunctions.unique thresh
        Expect.containsAll unique [0uy; 1uy; 2uy] "Otsu multi-threshold should label into 3 regions"
    }

    test "Moments thresholding produces binary result" {
        let arr = array2D [
            [0uy; 0uy; 100uy; 255uy; 255uy]
        ]
        let img = Image<uint8>.ofArray2D arr
        let thresh = ImageFunctions.momentsThreshold img

        let unique = ImageFunctions.unique thresh
        Expect.containsAll unique [0uy; 1uy] "Moments thresholding should produce binary image"
    }
  ]

[<Tests>]
let generateCoordinateAxisTests =
  testList "Coordinate axis generator" [

    test "2D X-axis has correct coordinate values" {
        let size = [3; 2]
        let axis = 0
        let img = ImageFunctions.generateCoordinateAxis axis size
        let arr = img.toArray2D()
        arr
        |> Array2D.mapi (fun x y value ->
            Expect.equal value (uint32 x) $"Pixel at (%d{x}, %d{y}) should be %d{x}")
        |> ignore
    }

    test "2D Y-axis has correct coordinate values" {
        let size = [2; 3]
        let axis = 1
        let img = ImageFunctions.generateCoordinateAxis axis size
        let arr = img.toArray2D()
        arr
        |> Array2D.mapi (fun x y value ->
            Expect.equal value (uint32 y) $"Pixel at (%d{x}, %d{y}) should be %d{y}")
        |> ignore
    }

    test "3D Z-axis has correct coordinate values" {
        let size = [2; 2; 3]
        let axis = 2
        let img = ImageFunctions.generateCoordinateAxis axis size
        let arr = img.toArray3D()
        Array3D.iteri (fun x y z value ->
            let expected = uint32 z
            Expect.equal value expected $"Pixel at (%d{x}, %d{y}, %d{z}) should be %d{z}") arr
    }
  ]

let expandTests =
  testList "expand" [

    testCase "Expand list shorter than dim" <| fun _ ->
      let result = expand 5u 0 [1; 2]
      Expect.equal result [1; 2; 0; 0; 0] "Should pad with zeros to length 5"

    testCase "Expand list equal to dim" <| fun _ ->
      let result = expand 3u "-" ["a"; "b"; "c"]
      Expect.equal result ["a"; "b"; "c"] "Should return unchanged list"

    testCase "Expand list longer than dim" <| fun _ ->
      let result = expand 2u 0 [1; 2; 3]
      Expect.equal result [1; 2; 3] "Should return original list unchanged"

    testCase "Expand empty list to length" <| fun _ ->
      let result = expand 4u 'x' []
      Expect.equal result ['x'; 'x'; 'x'; 'x'] "Should return list of four 'x'"
  ]

[<Tests>]
let stackTests =
  testList "ImageFunctions.stack" [

    testCase "stack joins images along z-axis" <| fun _ ->
      let img1 = Image.ofArray2D (array2D [[1;2];[3;4]])
      let img2 = Image.ofArray2D (array2D [[5;6];[7;8]])
      let stacked = stack [img1; img2]
      Expect.equal (stacked.GetSize()) [2u; 2u; 2u] "Stacked size should be 2x2x2"

    testCase "stack copies values" <| fun _ ->
      let img1 = Image.ofArray2D (array2D [[1;2];[3;4]])
      let img2 = Image.ofArray2D (array2D [[5;6];[7;8]])
      let stacked = stack [img1; img2]
      let result = stacked.toArray3D()
      Expect.equal [result[1,0,0];result[0,0,1]] [3;5] "Stacked size should be 2x2x2"

    testCase "Stacking 2D images results in 3D image along 3rd axis" <| fun _ ->
      let img1 = Image.ofArray2D (array2D [[1; 2]; [3; 4]])
      let img2 = Image.ofArray2D (array2D [[5; 6]; [7; 8]])
      let stacked = stack [img1; img2]
      Expect.equal (stacked.GetSize()) [2u; 2u; 2u] "Expected 2x2x2 (3D) image"

    testCase "Stacking 3D images results in 3D image" <| fun _ ->
      let img1 = Image.ofArray3D (Array3D.create 1 1 1 1) // size: 1x1x1
      let img2 = Image.ofArray3D (Array3D.create 1 1 2 1) // size: 1x1x1
      let stacked = stack [img1; img2]
      Expect.equal (stacked.GetSize()) [1u; 1u; 3u] "Expected 1x1x3 (3D) image"

    testCase "Stacking images with mismatched non-z dimensions fails" <| fun _ ->
      let img1 = Image.ofArray2D (array2D [[1; 2]])
      let img2 = Image.ofArray2D (array2D [[3; 4]; [5; 6]])
      Expect.throws (fun () -> stack [img1; img2] |> ignore) "Should fail due to mismatched Y dimension"

    testCase "Stacking single image still returns a valid image" <| fun _ ->
      let img = Image.ofArray2D (array2D [[9; 9]; [9; 9]])
      let stacked = stack [img]
      Expect.equal (stacked.GetSize()) [2u; 2u; 1u] "Expected 2x2x1 image from single input"
  ]

[<Tests>]
let imageFunctionTests =
  testList "ImageFunctions" [

    testCase "unique returns sorted distinct values" <| fun _ ->
      let img = Image.ofArray2D (array2D [[1; 2; 3]; [3; 2; 1]])
      let u = unique img
      Expect.equal u [1; 2; 3] "Should return sorted distinct values"

    testCase "histogram returns correct counts" <| fun _ ->
      let img = Image.ofArray2D (array2D [[1; 2; 2]; [3; 3; 3]])
      let hist = histogram img
      Expect.equal hist.[1] 1UL "One 1"
      Expect.equal hist.[2] 2UL "Two 2s"
      Expect.equal hist.[3] 3UL "Three 3s"

    testCase "addNormalNoise produces output of same size" <| fun _ ->
      let img = Image.ofArray2D (array2D [[1.0; 1.0]; [1.0; 1.0]])
      let noisy = addNormalNoise 0.0 1.0 img
      Expect.equal (img.GetSize()) (noisy.GetSize()) "Noise should not change size"

    testCase "threshold produces binary output" <| fun _ ->
      let img = Image.ofArray2D (array2D [[0.5; 1.5]; [2.5; 3.5]])
      let result = threshold 1.0 3.0 img
      let arr = result.toArray2D()
      Expect.equal arr.[0,0] 0.0 "Below lower threshold"
      Expect.equal arr.[0,1] 1.0 "Within threshold"
      Expect.equal arr.[1,0] 1.0 "Within threshold"
      Expect.equal arr.[1,1] 0.0 "Above upper threshold"

    testCase "extractSub extracts correct sub-region" <| fun _ ->
      let img = Image.ofArray2D (array2D [[1;2;3];[4;5;6];[7;8;9]])
      let sub = extractSub [1u;1u] [2u;2u] img
      let arr = sub.toArray2D()
      Expect.equal arr.[0,0] 5 "Top-left value"
      Expect.equal arr.[1,1] 9 "Bottom-right value"

    testCase "extractSlice extracts correct z slice" <| fun _ ->
      let img1 = Image.ofArray2D (array2D [[1;1];[1;1]])
      let img2 = Image.ofArray2D (array2D [[2;2];[2;2]])
      let img = stack [img1; img2]
      let slice = extractSlice 1u img
      let arr = (slice |> squeeze).toArray2D()
      Expect.equal arr.[0,0] 2 "Value from second slice"

    testCase "extractSlice extracts correct first slice" <| fun _ ->
      let img1 = Image.ofArray2D (array2D [[1;1];[1;1]])
      let img2 = Image.ofArray2D (array2D [[2;2];[2;2]])
      let img = stack [img1; img2]
      let slice = extractSlice 0u img
      let arr = (slice |> squeeze).toArray2D()
      Expect.equal arr.[0,0] 1 "Value from second slice"
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
    IsNotEqualTests
    LessThanTests
    LessEqualTests
    GreaterThanTests
    GreaterEqualTests
    ModulusTests
    PowTests
    BitwiseTests
    IndexingTests
    UnaryFunctionTests
    ImageProcessingTests
    MorphologyTests
    labelShapeStatisticsTests
    imageStatsAndThresholdTests
    generateCoordinateAxisTests
    expandTests
    stackTests
    imageFunctionTests
  ])