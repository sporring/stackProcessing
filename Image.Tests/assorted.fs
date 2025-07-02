module Tests.assorted

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

//    testCase "System.Numerics.Complex" <| fun _ ->
//      let result = fromType<System.Numerics.Complex>
//      Expect.equal result itk.simple.PixelIDValueEnum.sitkVectorFloat64 $"Expected {itk.simple.PixelIDValueEnum.sitkVectorFloat64}"

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

//    testCase "int32->System.Numerics.Complex" <| fun _ ->
//      let cast = ofCastItk<System.Numerics.Complex> imgInt32
//      let result = cast.GetPixelIDTypeAsString()
//      let expected = "vector of 64-bit float"
//      Expect.equal result expected $"Got {result} expected {expected}"

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

//    testCase "float->System.Numerics.Complex" <| fun _ ->
//      let cast = ofCastItk<System.Numerics.Complex> imgFloat64
//      let result = cast.GetPixelIDTypeAsString()
//      let expected = "vector of 64-bit float"
//      Expect.equal result expected $"Got {result} expected {expected}"

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
      let dim = img.GetDimensions()
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
      let dim = img.GetDimensions()
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
      let dim = img.GetDimensions()
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
      let dim = img.GetDimensions()
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

let array3DZip arr1 arr2 =
  Array3D.mapi (fun i j k v -> (v, arr2[i,j,k])) arr1

let floatArray3DFloatClose arr1 arr2 tol str = 
  array3DZip arr1 arr2 
  |> Array3D.iter (fun (a,b)-> Expect.isLessThan (abs (a-b)) tol str)

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
    // uint8
    testCase "uint8 image = image" <| fun _ ->
        let arr = array2D [| [| 1uy; 2uy |]; [| 3uy; 4uy |] |]
        let img1 = Image<byte>.ofArray2D arr 
        let img2 = Image<byte>.ofArray2D arr 
        Expect.isTrue (img1 = img2) "Expected images to be equal"
    // int16
    testCase "int16 image = image" <| fun _ ->
        let arr = array2D [| [| 1s; 2s |]; [| 3s; 4s |] |]
        let img1 = Image<int16>.ofArray2D arr
        let img2 = Image<int16>.ofArray2D arr 
        Expect.isTrue (img1 = img2) "Expected images to be equal"
    // uint16
    testCase "uint16 image = image" <| fun _ ->
        let arr = array2D [| [| 1us; 2us |]; [| 3us; 4us |] |]
        let img1 = Image<uint16>.ofArray2D arr 
        let img2 = Image<uint16>.ofArray2D arr 
        Expect.isTrue (img1 = img2) "Expected images to be equal"
    // int32
    testCase "int32 image = image" <| fun _ ->
        let arr = array2D [| [| 1; 2 |]; [| 3; 4 |] |]
        let img1 = Image<int>.ofArray2D arr 
        let img2 = Image<int>.ofArray2D arr 
        Expect.isTrue (img1 = img2) "Expected images to be equal"
    // uint32
    testCase "uint32 image = image" <| fun _ ->
        let arr = array2D [| [| 1u; 2u |]; [| 3u; 4u |] |]
        let img1 = Image<uint32>.ofArray2D arr 
        let img2 = Image<uint32>.ofArray2D arr 
        Expect.isTrue (img1 = img2) "Expected images to be equal"
    // int64
    testCase "int64 image = image" <| fun _ ->
        let arr = array2D [| [| 1L; 2L |]; [| 3L; 4L |] |]
        let img1 = Image<int64>.ofArray2D arr 
        let img2 = Image<int64>.ofArray2D arr 
        Expect.isTrue (img1 = img2) "Expected images to be equal"
    // uint64
    testCase "uint64 image = image" <| fun _ ->
        let arr = array2D [| [| 1UL; 2UL |]; [| 3UL; 4UL |] |]
        let img1 = Image<uint64>.ofArray2D arr 
        let img2 = Image<uint64>.ofArray2D arr 
        Expect.isTrue (img1 = img2) "Expected images to be equal"
    // float32
    testCase "float32 image = image" <| fun _ ->
        let arr = array2D [| [| 1.0f; 2.0f |]; [| 3.0f; 4.0f |] |]
        let img1 = Image<float32>.ofArray2D arr 
        let img2 = Image<float32>.ofArray2D arr 
        Expect.isTrue (img1 = img2) "Expected images to be equal"
    // float
    testCase "float image = image" <| fun _ ->
        let arr = array2D [| [| 1.0; 2.0 |]; [| 3.0; 4.0 |] |]
        let img1 = Image<float>.ofArray2D arr 
        let img2 = Image<float>.ofArray2D arr 
        Expect.isTrue (img1 = img2) "Expected images to be equal"
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
    // uint8
    testCase "uint8 identical images not unequal" <| fun _ ->
        let arr = array2D [| [| 1uy; 2uy |]; [| 3uy; 4uy |] |]
        let img1 = Image<uint8>.ofArray2D arr
        let img2 = Image<uint8>.ofArray2D arr
        Expect.isFalse (Image<uint8>.neq(img1, img2)) "Expected identical images to NOT be not equal"
    // int16
    testCase "int16 identical images not unequal" <| fun _ ->
        let arr = array2D [| [| 1s; 2s |]; [| 3s; 4s |] |]
        let img1 = Image<int16>.ofArray2D arr
        let img2 = Image<int16>.ofArray2D arr
        Expect.isFalse (Image<int16>.neq(img1, img2)) "Expected identical images to NOT be not equal"
    // int32
    testCase "int32 identical images not unequal" <| fun _ ->
        let arr = array2D [| [| 1; 2 |]; [| 3; 4 |] |]
        let img1 = Image<int>.ofArray2D arr
        let img2 = Image<int>.ofArray2D arr
        Expect.isFalse (Image<int>.neq(img1, img2)) "Expected identical images to NOT be not equal"
    // float
    testCase "float identical images not unequal" <| fun _ ->
        let arr = array2D [| [| 1.0; 2.0 |]; [| 3.0; 4.0 |] |]
        let img1 = Image<float>.ofArray2D arr
        let img2 = Image<float>.ofArray2D arr
        Expect.isFalse (Image<float>.neq(img1, img2)) "Expected identical float images to NOT be not equal"
  ]

[<Tests>]
let LessThanTests =
  testList "Image lessThan/lt Tests" [

    // int8
    testCase "int8 image < image" <| fun _ ->
        let img1 = Image<int8>.ofArray2D (array2D [| [| 1y; 2y |]; [| 3y; 4y |] |])
        let img2 = Image<int8>.ofArray2D (array2D [| [| 2y; 3y |]; [| 4y; 5y |] |])
        Expect.isTrue (Image<int8>.lt(img1, img2)) "Expected img1 < img2"
    testCase "int8 image < image (not true)" <| fun _ ->
        let img1 = Image<int8>.ofArray2D (array2D [| [| 5y; 6y |]; [| 7y; 8y |] |])
        let img2 = Image<int8>.ofArray2D (array2D [| [| 2y; 3y |]; [| 4y; 5y |] |])
        Expect.isFalse (Image<int8>.lt(img1, img2)) "Expected img1 NOT < img2"
  ]

[<Tests>]
let LessEqualTests =
  testList "Image isLessEqual/lte Tests" [

    // int8
    testCase "int8 image <= image" <| fun _ ->
        let img1 = Image<int8>.ofArray2D (array2D [| [| 1y; 2y |]; [| 3y; 4y |] |])
        let img2 = Image<int8>.ofArray2D (array2D [| [| 1y; 2y |]; [| 3y; 4y |] |])
        Expect.isTrue (Image<int8>.lte(img1, img2)) "Expected img1 <= img2 (equal values)"
    // int32
    testCase "int32 image <= image (true)" <| fun _ ->
        let img1 = Image<int32>.ofArray2D (array2D [| [| 1; 2 |]; [| 3; 4 |] |])
        let img2 = Image<int32>.ofArray2D (array2D [| [| 1; 2 |]; [| 3; 4 |] |])
        Expect.isTrue (Image<int32>.lte(img1, img2)) "Expected img1 <= img2"
  ]

[<Tests>]
let GreaterThanTests =
  testList "Image isGreater/gt Tests" [

    // int8
    testCase "int8 image > image" <| fun _ ->
        let img1 = Image<int8>.ofArray2D (array2D [| [| 5y; 6y |]; [| 7y; 8y |] |])
        let img2 = Image<int8>.ofArray2D (array2D [| [| 1y; 2y |]; [| 3y; 4y |] |])
        Expect.isTrue (Image<int8>.gt(img1, img2)) "Expected img1 > img2"
    // int32
    testCase "int32 image > image (false)" <| fun _ ->
        let img1 = Image<int>.ofArray2D (array2D [| [| 1; 2 |]; [| 3; 4 |] |])
        let img2 = Image<int>.ofArray2D (array2D [| [| 5; 6 |]; [| 7; 8 |] |])
        Expect.isFalse (Image<int>.gt(img1, img2)) "Expected img1 NOT > img2"
  ]

[<Tests>]
let GreaterEqualTests =
  testList "Image isGreaterEqual/gte Tests" [

    // int8
    testCase "int8 image >= image (equal)" <| fun _ ->
        let img = Image<int8>.ofArray2D (array2D [| [| 1y; 2y |]; [| 3y; 4y |] |])
        Expect.isTrue (Image<int8>.gte(img, img)) "Expected image >= itself"
    // int32
    testCase "int32 image >= image (false)" <| fun _ ->
        let img1 = Image<int>.ofArray2D (array2D [| [| 1; 2 |]; [| 3; 4 |] |])
        let img2 = Image<int>.ofArray2D (array2D [| [| 5; 6 |]; [| 7; 8 |] |])
        Expect.isFalse (Image<int>.gte(img1, img2)) "Expected img1 NOT >= img2"
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
    // XOR image ^^^ image
    testCase "Bitwise XOR: image ^^^ image" <| fun _ ->
        let img1 = Image<int>.ofArray2D (array2D [ [ 0b1010; 0b1111 ]; [ 0b0000; 0b0101 ] ])
        let img2 = Image<int>.ofArray2D (array2D [ [ 0b0101; 0b1010 ]; [ 0b1111; 0b0011 ] ])
        let result = Image<int>.op_ExclusiveOr(img1, img2)
        let expected = array2D [ [ 0b1111; 0b0101 ]; [ 0b1111; 0b0110 ] ]
        Expect.equal (result.toArray2D()) expected "Expected bitwise XOR of two images"
    // OR image ||| image
    testCase "Bitwise OR: image ||| image" <| fun _ ->
        let img1 = Image<int>.ofArray2D (array2D [ [ 0b0000; 0b1100 ]; [ 0b1010; 0b1001 ] ])
        let img2 = Image<int>.ofArray2D (array2D [ [ 0b1111; 0b0011 ]; [ 0b0101; 0b0110 ] ])
        let result = Image<int>.op_BitwiseOr(img1, img2)
        let expected = array2D [ [ 0b1111; 0b1111 ]; [ 0b1111; 0b1111 ] ]
        Expect.equal (result.toArray2D()) expected "Expected bitwise OR of two images"
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
        img.Set [0u; 0u] 99uy
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
        img.Set [1u; 0u; 1u] 77s
        let v = img[1, 0, 1]
        Expect.equal v 77s "Expected voxel at (1,0,1) to be 77"
  ]

[<Tests>]
let UnaryFunctionTests =
  testList "ImageFunctions unary image operator tests" [

    // abs
    testCase "abs of negative and positive values" <| fun _ ->
        let img = Image<float>.ofArray2D (array2D [ [ -1.0; 0.0 ]; [ 1.5; -2.5 ] ])
        let result = ImageFunctions.absImage img
        let expected = array2D [ [ 1.0; 0.0 ]; [ 1.5; 2.5 ] ]
        Expect.equal (result.toArray2D()) expected "Expected absolute values"

    // log
    testCase "log of values > 0" <| fun _ ->
        let img = Image<float>.ofArray2D (array2D [ [ 1.0; System.Math.E ]; [ 10.0; 100.0 ] ])
        let result = ImageFunctions.logImage img
        let expected = array2D [ [ 0.0; 1.0 ]; [ System.Math.Log 10.0; System.Math.Log 100.0 ] ]
        floatArray2DFloatClose (result.toArray2D()) expected 1e-10 "Expected natural log values"

    // log10
    testCase "log10 of powers of ten" <| fun _ ->
        let img = Image<float>.ofArray2D (array2D [ [ 1.0; 10.0 ]; [ 100.0; 1000.0 ] ])
        let result = ImageFunctions.log10Image img
        let expected = array2D [ [ 0.0; 1.0 ]; [ 2.0; 3.0 ] ]
        floatArray2DFloatClose (result.toArray2D()) expected 1e-10 "Expected base-10 log values"

    // exp
    testCase "exp of small numbers" <| fun _ ->
        let img = Image<float>.ofArray2D (array2D [ [ 0.0; 1.0 ]; [ 2.0; 3.0 ] ])
        let result = ImageFunctions.expImage img
        let expected = array2D [ [ 1.0; System.Math.Exp 1.0 ]; [ System.Math.Exp 2.0; System.Math.Exp 3.0 ] ]
        floatArray2DFloatClose (result.toArray2D()) expected 1e-10 "Expected exponentials"

    // sqrt
    testCase "sqrt of perfect squares" <| fun _ ->
        let img = Image<float>.ofArray2D (array2D [ [ 0.0; 1.0 ]; [ 4.0; 9.0 ] ])
        let result = ImageFunctions.sqrtImage img
        let expected = array2D [ [ 0.0; 1.0 ]; [ 2.0; 3.0 ] ]
        floatArray2DFloatClose (result.toArray2D()) expected 1e-10 "Expected square roots"

    // square
    testCase "square values" <| fun _ ->
        let img = Image<float>.ofArray2D (array2D [ [ 2.0; -3.0 ]; [ 4.0; -5.0 ] ])
        let result = ImageFunctions.squareImage img
        let expected = array2D [ [ 4.0; 9.0 ]; [ 16.0; 25.0 ] ]
        floatArray2DFloatClose (result.toArray2D()) expected 1e-10 "Expected squares"

    // sin
    testCase "sin of common angles" <| fun _ ->
        let img = Image<float>.ofArray2D (array2D [ [ 0.0; System.Math.PI / 2.0 ] ])
        let result = ImageFunctions.sinImage img
        let expected = array2D [ [ 0.0; 1.0 ] ]
        floatArray2DFloatClose (result.toArray2D()) expected 1e-10 "Expected sin values"

    // cos
    testCase "cos of common angles" <| fun _ ->
        let img = Image<float>.ofArray2D (array2D [ [ 0.0; System.Math.PI ] ])
        let result = ImageFunctions.cosImage img
        let expected = array2D [ [ 1.0; -1.0 ] ]
        floatArray2DFloatClose (result.toArray2D()) expected 1e-10 "Expected cos values"

    // tan
    testCase "tan of small angles" <| fun _ ->
        let img = Image<float>.ofArray2D (array2D [ [ 0.0; System.Math.PI / 4.0 ] ])
        let result = ImageFunctions.tanImage img
        let expected = array2D [ [ 0.0; 1.0 ] ]
        floatArray2DFloatClose (result.toArray2D()) expected 1e-10 "Expected tan values"

    // asin
    testCase "asin of valid inputs" <| fun _ ->
        let img = Image<float>.ofArray2D (array2D [ [ 0.0; 1.0 ] ])
        let result = ImageFunctions.asinImage img
        let expected = array2D [ [ 0.0; System.Math.PI / 2.0 ] ]
        floatArray2DFloatClose (result.toArray2D()) expected 1e-10 "Expected asin values"

    // acos
    testCase "acos of valid inputs" <| fun _ ->
        let img = Image<float>.ofArray2D (array2D [ [ 1.0; 0.0 ] ])
        let result = ImageFunctions.acosImage img
        let expected = array2D [ [ 0.0; System.Math.PI / 2.0 ] ]
        floatArray2DFloatClose (result.toArray2D()) expected 1e-10 "Expected acos values"

    // atan
    testCase "atan of known values" <| fun _ ->
        let img = Image<float>.ofArray2D (array2D [ [ 0.0; 1.0 ] ])
        let result = ImageFunctions.atanImage img
        let expected = array2D [ [ 0.0; System.Math.PI / 4.0 ] ]
        floatArray2DFloatClose (result.toArray2D()) expected 1e-10 "Expected atan values"

    // round
    testCase "round to nearest int" <| fun _ ->
        let img = Image<float>.ofArray2D (array2D [ [ 1.1; 2.5 ]; [ 3.9; 4.0 ] ])
        let result = ImageFunctions.roundImage img
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
        Expect.equal (result.GetDimensions()) 2u "Expected reduced dimensionality"

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
    testCase "conv 2D with identity kernel" <| fun _ ->
        let img = Image<float>.ofArray2D (array2D [ [ 1.0; 2.0 ]; [ 3.0; 4.0 ] ])
        let arr = Array2D.init 1 1 (fun m n -> if m=0 && n=0 then 1.0 else 0.0)
        let ker = Image<float>.ofArray2D arr
        let result = ImageFunctions.conv img ker
        let expected = img.toArray2D()
        floatArray2DFloatClose (result.toArray2D()) expected 1e-10 "Expected convolution with identity kernel"

    // conv
    testCase "convolve 2D with identity kernel" <| fun _ ->
        let img = Image<float>.ofArray2D (array2D [ [ 1.0; 2.0 ]; [ 3.0; 4.0 ] ])
        let arr = Array2D.init 1 1 (fun m n -> if m=0 && n=0 then 1.0 else 0.0)
        let ker = Image<float>.ofArray2D arr
        let result = ImageFunctions.convolve None None img ker
        let expected = img.toArray2D()
        floatArray2DFloatClose (result.toArray2D()) expected 1e-10 "Expected convolution with identity kernel"

    testCase "convolve 2D Valid" <| fun _ ->
        let img = Image<float>.ofArray2D (Array2D.create 3 3 1.0)
        let ker = Image<float>.ofArray2D (Array2D.create 2 2 2.0)
        let result = ImageFunctions.convolve (Some ImageFunctions.Valid) None img ker
        let sz = result.GetSize()
        let expected = [2u; 2u]
        Expect.equal sz expected "Expected a smaller result"

    testCase "convolve 2D Odd size" <| fun _ ->
        let img = Image<float>.ofArray2D (Array2D.create 3 3 1.0)
        let ker = Image<float>.ofArray2D (Array2D.create 1 2 2.0)
        let result = ImageFunctions.convolve (Some ImageFunctions.Valid) None img ker
        let sz = result.GetSize()
        let expected = [3u; 2u]
        Expect.equal sz expected "Expected a non-symmetric smaller result"

    testCase "convolve 2D Odd size reverse order" <| fun _ ->
        let img = Image<float>.ofArray2D (Array2D.create 1 2 2.0)
        let ker = Image<float>.ofArray2D (Array2D.create 3 3 1.0)
        let result = ImageFunctions.convolve None None img ker
        let sz = result.GetSize()
        let expected = [3u; 2u]
        Expect.equal sz expected "Expected a non-symmetric smaller result regardless of order"
(*
    // conv
    testCase "conv 3D with identity kernel" <| fun _ ->
        let img = Image<float>.ofArray3D (Array3D.init 2 2 2 (fun i j k -> float(k+2*j+4*i)))
        let arr = Array3D.init 1 1 1 (fun m n o -> if m=0 && n=0 && o=0 then 1.0 else 0.0)
        let ker = Image<float>.ofArray3D arr
        let result = ImageFunctions.conv img ker
        let expected = img.toArray3D()
        floatArray3DFloatClose (result.toArray3D()) expected 1e-10 "Expected conv with identity kernel"

    // conv
    testCase "convolve 3D with identity kernel" <| fun _ ->
        let img = Image<float>.ofArray3D (Array3D.init 2 2 2 (fun i j k -> float(k+2*j+4*i)))
        let arr = Array3D.init 1 1 1 (fun m n o -> if m=0 && n=0 && o=0 then 1.0 else 0.0)
        let ker = Image<float>.ofArray3D arr
        let result = ImageFunctions.convolve None None img ker
        let expected = img.toArray3D()
        floatArray3DFloatClose (result.toArray3D()) expected 1e-10 "Expected convolution with identity kernel"
*)
    testCase "convolve 3D Valid" <| fun _ ->
        let img = Image<float>.ofArray3D (Array3D.create 3 3 3 1.0)
        let ker = Image<float>.ofArray3D (Array3D.create 2 2 2 2.0)
        let result = ImageFunctions.convolve (Some ImageFunctions.Valid) None img ker
        let sz = result.GetSize()
        let expected = [2u; 2u; 2u]
        Expect.equal sz expected "Expected a smaller result"

    testCase "convolve 3D Odd size" <| fun _ ->
        let img = Image<float>.ofArray3D (Array3D.create 3 3 3 1.0)
        let ker = Image<float>.ofArray3D (Array3D.create 1 2 2 2.0)
        let result = ImageFunctions.convolve (Some ImageFunctions.Valid) None img ker
        let sz = result.GetSize()
        let expected = [3u; 2u; 2u]
        Expect.equal sz expected "Expected a non-symmetric smaller result"

    testCase "convolve 3D Odd size reverse order" <| fun _ ->
        let img = Image<float>.ofArray3D (Array3D.create 1 2 2 2.0)
        let ker = Image<float>.ofArray3D (Array3D.create 3 3 3 1.0)
        let result = ImageFunctions.convolve None None img ker
        let sz = result.GetSize()
        let expected = [3u; 2u; 2u]
        Expect.equal sz expected "Expected a non-symmetric smaller result regardless of order"

    // discreteGaussian
    testCase "2D discreteGaussian smooths image" <| fun _ ->
        let arr = Array2D.init 9 9 (fun m n -> if m=4 && n=4 then 1.0 else 0.0)
        let img = Image<float>.ofArray2D arr
        let blurred = ImageFunctions.discreteGaussian 2u 1.0 None None None img
        Expect.isTrue (
          blurred[2,2] < 1.0
          && blurred[2,2] > 0.0 
          && blurred[2,1] < blurred[2,2]
          && blurred[2,3] < blurred[2,2] 
          && blurred[1,2] < blurred[2,2] 
          && blurred[3,2] < blurred[2,2]
          ) "Expected smoothing at center"

    // discreteGaussian
    testCase "3D discreteGaussian smooths image" <| fun _ ->
        let arr = Array3D.init 9 9 9 (fun m n o -> if m=4 && n=4 && o=4 then 1.0 else 0.0)
        let img = Image<float>.ofArray3D arr
        let blurred = ImageFunctions.discreteGaussian 3u 1.0 None None None img
        Expect.isTrue (
          blurred[2,2,2] < 1.0
          && blurred[2,2,2] > 0.0 
          && blurred[2,1,1] < blurred[2,2,2]
          && blurred[2,3,3] < blurred[2,2,2] 
          && blurred[1,1,2] < blurred[2,2,2] 
          && blurred[3,2,2] < blurred[2,2,2]
          ) "Expected smoothing at center"

    // gradientConvolve
    testCase "gradientConvolve estimates gradient in x" <| fun _ ->
        let arr = Array2D.init 5 5 (fun m n -> if m=2 && n=2 then 1.0 else 0.0)
        let img = Image<float>.ofArray2D arr
        let grad = ImageFunctions.gradientConvolve 0u 1u img
        Expect.isTrue (
          grad[1,2] > 0.0
          && grad[3,2] < 0.0 
          ) "Expected positive gradient"

    testCase "finiteDiff 2D axis 0" <| fun _ ->
        let imgLst = List.map (ImageFunctions.finiteDiffFilter2D 0u) [1u;2u;3u;4u;5u]
        let sumLst = List.map ImageFunctions.sum imgLst
        let expected = 0.0
        let diff = List.fold (fun acc elm -> abs(elm-expected)) 0.0 sumLst
        let sz = imgLst[0].GetSize()
        Expect.isTrue (sz[0]>1u && sz[1]=1u) "shape should be nx1"
        Expect.floatClose Accuracy.high diff 0.0 "All finite diff filters should sum to 0.0"

    testCase "finiteDiff 2D axis 1" <| fun _ ->
        let imgLst = List.map (ImageFunctions.finiteDiffFilter2D 1u) [1u;2u;3u;4u;5u]
        let sumLst = List.map ImageFunctions.sum imgLst
        let expected = 0.0
        let diff = List.fold (fun acc elm -> abs(elm-expected)) 0.0 sumLst
        let sz = imgLst[0].GetSize()
        Expect.isTrue (sz[0]=1u && sz[1]>1u) "shape should be nx1"
        Expect.floatClose Accuracy.high diff 0.0 "All finite diff filters should sum to 0.0"

    testCase "finiteDiff 3D axis 0" <| fun _ ->
        let imgLst = List.map (ImageFunctions.finiteDiffFilter3D 0u) [1u;2u;3u;4u;5u]
        let sumLst = List.map ImageFunctions.sum imgLst
        let expected = 0.0
        let diff = List.fold (fun acc elm -> abs(elm-expected)) 0.0 sumLst
        let sz = imgLst[0].GetSize()
        Expect.isTrue (sz[0]>1u && sz[1]=1u && sz[2]=1u) "shape should be nx1"
        Expect.floatClose Accuracy.high diff 0.0 "All finite diff filters should sum to 0.0"

    testCase "finiteDiff 3D axis 1" <| fun _ ->
        let imgLst = List.map (ImageFunctions.finiteDiffFilter3D 1u) [1u;2u;3u;4u;5u]
        let sumLst = List.map ImageFunctions.sum imgLst
        let expected = 0.0
        let diff = List.fold (fun acc elm -> abs(elm-expected)) 0.0 sumLst
        let sz = imgLst[0].GetSize()
        Expect.isTrue (sz[0]=1u && sz[1]>1u && sz[2]=1u ) "shape should be nx1"
        Expect.floatClose Accuracy.high diff 0.0 "All finite diff filters should sum to 0.0"

    testCase "finiteDiff 3D axis 2" <| fun _ ->
        let imgLst = List.map (ImageFunctions.finiteDiffFilter3D 2u) [1u;2u;3u;4u;5u]
        let sumLst = List.map ImageFunctions.sum imgLst
        let expected = 0.0
        let diff = List.fold (fun acc elm -> abs(elm-expected)) 0.0 sumLst
        let sz = imgLst[0].GetSize()
        Expect.isTrue (sz[0]=1u && sz[1]=1u && sz[2]>1u) "shape should be nx1"
        Expect.floatClose Accuracy.high diff 0.0 "All finite diff filters should sum to 0.0"

    testCase "finiteDiff 4D axis 0" <| fun _ ->
        let imgLst = List.map (ImageFunctions.finiteDiffFilter4D 0u) [1u;2u;3u;4u;5u]
        let sumLst = List.map ImageFunctions.sum imgLst
        let expected = 0.0
        let diff = List.fold (fun acc elm -> abs(elm-expected)) 0.0 sumLst
        let sz = imgLst[0].GetSize()
        Expect.isTrue (sz[0]>1u && sz[1]=1u && sz[2]=1u && sz[3]=1u) "shape should be nx1"
        Expect.floatClose Accuracy.high diff 0.0 "All finite diff filters should sum to 0.0"

    testCase "finiteDiff 4D axis 1" <| fun _ ->
        let imgLst = List.map (ImageFunctions.finiteDiffFilter4D 1u) [1u;2u;3u;4u;5u]
        let sumLst = List.map ImageFunctions.sum imgLst
        let expected = 0.0
        let diff = List.fold (fun acc elm -> abs(elm-expected)) 0.0 sumLst
        let sz = imgLst[0].GetSize()
        Expect.isTrue (sz[0]=1u && sz[1]>1u && sz[2]=1u && sz[3]=1u) "shape should be nx1"
        Expect.floatClose Accuracy.high diff 0.0 "All finite diff filters should sum to 0.0"

    testCase "finiteDiff 4D axis 2" <| fun _ ->
        let imgLst = List.map (ImageFunctions.finiteDiffFilter4D 2u) [1u;2u;3u;4u;5u]
        let sumLst = List.map ImageFunctions.sum imgLst
        let expected = 0.0
        let diff = List.fold (fun acc elm -> abs(elm-expected)) 0.0 sumLst
        let sz = imgLst[0].GetSize()
        Expect.isTrue (sz[0]=1u && sz[1]=1u && sz[2]>1u && sz[3]=1u) "shape should be nx1"
        Expect.floatClose Accuracy.high diff 0.0 "All finite diff filters should sum to 0.0"

    testCase "finiteDiff 4D axis 3" <| fun _ ->
        let imgLst = List.map (ImageFunctions.finiteDiffFilter4D 3u) [1u;2u;3u;4u;5u]
        let sumLst = List.map ImageFunctions.sum imgLst
        let expected = 0.0
        let diff = List.fold (fun acc elm -> abs(elm-expected)) 0.0 sumLst
        let sz = imgLst[0].GetSize()
        Expect.isTrue (sz[0]=1u && sz[1]=1u && sz[2]=1u && sz[3]>1u) "shape should be nx1"
        Expect.floatClose Accuracy.high diff 0.0 "All finite diff filters should sum to 0.0"

  ]

[<Tests>]
let MorphologyTests =
  testList "ImageFunctions morphology tests" [

    testCase "binaryErode reduces foreground area" <| fun _ ->
      let img = Image<uint8>.ofArray2D (array2D [[0uy;1uy;1uy;0uy]])
      let eroded = ImageFunctions.binaryErode 1u img
      Expect.isTrue(eroded[0,1] < 1uy && eroded[0,2] < 1uy) "Foreground should shrink"

    testCase "binaryDilate expands foreground area" <| fun _ ->
      let img = Image<uint8>.ofArray2D (array2D [[0uy;1uy;0uy]])
      let dilated = ImageFunctions.binaryDilate 1u img
      Expect.isTrue(dilated[0,0] > 0uy && dilated[0,2] > 0uy) "Foreground should grow"

    testCase "binaryOpening removes small regions" <| fun _ ->
      let img = Image<uint8>.ofArray2D (array2D [[0uy;1uy;0uy;1uy;0uy]])
      let opened = ImageFunctions.binaryOpening 1u img
      Expect.isTrue(opened[0,1] < 1uy && opened[0,3] < 1uy) "Isolated pixels should be removed"

    testCase "binaryClosing fills small gaps" <| fun _ ->
      let img = Image<uint8>.ofArray2D (array2D [[1uy;0uy;1uy]])
      let closed = ImageFunctions.binaryClosing 1u img
      Expect.isTrue(closed[0,1] > 0uy) "Gap should be filled"

    testCase "binaryFillHoles fills a hole" <| fun _ ->
      let img = Image<uint8>.ofArray2D (array2D [[1uy;1uy;1uy]; [1uy;0uy;1uy]; [1uy;1uy;1uy]])
      let filled = ImageFunctions.binaryFillHoles img
      Expect.isTrue(filled[1,1] > 0uy) "Hole should be filled"

    testCase "connectedComponents labels two blobs" <| fun _ ->
      let img = Image<uint8>.ofArray2D (array2D [[0uy;1uy;1uy;0uy;1uy]])
      let cc = ImageFunctions.connectedComponents img
      Expect.isTrue(cc[0,0] <> cc[0,2] && cc[0,3] = 0uL) "Should label blobs separately"

    testCase "relabelComponents removes small component" <| fun _ ->
      let img = Image<uint8>.ofArray2D (array2D [[1uy;0uy;1uy]])
      let cc = ImageFunctions.connectedComponents img
      let relabeled = ImageFunctions.relabelComponents 2u cc
      Expect.isTrue(relabeled[0,0] = 0uL || relabeled[0,2] = 0uL) "Small blobs should be removed"

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

        Expect.equal (stats.NumPixels) 6u "Numpixels"
        Expect.floatClose Accuracy.low stats.Mean 3.5 "Mean"
        Expect.floatClose Accuracy.low stats.Std 1.870828693 "Standard deviation"
        Expect.floatClose Accuracy.low stats.Min 1.0 "Minimum"
        Expect.floatClose Accuracy.low stats.Max 6.0 "Maximum"
        Expect.floatClose Accuracy.low stats.Sum 21.0 "Sum"
        Expect.floatClose Accuracy.low stats.Var 3.5 "Variance"
    }

    test "Additive compute image statistics correctly" {
        let data1 = array2D [
            [1.0; 2.0; 3.0]
            [4.0; 5.0; 6.0]
        ]
        let data2 = array2D [
            [2.0; 3.0; 4.0]
            [5.0; 6.0; 7.0]
        ]
        let img1 = Image<float>.ofArray2D data1
        let stats1 = ImageFunctions.computeStats img1
        let img2 = Image<float>.ofArray2D data2
        let stats2 = ImageFunctions.computeStats img2
        let stats = ImageFunctions.addComputeStats stats1 stats2
        Expect.equal (stats.NumPixels) 12u $"Numpixels {stats.NumPixels}"
        Expect.floatClose Accuracy.low stats.Mean 4.0 $"Mean {stats.Mean}"
        Expect.floatClose Accuracy.low stats.Std 1.858640755 $"Standard deviation {stats.Std}"
        Expect.floatClose Accuracy.low stats.Min 1.0 $"Minimum {stats.Min}"
        Expect.floatClose Accuracy.low stats.Max 7.0 $"Maximum {stats.Max}"
        Expect.floatClose Accuracy.low stats.Sum 48.0 $"Sum {stats.Sum}"
        Expect.floatClose Accuracy.low stats.Var 3.454545455 $"Variance {stats.Var}"
    }

    test "Sum pixels" {
        let data = array2D [
            [1.0; 2.0; 3.0]
            [4.0; 5.0; 6.0]
        ]
        let img = Image<float>.ofArray2D data
        let result = ImageFunctions.sum(img)
        Expect.equal (result) 21.0 $"Sum {result}"
    }

    test "Product of pixels" {
        let data = array2D [
            [1.0; 2.0; 3.0]
            [4.0; 5.0; 6.0]
        ]
        let img = Image<float>.ofArray2D data
        let result = ImageFunctions.prod(img)
        Expect.equal (result) 720.0 $"Sum {result}"
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
      let result = ImageFunctions.expand 5u 0 [1; 2]
      Expect.equal result [1; 2; 0; 0; 0] "Should pad with zeros to length 5"

    testCase "Expand list equal to dim" <| fun _ ->
      let result = ImageFunctions.expand 3u "-" ["a"; "b"; "c"]
      Expect.equal result ["a"; "b"; "c"] "Should return unchanged list"

    testCase "Expand list longer than dim" <| fun _ ->
      let result = ImageFunctions.expand 2u 0 [1; 2; 3]
      Expect.equal result [1; 2; 3] "Should return original list unchanged"

    testCase "Expand empty list to length" <| fun _ ->
      let result = ImageFunctions.expand 4u 'x' []
      Expect.equal result ['x'; 'x'; 'x'; 'x'] "Should return list of four 'x'"
  ]

[<Tests>]
let stackTests =
  testList "ImageFunctions.stack" [

    testCase "stack joins images along z-axis" <| fun _ ->
      let img1 = Image.ofArray2D (array2D [[1;2];[3;4]])
      let img2 = Image.ofArray2D (array2D [[5;6];[7;8]])
      let stacked = ImageFunctions.stack [img1; img2]
      Expect.equal (stacked.GetSize()) [2u; 2u; 2u] "Stacked size should be 2x2x2"

    testCase "stack copies values" <| fun _ ->
      let img1 = Image.ofArray2D (array2D [[1;2];[3;4]])
      let img2 = Image.ofArray2D (array2D [[5;6];[7;8]])
      let stacked = ImageFunctions.stack [img1; img2]
      let result = stacked.toArray3D()
      Expect.equal [result[1,0,0];result[0,0,1]] [3;5] "Stacked size should be 2x2x2"

    testCase "Stacking 2D images results in 3D image along 3rd axis" <| fun _ ->
      let img1 = Image.ofArray2D (array2D [[1; 2]; [3; 4]])
      let img2 = Image.ofArray2D (array2D [[5; 6]; [7; 8]])
      let stacked = ImageFunctions.stack [img1; img2]
      Expect.equal (stacked.GetSize()) [2u; 2u; 2u] "Expected 2x2x2 (3D) image"

    testCase "Stacking 3D images results in 3D image" <| fun _ ->
      let img1 = Image.ofArray3D (Array3D.create 1 1 1 1) // size: 1x1x1
      let img2 = Image.ofArray3D (Array3D.create 1 1 2 1) // size: 1x1x1
      let stacked = ImageFunctions.stack [img1; img2]
      Expect.equal (stacked.GetSize()) [1u; 1u; 3u] "Expected 1x1x3 (3D) image"

    testCase "Stacking images with mismatched non-z dimensions fails" <| fun _ ->
      let img1 = Image.ofArray2D (array2D [[1; 2]])
      let img2 = Image.ofArray2D (array2D [[3; 4]; [5; 6]])
      Expect.throws (fun () -> ImageFunctions.stack [img1; img2] |> ignore) "Should fail due to mismatched Y dimension"

    testCase "Stacking single image still returns a valid image" <| fun _ ->
      let img = Image.ofArray2D (array2D [[9; 9]; [9; 9]])
      let stacked = ImageFunctions.stack [img]
      Expect.equal (stacked.GetSize()) [2u; 2u; 1u] "Expected 2x2x1 image from single input"

    testCase "stack-unstack identity" <| fun _ ->
      let img1 = Image.ofArray2D (array2D [[1;2];[3;4]])
      let img2 = Image.ofArray2D (array2D [[5;6];[7;8]])
      let stacked = ImageFunctions.stack [img1; img2]
      let unstacked = ImageFunctions.unstack stacked
      Expect.isTrue ((Image.eq (img1, unstacked[0])) && (Image.eq (img2, unstacked[1]))) "Stacking and unstacking are each other's inverse"
  ]

[<Tests>]
let someImageFunctionTests =
  testList "ImageFunctions" [

    testCase "unique returns sorted distinct values" <| fun _ ->
      let img = Image.ofArray2D (array2D [[1; 2; 3]; [3; 2; 1]])
      let u = ImageFunctions.unique img
      Expect.equal u [1; 2; 3] "Should return sorted distinct values"

    testCase "histogram returns correct counts" <| fun _ ->
      let img = Image.ofArray2D (array2D [[1; 2; 2]; [3; 3; 3]])
      let hist = ImageFunctions.histogram img
      Expect.equal hist.[1] 1UL "One 1"
      Expect.equal hist.[2] 2UL "Two 2s"
      Expect.equal hist.[3] 3UL "Three 3s"

    test "Adding histograms correctly" {
        let im1 = Image<int>.ofArray2D (Array2D.init 2 2 (fun i j -> i ))
        let im2 = Image<int>.ofArray2D (Array2D.init 2 2 (fun i j -> i+1 ))
        let expected = Map [(0,2uL);(1,4uL);(2,2uL)]
        let h1 = ImageFunctions.histogram im1
        let h2 = ImageFunctions.histogram im2
        let result = ImageFunctions.addHistogram h1 h2
        Expect.equal result expected $"Adding histograms {h1} {h2}"
    }

    testCase "addNormalNoise produces output of same size" <| fun _ ->
      let img = Image.ofArray2D (array2D [[1.0; 1.0]; [1.0; 1.0]])
      let noisy = ImageFunctions.addNormalNoise 0.0 1.0 img
      Expect.equal (img.GetSize()) (noisy.GetSize()) "Noise should not change size"

    testCase "threshold produces binary output" <| fun _ ->
      let img = Image.ofArray2D (array2D [[0.5; 1.5]; [2.5; 3.5]])
      let result = ImageFunctions.threshold 1.0 3.0 img
      let arr = result.toArray2D()
      Expect.equal arr.[0,0] 0.0 "Below lower threshold"
      Expect.equal arr.[0,1] 1.0 "Within threshold"
      Expect.equal arr.[1,0] 1.0 "Within threshold"
      Expect.equal arr.[1,1] 0.0 "Above upper threshold"

    testCase "extractSub extracts correct sub-region" <| fun _ ->
      let img = Image.ofArray2D (array2D [[1;2;3];[4;5;6];[7;8;9]])
      let sub = ImageFunctions.extractSub [1u;1u] [2u;2u] img
      let arr = sub.toArray2D()
      Expect.equal arr.[0,0] 5 "Top-left value"
      Expect.equal arr.[1,1] 9 "Bottom-right value"

    testCase "extractSlice extracts correct z slice" <| fun _ ->
      let img1 = Image.ofArray2D (array2D [[1;1];[1;1]])
      let img2 = Image.ofArray2D (array2D [[2;2];[2;2]])
      let img = ImageFunctions.stack [img1; img2]
      let slice = ImageFunctions.extractSlice 1u img
      let arr = (slice |> ImageFunctions.squeeze).toArray2D()
      Expect.equal arr.[0,0] 2 "Value from second slice"

    testCase "extractSlice extracts correct first slice" <| fun _ ->
      let img1 = Image.ofArray2D (array2D [[1;1];[1;1]])
      let img2 = Image.ofArray2D (array2D [[2;2];[2;2]])
      let img = ImageFunctions.stack [img1; img2]
      let slice = ImageFunctions.extractSlice 0u img
      let arr = (slice |> ImageFunctions.squeeze).toArray2D()
      Expect.equal arr.[0,0] 1 "Value from second slice"
  ]

