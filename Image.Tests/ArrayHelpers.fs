/// <summary>
/// Unit tests for array conversion and type utility functions from InternalHelpers.fs.
/// Includes tests for pixel type mapping, casting, and n-dimensional array interoperability.
/// </summary>

module Image.Tests.ArrayHelpers

open Expecto
open itk.simple
open Image
open Image.InternalHelpers

[<Tests>]
let arrayAndTypeTests =
  testList "Array and type interop tests" [

    // fromType<'T>
    testCase "fromType basic check" <| fun _ ->
      let idUInt8 = fromType<uint8>
      Expect.equal idUInt8 PixelIDValueEnum.sitkUInt8 "Expected sitkUInt8"

      let idFloat = fromType<float>
      Expect.equal idFloat PixelIDValueEnum.sitkFloat64 "Expected sitkFloat64"

    // ofCastItk<'T>
    testCase "ofCastItk to float32" <| fun _ ->
      let arr = Array2D.init 5 5 (fun _ _ -> 1uy)
      let img = Image<uint8>.ofArray2D arr
      let casted = ofCastItk<float32> (img.toSimpleITK())
      Expect.equal (casted.GetPixelID()) PixelIDValueEnum.sitkFloat32 "Pixel ID changed to float32"

    // GetArray2DFromImage
    testCase "GetArray2DFromImage basic" <| fun _ ->
      let arr = Array2D.zeroCreate<uint8> 2 2
      arr.[0,0] <- 1uy
      arr.[1,1] <- 2uy
      let img = Image<uint8>.ofArray2D arr
      let out = GetArray2DFromImage (img.toSimpleITK())
      Expect.equal out.[0,0] 1uy "Pixel 0,0"
      Expect.equal out.[1,1] 2uy "Pixel 1,1"

    // GetArray3DFromImage
    testCase "GetArray3DFromImage basic" <| fun _ ->
      let arr = Array3D.zeroCreate<uint16> 2 2 2
      arr.[1,1,1] <- 42us
      let img = Image<uint16>.ofArray3D arr
      let out = GetArray3DFromImage (img.toSimpleITK())
      Expect.equal out.[1,1,1] 42us "Pixel 1,1,1"

    // GetArray4DFromImage
    testCase "GetArray4DFromImage basic" <| fun _ ->
      let arr = Array4D.zeroCreate<int8> 2 2 1 2
      arr.[1,1,0,1] <- -5y
      let img = Image<int8>.ofArray4D arr
      let out = GetArray4DFromImage (img.toSimpleITK())
      Expect.equal out.[1,1,0,1] -5y "Pixel 1,1,0,1"

    // Array4Diteri
    testCase "Array4Diteri summation" <| fun _ ->
      let arr = Array4D.zeroCreate 2 2 1 2
      arr.[0,0,0,0] <- 1
      arr.[1,1,0,1] <- 3
      let sum = ref 0
      Array4Diteri (fun _ _ _ _ v -> sum := !sum + v) arr
      Expect.equal !sum 4 "Sum over 4D array"

    // array2dZip
    testCase "array2dZip elementwise" <| fun _ ->
      let a = array2D [ [1; 2]; [3; 4] ]
      let b = array2D [ [10; 20]; [30; 40] ]
      let zipped = array2dZip a b
      Expect.equal zipped.[1,1] (4,40) "Zipped element 1,1"

    // pixelIdToString
    testCase "pixelIdToString basic" <| fun _ ->
      let name = pixelIdToString PixelIDValueEnum.sitkInt32
      Expect.equal name "Int32" "Enum to string"
  ]
