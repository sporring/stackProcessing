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
