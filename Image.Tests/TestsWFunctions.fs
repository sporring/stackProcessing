module Tests

open Expecto
open ImageClass
open ImageFunctions
open itk.simple


[<Tests>]
let pixelTypeTests =
  testList "PixelType Tests" [
    testCase "ToSimpleITK UInt8" <| fun _ ->
      let value = PixelType.UInt8
      let result = value.ToSimpleITK()
      Expect.equal result PixelIDValueEnum.sitkUInt8 "Expected sitkUInt8"

    testCase "Zero value for UInt8" <| fun _ ->
      let value = PixelType.UInt8
      let result = value.Zero()
      Expect.equal result (box 0uy) "Expected boxed 0uy"
  ]



[<Tests>]
let imageFunctionsTests =
  testList "ImageFunctions Tests" [
    testCase "abs filter on simple image" <| fun _ ->
      let pixels = [| -1.0f; -2.0f; 3.0f; -4.0f |]
      let image = Image<float32>.ofArray(pixels, [|2; 2|])
      let result = ImageFunctions.abs image
      let expected = [| 1.0f; 2.0f; 3.0f; 4.0f |]
      Expect.sequenceEqual (result.toArray()) expected "Abs filter output mismatch"

    testCase "log filter on simple image" <| fun _ ->
      let pixels = [| 1.0f; 2.71828f; 7.389f; 20.0f |]
      let image = Image<float32>.ofArray(pixels, [|2; 2|])
      let result = ImageFunctions.log image
      let output = result.toArray()
      Expect.floatClose Accuracy.medium output.[1] 1.0 "Expected log(e) ≈ 1.0"
  ]


[<EntryPoint>]
let main argv =
  runTestsWithArgs defaultConfig argv (testList "All Tests" [ pixelTypeTests; imageFunctionsTests ])
