/// <summary>
/// Unit tests for array conversion and type utility functions from InternalHelpers.fs.
/// Includes tests for pixel type mapping, casting, and n-dimensional array interoperability.
/// </summary>

module Image.Tests.simpleFunctionTests

open Expecto
open itk.simple
open Image
open ImageFunctions

[<Tests>]
let simpleFunctionTests =
  testList "Simple arithmetic function tests" [

  testCase "sum for uint8 2x2 image" <| fun _ ->
    let arr = array2D [ [1uy; 2uy]; [3uy; 4uy] ]
    let img = Image<uint8>.ofArray2D arr
    Expect.equal (img |> sum) 10uy "Sum of 1+2+3+4 uint8"
  testCase "sum for int8 2x2 image" <| fun _ ->
    let arr = array2D [ [1y; 2y]; [3y; 4y] ]
    let img = Image<int8>.ofArray2D arr
    Expect.equal (img |> sum) 10y "Sum of 1+2+3+4 int8"
  testCase "sum for uint16 2x2 image" <| fun _ ->
    let arr = array2D [ [1us; 2us]; [3us; 4us] ]
    let img = Image<uint16>.ofArray2D arr
    Expect.equal (img |> sum) 10us "Sum of 1+2+3+4 uint16"
  testCase "sum for int16 2x2 image" <| fun _ ->
    let arr = array2D [ [1s; 2s]; [3s; 4s] ]
    let img = Image<int16>.ofArray2D arr
    Expect.equal (img |> sum) 10s "Sum of 1+2+3+4 int16"
  testCase "sum for uint 2x2 image" <| fun _ ->
    let arr = array2D [ [1u; 2u]; [3u; 4u] ]
    let img = Image<uint>.ofArray2D arr
    Expect.equal (img |> sum) 10u "Sum of 1+2+3+4 uint"
  testCase "sum for int 2x2 image" <| fun _ ->
    let arr = array2D [ [1; 2]; [3; 4] ]
    let img = Image<int>.ofArray2D arr
    Expect.equal (img |> sum) 10 "Sum of 1+2+3+4 int"
  testCase "sum for uint64 2x2 image" <| fun _ ->
    let arr = array2D [ [1uL; 2uL]; [3uL; 4uL] ]
    let img = Image<uint64>.ofArray2D arr
    Expect.equal (img |> sum) 10uL "Sum of 1+2+3+4 uint64"
  testCase "sum for int64 2x2 image" <| fun _ ->
    let arr = array2D [ [1L; 2L]; [3L; 4L] ]
    let img = Image<int64>.ofArray2D arr
    Expect.equal (img |> sum) 10L "Sum of 1+2+3+4 int64"
  testCase "sum for float32 2x2 image" <| fun _ ->
    let arr = array2D [ [1.0f; 2.0f]; [3.0f; 4.0f] ]
    let img = Image<float32>.ofArray2D arr
    Expect.floatClose Accuracy.veryHigh (img |> sum|>float) 10.0 "Sum of float32"
  testCase "sum for float 2x2 image" <| fun _ ->
    let arr = array2D [ [1.0; 2.0]; [3.0; 4.0] ]
    let img = Image<float>.ofArray2D arr
    Expect.floatClose Accuracy.veryHigh (img |> sum|>float) 10.0 "Sum of floats"

  ]