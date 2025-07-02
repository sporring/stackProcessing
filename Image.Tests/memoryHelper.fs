module Tests.memoryHelper

open Expecto
open itk.simple
open Image

[<Tests>]
let memoryHelperTests =
  testList "Memory helper tests" [

    testCase "memoryEstimate for uint8 2x2 image with 1 component" <| fun _ ->
    let arr = array2D [ [1uy; 2uy]; [3uy; 4uy] ]
    let img = Image<uint8>.ofArray2D arr
    let expectedSize = uint32 (2 * 2 * 1 * sizeof<uint8>)
    Expect.equal (img.memoryEstimate()) expectedSize "Memory estimate for 2x2 uint8"
  ]
  