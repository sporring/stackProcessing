module Tests.casting

open Expecto
open Image

let inline makeTestImage<^T when ^T: equality
                            and  ^T : (static member One  : ^T)
                            and  ^T : (static member ( + ) : ^T * ^T -> ^T)
                            > () =
  let one   : ^T = LanguagePrimitives.GenericOne
  let two   : ^T = one + one
  let three : ^T = two + one
  let four  : ^T = three + one
  Image< ^T>.ofArray2D (array2D [ [one; two]; [three; four] ])

let inline testOps<^T when ^T: equality
                      and  ^T: (static member op_Explicit: ^T -> float)
                      and  ^T : (static member One  : ^T)
                      and  ^T : (static member ( + ) : ^T * ^T -> ^T)
                      >
  (name: string) =
    let one   : ^T = LanguagePrimitives.GenericOne
    let two   : ^T = one + one
    let three : ^T = two + one

    testList name [
      testCase "image.toUInt8" <| fun _ ->
        let img1 = makeTestImage<^T>()
        let result = img1.toUInt8()
        Expect.equal (result.toArray2D().[1,0]) 3uy "uint8"
      testCase "image.toInt8" <| fun _ ->
        let img1 = makeTestImage<^T>()
        let result = img1.toInt8()
        Expect.equal (result.toArray2D().[1,0]) 3y "int8"
      testCase "image.toUInt16" <| fun _ ->
        let img1 = makeTestImage<^T>()
        let result = img1.toUInt16()
        Expect.equal (result.toArray2D().[1,0]) 3us "uint16"
      testCase "image.toInt16" <| fun _ ->
        let img1 = makeTestImage<^T>()
        let result = img1.toInt16()
        Expect.equal (result.toArray2D().[1,0]) 3s "int16"
      testCase "image.toUInt" <| fun _ ->
        let img1 = makeTestImage<^T>()
        let result = img1.toUInt()
        Expect.equal (result.toArray2D().[1,0]) 3u "uint"
      testCase "image.toInt" <| fun _ ->
        let img1 = makeTestImage<^T>()
        let result = img1.toInt()
        Expect.equal (result.toArray2D().[1,0]) 3 "int"
      testCase "image.toUInt64" <| fun _ ->
        let img1 = makeTestImage<^T>()
        let result = img1.toUInt64()
        Expect.equal (result.toArray2D().[1,0]) 3uL "uint64"
      testCase "image.toInt64" <| fun _ ->
        let img1 = makeTestImage<^T>()
        let result = img1.toInt64()
        Expect.equal (result.toArray2D().[1,0]) 3L "int64"
      testCase "image.toUFloat32" <| fun _ ->
        let img1 = makeTestImage<^T>()
        let result = img1.toFloat32()
        Expect.equal (result.toArray2D().[1,0]) 3.0f "float32"
      testCase "image.toFloat" <| fun _ ->
        let img1 = makeTestImage<^T>()
        let result = img1.toFloat()
        Expect.equal (result.toArray2D().[1,0]) 3.0 "float"

      testCase "memoryEstimate for ^T 2x2 image with 1 component" <| fun _ ->
        let img = makeTestImage<^T>()
        let expectedSize = uint32 (2 * 2 * 1 * sizeof<^T>)
        Expect.equal (Image<_>.memoryEstimateSItk img.Image) expectedSize "Memory estimate for 2x2 ^T"
    ]

[<Tests>]
let arithmeticTests =
  testList "Casting tests on Image<^T>" [
    // Run for different types
    testOps<uint8> "uint8 casts"
    testOps<int8> "int8 casts"
    testOps<uint16> "uint16 casts"
    testOps<int16> "int16 casts"
    testOps<uint32> "uint32 casts"
    testOps<int32> "int32 casts"
    testOps<uint64> "uint64 casts"
    testOps<int64> "int64 casts"
    testOps<float32> "float32 casts"
    testOps<float> "float64 casts"
  ]
