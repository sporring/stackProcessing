module Tests.VectorHelpers

open Expecto
open itk.simple
open Image.InternalHelpers

let float32Close (expected: float32) (actual: float32) (within: float32) msg =
  if abs (expected - actual) > within then
    failwithf "%s: expected %f but got %f (diff %f > %f)" msg expected actual (abs (expected - actual)) within

[<Tests>]
let vectorInteropTests =
  testList "Vector interop tests" [

    // UInt8
    testCase "RT_UInt8_nonEmpty" <| fun _ ->
      let v = toVectorUInt8 [0uy; 255uy]
      Expect.equal (v.Count) 2 "Count should be 2"
      Expect.equal (v.[0]) 0uy "First element"
      Expect.equal (v.[1]) 255uy "Second element"

    testCase "RT_UInt8_empty" <| fun _ ->
      let v = toVectorUInt8 []
      Expect.equal (v.Count) 0 "Empty vector"

    testCase "fromVectorUInt8" <| fun _ ->
      let v = new VectorUInt8()
      v.Add(10uy)
      v.Add(20uy)
      let lst = fromVectorUInt8 v
      Expect.sequenceEqual lst [10uy; 20uy] "UInt8 list"

    // Int8
    testCase "RT_Int8_mixedSigns" <| fun _ ->
      let v = toVectorInt8 [-128y; 0y; 127y]
      Expect.equal (v.Count) 3 "Count"
      Expect.equal (v.[0]) -128y "First"
      Expect.equal (v.[2]) 127y "Last"

    testCase "RT_Int8_empty" <| fun _ ->
      let v = toVectorInt8 []
      Expect.equal (v.Count) 0 "Empty"

    testCase "fromVectorInt8" <| fun _ ->
      let v = new VectorInt8()
      v.Add(-1y)
      v.Add(1y)
      let lst = fromVectorInt8 v
      Expect.sequenceEqual lst [-1y; 1y] "Int8 list"

    // UInt16
    testCase "RT_UInt16_basic" <| fun _ ->
      let data = [0us; 65535us]
      let v = toVectorUInt16 data
      let back = fromVectorUInt16 v
      Expect.sequenceEqual back data "UInt16 round-trip"

    // Int16
    testCase "RT_Int16_basic" <| fun _ ->
      let data = [-32768s; 0s; 32767s]
      let v = toVectorInt16 data
      let back = fromVectorInt16 v
      Expect.sequenceEqual back data "Int16 round-trip"

    // UInt32
    testCase "RT_UInt32_basic" <| fun _ ->
      let data = [0u; 123456u]
      let v = toVectorUInt32 data
      let back = fromVectorUInt32 v
      Expect.sequenceEqual back data "UInt32 round-trip"

    // Int32
    testCase "RT_Int32_basic" <| fun _ ->
      let data = [-1000; 0; 1000]
      let v = toVectorInt32 data
      let back = fromVectorInt32 v
      Expect.sequenceEqual back data "Int32 round-trip"

    // UInt64
    testCase "RT_UInt64_basic" <| fun _ ->
      let data = [0UL; 1234567890UL]
      let v = toVectorUInt64 data
      let back = fromVectorUInt64 v
      Expect.sequenceEqual back data "UInt64 round-trip"

    // Int64
    testCase "RT_Int64_basic" <| fun _ ->
      let data = [-9999999999L; 0L; 9999999999L]
      let v = toVectorInt64 data
      let back = fromVectorInt64 v
      Expect.sequenceEqual back data "Int64 round-trip"

    // Float32
    testCase "RT_F32_typical" <| fun _ ->
      let data = [-1.5f; 0.0f; 3.14f]
      let v = toVectorFloat32 data
      let back = fromVectorFloat32 v
      Expect.sequenceEqual back data "float32 round-trip"
      float32Close 3.14f back.[2] 1e-4f "Check PI"

    testCase "RT_F32_empty" <| fun _ ->
      let v = toVectorFloat32 []
      Expect.equal (v.Count) 0 "Empty"

    // Float64
    testCase "RT_F64_typical" <| fun _ ->
      let data = [-1.5; 0.0; 3.14159]
      let v = toVectorFloat64 data
      let back = fromVectorFloat64 v
      Expect.sequenceEqual back data "float64 round-trip"
      Expect.floatClose Accuracy.high back.[2] 3.14159 "Check double PI"

    // Generic fromItkVector
    testCase "fromItkVector mapping" <| fun _ ->
      let v = new VectorInt32()
      v.Add(1); v.Add(2); v.Add(3)
      let lst = fromItkVector ((+) 1) v
      Expect.sequenceEqual lst [2;3;4] "Mapped list"
  ]
