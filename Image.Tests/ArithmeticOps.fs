/// <summary>
/// Unit tests for arithmetic operator overloads on Image<'T>.
/// Covers simple +, -, *, /, %, bitwise ops with scalars and images.
/// </summary>

module Image.Tests.ArithmeticOps

open Expecto
open Image

let inline makeTestImage<^T when ^T: equality
                            and  ^T : (static member Zero : ^T)
                            and  ^T : (static member One  : ^T)
                            and  ^T : (static member ( + ) : ^T * ^T -> ^T)
                            and  ^T : (static member ( - ) : ^T * ^T -> ^T)
                            and  ^T : (static member ( * ) : ^T * ^T -> ^T)
                            and  ^T : (static member ( / ) : ^T * ^T -> ^T)
                            > () =
  let one   : ^T = LanguagePrimitives.GenericOne
  let two   : ^T = one + one
  let three : ^T = two + one
  let four  : ^T = three + one
  Image< ^T>.ofArray2D (array2D [ [one; two]; [three; four] ])

[<Tests>]
let arithmeticTests =
  testList "Arithmetic operations on Image<'T>" [

    let testOps<'T when 'T: equality
                  and  ^T : (static member Zero : ^T)
                  and  ^T : (static member One  : ^T)
                  and  ^T : (static member ( + ) : ^T * ^T -> ^T)
                  and  ^T : (static member ( - ) : ^T * ^T -> ^T)
                  and  ^T : (static member ( * ) : ^T * ^T -> ^T)
                  and  ^T : (static member ( / ) : ^T * ^T -> ^T)
                >
      (name: string) =

      testList name [
        testCase "image + scalar" <| fun _ ->
          let img = makeTestImage<'T>()
          let one = LanguagePrimitives.GenericOne<'T>
          let result = img + one
          Expect.equal (result.toArray2D().[0,0]) (one + one) "1 + 1"

        testCase "scalar + image" <| fun _ ->
          let img = makeTestImage<'T>()
          let ten = (LanguagePrimitives.GenericOne<'T> * 10)
          let four = (LanguagePrimitives.GenericOne<'T> * 4)
          let result = ten + img
          Expect.equal (result.toArray2D().[1,1]) (ten + four) "10 + 4"

        testCase "image + image" <| fun _ ->
          let img1 = makeTestImage<'T>()
          let img2 = makeTestImage<'T>()
          let result = img1 + img2
          Expect.equal (result.toArray2D().[1,0]) (LanguagePrimitives.GenericOne<'T> * 6) "3 + 3"

        testCase "image - scalar" <| fun _ ->
          let img = makeTestImage<'T>()
          let one = LanguagePrimitives.GenericOne<'T>
          let result = img - one
          Expect.equal (result.toArray2D().[0,1]) (LanguagePrimitives.GenericOne<'T>) "2 - 1"

        testCase "scalar - image" <| fun _ ->
          let img = makeTestImage<'T>()
          let ten = LanguagePrimitives.GenericOne<'T> * (LanguagePrimitives.GenericOne<'T> * 10)
          let result = ten - img
          Expect.equal (result.toArray2D().[1,1]) (ten - (LanguagePrimitives.GenericOne<'T> * 4)) "10 - 4"

        testCase "image - image" <| fun _ ->
          let img = makeTestImage<'T>()
          let result = img - img
          Expect.equal (result.toArray2D().[1,0]) LanguagePrimitives.GenericZero<'T> "3 - 3"

        testCase "image * scalar" <| fun _ ->
          let img = makeTestImage<'T>()
          let two = LanguagePrimitives.GenericOne<'T> * 2
          let result = img * two
          Expect.equal (result.toArray2D().[0,1]) (two * LanguagePrimitives.GenericOne<'T> * 2) "2 * 2"

        testCase "scalar * image" <| fun _ ->
          let img = makeTestImage<'T>()
          let three = LanguagePrimitives.GenericOne<'T> * 3
          let result = three * img
          Expect.equal (result.toArray2D().[1,0]) (three * LanguagePrimitives.GenericOne<'T> * 3) "3 * 3"

        testCase "image * image" <| fun _ ->
          let img = makeTestImage<'T>()
          let result = img * img
          Expect.equal (result.toArray2D().[1,1]) (LanguagePrimitives.GenericOne<'T> * 16) "4 * 4"

        testCase "image / scalar" <| fun _ ->
          let img = makeTestImage<'T>()
          let two = LanguagePrimitives.GenericOne<'T> * 2
          let result = img / two
          Expect.equal (result.toArray2D().[1,1]) (LanguagePrimitives.GenericOne<'T> * 2) "4 / 2"

        testCase "scalar / image" <| fun _ ->
          let img = makeTestImage<'T>()
          let eight = LanguagePrimitives.GenericOne<'T> * 8
          let result = eight / img
          Expect.equal (result.toArray2D().[0,1]) (eight / (LanguagePrimitives.GenericOne<'T> * 2)) "8 / 2"

        testCase "image / image" <| fun _ ->
          let img = makeTestImage<'T>()
          let result = img / img
          Expect.equal (result.toArray2D().[0,0]) LanguagePrimitives.GenericOne<'T> "1 / 1"

        testCase "image % scalar" <| fun _ ->
          let img = makeTestImage<'T>()
          let two = LanguagePrimitives.GenericOne<'T> * 2
          let result = img % two
          Expect.equal (result.toArray2D().[1,0]) LanguagePrimitives.GenericOne<'T> "3 % 2"

        testCase "scalar % image" <| fun _ ->
          let img = makeTestImage<'T>()
          let five = LanguagePrimitives.GenericOne<'T> * 5
          let result = five % img
          Expect.equal (result.toArray2D().[0,1]) LanguagePrimitives.GenericOne<'T> "5 % 2"

        testCase "image % image" <| fun _ ->
          let img = makeTestImage<'T>()
          let result = img % img
          Expect.equal (result.toArray2D().[1,1]) LanguagePrimitives.GenericZero<'T> "4 % 4"
      ]

    // Run for different types
    yield! testOps<uint8> "uint8 ops"
    yield! testOps<int8> "int8 ops"
    yield! testOps<uint16> "uint16 ops"
    yield! testOps<int16> "int16 ops"
    yield! testOps<uint32> "uint32 ops"
    yield! testOps<int32> "int32 ops"
    yield! testOps<uint64> "uint64 ops"
    yield! testOps<int64> "int64 ops"
    yield! testOps<float32> "float32 ops"
    yield! testOps<float> "float64 ops"
  ]
