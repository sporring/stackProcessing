module Tests.ArithmeticOps

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

let inline testOps<^T when ^T: equality
                and  ^T: (static member op_Explicit: ^T -> float)
                and  ^T : (static member Zero : ^T)
                and  ^T : (static member One  : ^T)
                and  ^T : (static member ( + ) : ^T * ^T -> ^T)
                and  ^T : (static member ( - ) : ^T * ^T -> ^T)
                and  ^T : (static member ( * ) : ^T * ^T -> ^T)
                and  ^T : (static member ( / ) : ^T * ^T -> ^T)
            >
  (name: string) =
    let zero  : ^T = LanguagePrimitives.GenericZero
    let one   : ^T = LanguagePrimitives.GenericOne
    let two   : ^T = one + one
    let three : ^T = two + one
    let four  : ^T = three + one
    let five  : ^T = four + one
    let six   : ^T = five + one
    let seven : ^T = six + one
    let eight : ^T = seven + one
    let nine : ^T  = eight + one
    let sixteen : ^T  = nine + seven
    let twentyfive : ^T  = sixteen + nine

    testList name [
      testCase "image + scalar" <| fun _ ->
        let img = makeTestImage<'T>()
        let result = ImageFunctions.imageAddScalar<^T> img one
        Expect.equal (result.toArray2D().[0,0]) (one + one) "1 + 1"

      testCase "scalar + image" <| fun _ ->
        let img = makeTestImage<'T>()
        let result = ImageFunctions.scalarAddImage three img
        Expect.equal (result.toArray2D().[1,1]) seven "3 + 4"

      testCase "image + image" <| fun _ ->
        let img1 = makeTestImage<'T>()
        let img2 = makeTestImage<'T>()
        let result = img1 + img2
        Expect.equal (result.toArray2D().[1,0]) six "3 + 3"

      testCase "image - scalar" <| fun _ ->
        let img = makeTestImage<'T>()
        let one = LanguagePrimitives.GenericOne<'T>
        let result = ImageFunctions.imageSubScalar img one
        Expect.equal (result.toArray2D().[0,1]) one "2 - 1"

      testCase "scalar - image" <| fun _ ->
        let img = makeTestImage<'T>()
        let result = ImageFunctions.scalarSubImage six img
        Expect.equal (result.toArray2D().[1,1]) two "6 - 4"

      testCase "image - image" <| fun _ ->
        let img = makeTestImage<'T>()
        let result = img - img
        Expect.equal (result.toArray2D().[1,0]) zero "3 - 3"

      testCase "image * scalar" <| fun _ ->
        let img = makeTestImage<'T>()
        let result = ImageFunctions.imageMulScalar img two
        Expect.equal (result.toArray2D().[0,1]) four "2 * 2"

      testCase "scalar * image" <| fun _ ->
        let img = makeTestImage<'T>()
        let result = ImageFunctions.scalarMulImage three img
        Expect.equal (result.toArray2D().[0,1]) six "2 * 3"

      testCase "image * image" <| fun _ ->
        let img = makeTestImage<'T>()
        let result = img * img
        Expect.equal (result.toArray2D().[0,1]) four "2 * 2"

      testCase "image / scalar" <| fun _ ->
        let img = makeTestImage<'T>()
        let result = ImageFunctions.imageDivScalar img two
        Expect.equal (result.toArray2D().[1,1]) two "4 / 2"

      testCase "scalar / image" <| fun _ ->
        let img = makeTestImage<'T>()
        let result = ImageFunctions.scalarDivImage eight img
        Expect.equal (result.toArray2D().[0,1]) four "8 / 2"

      testCase "image / image" <| fun _ ->
        let img = makeTestImage<'T>()
        let result = img / img
        Expect.equal (result.toArray2D().[0,0]) one "1 / 1"

      testCase "Image ^ scalar" <| fun _ ->
          let baseImg = Image<'T>.ofArray2D (array2D [ [ one; three ]; [ four; five ] ])
          let result = ImageFunctions.imagePowScalar(baseImg, two)
          let expected = array2D [ [ one; nine ]; [ sixteen; twentyfive ] ]
          Expect.equal (result.toArray2D()) expected "Expected image ^ scalar result"

      testCase "scalar ^ Image" <| fun _ ->
          let baseImg = Image<'T>.ofArray2D (array2D [ [ one; two ]; [ three; four ] ])
          let result = ImageFunctions.scalarPowImage(two, baseImg)
          let expected = array2D [ [ two; four ]; [ eight; sixteen ] ]
          Expect.equal (result.toArray2D()) expected "Expected image ^ scalar result"

      testCase "Image ^ image" <| fun _ ->
          let baseImg = Image<'T>.ofArray2D (array2D [ [ one; three ]; [ four; five ] ])
          let exponentImg = Image<'T>.ofArray2D (array2D [ [ three; two ]; [ two; one ] ])
          let result = Image<'T>.Pow(baseImg, exponentImg)
          let expected = array2D [ [ one; nine ]; [ sixteen; five ] ]
          Expect.equal (result.toArray2D()) expected "Expected image ^ image result"

    ]

let isClose (im1: Image<float>) (im2: Image<float>): bool = (im1 - im2 |> ImageFunctions.absImage |> ImageFunctions.sum) < 1.0e-6
let errorMsg img1 img2 result expected absDiff sum =
  $"img1->{ImageFunctions.dump(img1)})\nimg2->{ImageFunctions.dump(img2)}\nresult->{ImageFunctions.dump(result)}\nexpected->{ImageFunctions.dump(expected)}\nabsDiff->{ImageFunctions.dump(absDiff)}\nsum->{sum}"


[<Tests>]
let arithmeticTests =
  testList "Arithmetic operations on Image<'T>" [
    // Run for different types
    testOps<uint8> "uint8 ops"
    testOps<int8> "int8 ops"
    testOps<uint16> "uint16 ops"
    testOps<int16> "int16 ops"
    testOps<uint32> "uint32 ops"
    testOps<int32> "int32 ops"
    testOps<uint64> "uint64 ops"
    testOps<int64> "int64 ops"
    testOps<float32> "float32 ops"
    testOps<float> "float64 ops"

    testCase "signed image + image" <| fun _ ->
      let img1 =     Image<float>.ofArray2D (array2D [ [-1.0; 2.0]; [-3.0; 4.0] ])
      let img2 =     Image<float>.ofArray2D (array2D [ [ 2.0; 3.0]; [ 4.0; 1.0] ])
      let expected = Image<float>.ofArray2D (array2D [ [ 1.0; 5.0]; [ 1.0; 5.0] ])
      let result = img1 + img2
      let diff = result - expected
      let absDiff = diff |> ImageFunctions.absImage
      let sum = absDiff |> ImageFunctions.sum
      Expect.isTrue (isClose expected result) $"float negative and positive:\n{errorMsg img1 img2 result expected absDiff sum}"
    testCase "signed image - image" <| fun _ ->
      let img1 =     Image<float>.ofArray2D (array2D [ [-1.0; 2.0]; [-3.0; 4.0] ])
      let img2 =     Image<float>.ofArray2D (array2D [ [ 2.0; 3.0]; [ 4.0; 1.0] ])
      let expected = Image<float>.ofArray2D (array2D [ [-3.0;-1.0]; [-7.0; 3.0] ])
      let result = img1 - img2
      let diff = result - expected
      let absDiff = diff |> ImageFunctions.absImage
      let sum = absDiff |> ImageFunctions.sum
      Expect.isTrue (isClose expected result) $"float negative and positive:\n{errorMsg img1 img2 result expected absDiff sum}"
    testCase "signed image * image" <| fun _ ->
      let img1 =     Image<float>.ofArray2D (array2D [ [-1.0; 2.0]; [-3.0; 4.0] ])
      let img2 =     Image<float>.ofArray2D (array2D [ [ 2.0; 3.0]; [ 4.0; 1.0] ])
      let expected = Image<float>.ofArray2D (array2D [ [-2.0; 6.0]; [-12.0; 4.0] ])
      let result = img1 * img2
      let diff = result - expected
      let absDiff = diff |> ImageFunctions.absImage
      let sum = absDiff |> ImageFunctions.sum
      Expect.isTrue (isClose expected result) $"float negative and positive:\n{errorMsg img1 img2 result expected absDiff sum}"
    testCase "signed image / image" <| fun _ ->
      let img1 =     Image<float>.ofArray2D (array2D [ [-1.0; 2.0]; [-3.0; 4.0] ])
      let img2 =     Image<float>.ofArray2D (array2D [ [ 2.0; 3.0]; [ 4.0; 1.0] ])
      let expected = Image<float>.ofArray2D (array2D [ [-0.5; 2.0/3.0]; [-3.0/4.0; 4.0] ])
      let result = img1 / img2
      let diff = result - expected
      let absDiff = diff |> ImageFunctions.absImage
      let sum = absDiff |> ImageFunctions.sum
      Expect.isTrue (isClose expected result) $"float negative and positive:\n{errorMsg img1 img2 result expected absDiff sum}"
  ]
