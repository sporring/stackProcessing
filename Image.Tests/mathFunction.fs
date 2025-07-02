module Tests.mathFunction

open Expecto
open System
open Image

let inline makeImage<'T when 'T : equality> (values: 'T list) : Image<'T> =
    let img = Image<'T>([ uint32 values.Length; 1u ])
    values |> List.iteri (fun i v -> img.Set [uint32 i; 0u] v)
    img

let inline expectImageEqual (expected: Image<'T>) (actual: Image<'T>) (label: string) =
    let zipped = Image.zip [expected; actual]
    let eq = zipped.forAll (fun vList -> vList[0] = vList[0])
    Expect.isTrue eq label

[<Tests>]
let mathFunctionTests =
    testList "ImageFunctions: Unary operations" [

        // absImage
        testCase "absImage float32" <| fun _ ->
            let img = makeImage<float32> [ -1.5f; 0.0f; 2.5f ]
            let result = ImageFunctions.absImage img
            let expected = makeImage [ 1.5f; 0.0f; 2.5f ]
            expectImageEqual expected result "absImage float32"

        testCase "absImage float" <| fun _ ->
            let img = makeImage<float> [ -1.5; 0.0; 2.5 ]
            let result = ImageFunctions.absImage img
            let expected = makeImage [ 1.5; 0.0; 2.5 ]
            expectImageEqual expected result "absImage float"

        testCase "absImage int32" <| fun _ ->
            let img = makeImage<int32> [ -2; 0; 3 ]
            let result = ImageFunctions.absImage img
            let expected = makeImage [ 2; 0; 3 ]
            expectImageEqual expected result "absImage int"

        // logImage
        testCase "logImage float" <| fun _ ->
            let img = makeImage<float> [ 1.0; Math.E ]
            let result = ImageFunctions.logImage img
            let expected = makeImage [ 0.0; 1.0 ]
            expectImageEqual expected result "logImage float"

        // log10Image
        testCase "log10Image float32" <| fun _ ->
            let img = makeImage<float32> [ 1.0f; 100.0f ]
            let result = ImageFunctions.log10Image img
            let expected = makeImage [ 0.0f; 2.0f ]
            expectImageEqual expected result "log10Image float32"

        // expImage
        testCase "expImage float" <| fun _ ->
            let img = makeImage<float> [ 0.0; 1.0 ]
            let result = ImageFunctions.expImage img
            let expected = makeImage [ 1.0; Math.E ]
            expectImageEqual expected result "expImage float"

        // sqrtImage
        testCase "sqrtImage float" <| fun _ ->
            let img = makeImage<float> [ 0.0; 4.0 ]
            let result = ImageFunctions.sqrtImage img
            let expected = makeImage [ 0.0; 2.0 ]
            expectImageEqual expected result "sqrtImage float"

        // squareImage
        testCase "squareImage int32" <| fun _ ->
            let img = makeImage<int32> [ -2; 3 ]
            let result = ImageFunctions.squareImage img
            let expected = makeImage [ 4; 9 ]
            expectImageEqual expected result "squareImage int32"

        // trig
        testCase "sinImage float" <| fun _ ->
            let img = makeImage [ 0.0; Math.PI / 2.0 ]
            let result = ImageFunctions.sinImage img
            let expected = makeImage [ 0.0; 1.0 ]
            expectImageEqual expected result "sinImage float"

        testCase "cosImage float32" <| fun _ ->
            let img = makeImage [ 0.0f; MathF.PI ]
            let result = ImageFunctions.cosImage img
            let expected = makeImage [ 1.0f; -1.0f ]
            expectImageEqual expected result "cosImage float32"

        testCase "tanImage float" <| fun _ ->
            let img = makeImage [ 0.0; 1.0 ]
            let result = ImageFunctions.tanImage img
            let expected = makeImage [ Math.Tan 0.0; Math.Tan 1.0 ]
            expectImageEqual expected result "tanImage float"

        testCase "asinImage float" <| fun _ ->
            let img = makeImage [ 0.0; 1.0 ]
            let result = ImageFunctions.asinImage img
            let expected = makeImage [ 0.0; Math.Asin 1.0 ]
            expectImageEqual expected result "asinImage float"

        testCase "acosImage float" <| fun _ ->
            let img = makeImage [ 1.0; 0.0 ]
            let result = ImageFunctions.acosImage img
            let expected = makeImage [ 0.0; Math.PI / 2.0 ]
            expectImageEqual expected result "acosImage float"

        testCase "atanImage float" <| fun _ ->
            let img = makeImage [ 0.0; 1.0 ]
            let result = ImageFunctions.atanImage img
            let expected = makeImage [ 0.0; Math.Atan 1.0 ]
            expectImageEqual expected result "atanImage float"

        testCase "roundImage float32" <| fun _ ->
            let img = makeImage [ 1.4f; 2.6f ]
            let result = ImageFunctions.roundImage img
            let expected = makeImage [ 1.0f; 3.0f ]
            expectImageEqual expected result "roundImage float32"
    ]
