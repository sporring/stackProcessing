module Tests.ArithmeticSupportOps

open Expecto
open Image
open System

let isClose (im1: Image<float>) (im2: Image<float>): bool = (im1 - im2 |> ImageFunctions.absImage |> ImageFunctions.sum) < 1.0e-6

let inline testOps
        (label : string)
        (img : Image<float>)
        (scalar : float)
        ((imageOp,imageExpected) : (Image< float > -> float -> Image< float >)*Image<float>)
        ((scalarOp,scalarExpected) : (float -> Image< float > -> Image< float >)*Image<float>) =
    
    testCase label <| fun _ ->
        let imageResult = imageOp img scalar
        let scalarResult = scalarOp scalar img
        let diff1 = imageResult - imageExpected
        let absDiff1 = diff1 |> ImageFunctions.absImage
        let sum1 = absDiff1 |> ImageFunctions.sum
        let diff2 = scalarResult - imageExpected
        let absDiff2 = diff2 |> ImageFunctions.absImage
        let sum2 = absDiff2 |> ImageFunctions.sum
        Expect.isTrue (isClose imageResult imageExpected) $"{label}: imageOpScalar mismatch\nimageResult->{ImageFunctions.dump(imageResult)}\nimageExpected->{ImageFunctions.dump(imageExpected)}\ndiff->{ImageFunctions.dump(diff1)}\nabsDiff->{ImageFunctions.dump(absDiff1)}\nsum->{sum1}"
        Expect.isTrue (isClose scalarResult scalarExpected) $"{label}: scalarOpImage mismatch\nimageResult->{ImageFunctions.dump(imageResult)}\nscalarExpected->{ImageFunctions.dump(scalarExpected)})\ndiff->{ImageFunctions.dump(diff2)}\nabsDiff->{ImageFunctions.dump(absDiff2)}\nsum->{sum2}"

[<Tests>]
let tests =
    testList "ImageFunctions arithmetic tests" [
        let img = Image<float>.ofArray2D (array2D         [ [ 1.0; 2.0 ]; [ 3.0; 4.0 ] ])
        let imgPlusTwo = Image<float>.ofArray2D (array2D  [ [ 3.0; 4.0 ]; [ 5.0; 6.0 ] ])
        let imgMinusTwo = Image<float>.ofArray2D (array2D [ [-1.0; 0.0 ]; [ 1.0; 2.0 ] ])
        let twoMinusImg = Image<float>.ofArray2D (array2D [ [ 1.0; 0.0 ]; [-1.0;-2.0 ] ])
        let imgMulTwo = Image<float>.ofArray2D (array2D   [ [ 2.0; 4.0 ]; [ 6.0; 8.0 ] ])
        let imgDivTwo = Image<float>.ofArray2D (array2D   [ [ 0.5; 1.0 ]; [ 1.5; 2.0 ] ])
        let twoDivImg = Image<float>.ofArray2D (array2D   [ [ 2.0; 1.0 ]; [ 2.0/3.0; 2.0/4.0 ] ])
        testOps "Add" img 2.0      (ImageFunctions.imageAddScalar, imgPlusTwo)  (ImageFunctions.scalarAddImage, imgPlusTwo)
        testOps "Subtract" img 2.0 (ImageFunctions.imageSubScalar, imgMinusTwo) (ImageFunctions.scalarSubImage, twoMinusImg)
        testOps "Multiply" img 2.0 (ImageFunctions.imageMulScalar, imgMulTwo)   (ImageFunctions.scalarMulImage, imgMulTwo)
        testOps "Divide" img 2.0   (ImageFunctions.imageDivScalar, imgDivTwo)   (ImageFunctions.scalarDivImage, twoDivImg)

        testCase "Power (image ^ scalar)" <| fun _ ->
            let img = Image<float>.ofArray2D (array2D      [ [ 1.0; 2.0 ]; [ 3.0; 4.0 ] ])
            let expected = Image<float>.ofArray2D (array2D [ [ 1.0; 8.0 ]; [ 27.0; 64.0 ] ])
            let result = ImageFunctions.imagePowScalar (img, 3.0)
            let diff = result - expected
            let absDiff = diff |> ImageFunctions.absImage
            let sum = absDiff |> ImageFunctions.sum
            Expect.isTrue (isClose expected result) $"image ^ scalar result mismatc:\nimg->{ImageFunctions.dump(img)})\nresult->{ImageFunctions.dump(result)}\nexpected->{ImageFunctions.dump(expected)}\nabsDiff->{ImageFunctions.dump(absDiff)}\nsum->{sum}"

        testCase "Power (scalar ^ image)" <| fun _ ->
            let img = Image<float>.ofArray2D (array2D      [ [ 1.0; 2.0 ]; [ 3.0; 4.0 ] ])
            let expected = Image<float>.ofArray2D (array2D [ [ 3.0; 9.0 ]; [ 27.0; 81.0 ] ])
            let result = ImageFunctions.scalarPowImage (3.0, img)
            let diff = result - expected
            let absDiff = diff |> ImageFunctions.absImage
            let sum = absDiff |> ImageFunctions.sum
            Expect.isTrue (isClose expected result) $"scalar ^ image result mismatc: \nimg->{ImageFunctions.dump(img)}\nresult->{ImageFunctions.dump(result)}\nexpected->{ImageFunctions.dump(expected)}\nabsDiff->{ImageFunctions.dump(absDiff)}\nsum->{sum}"
    ]
