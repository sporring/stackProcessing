module Tests.StackProcessingCorrectnessTests

open System
open System.IO
open Expecto
open Image
open StackProcessing

let private tempDirectory name =
    let path = Path.Combine(Path.GetTempPath(), $"stackprocessing-{name}-{Guid.NewGuid():N}")
    Directory.CreateDirectory(path) |> ignore
    path

let private makeFloat32Volume side =
    Array3D.init side side side (fun x y z ->
        let xf = float x
        let yf = float y
        let zf = float z
        single (Math.Sin(xf * 0.11) + Math.Cos(yf * 0.07) + zf * 0.013 + xf * yf * 0.0003))
    |> Image<float32>.ofArray3D

let private makePositiveFloat32Volume side =
    Array3D.init side side side (fun x y z ->
        single ((x * 3 + y * 5 + z * 7) % 201))
    |> Image<float32>.ofArray3D

let private makeStrictPositiveFloat32Volume side =
    Array3D.init side side side (fun x y z ->
        let xf = float x
        let yf = float y
        let zf = float z
        single (1.5 + Math.Sin(xf * 0.05) * 0.2 + Math.Cos(yf * 0.07) * 0.2 + zf * 0.01))
    |> Image<float32>.ofArray3D

let private makeTrigonometricFloat32Volume side =
    Array3D.init side side side (fun x y z ->
        let xf = float x
        let yf = float y
        let zf = float z
        single (0.55 * Math.Sin(xf * 0.17) + 0.25 * Math.Cos(yf * 0.13) + 0.01 * zf))
    |> Image<float32>.ofArray3D

let private makeFloat64Volume side =
    Array3D.init side side side (fun x y z ->
        let xf = float x
        let yf = float y
        let zf = float z
        Math.Sin(xf * 0.11) + Math.Cos(yf * 0.07) + zf * 0.013 + xf * yf * 0.0003)
    |> Image<float>.ofArray3D

let private makeBinaryVolume side =
    Array3D.init side side side (fun x y z ->
        let dx = x - side / 2
        let dy = y - side / 2
        let dz = z - side / 2
        if dx * dx + dy * dy + dz * dz < (side / 4) * (side / 4) then 255uy else 0uy)
    |> Image<uint8>.ofArray3D

let private makeAveragingKernel () =
    Array3D.create 3 3 3 (single (1.0 / 27.0))
    |> Image<float32>.ofArray3D

let private makeAveragingKernel64 () =
    Array3D.create 3 3 3 (1.0 / 27.0)
    |> Image<float>.ofArray3D

let private makeBinaryVolumeWithHole side =
    Array3D.init side side side (fun x y z ->
        let dx = x - side / 2
        let dy = y - side / 2
        let dz = z - side / 2
        let outer = dx * dx + dy * dy + dz * dz < (side / 3) * (side / 3)
        let inner = dx * dx + dy * dy + dz * dz < (side / 7) * (side / 7)
        if outer && not inner then 255uy else 0uy)
    |> Image<uint8>.ofArray3D

let private writeVolumeAsSlices directory suffix (volume: Image<'T>) =
    let slices = ImageFunctions.unstack 2u volume

    try
        slices
        |> List.iteri (fun index slice ->
            let fileName = Path.Combine(directory, sprintf "image_%03d%s" index suffix)
            slice.toFile(fileName))
    finally
        slices |> List.iter (fun slice -> slice.decRefCount())

let private readVolumeFromSlices<'T when 'T: equality> directory suffix =
    let slices =
        Directory.GetFiles(directory, "*" + suffix)
        |> Array.sort
        |> Array.map Image<'T>.ofFile
        |> Array.toList

    try
        ImageFunctions.stack slices
    finally
        slices |> List.iter (fun slice -> slice.decRefCount())

let private maxAbsDifference (left: Image<'T>) (right: Image<'T>) =
    let leftSize = left.GetSize()
    let rightSize = right.GetSize()

    if leftSize <> rightSize then
        failwith $"Cannot compare images with different sizes: {leftSize} vs {rightSize}"

    let leftFloat = left.toFloat()
    let rightFloat = right.toFloat()

    try
        let mutable maxDiff = 0.0

        for x in 0 .. int leftSize.[0] - 1 do
            for y in 0 .. int leftSize.[1] - 1 do
                for z in 0 .. int leftSize.[2] - 1 do
                    let diff = Math.Abs(leftFloat.[x, y, z] - rightFloat.[x, y, z])
                    if diff > maxDiff then
                        maxDiff <- diff

        maxDiff
    finally
        leftFloat.decRefCount()
        rightFloat.decRefCount()

let private compareImages name tolerance inputDir outputDir (expected: Image<'T>) (actual: Image<'T>) =
    let actualSize = actual.GetSize()
    let expectedSize = expected.GetSize()
    let mutable keepTempDirs = false

    if actualSize <> expectedSize then
        keepTempDirs <- true

    Expect.equal actualSize expectedSize $"{name}: streaming and direct results should produce the same volume shape. Direct size: {expectedSize}; streamed size: {actualSize}. Input slices: {inputDir}; output slices: {outputDir}."

    let maxDiff = maxAbsDifference expected actual

    if maxDiff >= tolerance then
        keepTempDirs <- true

    Expect.isLessThan maxDiff tolerance $"{name}: streaming result should match direct 3D result. Max difference: {maxDiff}; tolerance: {tolerance}. Input slices: {inputDir}; output slices: {outputDir}."

    keepTempDirs

let private runSlicePipeline name suffix (input: Image<'In>) (stage: Stage<Image<'In>, Image<'Out>>) =
    let inputDir = tempDirectory $"{name}-input"
    let outputDir = tempDirectory $"{name}-output"

    writeVolumeAsSlices inputDir suffix input

    source (2UL * 1024UL * 1024UL * 1024UL)
    |> read<'In> inputDir suffix
    >=> stage
    >=> write outputDir suffix
    |> sink

    let actual = readVolumeFromSlices<'Out> outputDir suffix
    actual, inputDir, outputDir

let private cleanupResult keepTempDirs inputDir outputDir =
    if keepTempDirs then
        printfn $"[StackProcessing.Tests] keeping temp directories for inspection: {inputDir}; {outputDir}"
    else
        if Directory.Exists inputDir then Directory.Delete(inputDir, true)
        if Directory.Exists outputDir then Directory.Delete(outputDir, true)

let private assertStreamingMatchesDirect name suffix tolerance (input: Image<'In>) (stage: Stage<Image<'In>, Image<'Out>>) (direct: Image<'In> -> Image<'Out>) =
    let mutable inputDir = ""
    let mutable outputDir = ""
    let mutable keepTempDirs = false
    let mutable actualOpt : Image<'Out> option = None
    let mutable expectedOpt : Image<'Out> option = None

    try
        let expected = direct input
        let actual, iDir, oDir = runSlicePipeline name suffix input stage

        inputDir <- iDir
        outputDir <- oDir
        actualOpt <- Some actual
        expectedOpt <- Some expected

        keepTempDirs <- compareImages name tolerance inputDir outputDir expected actual
    finally
        actualOpt |> Option.iter (fun image -> image.decRefCount())
        expectedOpt |> Option.iter (fun image -> image.decRefCount())
        cleanupResult keepTempDirs inputDir outputDir

let private compareStats tolerance (expected: ImageStats) (actual: ImageStats) =
    let expectClose label (actual: float) (expected: float) =
        let diff = Math.Abs(actual - expected)
        Expect.isLessThan diff tolerance $"Streaming and direct stats should have the same {label}. Actual: {actual}; expected: {expected}; difference: {diff}; tolerance: {tolerance}."

    Expect.equal actual.NumPixels expected.NumPixels "Streaming and direct stats should count the same number of pixels."
    expectClose "mean" actual.Mean expected.Mean
    expectClose "standard deviation" actual.Std expected.Std
    expectClose "minimum" actual.Min expected.Min
    expectClose "maximum" actual.Max expected.Max
    expectClose "sum" actual.Sum expected.Sum
    expectClose "variance" actual.Var expected.Var

let private expectoTestCase = testCase

let private testCase name body =
    expectoTestCase name (fun arg ->
        try
            body arg
        finally
            GC.Collect()
            GC.WaitForPendingFinalizers()
            GC.Collect())

let stackProcessingCorrectnessSuite =
    testSequenced <| testList "StackProcessing image correctness" [
        testCase "streamed valid 3D convolution matches direct 3D SimpleITK convolution" <| fun _ ->
            let suffix = ".tiff"
            let volume = makeFloat32Volume 16
            let kernel = makeAveragingKernel ()

            try
                assertStreamingMatchesDirect
                    "convolve-valid"
                    suffix
                    1.0e-4
                    volume
                    (convolve kernel (Some ImageFunctions.Valid) None (Some 15u))
                    (fun input -> ImageFunctions.convolve (Some ImageFunctions.Valid) None input kernel)
            finally
                kernel.decRefCount()
                volume.decRefCount()

        testCase "streamed threshold matches direct 3D threshold" <| fun _ ->
            let suffix = ".tiff"
            let volume = makeFloat32Volume 12

            try
                assertStreamingMatchesDirect
                    "threshold"
                    suffix
                    0.5
                    volume
                    (threshold 0.2 2.0)
                    (ImageFunctions.threshold 0.2 2.0)
            finally
                volume.decRefCount()

        ptestCase "streamed gaussian stages match direct 3D SimpleITK gaussian convolution" <| fun _ ->
            let suffix = ".mha"
            let volume = makeFloat64Volume 8

            try
                assertStreamingMatchesDirect
                    "discrete-gaussian"
                    suffix
                    1.0e-8
                    volume
                    (discreteGaussian 0.5 None None (Some 7u))
                    (ImageFunctions.discreteGaussian 3u 0.5 (Some 3u) None None)

                assertStreamingMatchesDirect
                    "conv-gauss"
                    suffix
                    1.0e-8
                    volume
                    (convGauss 0.5)
                    (ImageFunctions.discreteGaussian 3u 0.5 (Some 3u) None None)
            finally
                volume.decRefCount()

        ptestCase "streamed conv matches direct 3D convolution" <| fun _ ->
            let suffix = ".tiff"
            let volume = makeFloat32Volume 16
            let kernel = makeAveragingKernel ()

            try
                assertStreamingMatchesDirect
                    "conv"
                    suffix
                    1.0e-8
                    volume
                    (conv kernel)
                    (fun input -> ImageFunctions.conv input kernel)
            finally
                kernel.decRefCount()
                volume.decRefCount()

        ptestCase "streamed finiteDiff matches direct 3D finite difference convolution" <| fun _ ->
            let suffix = ".mha"
            let volume = makeFloat64Volume 8

            try
                assertStreamingMatchesDirect
                    "finite-diff-z"
                    suffix
                    1.0e-8
                    volume
                    (finiteDiff 0.0 2u 1u)
                    (fun input ->
                        let finiteKernel = ImageFunctions.finiteDiffFilter3D 0.0 2u 1u
                        try
                            ImageFunctions.conv input finiteKernel
                        finally
                            finiteKernel.decRefCount())
            finally
                volume.decRefCount()

        testCase "streamed binary dilation matches direct 3D binary dilation" <| fun _ ->
            let suffix = ".tiff"
            let volume = makeBinaryVolume 14

            try
                assertStreamingMatchesDirect
                    "binary-dilate"
                    suffix
                    0.5
                    volume
                    (dilate 1u)
                    (ImageFunctions.binaryDilate 1u)
            finally
                volume.decRefCount()

        testCase "streamed binary morphology stages match direct 3D morphology" <| fun _ ->
            let suffix = ".tiff"
            let volume = makeBinaryVolume 10

            let cases : (string * Stage<Image<uint8>, Image<uint8>> * (Image<uint8> -> Image<uint8>)) list =
                [ "binary-erode", erode 1u, ImageFunctions.binaryErode 1u
                  "binary-opening", opening 1u, ImageFunctions.binaryOpening 1u
                  "binary-closing", closing 1u, ImageFunctions.binaryClosing 1u ]

            try
                for name, stage, direct in cases do
                    assertStreamingMatchesDirect name suffix 0.5 volume stage direct
            finally
                volume.decRefCount()

        testCase "streamed full-stack binary and threshold stages match direct 3D filters" <| fun _ ->
            let suffix = ".tiff"
            let binary = makeBinaryVolumeWithHole 10
            let scalar = makePositiveFloat32Volume 10

            try
                assertStreamingMatchesDirect
                    "binary-fill-holes"
                    suffix
                    0.5
                    binary
                    (binaryFillHoles 10u)
                    ImageFunctions.binaryFillHoles

                assertStreamingMatchesDirect
                    "otsu-threshold"
                    suffix
                    1.5
                    scalar
                    (otsuThreshold 10u)
                    ImageFunctions.otsuThreshold

                assertStreamingMatchesDirect
                    "moments-threshold"
                    suffix
                    1.5
                    scalar
                    (momentsThreshold 10u)
                    ImageFunctions.momentsThreshold
            finally
                binary.decRefCount()
                scalar.decRefCount()

        ptestCase "streamed watershed matches direct 3D watershed" <| fun _ ->
            let suffix = ".mha"
            let grayscale = makePositiveFloat32Volume 8

            try
                assertStreamingMatchesDirect
                    "watershed"
                    suffix
                    0.5
                    grayscale
                    (watershed 0.0 8u)
                    (ImageFunctions.watershed 0.0)
            finally
                grayscale.decRefCount()

        ptestCase "streamed signedDistanceMap matches direct 3D distance map" <| fun _ ->
            let suffix = ".mha"
            let binary = makeBinaryVolume 8

            try
                assertStreamingMatchesDirect
                    "signed-distance-map"
                    suffix
                    1.0e-8
                    binary
                    (signedDistanceMap 8u)
                    (ImageFunctions.signedDistanceMap 0uy 1uy)
            finally
                binary.decRefCount()

        testCase "streamed unary math functions match direct 3D ImageFunctions" <| fun _ ->
            let suffix = ".tiff"
            let volume = makeStrictPositiveFloat32Volume 8

            let cases : (string * Stage<Image<float32>, Image<float32>> * (Image<float32> -> Image<float32>)) list =
                [ "abs", StackProcessing.abs<float32>, ImageFunctions.absImage
                  "sqrt", StackProcessing.sqrt<float32>, ImageFunctions.sqrtImage
                  "square", StackProcessing.square<float32>, ImageFunctions.squareImage
                  "exp", StackProcessing.exp<float32>, ImageFunctions.expImage
                  "log", StackProcessing.log<float32>, ImageFunctions.logImage
                  "round", StackProcessing.round<float32>, ImageFunctions.roundImage ]

            try
                for name, stage, direct in cases do
                    assertStreamingMatchesDirect $"unary-{name}" suffix 1.0e-4 volume stage direct
            finally
                volume.decRefCount()

        testCase "streamed trigonometric inverse functions match direct 3D ImageFunctions" <| fun _ ->
            let suffix = ".tiff"
            let volume = makeTrigonometricFloat32Volume 8

            let cases : (string * Stage<Image<float32>, Image<float32>> * (Image<float32> -> Image<float32>)) list =
                [ "acos", StackProcessing.acos<float32>, ImageFunctions.acosImage
                  "asin", StackProcessing.asin<float32>, ImageFunctions.asinImage
                  "atan", StackProcessing.atan<float32>, ImageFunctions.atanImage
                  "cos", StackProcessing.cos<float32>, ImageFunctions.cosImage
                  "sin", StackProcessing.sin<float32>, ImageFunctions.sinImage
                  "tan", StackProcessing.tan<float32>, ImageFunctions.tanImage
                  "log10", StackProcessing.log10<float32>, ImageFunctions.log10Image ]

            try
                for name, stage, direct in cases do
                    assertStreamingMatchesDirect $"unary-{name}" suffix 1.0e-4 volume stage direct
            finally
                volume.decRefCount()

        testCase "streamed scalar arithmetic families match direct 3D ImageFunctions" <| fun _ ->
            let suffix = ".tiff"
            let volume = makeStrictPositiveFloat32Volume 8

            let cases : (string * Stage<Image<float32>, Image<float32>> * (Image<float32> -> Image<float32>)) list =
                [ "image-add-scalar", imageAddScalar 2.25f, fun input -> ImageFunctions.imageAddScalar input 2.25f
                  "image-sub-scalar", imageSubScalar 0.75f, fun input -> ImageFunctions.imageSubScalar input 0.75f
                  "scalar-sub-image", scalarSubImage 5.0f, fun input -> ImageFunctions.scalarSubImage 5.0f input
                  "image-mul-scalar", imageMulScalar 2.5f, fun input -> ImageFunctions.imageMulScalar input 2.5f
                  "scalar-div-image", scalarDivImage 4.0f, fun input -> ImageFunctions.scalarDivImage 4.0f input ]

            try
                for name, stage, direct in cases do
                    assertStreamingMatchesDirect name suffix 1.0e-4 volume stage direct
            finally
                volume.decRefCount()

        testCase "streamed remaining scalar arithmetic families match direct 3D ImageFunctions" <| fun _ ->
            let suffix = ".tiff"
            let volume = makeStrictPositiveFloat32Volume 8

            let cases : (string * Stage<Image<float32>, Image<float32>> * (Image<float32> -> Image<float32>)) list =
                [ "scalar-add-image", scalarAddImage 1.5f, fun input -> ImageFunctions.scalarAddImage 1.5f input
                  "scalar-mul-image", scalarMulImage 1.75f, fun input -> ImageFunctions.scalarMulImage 1.75f input
                  "image-div-scalar", imageDivScalar 2.0f, fun input -> ImageFunctions.imageDivScalar input 2.0f ]

            try
                for name, stage, direct in cases do
                    assertStreamingMatchesDirect name suffix 1.0e-4 volume stage direct
            finally
                volume.decRefCount()

        testCase "streamed addNormalNoise with zero variance matches direct 3D no-op noise" <| fun _ ->
            let suffix = ".tiff"
            let volume = makePositiveFloat32Volume 8

            try
                assertStreamingMatchesDirect
                    "add-normal-noise-zero"
                    suffix
                    1.0e-4
                    volume
                    (addNormalNoise 0.0 0.0)
                    (ImageFunctions.addNormalNoise 0.0 0.0)
            finally
                volume.decRefCount()

        testCase "streamed cast matches direct 3D cast" <| fun _ ->
            let suffix = ".tiff"
            let volume = makePositiveFloat32Volume 12

            try
                assertStreamingMatchesDirect
                    "cast-float32-uint8"
                    suffix
                    0.5
                    volume
                    (cast<float32, uint8>)
                    (fun input -> input.castTo<uint8>())
            finally
                volume.decRefCount()

        testCase "streamed permuteAxes xy swap matches direct 3D permuteAxes" <| fun _ ->
            let suffix = ".tiff"
            let volume = makePositiveFloat32Volume 8

            try
                assertStreamingMatchesDirect
                    "permute-axes-xy"
                    suffix
                    0.5
                    volume
                    (permuteAxes (1u, 0u, 2u) 4u)
                    (ImageFunctions.permuteAxes [1u; 0u; 2u])
            finally
                volume.decRefCount()

        testCase "streamed computeStats matches direct 3D computeStats" <| fun _ ->
            let inputDir = tempDirectory "compute-stats-input"
            let suffix = ".tiff"
            let volume = makeFloat32Volume 24

            try
                writeVolumeAsSlices inputDir suffix volume

                let actual =
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> read<float32> inputDir suffix
                    >=> computeStats ()
                    |> drain

                let expected = ImageFunctions.computeStats volume
                compareStats 1.0e-5 expected actual
            finally
                volume.decRefCount()
                if Directory.Exists inputDir then Directory.Delete(inputDir, true)
    ]
