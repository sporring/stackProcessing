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

let private makeBimodalFloat32Volume side =
    Array3D.init side side side (fun x _ _ ->
        if x < side / 2 then 0.0f else 10.0f)
    |> Image<float32>.ofArray3D

let private makeStrictPositiveFloat32Volume side =
    Array3D.init side side side (fun x y z ->
        let xf = float x
        let yf = float y
        let zf = float z
        single (1.5 + Math.Sin(xf * 0.05) * 0.2 + Math.Cos(yf * 0.07) * 0.2 + zf * 0.01))
    |> Image<float32>.ofArray3D

let private makeSecondStrictPositiveFloat32Volume side =
    Array3D.init side side side (fun x y z ->
        let xf = float x
        let yf = float y
        let zf = float z
        single (2.25 + Math.Cos(xf * 0.09) * 0.15 + Math.Sin(yf * 0.04) * 0.12 + zf * 0.02))
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

let private makeSeparatedBinaryComponentsVolume () =
    Array3D.init 8 8 8 (fun x y z ->
        let crossingSlabComponent =
            x >= 1 && x <= 2 &&
            y >= 1 && y <= 2 &&
            z >= 1 && z <= 4
        let laterComponent =
            x >= 5 && x <= 6 &&
            y >= 5 && y <= 6 &&
            z >= 6 && z <= 7
        if crossingSlabComponent || laterComponent then 255uy else 0uy)
    |> Image<uint8>.ofArray3D

let private makeAveragingKernel () =
    Array3D.create 3 3 3 (single (1.0 / 27.0))
    |> Image<float32>.ofArray3D

let private makeAveragingKernel64 () =
    Array3D.create 3 3 3 (1.0 / 27.0)
    |> Image<float>.ofArray3D

let private makeAveragingKernelSingletonZ () =
    Array3D.create 3 3 1 (single (1.0 / 9.0))
    |> Image<float32>.ofArray3D

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

let private runPairSlicePipeline name suffix (left: Image<'In>) (right: Image<'In>) (combine: Image<'In> -> Image<'In> -> Image<'Out>) =
    let leftDir = tempDirectory $"{name}-left-input"
    let rightDir = tempDirectory $"{name}-right-input"
    let outputDir = tempDirectory $"{name}-output"

    writeVolumeAsSlices leftDir suffix left
    writeVolumeAsSlices rightDir suffix right

    let leftPlan = source (2UL * 1024UL * 1024UL * 1024UL) |> read<'In> leftDir suffix
    let rightPlan = source (2UL * 1024UL * 1024UL * 1024UL) |> read<'In> rightDir suffix

    zip leftPlan rightPlan
    >>=> combine
    >=> write outputDir suffix
    |> sink

    let actual = readVolumeFromSlices<'Out> outputDir suffix
    actual, leftDir, rightDir, outputDir

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

let private assertStreamingStagesMatch name suffix tolerance (input: Image<'In>) (expectedStage: Stage<Image<'In>, Image<'Out>>) (actualStage: Stage<Image<'In>, Image<'Out>>) =
    let mutable expectedInputDir = ""
    let mutable expectedOutputDir = ""
    let mutable actualInputDir = ""
    let mutable actualOutputDir = ""
    let mutable keepTempDirs = false
    let mutable expectedOpt : Image<'Out> option = None
    let mutable actualOpt : Image<'Out> option = None

    try
        let expected, eInputDir, eOutputDir = runSlicePipeline $"{name}-expected" suffix input expectedStage
        let actual, aInputDir, aOutputDir = runSlicePipeline $"{name}-actual" suffix input actualStage

        expectedInputDir <- eInputDir
        expectedOutputDir <- eOutputDir
        actualInputDir <- aInputDir
        actualOutputDir <- aOutputDir
        expectedOpt <- Some expected
        actualOpt <- Some actual

        let expectedSize = expected.GetSize()
        let actualSize = actual.GetSize()

        if expectedSize <> actualSize then
            keepTempDirs <- true

        Expect.equal actualSize expectedSize $"{name}: compared streaming stages should produce the same volume shape."

        let maxDiff = maxAbsDifference expected actual

        if maxDiff >= tolerance then
            keepTempDirs <- true

        Expect.isLessThan maxDiff tolerance $"{name}: compared streaming stages should produce the same pixels. Max difference: {maxDiff}."
    finally
        expectedOpt |> Option.iter (fun image -> image.decRefCount())
        actualOpt |> Option.iter (fun image -> image.decRefCount())

        cleanupResult keepTempDirs expectedInputDir expectedOutputDir
        cleanupResult keepTempDirs actualInputDir actualOutputDir

let private assertPairStreamingMatchesDirect name suffix tolerance (left: Image<'In>) (right: Image<'In>) (combine: Image<'In> -> Image<'In> -> Image<'Out>) (direct: Image<'In> -> Image<'In> -> Image<'Out>) =
    let mutable leftDir = ""
    let mutable rightDir = ""
    let mutable outputDir = ""
    let mutable keepTempDirs = false
    let mutable actualOpt : Image<'Out> option = None
    let mutable expectedOpt : Image<'Out> option = None

    try
        let expected = direct left right
        let actual, lDir, rDir, oDir = runPairSlicePipeline name suffix left right combine

        leftDir <- lDir
        rightDir <- rDir
        outputDir <- oDir
        actualOpt <- Some actual
        expectedOpt <- Some expected

        keepTempDirs <- compareImages name tolerance $"{leftDir}; {rightDir}" outputDir expected actual
    finally
        actualOpt |> Option.iter (fun image -> image.decRefCount())
        expectedOpt |> Option.iter (fun image -> image.decRefCount())
        cleanupResult keepTempDirs leftDir outputDir
        if keepTempDirs then
            if rightDir <> "" then printfn $"[StackProcessing.Tests] keeping temp directory for inspection: {rightDir}"
        elif Directory.Exists rightDir then
            Directory.Delete(rightDir, true)

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

let private compareComponentStatistics tolerance (expected: ImageFunctions.LabelShapeStatistics list) (actual: ComponentStatistics list) =
    let expected =
        expected
        |> List.map (fun stats ->
            let boundingBox = stats.BoundingBox
            let minX = boundingBox[0]
            let minY = boundingBox[1]
            let minZ = boundingBox[2]
            let maxX = minX + boundingBox[3] - 1u
            let maxY = minY + boundingBox[4] - 1u
            let maxZ = minZ + boundingBox[5] - 1u
            stats.NumberOfPixels, stats.Centroid, (minX, maxX, minY, maxY, minZ, maxZ))
        |> List.sortBy (fun (_, centroid, _) -> centroid)

    let actual =
        actual
        |> List.map (fun stats ->
            let centroid =
                let n = float stats.NumberOfPixels
                [ float stats.SumX / n
                  float stats.SumY / n
                  float stats.SumZ / n ]
            stats.NumberOfPixels, centroid, (stats.MinX, stats.MaxX, stats.MinY, stats.MaxY, stats.MinZ, stats.MaxZ))
        |> List.sortBy (fun (_, centroid, _) -> centroid)

    Expect.equal actual.Length expected.Length "Streaming and direct connected-component statistics should find the same number of components."

    List.zip expected actual
    |> List.iteri (fun index ((expectedPixels, expectedCentroid, expectedBounds), (actualPixels, actualCentroid, actualBounds)) ->
        Expect.equal actualPixels expectedPixels $"Component {index} should have the same voxel count."
        Expect.equal actualBounds expectedBounds $"Component {index} should have the same bounding box."
        List.zip expectedCentroid actualCentroid
        |> List.iteri (fun axis (expectedValue, actualValue) ->
            let diff = Math.Abs(expectedValue - actualValue)
            Expect.isLessThan diff tolerance $"Component {index} centroid axis {axis} should match direct 3D SimpleITK statistics. Actual: {actualValue}; expected: {expectedValue}; difference: {diff}; tolerance: {tolerance}."))

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

        testCase "streamed singleton-z 3D convolution matches direct 3D SimpleITK convolution" <| fun _ ->
            let suffix = ".tiff"
            let volume = makeFloat32Volume 12
            let kernel = makeAveragingKernelSingletonZ ()

            try
                assertStreamingMatchesDirect
                    "convolve-singleton-z-valid"
                    suffix
                    1.0e-4
                    volume
                    (convolve kernel (Some ImageFunctions.Valid) None None)
                    (fun input -> ImageFunctions.convolve (Some ImageFunctions.Valid) None input kernel)
            finally
                kernel.decRefCount()
                volume.decRefCount()

        testCase "streamed valid 3D convolution handles partial final window" <| fun _ ->
            let suffix = ".tiff"
            let volume = makeFloat32Volume 12
            let kernel = makeAveragingKernel ()

            try
                assertStreamingMatchesDirect
                    "convolve-valid-partial-final-window"
                    suffix
                    1.0e-4
                    volume
                    (convolve kernel (Some ImageFunctions.Valid) None (Some 5u))
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

        testCase "windowed threshold via slab matches regular streamed threshold" <| fun _ ->
            let suffix = ".tiff"
            let volume = makeFloat32Volume 12

            try
                assertStreamingStagesMatch
                    "windowed-threshold"
                    suffix
                    0.5
                    volume
                    (threshold 0.2 2.0)
                    (windowedThreshold 5u 0.2 2.0)
            finally
                volume.decRefCount()

        testCase "streamed valid smoothWGauss matches direct 3D Gaussian smoothing" <| fun _ ->
            let suffix = ".mha"
            let volume = makeFloat64Volume 8

            try
                assertStreamingMatchesDirect
                    "smooth-gauss-valid"
                    suffix
                    1.0e-8
                    volume
                    (smoothWGauss 0.5 (Some ImageFunctions.Valid) None (Some 7u))
                    (ImageFunctions.discreteGaussian 3u 0.5 (Some 3u) (Some ImageFunctions.Valid) None)
            finally
                volume.decRefCount()

        testCase "streamed conv matches direct 3D convolution" <| fun _ ->
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

        testCase "streamed finiteDiff matches direct 3D finite difference convolution" <| fun _ ->
            let suffix = ".mha"
            let volume = makeFloat64Volume 8

            try
                assertStreamingMatchesDirect
                    "finite-diff-z"
                    suffix
                    1.0e-8
                    volume
                    (finiteDiff 2u 1u)
                    (fun input ->
                        let finiteKernel = ImageFunctions.finiteDiffFilter3D 2u 1u
                        try
                            ImageFunctions.conv input finiteKernel
                        finally
                            finiteKernel.decRefCount())
            finally
                volume.decRefCount()

        testCase "streamed finiteDiff with xy direction keeps singleton-z kernels slice-local" <| fun _ ->
            let suffix = ".mha"
            let volume = makeFloat64Volume 8

            try
                assertStreamingMatchesDirect
                    "finite-diff-x"
                    suffix
                    1.0e-8
                    volume
                    (finiteDiff 0u 2u)
                    (fun input ->
                        let finiteKernel = ImageFunctions.finiteDiffFilter3D 0u 2u
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

        testCase "streamed zonohedral dilation matches direct zonohedral dilation" <| fun _ ->
            let suffix = ".tiff"
            let volume =
                Array3D.init 14 14 14 (fun x y z ->
                    let dx = x - 7
                    let dy = y - 7
                    let dz = z - 7
                    if dx * dx + dy * dy + dz * dz < 9 then 1uy else 0uy)
                |> Image<uint8>.ofArray3D

            try
                assertStreamingMatchesDirect
                    "binary-dilate-zonohedral"
                    suffix
                    0.5
                    volume
                    (dilateZonohedral 2u None)
                    (ImageFunctions.binaryDilateZonohedralNative 2u)
            finally
                volume.decRefCount()

        testCase "streamed zonohedral morphology stages match direct zonohedral morphology" <| fun _ ->
            let suffix = ".tiff"
            let volume =
                Array3D.init 14 14 14 (fun x y z ->
                    let dx = x - 7
                    let dy = y - 7
                    let dz = z - 7
                    if dx * dx + dy * dy + dz * dz < 16 then 1uy else 0uy)
                |> Image<uint8>.ofArray3D

            let directOpening (image: Image<uint8>) =
                let eroded = ImageFunctions.binaryErodeZonohedralNative 2u image
                try
                    ImageFunctions.binaryDilateZonohedralNative 2u eroded
                finally
                    eroded.decRefCount()

            let cases : (string * Stage<Image<uint8>, Image<uint8>> * (Image<uint8> -> Image<uint8>)) list =
                [ "binary-erode-zonohedral", erodeZonohedral 2u None, ImageFunctions.binaryErodeZonohedralNative 2u
                  "binary-opening-zonohedral", openingZonohedral 2u None, directOpening ]

            try
                for name, stage, direct in cases do
                    assertStreamingMatchesDirect name suffix 0.5 volume stage direct
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

        testCase "sampled streamed Otsu threshold separates a bimodal stack" <| fun _ ->
            let suffix = ".tiff"
            let inputDir = tempDirectory "otsu-threshold-input"
            let outputDir = tempDirectory "otsu-threshold-output"
            let volume = makeBimodalFloat32Volume 8
            let mutable actualOpt : Image<uint8> option = None
            let mutable expectedOpt : Image<uint8> option = None
            let mutable keepTempDirs = false

            try
                writeVolumeAsSlices inputDir suffix volume

                let thresholdValue =
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> readRandom<float32> 4u inputDir suffix
                    >=> imHistogram ()
                    |> drain
                    |> otsuThresholdFromHistogram

                source (2UL * 1024UL * 1024UL * 1024UL)
                |> read<float32> inputDir suffix
                >=> threshold thresholdValue infinity
                >=> write outputDir suffix
                |> sink

                let actual = readVolumeFromSlices<uint8> outputDir suffix
                let expected = ImageFunctions.threshold 5.0 infinity volume
                actualOpt <- Some actual
                expectedOpt <- Some expected

                keepTempDirs <- compareImages "otsu-threshold" 0.5 inputDir outputDir expected actual
            finally
                actualOpt |> Option.iter (fun image -> image.decRefCount())
                expectedOpt |> Option.iter (fun image -> image.decRefCount())
                volume.decRefCount()
                cleanupResult keepTempDirs inputDir outputDir

        testCase "sampled streamed moments threshold separates a bimodal stack" <| fun _ ->
            let suffix = ".tiff"
            let inputDir = tempDirectory "moments-threshold-input"
            let outputDir = tempDirectory "moments-threshold-output"
            let volume = makeBimodalFloat32Volume 8
            let mutable actualOpt : Image<uint8> option = None
            let mutable expectedOpt : Image<uint8> option = None
            let mutable keepTempDirs = false

            try
                writeVolumeAsSlices inputDir suffix volume

                let thresholdValue =
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> readRandom<float32> 4u inputDir suffix
                    >=> imHistogram ()
                    |> drain
                    |> momentsThresholdFromHistogram

                source (2UL * 1024UL * 1024UL * 1024UL)
                |> read<float32> inputDir suffix
                >=> threshold thresholdValue infinity
                >=> write outputDir suffix
                |> sink

                let actual = readVolumeFromSlices<uint8> outputDir suffix
                let expected = ImageFunctions.threshold 2.5 infinity volume
                actualOpt <- Some actual
                expectedOpt <- Some expected

                keepTempDirs <- compareImages "moments-threshold" 0.5 inputDir outputDir expected actual
            finally
                actualOpt |> Option.iter (fun image -> image.decRefCount())
                expectedOpt |> Option.iter (fun image -> image.decRefCount())
                volume.decRefCount()
                cleanupResult keepTempDirs inputDir outputDir

        testCase "streamed signedDistanceBand matches sampled direct 3D distance map values inside the finite band" <| fun _ ->
            let suffix = ".mha"
            let binary = makeBinaryVolume 8
            let mutable inputDir = ""
            let mutable outputDir = ""
            let mutable actualOpt : Image<float> option = None
            let mutable expectedOpt : Image<float> option = None

            try
                let actual, iDir, oDir = runSlicePipeline "signed-distance-band" suffix binary (signedDistanceBand 8u 8u)
                let expected = ImageFunctions.bandSignedDistanceMap 8u binary
                inputDir <- iDir
                outputDir <- oDir
                actualOpt <- Some actual
                expectedOpt <- Some expected

                Expect.equal (actual.GetSize()) (binary.GetSize()) $"signedDistanceBand should preserve the input stack shape. Input slices: {inputDir}; output slices: {outputDir}."

                let sampledPoints =
                    [ 4, 4, 4
                      5, 4, 4
                      6, 4, 4
                      4, 6, 4
                      4, 4, 6
                      3, 4, 4
                      4, 3, 4
                      4, 4, 3 ]

                for x, y, z in sampledPoints do
                    let actualValue = actual.[x, y, z]
                    let expectedValue = expected.[x, y, z]
                    Expect.isFalse (Double.IsNaN actualValue) $"Sampled streamed distance at ({x},{y},{z}) should be inside the finite band."
                    Expect.isFalse (Double.IsNaN expectedValue) $"Sampled direct distance at ({x},{y},{z}) should be inside the finite band."
                    let diff = Math.Abs(actualValue - expectedValue)
                    Expect.isLessThan diff 1.0e-8 $"Sampled signed distance at ({x},{y},{z}) should match direct band distance. Actual: {actualValue}; expected: {expectedValue}."
            finally
                actualOpt |> Option.iter (fun image -> image.decRefCount())
                expectedOpt |> Option.iter (fun image -> image.decRefCount())
                binary.decRefCount()
                cleanupResult false inputDir outputDir

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

                assertStreamingMatchesDirect
                    "unary-sqrt-windowed"
                    suffix
                    1.0e-4
                    volume
                    (sqrtWindowed<float32> 3u)
                    ImageFunctions.sqrtImage
            finally
                volume.decRefCount()

        testCase "streamed pair arithmetic and extrema match direct 3D Image operators" <| fun _ ->
            let suffix = ".tiff"
            let left = makeStrictPositiveFloat32Volume 8
            let right = makeSecondStrictPositiveFloat32Volume 8

            let cases : (string * (Image<float32> -> Image<float32> -> Image<float32>) * (Image<float32> -> Image<float32> -> Image<float32>)) list =
                [ "add-pair", addPair, fun a b -> a + b
                  "sub-pair", subPair, fun a b -> a - b
                  "mul-pair", mulPair, fun a b -> a * b
                  "div-pair", divPair, fun a b -> a / b
                  "max-of-pair", maxOfPair, Image.maximumImage
                  "min-of-pair", minOfPair, Image.minimumImage ]

            try
                for name, combine, direct in cases do
                    assertPairStreamingMatchesDirect name suffix 1.0e-4 left right combine direct
            finally
                left.decRefCount()
                right.decRefCount()

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

        testCase "streamed permuteAxes yz swap matches direct 3D permuteAxes" <| fun _ ->
            let suffix = ".tiff"
            let volume = makePositiveFloat32Volume 8

            try
                assertStreamingMatchesDirect
                    "permute-axes-yz"
                    suffix
                    0.5
                    volume
                    (permuteAxes (0u, 2u, 1u) 4u)
                    (ImageFunctions.permuteAxes [0u; 2u; 1u])
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

        testCase "streamed histogram matches direct 3D histogram" <| fun _ ->
            let inputDir = tempDirectory "histogram-input"
            let suffix = ".tiff"
            let volume = makeBinaryVolume 12

            try
                writeVolumeAsSlices inputDir suffix volume

                let actual =
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> read<uint8> inputDir suffix
                    >=> imHistogram ()
                    |> drain

                let expected = ImageFunctions.histogram volume
                Expect.equal actual.Counts expected "Streaming histogram should match direct 3D histogram."
            finally
                volume.decRefCount()
                if Directory.Exists inputDir then Directory.Delete(inputDir, true)

        testCase "streamed connectedComponents matches direct 3D connected components for one full window" <| fun _ ->
            let inputDir = tempDirectory "connected-components-input"
            let suffix = ".tiff"
            let volume = makeBinaryVolume 10
            let mutable labelsOpt : Image<uint64> option = None

            try
                writeVolumeAsSlices inputDir suffix volume

                let actual =
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> read<uint8> inputDir suffix
                    >=> connectedComponents (Some 10u)
                    |> drainList

                let expected = ImageFunctions.connectedComponents volume
                labelsOpt <- Some expected.Labels

                try
                    Expect.equal actual.Length 1 "Full-window connectedComponents should emit one label block."
                    let actualLabels, actualObjectCount = actual.Head
                    Expect.equal actualObjectCount expected.ObjectCount "Streaming and direct connectedComponents should report the same object count."
                    let keepTempDirs = compareImages "connected-components-labels" 0.5 inputDir "" expected.Labels actualLabels
                    if keepTempDirs then
                        printfn $"[StackProcessing.Tests] keeping temp directory for inspection: {inputDir}"
                    actualLabels.decRefCount()
                finally
                    actual.Tail |> List.iter (fun (image, _) -> image.decRefCount())
            finally
                labelsOpt |> Option.iter (fun image -> image.decRefCount())
                volume.decRefCount()
                if Directory.Exists inputDir then Directory.Delete(inputDir, true)

        testCase "streamed connected component statistics match direct 3D SimpleITK label shape statistics" <| fun _ ->
            let inputDir = tempDirectory "connected-component-stats-input"
            let suffix = ".tiff"
            let volume = makeSeparatedBinaryComponentsVolume ()
            let mutable expectedLabelsOpt : Image<uint64> option = None

            try
                writeVolumeAsSlices inputDir suffix volume

                let actual =
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> read<uint8> inputDir suffix
                    >=> connectedComponents (Some 3u)
                    >=> makeConnectedComponentTranslationTable (Some 3u)
                    |> drain

                let expectedLabels = (ImageFunctions.connectedComponents volume).Labels
                expectedLabelsOpt <- Some expectedLabels
                let expected =
                    ImageFunctions.labelShapeStatistics expectedLabels
                    |> Map.toList
                    |> List.map snd

                compareComponentStatistics 1.0e-6 expected actual.Statistics
            finally
                expectedLabelsOpt |> Option.iter (fun image -> image.decRefCount())
                volume.decRefCount()
                if Directory.Exists inputDir then Directory.Delete(inputDir, true)

        testCase "streamed relabelComponents matches direct 3D relabeling for one full window" <| fun _ ->
            let suffix = ".mha"
            let binary = makeBinaryVolume 10
            let mutable labelsOpt : Image<uint64> option = None

            try
                let labels = (ImageFunctions.connectedComponents binary).Labels
                labelsOpt <- Some labels

                assertStreamingMatchesDirect
                    "relabel-components"
                    suffix
                    0.5
                    labels
                    (relabelComponents 1u (Some 10u))
                    (ImageFunctions.relabelComponents 1u)
            finally
                labelsOpt |> Option.iter (fun image -> image.decRefCount())
                binary.decRefCount()
    ]
