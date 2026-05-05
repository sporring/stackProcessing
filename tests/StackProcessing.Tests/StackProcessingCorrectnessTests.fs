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

[<Tests>]
let stackProcessingCorrectnessSuite =
    testList "StackProcessing image correctness" [
        testCase "streamed valid 3D convolution matches direct 3D SimpleITK convolution" <| fun _ ->
            let suffix = ".tiff"
            let volume = makeFloat32Volume 32
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
            let volume = makeFloat32Volume 28

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

        testCase "streamed image-scalar multiplication matches direct 3D multiplication" <| fun _ ->
            let suffix = ".tiff"
            let volume = makeFloat32Volume 28

            try
                assertStreamingMatchesDirect
                    "image-mul-scalar"
                    suffix
                    1.0e-5
                    volume
                    (imageMulScalar 2.5f)
                    (fun input -> ImageFunctions.imageMulScalar input 2.5f)
            finally
                volume.decRefCount()

        testCase "streamed cast matches direct 3D cast" <| fun _ ->
            let suffix = ".tiff"
            let volume = makePositiveFloat32Volume 24

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

        testCase "streamed binary dilation matches direct 3D binary dilation" <| fun _ ->
            let suffix = ".tiff"
            let volume = makeBinaryVolume 28

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
