module Tests.StackProcessingSupportTests

open System
open System.IO
open Expecto
open Image
open PureHDF
open SlimPipeline
open StackProcessing
open StackProcessingCost
open TinyLinAlg

let private tempDirectory name =
    let path = Path.Combine(Path.GetTempPath(), $"stackprocessing-{name}-{Guid.NewGuid():N}")
    Directory.CreateDirectory(path) |> ignore
    path

let private deleteDirectory path =
    if Directory.Exists path then
        Directory.Delete(path, true)

let private makeSlice width height z =
    Array2D.init width height (fun x y -> uint8 ((x + 2 * y + 3 * z) % 251))
    |> Image<uint8>.ofArray2D

let private makeBinarySlice width height z =
    Array2D.init width height (fun x y ->
        let inFirst = x >= 1 && x <= 2 && y >= 1 && y <= 2 && z <= 1
        let inSecond = x >= 4 && x <= 5 && y >= 4 && y <= 5 && z >= 2
        if inFirst || inSecond then 255uy else 0uy)
    |> Image<uint8>.ofArray2D

let private makeFloatSlice width height z =
    Array2D.init width height (fun x y -> float32 x + 10.0f * float32 y + 100.0f * float32 z)
    |> Image<float32>.ofArray2D

let private identity3 =
    { m00 = 1.0; m01 = 0.0; m02 = 0.0
      m10 = 0.0; m11 = 1.0; m12 = 0.0
      m20 = 0.0; m21 = 0.0; m22 = 1.0 }

let private imageGeom width height depth : StackAffineResampler.ImageGeom =
    { W = width
      H = height
      D = depth
      Origin = v3 0.0 0.0 0.0
      Spacing = v3 1.0 1.0 1.0
      Direction = identity3 }

let private writeSlices directory suffix (slices: Image<'T> list) =
    slices
    |> List.iteri (fun index image ->
        let fileName = Path.Combine(directory, sprintf "image_%03d%s" index suffix)
        image.toFile(fileName))

let private writeNexusStack (path: string) (datasetPath: string) (data: uint16[,,]) =
    let parts: string[] = datasetPath.Trim('/').Split('/', StringSplitOptions.RemoveEmptyEntries)
    let file = H5File()
    let mutable group = file :> H5Group

    for part in parts[0 .. parts.Length - 2] do
        let next = H5Group()
        group.Add(part, next)
        group <- next

    group.Add(parts[parts.Length - 1], data)
    file.Write(filePath = path)

let private disposeImages images =
    images |> List.iter (fun (image: Image<'T>) -> image.decRefCount())

let private image2D f =
    Array2D.init 2 2 f |> Image<float32>.ofArray2D

let private expectFloat32Close actual expected message =
    Expect.isLessThan (Math.Abs(float actual - float expected)) 1.0e-5 message

let private expectComplexClose (actual: System.Numerics.Complex) (expected: System.Numerics.Complex) message =
    Expect.floatClose Accuracy.high actual.Real expected.Real $"{message} real"
    Expect.floatClose Accuracy.high actual.Imaginary expected.Imaginary $"{message} imaginary"

let private point x y z : CoordinatePoint =
    { X = x; Y = y; Z = z; Scale = Double.NaN; Response = Double.NaN }

let private pointDistance (a: CoordinatePoint) (b: CoordinatePoint) =
    let dx = a.X - b.X
    let dy = a.Y - b.Y
    let dz = a.Z - b.Z
    Math.Sqrt(dx * dx + dy * dy + dz * dz)

let private permutations values =
    let rec loop prefix remaining =
        seq {
            match remaining with
            | [] -> yield List.rev prefix
            | _ ->
                for index in 0 .. List.length remaining - 1 do
                    let value = remaining[index]
                    let rest = remaining |> List.removeAt index
                    yield! loop (value :: prefix) rest
        }

    loop [] values

let private bruteForceMatchingDistance fixedPoints movingPoints =
    permutations movingPoints
    |> Seq.map (fun candidate ->
        (fixedPoints, candidate)
        ||> List.map2 pointDistance
        |> List.average)
    |> Seq.min

let private expectoTestCase = testCase

let private testCase name body =
    expectoTestCase name (fun arg ->
        try
            body arg
        finally
            GC.Collect()
            GC.WaitForPendingFinalizers()
            GC.Collect())

let private scalarPlan values =
    let items = values |> List.toArray
    let stage =
        Stage.init
            "scalar source"
            (uint items.Length)
            (fun index -> items[index])
            (ProfileTransition.create Unit Streaming)
            (fun _ -> 0UL)
            id

    Plan.create (Some stage) 1024UL 0UL 1UL (uint64 items.Length) false

let private imagePlan (images: Image<'T> list) =
    images |> List.iter (fun image -> image.incRefCount())
    let items = images |> List.toArray
    let stage =
        Stage.init
            "image source"
            (uint items.Length)
            (fun index -> items[index])
            (ProfileTransition.create Unit Streaming)
            (fun _ -> 0UL)
            id

    Plan.create (Some stage) (2UL * 1024UL * 1024UL * 1024UL) 0UL 1UL (uint64 items.Length) false

let private scalarStage name f =
    Stage.map name (fun _ value -> f value) (fun _ -> 0UL) id

let stackProcessingSupportSuite =
    testSequenced <| testList "StackProcessing support coverage" [
        testCase ">=> composes a plan with a stage" <| fun _ ->
            let actual =
                scalarPlan [ 1; 2; 3 ]
                >=> scalarStage "increment" ((+) 1)
                |> drainList

            Expect.equal actual [ 2; 3; 4 ] ">=> should apply the stage to every stream element."

        testCase "--> composes stages before plan execution" <| fun _ ->
            let stage =
                scalarStage "increment" ((+) 1)
                --> scalarStage "double" ((*) 2)

            let actual =
                scalarPlan [ 1; 2; 3 ]
                >=> stage
                |> drainList

            Expect.equal actual [ 4; 6; 8 ] "--> should compose stages left-to-right."

        testCase ">=>> forks a synchronized stream into two stages" <| fun _ ->
            let actual =
                scalarPlan [ 1; 2; 3 ]
                >=>> (scalarStage "increment" ((+) 1), scalarStage "double" ((*) 2))
                |> drainList

            Expect.equal actual [ 2, 2; 3, 4; 4, 6 ] ">=>> should produce paired branch outputs."

        testCase ">=>> does not deadlock when one branch waits for a window" <| fun _ ->
            let delayed =
                Stage.window "delayed window" 3u 1u (fun _ _ -> 0) 1u
                --> Stage.flattenWindow "delayed flatten"

            let actual =
                scalarPlan [ 1; 2; 3 ]
                >=>> (scalarStage "fast" ((+) 10), delayed)
                |> drainList

            Expect.equal actual [ 11, 1; 12, 2; 13, 3 ] ">=>> should queue early requests from the fast branch."

        testCase ">=>> rejects branches with different slice domains" <| fun _ ->
            Expect.throws
                (fun () ->
                    scalarPlan [ 1; 2; 3 ]
                    >=>> (scalarStage "preserve" id, Stage.skip "skip first" 1u)
                    |> ignore)
                ">=>> should reject synchronization when one branch skips slices."

        testCase ">>=> maps synchronized pairs to a single stream" <| fun _ ->
            let actual =
                scalarPlan [ 1; 2; 3 ]
                >=>> (scalarStage "increment" ((+) 1), scalarStage "double" ((*) 2))
                >>=> (+)
                |> drainList

            Expect.equal actual [ 4; 7; 10 ] ">>=> should combine paired values."

        testCase ">>=>> applies synchronized stages to paired streams" <| fun _ ->
            let actual =
                scalarPlan [ 1; 2; 3 ]
                >=>> (scalarStage "left" id, scalarStage "right" ((*) 10))
                >>=>> (scalarStage "left plus one" ((+) 1), scalarStage "right plus five" ((+) 5))
                |> drainList

            Expect.equal actual [ 2, 15; 3, 25; 4, 35 ] ">>=>> should run both tuple branches and zip their outputs."

        testCase ">>=>> does not deadlock when one tuple branch waits for a window" <| fun _ ->
            let delayed =
                Stage.window "delayed pair window" 3u 1u (fun _ _ -> 0) 1u
                --> Stage.flattenWindow "delayed pair flatten"

            let actual =
                scalarPlan [ 1; 2; 3 ]
                >=>> (scalarStage "left" id, scalarStage "right" ((*) 10))
                >>=>> (scalarStage "fast pair branch" ((+) 100), delayed)
                |> drainList

            Expect.equal actual [ 101, 10; 102, 20; 103, 30 ] ">>=>> should queue early requests from either tuple branch."

        testCase ">>=>> rejects tuple branches with different slice domains" <| fun _ ->
            Expect.throws
                (fun () ->
                    scalarPlan [ 1; 2; 3 ]
                    >=>> (scalarStage "left" id, scalarStage "right" ((*) 10))
                    >>=>> (Stage.skip "skip left" 1u, scalarStage "right preserve" id)
                    |> ignore)
                ">>=>> should reject synchronization when one tuple branch skips slices."

        testCase "command line source parses non-debug arguments" <| fun _ ->
            let plan, rest = commandLineSource 1024UL [| "alpha"; "beta" |]
            Expect.isFalse plan.debug "commandLineSource should leave debug off without -d."
            Expect.equal (rest |> Array.toList) [ "alpha"; "beta" ] "commandLineSource should return non-debug arguments."

        testCase "command line source parses debug level and remaining arguments" <| fun _ ->
            try
                let plan, rest = commandLineSource 1024UL [| "-d"; "2"; "alpha" |]

                Expect.isTrue plan.debug "commandLineSource should enable debug when -d is present."
                Expect.equal plan.debugLevel 2u "commandLineSource should store the requested debug level."
                Expect.equal (rest |> Array.toList) [ "alpha" ] "commandLineSource should remove the debug arguments."
            finally
                DebugLevel.set 0u
                Image<uint8>.setDebugLevel 0u

        testCase "command line source rejects non-numeric debug level" <| fun _ ->
            Expect.throws
                (fun () ->
                    commandLineSource 1024UL [| "-d"; "verbose" |]
                    |> ignore)
                "commandLineSource should validate the debug level after -d."

        testCase "releaseAfterWith releases resources after success and failure" <| fun _ ->
            let mutable releases = 0
            let ops =
                { Retain = fun _ -> ()
                  Release = fun _ -> releases <- releases + 1
                  MemoryOf = fun value -> Some(uint64 value) }

            let actual = StackCore.releaseAfterWith ops ((+) 1) 41
            Expect.equal actual 42 "releaseAfterWith should return the wrapped result."
            Expect.equal releases 1 "releaseAfterWith should release after a successful call."

            Expect.throws
                (fun () ->
                    StackCore.releaseAfterWith ops (fun _ -> failwith "boom") 7
                    |> ignore)
                "releaseAfterWith should rethrow the wrapped exception."
            Expect.equal releases 2 "releaseAfterWith should also release when the wrapped function fails."

        testCase "StackProcessingCost estimates memory and time cost components" <| fun _ ->
            let input = Pair(5UL, 7UL)
            Expect.equal (inputVoxels input) 12UL "inputVoxels should sum paired stream element sizes."
            Expect.equal (pixelTypeName<uint16>) "UInt16" "pixelTypeName should use the CLR pixel type name."

            let bytes = imageBytes<uint16> 12UL
            Expect.equal bytes 24UL "imageBytes should estimate bytes for the supplied pixel type."
            Expect.equal (sliceBytes<uint16> 3u 4u) bytes "sliceBytes should multiply x/y dimensions before estimating bytes."

            let shape =
                { InputImages = 1UL
                  OutputImages = 2UL
                  WorkImages = 3UL
                  RetainedImages = 4UL }
            let memory = imageStageMemory<uint16> Map shape
            let memoryEstimate = memory.Estimate input

            Expect.equal memoryEstimate.InputLive bytes "Input live bytes should scale by the input image count."
            Expect.equal memoryEstimate.OutputLive (bytes * 2UL) "Output live bytes should scale by the output image count."
            Expect.equal memoryEstimate.WorkLive (bytes * 3UL) "Work live bytes should scale by the work image count."
            Expect.equal memoryEstimate.RetainedLive (bytes * 4UL) "Retained live bytes should scale by the retained image count."
            Expect.equal memoryEstimate.Peak (bytes * 10UL) "Peak should include every image category."

            let costModel = imageMapCost<uint16> "map.u16" (fun actualInput -> float (inputVoxels actualInput) * 2.0)
            let costEstimate = StageCostModel.estimate costModel input
            Expect.equal costEstimate.Memory.Peak (bytes * 2UL) "imageMapCost should account for one input and one output image."
            Expect.equal costEstimate.Time.NativeCostUnits 24.0 "imageMapCost should use the supplied native cost estimator."
            Expect.equal costEstimate.Time.CalibrationKey (Some "map.u16") "imageMapCost should retain the calibration key."

        testCase "StackProcessingCost builds IO models and installs cost models on stages" <| fun _ ->
            let readModel = imageIoCost<uint8> "read" Source "read.key" (fun _ -> 128UL) (fun _ -> 2UL)
            let readEstimate = readModel.Estimate (Single 1UL)
            Expect.equal readEstimate.IoReadBytes 128UL "Read IO cost should estimate read bytes."
            Expect.equal readEstimate.IoReadOps 2UL "Read IO cost should estimate read ops."
            Expect.equal readEstimate.CalibrationKey (Some "read.key") "Read IO cost should retain the calibration key."

            let writeModel = imageIoCost<uint8> "write" Sink "write.key" (fun _ -> 256UL) (fun _ -> 3UL)
            let writeEstimate = writeModel.Estimate (Single 1UL)
            Expect.equal writeEstimate.IoWriteBytes 256UL "Write IO cost should estimate write bytes."
            Expect.equal writeEstimate.IoWriteOps 3UL "Write IO cost should estimate write ops."

            let unknownModel = imageIoCost<uint8> "metadata" (Custom "metadata") "metadata" (fun _ -> 512UL) (fun _ -> 4UL)
            Expect.equal (unknownModel.Estimate (Single 1UL)) StageTimeCostEstimate.zero "Unknown IO cost kinds should be zero-cost."

            let stage = scalarStage "identity" id
            let memoryModel = StageMemoryModel.fromSinglePeak Map (fun _ -> 99UL)
            let timeModel = StageTimeCostModel.cpu Map (Some "cpu.key") (fun _ -> 7.0)
            let costModel = StageCostModel.create memoryModel timeModel
            let updated = withCostModel costModel stage

            Expect.equal (updated.MemoryNeed (Single 1UL)) (Single 99UL) "withCostModel should update the legacy memory need."
            Expect.equal (updated.CostModel.Time.Estimate (Single 1UL)).CpuCostUnits 7.0 "withCostModel should install the supplied time cost model."

        testCase "StackProcessingCost loads the first available time calibration file" <| fun _ ->
            let tempDir = tempDirectory "time-calibration"
            let missing = Path.Combine(tempDir, "missing.json")
            let calibrationPath = Path.Combine(tempDir, "calibration.json")

            try
                StageTimeCalibration.clear()
                File.WriteAllText(calibrationPath, """{"calibrations":{"stage":{"cpuMillisecondsPerUnit":2.5}}}""")

                Expect.isFalse (tryLoadTimeCalibration missing) "Missing calibration files should not load."
                Expect.isTrue (tryLoadFirstTimeCalibration [ missing; calibrationPath ]) "The first existing calibration file should load."

                let estimate = StageTimeCostEstimate.create 4.0 0.0 0UL 0UL 0UL 0UL (Some "stage")
                Expect.equal (StageTimeCalibration.estimateMilliseconds estimate) (Some 10.0) "Loaded coefficients should estimate milliseconds."
            finally
                StageTimeCalibration.clear()
                deleteDirectory tempDir

        testCase "window and flatten expose padded windows for stack slices" <| fun _ ->
            let images = [ for z in 0 .. 2 -> makeSlice 3 3 z ]

            try
                let windows =
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> zero<uint8> 3u 3u 3u
                    >=> StackCore.window 3u 1u 1u
                    |> drainList

                try
                    Expect.isGreaterThanOrEqual windows.Length 3 "window should produce padded windows."
                    Expect.equal windows[0].Items.Length 3 "Each window should contain the requested number of images."
                finally
                    windows |> List.collect _.Items |> disposeImages

                let flattened =
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> zero<uint8> 3u 3u 2u
                    >=> StackCore.window 2u 0u 2u
                    >=> StackCore.flatten ()
                    |> drainList

                try
                    Expect.equal flattened.Length 2 "flatten should convert a windowed list stream back to image elements for non-overlapping windows."
                finally
                    disposeImages flattened
            finally
                disposeImages images

        testCase "normalNoise source emits requested shape and zero-variance value" <| fun _ ->
            let slices =
                source (2UL * 1024UL * 1024UL * 1024UL)
                |> StackProcessing.normalNoise<float> 4u 3u 2u 5.0 0.0
                |> drainList

            try
                Expect.equal slices.Length 2 "normalNoise should emit the requested depth."
                slices
                |> List.iter (fun image ->
                    Expect.equal (image.GetSize()) [ 4u; 3u ] "normalNoise should emit the requested x/y size."
                    for x in 0 .. 3 do
                        for y in 0 .. 2 do
                            Expect.floatClose Accuracy.high image[x, y] 0.0 "Zero-variance normalNoise should bypass SimpleITK and preserve the zero source.")
            finally
                disposeImages slices

        testCase "additional noise sources emit requested shape" <| fun _ ->
            let expectNoiseSource label (makePlan: unit -> Plan<unit, Image<float>>) =
                let slices = makePlan () |> drainList
                try
                    Expect.equal slices.Length 2 $"{label} should emit the requested depth."
                    slices
                    |> List.iter (fun image ->
                        Expect.equal (image.GetSize()) [ 4u; 3u ] $"{label} should emit the requested x/y size.")
                finally
                    disposeImages slices

            expectNoiseSource
                "saltAndPepperNoise"
                (fun () ->
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> StackProcessing.saltAndPepperNoise<float> 4u 3u 2u 0.0)

            expectNoiseSource
                "shotNoise"
                (fun () ->
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> StackProcessing.shotNoise<float> 4u 3u 2u 0.0)

            expectNoiseSource
                "speckleNoise"
                (fun () ->
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> StackProcessing.speckleNoise<float> 4u 3u 2u 0.0)

        testCase "StackIO reports stack metadata and read/readRandom return slices" <| fun _ ->
            let inputDir = tempDirectory "io-read"
            let suffix = ".tiff"
            let slices = [ for z in 0 .. 2 -> makeSlice 5 4 z ]

            try
                writeSlices inputDir suffix slices

                Expect.equal (getStackDepth inputDir suffix) 3u "Stack depth should count matching files."
                Expect.equal (getStackSize inputDir suffix) (5u, 4u, 3u) "Stack size should combine slice size and file count."
                Expect.equal (getStackWidth inputDir suffix) 5UL "Stack width should come from slice metadata."
                Expect.equal (getStackHeight inputDir suffix) 4UL "Stack height should come from slice metadata."

                File.WriteAllBytes(Path.Combine(inputDir, "alias_000.jpg"), [| 0uy |])
                File.WriteAllBytes(Path.Combine(inputDir, "alias_001.jpeg"), [| 0uy |])
                Expect.equal (getStackDepth inputDir ".jpg") 2u "JPG stack depth should include .jpg and .jpeg aliases."
                Expect.equal (getStackDepth inputDir ".jpeg") 2u "JPEG stack depth should include .jpg and .jpeg aliases."

                let allSlices =
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> read<uint8> inputDir suffix
                    |> drainList

                try
                    Expect.equal allSlices.Length 3 "read should stream every matching slice."
                    Expect.equal (allSlices[0].GetSize()) [ 5u; 4u ] "read slices should preserve the slice shape."
                    Expect.equal (allSlices[1].[0, 0]) 3uy "read should preserve sorted filename order."
                finally
                    disposeImages allSlices

                let randomSlices =
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> readRandom<uint8> 2u inputDir suffix
                    |> drainList

                try
                    Expect.equal randomSlices.Length 2 "readRandom should return the requested count."
                    randomSlices |> List.iter (fun image -> Expect.equal (image.GetSize()) [ 5u; 4u ] "readRandom slices should preserve shape.")
                finally
                    disposeImages randomSlices

                let rangeSlices =
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> readRange<uint8> "1" 2 "end" inputDir suffix
                    |> drainList

                try
                    Expect.equal rangeSlices.Length 1 "readRange should read first, first+step, ... up to last."
                    Expect.equal (rangeSlices[0].[0, 0]) 3uy "readRange should preserve the requested sorted slice order."
                finally
                    disposeImages rangeSlices

                let clampedRangeSlices =
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> readRange<uint8> "-10" 2 "end+10" inputDir suffix
                    |> drainList

                try
                    Expect.equal clampedRangeSlices.Length 2 "readRange should clamp endpoints outside the stack."
                    Expect.equal (clampedRangeSlices[0].[0, 0]) 0uy "A clamped first endpoint should begin at slice zero."
                    Expect.equal (clampedRangeSlices[1].[0, 0]) 6uy "A clamped last endpoint should allow the final stepped slice."
                finally
                    disposeImages clampedRangeSlices

                let reverseRangeSlices =
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> readRange<uint8> "end" -1 "0" inputDir suffix
                    |> drainList

                try
                    Expect.equal reverseRangeSlices.Length 3 "readRange should support negative steps."
                    Expect.equal (reverseRangeSlices[0].[0, 0]) 6uy "A reverse range should start at the clamped first endpoint."
                    Expect.equal (reverseRangeSlices[2].[0, 0]) 0uy "A reverse range should stop at the requested lower endpoint."
                finally
                    disposeImages reverseRangeSlices
            finally
                disposeImages slices
                deleteDirectory inputDir

        testCase "createPadding and crop update x/y/z volume geometry in streaming order" <| fun _ ->
            let slices = [ for z in 0 .. 2 -> makeSlice 3 3 z ]

            let padded =
                imagePlan slices
                >=> createPadding<uint8> 1u 2u 1u 0u 1u 1u 7.0
                |> drainList

            try
                Expect.equal padded.Length 5 "createPadding should add before/after z slices."
                padded |> List.iter (fun image -> Expect.equal (image.GetSize()) [ 6u; 4u ] "createPadding should pad x/y slice dimensions.")
                Expect.equal padded[0].[0, 0] 7uy "The prepended z padding slice should contain the padding value."
                Expect.equal padded[1].[1, 1] slices[0].[0, 0] "The original first pixel should be offset by x/y padding."
                Expect.equal padded[4].[5, 3] 7uy "The appended z padding slice should contain the padding value."
            finally
                disposeImages padded

            let cropped =
                imagePlan slices
                >=> crop<uint8> 1u 1u 1u 0u 1u 1u
                |> drainList

            try
                Expect.equal cropped.Length 1 "crop should trim the requested z prefix and suffix."
                Expect.equal (cropped[0].GetSize()) [ 1u; 2u ] "crop should remove x/y borders from every retained slice."
                Expect.equal cropped[0].[0, 0] slices[1].[1, 1] "crop should preserve the correct interior pixels."
            finally
                disposeImages cropped
                disposeImages slices

        testCase "marchingCubes streams triangle sets and writeMesh writes OBJ" <| fun _ ->
            let inputDir = tempDirectory "mesh-input"
            let outputDir = tempDirectory "mesh-output"
            let suffix = ".tiff"
            let outputPath = Path.Combine(outputDir, "surface.obj")
            let slices =
                [ Image<uint8>.ofArray2D(Array2D.create 3 3 0uy)
                  Image<uint8>.ofArray2D(Array2D.init 3 3 (fun x y -> if x >= 1 && y >= 1 then 1uy else 0uy))
                  Image<uint8>.ofArray2D(Array2D.create 3 3 0uy) ]

            try
                writeSlices inputDir suffix slices

                let triangleSets =
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> read<uint8> inputDir suffix
                    >=> marchingCubes<uint8> 0.5
                    |> drainList

                Expect.isTrue (triangleSets |> List.exists (fun triangleSet -> not triangleSet.Triangles.IsEmpty)) "marchingCubes should emit at least one triangle set for a crossing surface."

                source (2UL * 1024UL * 1024UL * 1024UL)
                |> read<uint8> inputDir suffix
                >=> marchingCubes<uint8> 0.5
                >=> writeMesh outputPath "auto"
                |> drain

                let meshText = File.ReadAllText(outputPath)
                Expect.stringContains meshText "v " "OBJ mesh should contain vertices."
                Expect.stringContains meshText "f " "OBJ mesh should contain faces."
            finally
                disposeImages slices
                deleteDirectory inputDir
                deleteDirectory outputDir

        testCase "surfaceArea scales triangle coordinates before summing areas" <| fun _ ->
            let triangleSet: TriangleSet =
                { Triangles =
                    [ { A = { X = 0.0; Y = 0.0; Z = 0.0 }
                        B = { X = 1.0; Y = 0.0; Z = 0.0 }
                        C = { X = 0.0; Y = 1.0; Z = 0.0 } }
                      { A = { X = 0.0; Y = 0.0; Z = 0.0 }
                        B = { X = 0.0; Y = 1.0; Z = 0.0 }
                        C = { X = 0.0; Y = 0.0; Z = 1.0 } } ] }

            let actual =
                scalarPlan [ triangleSet ]
                >=> surfaceArea 2.0 3.0 5.0
                |> drain

            Expect.floatClose Accuracy.high actual 10.5 "surfaceArea should sum areas after anisotropic x/y/z scaling."

        testCase "volume reduces UInt8 0-1 mask slices to physical volume" <| fun _ ->
            let slices =
                [ Image<uint8>.ofArray2D(array2D [ [ 1uy; 0uy ]; [ 1uy; 1uy ] ])
                  Image<uint8>.ofArray2D(array2D [ [ 0uy; 1uy ]; [ 0uy; 0uy ] ]) ]

            try
                let actual =
                    imagePlan slices
                    >=> volume 2.0 3.0 4.0
                    |> drain

                Expect.floatClose Accuracy.high actual 96.0 "volume should multiply foreground voxel count by physical voxel volume."
            finally
                disposeImages slices

        testCase "point-set CSV read and write round-trip point sets" <| fun _ ->
            let outputDir = tempDirectory "point-set"
            let path = Path.Combine(outputDir, "points.csv")
            let points: PointSet =
                { Points =
                    [ { X = 1.0; Y = 2.0; Z = 3.0; Scale = 1.6; Response = 0.25 }
                      { X = 4.5; Y = 5.5; Z = 6.5; Scale = 2.0; Response = -0.125 } ] }

            try
                scalarPlan [ points ]
                >=> writePointSet path ".csv"
                |> sink

                let reread =
                    source 1024UL
                    |> readPointSet path
                    |> drain

                Expect.equal reread points "Point-set CSV should preserve x,y,z,scale,response values."
            finally
                deleteDirectory outputDir

        testCase "pointPairDistances returns a vectorized physical distance matrix" <| fun _ ->
            let outputDir = tempDirectory "distance-matrix"
            let outputPath = Path.Combine(outputDir, "distances.csv")
            let points: PointSet =
                { Points =
                    [ point 0.0 0.0 0.0
                      point 1.0 0.0 0.0
                      point 1.0 2.0 2.0 ] }

            try
                let distances =
                    scalarPlan [ points ]
                    >=> pointPairDistances 2.0 3.0 4.0
                    |> drain

                let matrix = unvectorizeMatrix distances
                Expect.equal distances.Rows 3u "pointPairDistances should preserve the row count."
                Expect.equal distances.Columns 3u "pointPairDistances should preserve the column count."
                Expect.floatClose Accuracy.high matrix[0, 1] 2.0 "x distance should use xUnit."
                Expect.floatClose Accuracy.high matrix[1, 2] 10.0 "y/z distance should use yUnit and zUnit."
                Expect.floatClose Accuracy.high matrix[2, 1] 10.0 "distance matrix should be symmetric."

                let vectorizedAgain = vectorizeMatrix matrix
                Expect.equal vectorizedAgain distances "matrix vectorization should round-trip through unvectorizeMatrix."

                scalarPlan [ distances ]
                >=> writeMatrix outputPath ".csv"
                |> sink

                let csv = File.ReadAllLines(outputPath)
                Expect.equal csv.Length 3 "writeMatrix should write one row per matrix row."
                Expect.stringContains csv[0] "2" "writeMatrix should include matrix values."
            finally
                deleteDirectory outputDir

        testCase "dogKeypoints detects streamed Difference-of-Gaussian point candidates" <| fun _ ->
            let slices =
                [ for z in 0 .. 8 ->
                    let image =
                        Array2D.init 9 9 (fun x y -> if x = 4 && y = 4 && z = 4 then 255uy else 0uy)
                        |> Image<uint8>.ofArray2D
                    image.index <- z
                    image ]

            try
                let pointSets =
                    imagePlan slices
                    >=> dogKeypoints<uint8> 0.5 1.2 4u 0.0001 1u
                    |> drainList

                let points = pointSets |> List.collect _.Points
                Expect.isTrue (points |> List.exists (fun p -> p.X = 4.0 && p.Y = 4.0 && p.Z = 4.0)) "A centered impulse should produce a keypoint near the impulse coordinate."
            finally
                disposeImages slices

        testCase "siftKeypoints exposes SIFT-style streamed point detection" <| fun _ ->
            let slices =
                [ for z in 0 .. 8 ->
                    let image =
                        Array2D.init 9 9 (fun x y -> if x = 4 && y = 4 && z = 4 then 255uy else 0uy)
                        |> Image<uint8>.ofArray2D
                    image.index <- z
                    image ]

            try
                let pointSets =
                    imagePlan slices
                    >=> siftKeypoints<uint8> 0.5 1.2 4u 0.0001 1u
                    |> drainList

                let points = pointSets |> List.collect _.Points
                Expect.isTrue (points |> List.exists (fun p -> p.X = 4.0 && p.Y = 4.0 && p.Z = 4.0)) "A centered impulse should produce a SIFT-style keypoint near the impulse coordinate."
            finally
                disposeImages slices

        testCase "earthMoversDistance agrees with brute-force matching for equal-sized point sets" <| fun _ ->
            let fixedPoints =
                [ point 0.0 0.0 0.0
                  point 0.0 3.0 0.0
                  point 4.0 0.0 1.0
                  point 2.0 2.0 2.0 ]

            let movingPoints =
                [ point 4.2 0.1 1.2
                  point -0.1 3.3 -0.2
                  point 2.5 2.1 2.4
                  point 0.3 -0.2 0.1 ]

            let expected = bruteForceMatchingDistance fixedPoints movingPoints
            let actual = earthMoversDistance fixedPoints movingPoints

            Expect.floatClose Accuracy.high actual expected "Equal-cardinality EMD should use the optimal point matching."

        testCase "earthMoversDistance transports equal mass when point set sizes differ" <| fun _ ->
            let fixedPoints = [ point 0.0 0.0 0.0 ]
            let movingPoints = [ point 0.0 0.0 0.0; point 10.0 0.0 0.0 ]

            Expect.floatClose Accuracy.high (earthMoversDistance fixedPoints movingPoints) 5.0 "One fixed point should receive half the moving mass from each moving point."

        testCase "inverseAffine round-trips nontrivial affine points" <| fun _ ->
            let affine: Affine =
                { A =
                    { m00 = 1.2; m01 = 0.1; m02 = -0.05
                      m10 = -0.2; m11 = 0.9; m12 = 0.15
                      m20 = 0.05; m21 = -0.1; m22 = 1.1 }
                  T = TinyLinAlg.v3 2.0 -3.0 0.5
                  C = TinyLinAlg.v3 1.0 2.0 -1.0 }

            let original: PointSet =
                { Points = [ point 4.0 -2.0 3.0; point -1.0 5.0 2.5 ] }
            let roundTripped =
                original
                |> transformPointSet affine
                |> transformPointSet (inverseAffine affine)

            Expect.isLessThan (earthMoversDistance original.Points roundTripped.Points) 1.0e-10 "inverseAffine should invert affinePoint for resampler-style backward transforms."

        testCase "affine registration aligns translated point sets and exposes resampler-compatible inverse" <| fun _ ->
            let moving =
                [ point 0.0 0.0 0.0
                  point 1.0 0.0 0.0
                  point 0.0 1.0 0.0
                  point 0.0 0.0 1.0 ]

            let fixedPoints =
                moving
                |> List.map (fun p -> point (p.X + 2.0) (p.Y - 1.0) (p.Z + 0.5))

            let shuffledFixed =
                [ fixedPoints[2]; fixedPoints[0]; fixedPoints[3]; fixedPoints[1] ]

            let emd = earthMoversDistance fixedPoints shuffledFixed
            Expect.floatClose Accuracy.high emd 0.0 "EMD should match equal point sets independent of order."

            let result =
                affineRegistration
                    { defaultAffineRegistrationOptions with MaxIterations = 5 }
                    fixedPoints
                    moving

            Expect.isLessThan result.Distance 1.0e-8 "The optimizer should find the centroid translation immediately for a pure translation."

            let transformedMoving =
                transformPointSet result.Transform { Points = moving }
                |> _.Points

            Expect.isLessThan (earthMoversDistance fixedPoints transformedMoving) 1.0e-8 "The forward registration transform should map moving points to fixed points."

            let fixedAsMovingCoordinates =
                transformPointSet result.InverseTransform { Points = fixedPoints }
                |> _.Points

            Expect.isLessThan (earthMoversDistance moving fixedAsMovingCoordinates) 1.0e-8 "The inverse transform should map fixed-grid coordinates back to moving coordinates for StackAffineResampler."

        testCase "resampleAffineTrilinearSlices samples chunked slabs with the supplied output-to-input affine" <| fun _ ->
            let chunkDirectory = tempDirectory "affine-resampler-chunks"
            let slices =
                [ for z in 0 .. 3 ->
                    let image = makeFloatSlice 5 4 z
                    image.index <- z
                    image ]

            try
                imagePlan slices
                >=> writeInSlabs chunkDirectory ".tiff" 2u 2u 2u
                >=> ignoreSingles ()
                |> sink

                let lerp (a: float32) (b: float32) (t: float32) =
                    a + (b - a) * t

                let transform: Affine =
                    { A = identity3
                      T = v3 0.5 1.0 0.0
                      C = v3 0.0 0.0 0.0 }

                let output =
                    resampleAffineTrilinearSlices
                        chunkDirectory
                        ".tiff"
                        lerp
                        2
                        (imageGeom 5 4 4)
                        (imageGeom 3 2 3)
                        transform
                        -1.0f
                    |> Seq.toList

                Expect.equal (output |> List.map fst) [ 0; 1; 2 ] "The resampler should emit the requested output slices in z order."

                for k, image in output do
                    for y in 0 .. int (image.GetHeight()) - 1 do
                        for x in 0 .. int (image.GetWidth()) - 1 do
                            let expected =
                                (float32 x + 0.5f)
                                + 10.0f * (float32 y + 1.0f)
                                + 100.0f * float32 k

                            expectFloat32Close image[x, y] expected $"Trilinear sampling should match the known linear volume at ({x},{y},{k})."

                output |> List.map snd |> disposeImages
            finally
                disposeImages slices
                deleteDirectory chunkDirectory

        testCase "streamConnectedObjects emits completed objects without waiting for the full stack" <| fun _ ->
            let slices =
                [ for z in 0 .. 3 ->
                    let image =
                        Array2D.init 5 5 (fun x y ->
                            let objectCompletedAfterNextSlice = z = 0 && x = 1 && y = 1
                            let objectSpanningTwoSlices = (z = 1 || z = 2) && x = 3 && y = 3
                            if objectCompletedAfterNextSlice || objectSpanningTwoSlices then 255uy else 0uy)
                        |> Image<uint8>.ofArray2D
                    image.index <- z
                    image ]

            try
                let batches =
                    imagePlan slices
                    >=> streamConnectedObjects<uint8> ObjectConnectivity.Six
                    |> drainList

                Expect.equal (batches |> List.map List.length) [ 0; 1; 0; 1 ] "Objects should be emitted when the next slice proves they no longer continue."

                let objects = batches |> List.collect id
                Expect.equal objects.Length 2 "Both completed objects should be emitted."
                Expect.equal objects[0].Size 1UL "The one-slice object should contain one pixel."
                Expect.equal objects[0].Bounds.MinZ 0 "The first object should come from z=0."
                Expect.equal objects[0].Bounds.MaxZ 0 "The first object should be complete at z=0."
                Expect.equal objects[1].Size 2UL "The second object should span two slices."
                Expect.equal objects[1].Bounds.MinZ 1 "The spanning object should start at z=1."
                Expect.equal objects[1].Bounds.MaxZ 2 "The spanning object should end at z=2."
                Expect.equal objects[1].Positions [ { X = 3; Y = 3; Z = 1 }; { X = 3; Y = 3; Z = 2 } ] "Object pixels should use exact integer positions."
            finally
                disposeImages slices

        testCase "streamConnectedObjects honors 26-connectivity across diagonal slice contacts" <| fun _ ->
            let slices =
                [ for z in 0 .. 1 ->
                    let image =
                        Array2D.init 4 4 (fun x y ->
                            if (z = 0 && x = 1 && y = 1) || (z = 1 && x = 2 && y = 2) then 255uy else 0uy)
                        |> Image<uint8>.ofArray2D
                    image.index <- z
                    image ]

            try
                let sixObjects =
                    imagePlan slices
                    >=> streamConnectedObjects<uint8> ObjectConnectivity.Six
                    |> drainList
                    |> List.collect id

                let twentySixObjects =
                    imagePlan slices
                    >=> streamConnectedObjects<uint8> ObjectConnectivity.TwentySix
                    |> drainList
                    |> List.collect id

                Expect.equal sixObjects.Length 2 "Face connectivity should keep diagonal z-neighbors separate."
                Expect.equal twentySixObjects.Length 1 "26-connectivity should merge diagonal z-neighbors."
                Expect.equal twentySixObjects[0].Size 2UL "The merged object should contain both pixels."
            finally
                disposeImages slices

        testCase "removeSmallObjects removes completed small foreground components" <| fun _ ->
            let slices =
                [ for z in 0 .. 4 ->
                    let image =
                        Array2D.init 7 7 (fun x y ->
                            let smallObject = x = 1 && y = 1 && (z = 1 || z = 2)
                            let largeObject = x = 4 && y = 4 && z >= 1 && z <= 3
                            if smallObject || largeObject then 1uy else 0uy)
                        |> Image<uint8>.ofArray2D
                    image.index <- z
                    image ]
            let mutable cleaned: Image<uint8> list = []

            try
                cleaned <-
                    imagePlan slices
                    >=> removeSmallObjects 2UL ObjectConnectivity.Six
                    |> drainList

                let z1 = cleaned |> List.find (fun image -> image.index = 1)
                let z2 = cleaned |> List.find (fun image -> image.index = 2)
                let z3 = cleaned |> List.find (fun image -> image.index = 3)

                Expect.equal z1[1, 1] 0uy "The two-voxel object should be removed from its first slice."
                Expect.equal z2[1, 1] 0uy "The two-voxel object should be removed from its second slice."
                Expect.equal z1[4, 4] 1uy "The larger object should be preserved."
                Expect.equal z2[4, 4] 1uy "The larger object should be preserved through the middle slice."
                Expect.equal z3[4, 4] 1uy "The larger object should be preserved at completion."
            finally
                disposeImages cleaned
                disposeImages slices

        testCase "fillSmallHoles fills enclosed small background components and preserves exterior background" <| fun _ ->
            let slices =
                [ for z in 0 .. 4 ->
                    let image =
                        Array2D.init 7 7 (fun x y ->
                            let exteriorBackground = x = 0
                            let enclosedHole = x = 3 && y = 3 && z = 2
                            if exteriorBackground || enclosedHole then 0uy else 1uy)
                        |> Image<uint8>.ofArray2D
                    image.index <- z
                    image ]
            let mutable filled: Image<uint8> list = []

            try
                filled <-
                    imagePlan slices
                    >=> fillSmallHoles 1UL ObjectConnectivity.Six
                    |> drainList

                let z2 = filled |> List.find (fun image -> image.index = 2)
                Expect.equal z2[3, 3] 1uy "The one-voxel enclosed hole should be filled."
                Expect.equal z2[0, 3] 0uy "Background touching the x-y image border is exterior and should be preserved."
            finally
                disposeImages filled
                disposeImages slices

        testCase "paintObjects turns streamed object positions into UInt8 mask slices" <| fun _ ->
            let objects: StreamedObject list =
                [ { Label = 1UL
                    Positions = [ { X = 1; Y = 2; Z = 0 }; { X = 3; Y = 1; Z = 0 }; { X = 2; Y = 2; Z = 2 } ]
                    Bounds = { MinX = 1; MaxX = 3; MinY = 1; MaxY = 2; MinZ = 0; MaxZ = 2 }
                    Size = 3UL } ]

            let painted =
                scalarPlan [ objects ]
                >=> paintObjects 5u 4u
                |> drainList

            try
                Expect.equal (painted |> List.map _.index) [ 0; 2 ] "Painting should emit only z slices that contain object positions."
                Expect.equal (painted[0][1, 2]) 1uy "First painted position should be one."
                Expect.equal (painted[0][3, 1]) 1uy "Second painted position should be one."
                Expect.equal (painted[0][0, 0]) 0uy "Background should be zero."
                Expect.equal (painted[1][2, 2]) 1uy "The z=2 position should be painted in the second emitted image."
            finally
                disposeImages painted

        testCase "paintObjectsCropped turns streamed objects into minimal local UInt8 masks" <| fun _ ->
            let objects: StreamedObject list =
                [ { Label = 7UL
                    Positions = [ { X = 4; Y = 5; Z = 10 }; { X = 6; Y = 5; Z = 10 }; { X = 5; Y = 6; Z = 11 } ]
                    Bounds = { MinX = 4; MaxX = 6; MinY = 5; MaxY = 6; MinZ = 10; MaxZ = 11 }
                    Size = 3UL } ]

            let painted =
                scalarPlan [ objects ]
                >=> paintObjectsCropped
                |> drainList

            try
                Expect.equal painted.Length 2 "A two-z object should emit two local slices."
                Expect.equal (painted |> List.map _.index) [ 0; 1 ] "Cropped masks should use local z indices."
                Expect.equal (painted[0].GetWidth()) 3u "The local mask width should match the object x bounds."
                Expect.equal (painted[0].GetHeight()) 2u "The local mask height should match the object y bounds."
                Expect.equal (painted[0][0, 0]) 1uy "The first point should be translated to the local origin."
                Expect.equal (painted[0][2, 0]) 1uy "The second point should be translated by MinX/MinY."
                Expect.equal (painted[1][1, 1]) 1uy "The z=11 point should appear in local z=1."
            finally
                disposeImages painted

        testCase "signedDistanceBand emits finite values near boundaries and NaN outside the band" <| fun _ ->
            let slices =
                [ for z in 0 .. 4 ->
                    let image =
                        Array2D.init 9 9 (fun x y ->
                            if x >= 3 && x <= 5 && y >= 3 && y <= 5 && z >= 1 && z <= 3 then 1uy else 0uy)
                        |> Image<uint8>.ofArray2D
                    image.index <- z
                    image ]
            let mutable distances: Image<float> list = []

            try
                distances <-
                    imagePlan slices
                    >=> signedDistanceBand 2u 1u
                    |> drainList

                let middle = distances |> List.find (fun image -> image.index = 2)
                Expect.isTrue (Double.IsNaN middle[0, 0]) "Pixels outside the requested band should be NaN."
                Expect.isFalse (Double.IsNaN middle[3, 3]) "Object boundary pixels should be finite."
                let finiteValues =
                    middle.toArray2D()
                    |> Seq.cast<float>
                    |> Seq.filter (Double.IsNaN >> not)
                    |> Seq.toList

                Expect.isNonEmpty finiteValues "The finite band should contain measurable distances."
                Expect.isTrue (finiteValues |> List.exists (fun value -> Math.Abs value > 0.0)) "The finite band should include non-zero distances away from the boundary."
            finally
                disposeImages distances
                disposeImages slices

        testCase "measureObjects derives per-object measurements and reducers summarize sizes" <| fun _ ->
            let firstBatch: StreamedObject list =
                [ { Label = 1UL
                    Positions = [ { X = 1; Y = 1; Z = 0 }; { X = 2; Y = 1; Z = 0 } ]
                    Bounds = { MinX = 1; MaxX = 2; MinY = 1; MaxY = 1; MinZ = 0; MaxZ = 0 }
                    Size = 2UL }
                  { Label = 2UL
                    Positions = [ { X = 4; Y = 4; Z = 0 }; { X = 4; Y = 5; Z = 0 }; { X = 4; Y = 6; Z = 0 } ]
                    Bounds = { MinX = 4; MaxX = 4; MinY = 4; MaxY = 6; MinZ = 0; MaxZ = 0 }
                    Size = 3UL } ]

            let secondBatch: StreamedObject list =
                [ { Label = 3UL
                    Positions = [ { X = 0; Y = 0; Z = 2 }; { X = 1; Y = 0; Z = 2 }; { X = 0; Y = 1; Z = 2 }; { X = 1; Y = 1; Z = 2 } ]
                    Bounds = { MinX = 0; MaxX = 1; MinY = 0; MaxY = 1; MinZ = 2; MaxZ = 2 }
                    Size = 4UL } ]

            let measurements =
                scalarPlan [ firstBatch; secondBatch ]
                >=> measureObjects
                |> drainList

            Expect.equal (measurements.[0].[0].Width) 2UL "Width should be derived from x bounds."
            Expect.equal (measurements.[0].[1].Height) 3UL "Height should be derived from y bounds."
            Expect.equal (measurements.[1].[0].Depth) 1UL "Depth should be derived from z bounds."

            let stats =
                scalarPlan [ firstBatch; secondBatch ]
                >=> measureObjects
                >=> objectSizeStats
                |> drain

            Expect.equal stats.Count 3UL "Three objects should be summarized."
            Expect.floatClose Accuracy.high stats.Mean 3.0 "Mean size should be calculated online."
            Expect.floatClose Accuracy.high stats.Variance 1.0 "Sample variance of sizes 2,3,4 should be one."
            Expect.equal stats.Minimum 2UL "Minimum size should be tracked."
            Expect.equal stats.Maximum 4UL "Maximum size should be tracked."

            let histogram =
                scalarPlan [ firstBatch; secondBatch ]
                >=> measureObjects
                >=> objectSizeHistogram 2UL
                |> drain

            Expect.equal histogram (Map.ofList [ 1UL, 2UL; 2UL, 1UL ]) "Size histogram should count bin index size/binWidth."

        testCase "writeInSlabs creates chunk files and chunk metadata can be read" <| fun _ ->
            let inputDir = tempDirectory "chunks-input"
            let chunkDir = tempDirectory "chunks-output"
            let suffix = ".tiff"
            let chunkSuffix = ".mha"
            let slices = [ for z in 0 .. 3 -> makeSlice 4 4 z ]
            let mutable slabs: Image<uint8> list = []
            let mutable rereadSlices: Image<uint8> list = []

            try
                writeSlices inputDir suffix slices

                source (2UL * 1024UL * 1024UL * 1024UL)
                |> read<uint8> inputDir suffix
                >=> writeInSlabs chunkDir chunkSuffix 2u 2u 2u
                |> sink

                let info = getChunkInfo chunkDir chunkSuffix
                Expect.equal info.chunks [ 2; 2; 2 ] "4x4x4 data written as 2x2x2 chunks should create a 2x2x2 chunk grid."
                Expect.equal info.size [ 4UL; 4UL; 4UL ] "Chunk metadata should reconstruct full volume size."
                Expect.isTrue (File.Exists(getChunkFilename chunkDir chunkSuffix 1 1 1)) "The final chunk file should exist."

                slabs <-
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> readSlabStacked<uint8> chunkDir chunkSuffix
                    |> drainList

                Expect.equal slabs.Length 2 "readSlabStacked should emit one full x-y slab per z chunk."
                Expect.equal (slabs[0].GetSize()) [ 4u; 4u; 2u ] "The first stacked slab should span the full x-y extent and the chunk z depth."

                rereadSlices <-
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> readSlab<uint8> chunkDir chunkSuffix
                    |> drainList

                Expect.equal rereadSlices.Length 4 "readSlab should unstack slabs into the normal 2D slice stream."
            finally
                disposeImages slabs
                disposeImages rereadSlices
                disposeImages slices
                deleteDirectory inputDir
                deleteDirectory chunkDir

        testCase "writeZarr writes OME-Zarr and readZarrSlab reads it back as slices and slabs" <| fun _ ->
            let inputDir = tempDirectory "zarr-input"
            let rootDir = tempDirectory "zarr-output"
            let suffix = ".tiff"
            let zarrPath = Path.Combine(rootDir, "roundtrip.zarr")
            let zarrDebugPath = Path.Combine(Directory.GetCurrentDirectory(), @"C:\Users\Public\biolog.txt")
            let slices = [ for z in 0 .. 3 -> makeSlice 5 4 z ]
            let mutable rereadSlices: Image<uint8> list = []
            let mutable rereadSlabs: Image<uint8> list = []

            try
                if File.Exists zarrDebugPath then
                    File.Delete zarrDebugPath

                writeSlices inputDir suffix slices

                source (2UL * 1024UL * 1024UL * 1024UL)
                |> read<uint8> inputDir suffix
                >=> writeZarr zarrPath "roundtrip" 4u 3u 2u 2u 1.0 1.0 2.0 0
                |> sink

                let info = getZarrInfo zarrPath 0 0
                Expect.equal info.size [ 5UL; 4UL; 4UL ] "Zarr metadata should expose x/y/z image size."
                Expect.equal info.topLeftInfo.componentType "uint8" "Zarr metadata should expose the dataset dtype."

                rereadSlabs <-
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> readZarrSlabStacked<uint8> zarrPath 2u 0 0 0 0 0
                    |> drainList

                Expect.equal rereadSlabs.Length 2 "readZarrSlabStacked should emit one stacked slab per requested z block."
                Expect.equal (rereadSlabs[0].GetSize()) [ 5u; 4u; 2u ] "The first Zarr slab should retain x/y and requested z depth."

                rereadSlices <-
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> readZarrSlab<uint8> zarrPath 2u 0 0 0 0 0
                    |> drainList

                Expect.equal rereadSlices.Length 4 "readZarrSlab should unstack slabs into a normal slice stream."
                let pixels = rereadSlices[3].toArray2D()
                Expect.equal pixels[4, 3] (uint8 ((4 + 2 * 3 + 3 * 3) % 251)) "Round-tripped Zarr pixel values should match the source stack."
                Expect.isFalse (File.Exists zarrDebugPath) "ZarrNET debug logging should not create a Windows-style biolog.txt side-effect."
            finally
                disposeImages rereadSlices
                disposeImages rereadSlabs
                disposeImages slices
                deleteDirectory inputDir
                deleteDirectory rootDir

        testCase "readNexusSlab reads a rank-3 HDF5 detector dataset as slices and slabs" <| fun _ ->
            let rootDir = tempDirectory "nexus-input"
            let nexusPath = Path.Combine(rootDir, "scan.h5")
            let datasetPath = "/entry/data/data"
            let data =
                Array3D.init 4 3 5 (fun z y x -> uint16 (x + 10 * y + 100 * z))
            let mutable rereadSlices: Image<uint16> list = []
            let mutable rereadSlabs: Image<uint16> list = []

            try
                writeNexusStack nexusPath datasetPath data

                let info = getNexusInfo nexusPath datasetPath 0 1 2
                Expect.equal info.size [ 5UL; 3UL; 4UL ] "NeXus metadata should expose x/y/z image size according to the axis mapping."

                rereadSlabs <-
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> readNexusSlabStacked<uint16> nexusPath datasetPath 2u 0 1 2
                    |> drainList

                Expect.equal rereadSlabs.Length 2 "readNexusSlabStacked should emit one stacked slab per requested frame block."
                Expect.equal (rereadSlabs[0].GetSize()) [ 5u; 3u; 2u ] "The first NeXus slab should retain x/y and requested z depth."

                rereadSlices <-
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> readNexusSlab<uint16> nexusPath datasetPath 2u 0 1 2
                    |> drainList

                Expect.equal rereadSlices.Length 4 "readNexusSlab should unstack slabs into a normal slice stream."
                let pixels = rereadSlices[3].toArray2D()
                Expect.equal pixels[4, 2] (uint16 (4 + 10 * 2 + 100 * 3)) "NeXus pixel values should match the source HDF5 dataset."
            finally
                disposeImages rereadSlices
                disposeImages rereadSlabs
                deleteDirectory rootDir

        testCase "writeNexus writes a rank-3 HDF5 detector dataset incrementally" <| fun _ ->
            let inputDir = tempDirectory "nexus-write-input"
            let rootDir = tempDirectory "nexus-write-output"
            let suffix = ".tiff"
            let nexusPath = Path.Combine(rootDir, "written.h5")
            let datasetPath = "/entry/data/data"
            let slices =
                [ for z in 0 .. 3 ->
                    Array2D.init 5 3 (fun x y -> uint16 (x + 10 * y + 100 * z))
                    |> Image<uint16>.ofArray2D ]
            let mutable rereadSlices: Image<uint16> list = []

            try
                writeSlices inputDir suffix slices

                source (2UL * 1024UL * 1024UL * 1024UL)
                |> read<uint16> inputDir suffix
                >=> writeNexus nexusPath datasetPath 4u 5u 3u 2u 0 1 2
                |> sink

                let info = getNexusInfo nexusPath datasetPath 0 1 2
                Expect.equal info.size [ 5UL; 3UL; 4UL ] "writeNexus metadata should expose x/y/z image size."

                rereadSlices <-
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> readNexusSlab<uint16> nexusPath datasetPath 2u 0 1 2
                    |> drainList

                Expect.equal rereadSlices.Length 4 "writeNexus output should be readable by readNexusSlab."
                let pixels = rereadSlices[2].toArray2D()
                Expect.equal pixels[4, 2] (uint16 (4 + 10 * 2 + 100 * 2)) "writeNexus should preserve pixel values."
            finally
                disposeImages rereadSlices
                disposeImages slices
                deleteDirectory inputDir
                deleteDirectory rootDir

        testCase "resize changes x y z size while preserving first coordinate" <| fun _ ->
            let inputDir = tempDirectory "resize-input"
            let suffix = ".tiff"
            let slices =
                [ for z in 0 .. 2 ->
                    Array2D.init 3 3 (fun x y -> uint16 (x + 10 * y + 100 * z))
                    |> Image<uint16>.ofArray2D ]
            let mutable resized: Image<uint16> list = []

            try
                writeSlices inputDir suffix slices

                resized <-
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> read<uint16> inputDir suffix
                    |> resize<uint16> 5u 4u 5u "NearestNeighbor"
                    |> drainList

                Expect.equal resized.Length 5 "resize should emit the requested z size."
                Expect.equal (resized[0].GetSize()) [ 5u; 4u ] "resize should emit the requested x-y slice size."
                let first = resized[0].toArray2D()
                Expect.equal first[0, 0] 0us "resize should keep the first input coordinate anchored at output 0,0,0."
            finally
                disposeImages resized
                disposeImages slices
                deleteDirectory inputDir

        testCase "resample factors change x y z size while preserving first coordinate" <| fun _ ->
            let inputDir = tempDirectory "resample-input"
            let suffix = ".tiff"
            let slices =
                [ for z in 0 .. 2 ->
                    Array2D.init 4 2 (fun x y -> float32 (x + 10 * y + 100 * z))
                    |> Image<float32>.ofArray2D ]
            let mutable resampled: Image<float32> list = []

            try
                writeSlices inputDir suffix slices

                resampled <-
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> read<float32> inputDir suffix
                    |> resample<float32> 0.5 2.0 2.0 "Linear"
                    |> drainList

                Expect.equal resampled.Length 6 "resample should scale the z size by the z factor."
                Expect.equal (resampled[0].GetSize()) [ 2u; 4u ] "resample should scale the x-y slice size by the x/y factors."
                let first = resampled[0].toArray2D()
                expectFloat32Close first[0, 0] 0.0f "resample should keep the first input coordinate anchored at output 0,0,0."
                let halfwayZ = resampled[1].toArray2D()
                expectFloat32Close halfwayZ[0, 0] 50.0f "Linear z resampling should interpolate midway between the first two input slices."
            finally
                disposeImages resampled
                disposeImages slices
                deleteDirectory inputDir

        testCase "image pair helpers perform arithmetic and extrema" <| fun _ ->
            let runPair f =
                let a = image2D (fun x y -> float32 (1 + x + 2 * y))
                let b = image2D (fun x y -> 2.0f * float32 (1 + x + 2 * y))
                f a b

            let results =
                [ runPair addPair
                  runPair (fun a b -> subPair b a)
                  runPair mulPair
                  runPair (fun a b -> divPair b a)
                  runPair maxOfPair
                  runPair minOfPair ]

            try
                expectFloat32Close results[0].[1, 1] 12.0f "addPair should add images."
                expectFloat32Close results[1].[1, 1] 4.0f "subPair should subtract images."
                expectFloat32Close results[2].[1, 1] 32.0f "mulPair should multiply images."
                expectFloat32Close results[3].[1, 1] 2.0f "divPair should divide images."
                expectFloat32Close results[4].[1, 1] 8.0f "maxOfPair should choose the larger value."
                expectFloat32Close results[5].[1, 1] 4.0f "minOfPair should choose the smaller value."

                let extremaImage = image2D (fun x y -> 2.0f * float32 (1 + x + 2 * y))
                let minValue, maxValue = getMinMax extremaImage
                Expect.equal (minValue, maxValue) (2.0, 8.0) "getMinMax should report extrema."
                extremaImage.decRefCount()
            finally
                results |> List.iter (fun image -> image.decRefCount())

        testCase "histogram reducer combines streamed slices" <| fun _ ->
            let inputDir = tempDirectory "histogram-input"
            let suffix = ".tiff"
            let slices =
                [ array2D [ [ 0uy; 1uy ]; [ 1uy; 2uy ] ] |> Image<uint8>.ofArray2D
                  array2D [ [ 2uy; 2uy ]; [ 3uy; 3uy ] ] |> Image<uint8>.ofArray2D ]

            try
                writeSlices inputDir suffix slices

                let actual =
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> read<uint8> inputDir suffix
                    >=> histogram ()
                    |> drain

                Expect.equal actual (Map.ofList [ 0uy, 1UL; 1uy, 2UL; 2uy, 3UL; 3uy, 2UL ]) "histogram should fold all streamed slices."
            finally
                disposeImages slices
                deleteDirectory inputDir

        testCase "histogram threshold estimators return scalar thresholds for the standard threshold stage" <| fun _ ->
            let histogram = Map.ofList [ 0.0f, 4UL; 10.0f, 4UL ]

            let otsu = otsuThresholdFromHistogram histogram
            let moments = momentsThresholdFromHistogram histogram

            Expect.isGreaterThan otsu 0.0 "Otsu threshold should lie between the two modes."
            Expect.isLessThan otsu 10.0 "Otsu threshold should lie between the two modes."
            Expect.isGreaterThan moments 0.0 "Moments threshold should lie between the two modes."
            Expect.isLessThan moments 10.0 "Moments threshold should lie between the two modes."

        testCase "sumProjection reduces a stack to one transformed Float64 image" <| fun _ ->
            let slices =
                [ array2D [ [ 1s; -2s ]; [ 3s; -4s ] ] |> Image<int16>.ofArray2D
                  array2D [ [ 2s; -3s ]; [ 4s; -5s ] ] |> Image<int16>.ofArray2D ]
            let mutable projections: Image<float> list = []

            try
                projections <-
                    imagePlan slices
                    >=> sumProjection<int16> "Abs"
                    |> drainList

                Expect.equal projections.Length 1 "sumProjection should emit one projection image."
                let projection = projections[0]
                Expect.equal projection.index 0 "The projection image should use index zero."
                expectFloat32Close (float32 projection[0, 0]) 3.0f "Projection should sum transformed values at x=0,y=0."
                expectFloat32Close (float32 projection[0, 1]) 5.0f "Projection should sum absolute values at x=0,y=1."
                expectFloat32Close (float32 projection[1, 0]) 7.0f "Projection should sum absolute values at x=1,y=0."
                expectFloat32Close (float32 projection[1, 1]) 9.0f "Projection should sum absolute values at x=1,y=1."
            finally
                disposeImages projections
                disposeImages slices

        testCase "map2pairs and pair conversions transform histogram data" <| fun _ ->
            let inputDir = tempDirectory "histogram-pairs-input"
            let suffix = ".tiff"
            let slices =
                [ [| [| 1uy; 1uy |]; [| 3uy; 3uy |] |]
                  [| [| 3uy; 3uy |]; [| 3uy; 3uy |] |] ]
                |> List.map (fun rows -> array2D rows |> Image<uint8>.ofArray2D)

            try
                writeSlices inputDir suffix slices

                let pairs =
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> read<uint8> inputDir suffix
                    >=> histogram ()
                    >=> map2pairs<uint8, uint64>
                    >=> pairs2floats<uint8, uint64>
                    |> drain

                Expect.equal pairs [ 1.0, 2.0; 3.0, 6.0 ] "map2pairs followed by pairs2floats should expose histogram coordinates."
            finally
                disposeImages slices
                deleteDirectory inputDir

        testCase "zip, teeSnd, and pair ignore support paired pipelines" <| fun _ ->
            let leftDir = tempDirectory "zip-left"
            let rightDir = tempDirectory "zip-right"
            let suffix = ".tiff"
            let left = [ for z in 0 .. 1 -> makeSlice 3 3 z ]
            let right = [ for z in 0 .. 1 -> makeSlice 3 3 (z + 1) ]

            try
                writeSlices leftDir suffix left
                writeSlices rightDir suffix right

                let leftPlan = source (2UL * 1024UL * 1024UL * 1024UL) |> read<uint8> leftDir suffix
                let rightPlan = source (2UL * 1024UL * 1024UL * 1024UL) |> read<uint8> rightDir suffix

                let pairs =
                    zip leftPlan rightPlan
                    >=> teeSnd (tap "right")
                    |> drainList

                Expect.equal pairs.Length 2 "zip should produce synchronized pairs."
                pairs |> List.iter (fun (l, r) -> l.decRefCount(); r.decRefCount())
            finally
                disposeImages left
                disposeImages right
                deleteDirectory leftDir
                deleteDirectory rightDir

        testCase "connected component tuple stream writes labels and builds a translation table" <| fun _ ->
            let inputDir = tempDirectory "components-input"
            let labelDir = tempDirectory "components-labels"
            let suffix = ".tiff"
            let labelSuffix = ".mha"
            let slices = [ for z in 0 .. 3 -> makeBinarySlice 8 8 z ]

            try
                writeSlices inputDir suffix slices

                let table =
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> read<uint8> inputDir suffix
                    >=> connectedComponents 2u
                    >=> teeFst (writeSlabSlices labelDir labelSuffix 2u)
                    >=> makeConnectedComponentTranslationTable 2u
                    |> drain

                Expect.isNonEmpty table.Labels "Connected component translation table should contain label mappings."
                Expect.isNonEmpty table.Statistics "Connected component translation table should include streaming component statistics."
                Expect.isTrue (File.Exists(Path.Combine(labelDir, "image_000.mha"))) "writeSlabSlices should write label slices from the tuple stream."
                Expect.isTrue (table.Labels |> List.exists (fun (_, sourceLabel, targetLabel) -> sourceLabel <> 0UL && targetLabel <> 0UL)) "The table should contain foreground label mappings."
                Expect.isTrue (table.Statistics |> List.forall (fun stats -> stats.Label <> 0UL && stats.NumberOfPixels > 0UL)) "Statistics should be reduced to non-background global labels."
                let expectedForeground =
                    slices
                    |> List.sumBy (Image.fold (fun count value -> if value = 0uy then count else count + 1UL) 0UL)
                let actualForeground = table.Statistics |> List.sumBy _.NumberOfPixels
                Expect.equal actualForeground expectedForeground "Streaming component statistics should account for every foreground voxel exactly once."
            finally
                disposeImages slices
                deleteDirectory inputDir
                deleteDirectory labelDir

        testCase "StackProcessing facade reads files through direct reader stages" <| fun _ ->
            let inputDir = tempDirectory "facade-readers"
            let suffix = ".tiff"
            let slices = [ for z in 0 .. 1 -> makeSlice 4 3 z ]

            try
                writeSlices inputDir suffix slices
                let files = Directory.GetFiles(inputDir, "*" + suffix) |> Array.sort

                let single =
                    scalarPlan [ files[0] ]
                    >=> StackProcessing.readFiles<uint8> false
                    |> drain

                try
                    Expect.equal (single.GetSize()) [ 4u; 3u ] "readFiles should read a single public filename stream element."
                finally
                    single.decRefCount()

                let shaped =
                    scalarPlan [ files[1] ]
                    >=> StackProcessing.readFilesWithShape<uint8> false 4u 3u
                    |> drain

                try
                    Expect.equal shaped.[0, 0] 3uy "readFilesWithShape should read with the supplied shape estimate."
                finally
                    shaped.decRefCount()

                let left, right =
                    scalarPlan [ files[0], files[1] ]
                    >=> StackProcessing.readFilePairs<uint8> false
                    |> drain

                try
                    Expect.equal left.[0, 0] 0uy "readFilePairs should read the first file."
                    Expect.equal right.[0, 0] 3uy "readFilePairs should read the second file."
                finally
                    left.decRefCount()
                    right.decRefCount()

                let filtered =
                    StackProcessing.source (2UL * 1024UL * 1024UL * 1024UL)
                    |> StackProcessing.readFiltered<uint8> inputDir suffix (fun names -> names |> Array.take 1)
                    |> StackProcessing.drainList

                try
                    Expect.equal filtered.Length 1 "readFiltered should expose the public filtered stack reader."
                    Expect.equal filtered[0].[0, 0] 0uy "readFiltered should preserve sorted filtered order."
                finally
                    disposeImages filtered
            finally
                disposeImages slices
                deleteDirectory inputDir

        testCase "StackProcessing facade exposes scalar image arithmetic wrappers" <| fun _ ->
            let image = image2D (fun x y -> float32 (x + y + 2))

            let run (stage: Stage<Image<float32>, Image<float32>>) =
                let result =
                    imagePlan [ image ]
                    >=> stage
                    |> drain
                let value = result.[0, 0]
                result.decRefCount()
                value

            try
                expectFloat32Close (run (StackProcessing.scalarAddImage<float32> 3.0f)) 5.0f "scalarAddImage should add scalar on the left."
                expectFloat32Close (run (StackProcessing.imageAddScalar<float32> 3.0f)) 5.0f "imageAddScalar should add scalar on the right."
                expectFloat32Close (run (StackProcessing.scalarSubImage<float32> 10.0f)) 8.0f "scalarSubImage should subtract the image from the scalar."
                expectFloat32Close (run (StackProcessing.imageSubScalar<float32> 1.0f)) 1.0f "imageSubScalar should subtract the scalar from the image."
                expectFloat32Close (run (StackProcessing.scalarMulImage<float32> 4.0f)) 8.0f "scalarMulImage should multiply."
                expectFloat32Close (run (StackProcessing.imageMulScalar<float32> 4.0f)) 8.0f "imageMulScalar should multiply."
                expectFloat32Close (run (StackProcessing.scalarDivImage<float32> 8.0f)) 4.0f "scalarDivImage should divide scalar by image."
                expectFloat32Close (run (StackProcessing.imageDivScalar<float32> 2.0f)) 1.0f "imageDivScalar should divide image by scalar."
            finally
                image.decRefCount()

        testCase "StackProcessing facade exposes pair comparisons and mask wrappers" <| fun _ ->
            let runPairStage (stage: Stage<Image<uint8> * Image<uint8>, Image<uint8>>) (leftValue: uint8) (rightValue: uint8) =
                let left = Image<uint8>.ofArray2D (Array2D.create 2 2 leftValue)
                let right = Image<uint8>.ofArray2D (Array2D.create 2 2 rightValue)
                let result =
                    scalarPlan [ left, right ]
                    >=> stage
                    |> drain
                let value = result.[0, 0]
                result.decRefCount()
                value

            Expect.equal (runPairStage StackProcessing.equal<uint8> 7uy 7uy) 1uy "equal should flag equal pixels."
            Expect.equal (runPairStage StackProcessing.notEqual<uint8> 7uy 8uy) 1uy "notEqual should flag differing pixels."
            Expect.equal (runPairStage StackProcessing.greater<uint8> 9uy 8uy) 1uy "greater should flag greater pixels."
            Expect.equal (runPairStage StackProcessing.greaterEqual<uint8> 8uy 8uy) 1uy "greaterEqual should include equality."
            Expect.equal (runPairStage StackProcessing.less<uint8> 7uy 8uy) 1uy "less should flag lesser pixels."
            Expect.equal (runPairStage StackProcessing.lessEqual<uint8> 8uy 8uy) 1uy "lessEqual should include equality."
            Expect.equal (runPairStage StackProcessing.maskAnd 255uy 0uy) 0uy "maskAnd should combine binary masks."
            Expect.equal (runPairStage StackProcessing.maskOr 255uy 0uy) 255uy "maskOr should combine binary masks."
            Expect.equal (runPairStage StackProcessing.maskXor 255uy 0uy) 255uy "maskXor should combine binary masks."

            let mask = Image<uint8>.ofArray2D (Array2D.create 2 2 0uy)
            let inverted =
                imagePlan [ mask ]
                >=> StackProcessing.maskNot
                |> drain
            try
                Expect.equal inverted.[0, 0] 1uy "maskNot should invert binary masks."
            finally
                inverted.decRefCount()
                mask.decRefCount()

        testCase "StackProcessing facade exposes vector image wrappers" <| fun _ ->
            let makeX () = Image<float>.ofArray2D (array2D [ [ 1.0; 2.0 ]; [ 3.0; 4.0 ] ])
            let makeY () = Image<float>.ofArray2D (array2D [ [ 10.0; 20.0 ]; [ 30.0; 40.0 ] ])
            let makeZ () = Image<float>.ofArray2D (array2D [ [ 100.0; 200.0 ]; [ 300.0; 400.0 ] ])
            let makeZero () = Image<float>.ofArray2D (array2D [ [ 0.0; 0.0 ]; [ 0.0; 0.0 ] ])
            let makeOne () = Image<float>.ofArray2D (array2D [ [ 1.0; 1.0 ]; [ 1.0; 1.0 ] ])
            let makeVector3ViaStage zImage =
                let x, y = makeX (), makeY ()
                let vector2 = ImageFunctions.toVectorImage [ x; y ]
                x.decRefCount()
                y.decRefCount()

                scalarPlan [ vector2, zImage ]
                >=> StackProcessing.appendVectorElement
                |> drain

            let vector2 =
                scalarPlan [ makeX (), makeY () ]
                >=> StackProcessing.toVectorImage<float>
                |> drain

            try
                Expect.equal vector2.[1, 1] [ 4.0; 40.0 ] "toVectorImage should combine synchronized scalar images into vector pixels."

                let second =
                    imagePlan [ vector2 ]
                    >=> StackProcessing.vectorElement<float> 1u
                    |> drain

                try
                    Expect.equal second.[1, 1] 40.0 "vectorElement should extract a scalar component stream."
                finally
                    second.decRefCount()

                let mapped =
                    imagePlan [ vector2 ]
                    >=> StackProcessing.vectorMapElements "square"
                    |> drain

                try
                    Expect.equal mapped.[1, 1] [ 16.0; 1600.0 ] "vectorMapElements should map scalar functions over every component."
                finally
                    mapped.decRefCount()
            finally
                vector2.decRefCount()

            let vector3a = makeVector3ViaStage (makeZ ())

            let unitY =
                let zero, one = makeZero (), makeOne ()
                let vector2 = ImageFunctions.toVectorImage [ zero; one ]
                zero.decRefCount()
                one.decRefCount()
                scalarPlan [ vector2, makeZero () ]
                >=> StackProcessing.appendVectorElement
                |> drain

            let cross =
                scalarPlan [ vector3a, unitY ]
                >=> StackProcessing.vectorCross3D
                |> drain

            try
                Expect.equal cross.[1, 1] [ -400.0; 0.0; 4.0 ] "vectorCross3D should compute a cross product per pixel."
            finally
                cross.decRefCount()

            let vector3b = makeVector3ViaStage (makeZ ())

            let unitY2 =
                let zero, one = makeZero (), makeOne ()
                let vector2 = ImageFunctions.toVectorImage [ zero; one ]
                zero.decRefCount()
                one.decRefCount()
                scalarPlan [ vector2, makeZero () ]
                >=> StackProcessing.appendVectorElement
                |> drain

            let dot =
                scalarPlan [ vector3b, unitY2 ]
                >=> StackProcessing.vectorDot
                |> drain

            try
                Expect.equal dot.[1, 1] 40.0 "vectorDot should compute a dot product per pixel."
            finally
                dot.decRefCount()

        testCase "StackProcessing gradient emits streamed 3-vector finite differences" <| fun _ ->
            let makeDoubleSlice z =
                Array2D.init 5 5 (fun x y -> float x + 10.0 * float y + 100.0 * float z)
                |> Image<float>.ofArray2D

            let inputSlices = [ 0 .. 4 ] |> List.map makeDoubleSlice
            let directSlices = [ 0 .. 4 ] |> List.map makeDoubleSlice

            let actual =
                imagePlan inputSlices
                >=> StackProcessing.gradient 1u (Some 3u)
                |> drainList

            let volume = ImageFunctions.stack directSlices
            let expectedVolume = ImageFunctions.gradientVector3D 1u volume
            let expected = ImageFunctions.unstack 2u expectedVolume

            try
                Expect.equal actual.Length inputSlices.Length "gradient should preserve slice count."
                Expect.equal (actual.Head.GetNumberOfComponentsPerPixel()) 3u "gradient should emit 3-component vector pixels."
                List.zip actual expected
                |> List.iteri (fun z (streamed, direct) ->
                    Expect.equal (streamed.GetSize()) (direct.GetSize()) $"gradient slice {z} should preserve x/y shape."
                    Expect.equal streamed.[2, 2] direct.[2, 2] $"gradient slice {z} should match direct 3D finite differences at an interior pixel.")
            finally
                disposeImages actual
                disposeImages expected
                expectedVolume.decRefCount()
                volume.decRefCount()
                disposeImages directSlices

        testCase "StackProcessing structureTensor emits four 3-vector images per input slice" <| fun _ ->
            let makeDoubleSlice z =
                Array2D.init 5 5 (fun x _ -> float x + 0.0 * float z)
                |> Image<float>.ofArray2D

            let inputSlices = [ 0 .. 4 ] |> List.map makeDoubleSlice
            let actual =
                imagePlan inputSlices
                >=> StackProcessing.structureTensor 0.0 0.0
                |> drainList

            try
                Expect.equal actual.Length (4 * inputSlices.Length) "structureTensor should emit eigenvalues plus three eigenvector images for each input slice."
                actual |> List.iter (fun image -> Expect.equal (image.GetNumberOfComponentsPerPixel()) 3u "structureTensor outputs should be 3-vector images.")
                Expect.equal actual[8].[2, 2] [ 1.0; 0.0; 0.0 ] "The middle eigenvalue image should detect a pure x-gradient."
                Expect.equal actual[9].[2, 2] [ 1.0; 0.0; 0.0 ] "The first eigenvector image should point along x for a pure x-gradient."
            finally
                disposeImages actual

            let eigenvalues =
                imagePlan ([ 0 .. 4 ] |> List.map makeDoubleSlice)
                >=> StackProcessing.structureTensor 0.0 0.0
                >=> StackProcessing.selectGroupedOutput 4u 0u
                |> drainList

            try
                Expect.equal eigenvalues.Length inputSlices.Length "selectGroupedOutput should select one of the four output streams."
                Expect.equal eigenvalues[2].[2, 2] [ 1.0; 0.0; 0.0 ] "selectGroupedOutput 4 0 should select eigenvalues."
            finally
                disposeImages eigenvalues

        testCase "StackProcessing PCA reduces 3-vector images to eigensystem streams" <| fun _ ->
            let vectors = new Image<float list>([ 2u; 1u ], 3u, "pca-input", 0)
            vectors.[0, 0] <- [ 1.0; 0.0; 0.0 ]
            vectors.[1, 0] <- [ -1.0; 0.0; 0.0 ]

            let actual =
                imagePlan [ vectors ]
                >=> StackProcessing.PCA 3u
                |> drainList

            try
                Expect.equal actual.Length 4 "PCA should emit eigenvalues plus three eigenvector images."
                actual |> List.iter (fun image ->
                    Expect.equal (image.GetSize()) [ 1u; 1u ] "PCA outputs should be singleton images."
                    Expect.equal (image.GetNumberOfComponentsPerPixel()) 3u "PCA outputs should be 3-vector images.")
                Expect.equal actual[0].[0, 0] [ 1.0; 0.0; 0.0 ] "PCA eigenvalues should capture variance along x."
                Expect.floatClose Accuracy.high (System.Math.Abs(actual[1].[0, 0].[0])) 1.0 "The first PCA eigenvector should point along x, up to sign."
            finally
                disposeImages actual

            let vectors2D = new Image<float list>([ 2u; 1u ], 2u, "pca-input-2d", 0)
            vectors2D.[0, 0] <- [ 2.0; 0.0 ]
            vectors2D.[1, 0] <- [ -2.0; 0.0 ]

            let actual2D =
                imagePlan [ vectors2D ]
                >=> StackProcessing.PCA 2u
                |> drainList

            try
                Expect.equal actual2D.Length 3 "2-component PCA should emit eigenvalues plus two eigenvector images."
                actual2D |> List.iter (fun image -> Expect.equal (image.GetNumberOfComponentsPerPixel()) 2u "2-component PCA outputs should be 2-vector images.")
                Expect.equal actual2D[0].[0, 0] [ 4.0; 0.0 ] "2-component PCA eigenvalues should capture variance along x."
            finally
                disposeImages actual2D

        testCase "StackProcessing facade exposes complex image wrappers" <| fun _ ->
            let makeReal () = Image<float>.ofArray2D (array2D [ [ 3.0; 1.0 ]; [ -2.0; 0.0 ] ])
            let makeImag () = Image<float>.ofArray2D (array2D [ [ 4.0; -1.0 ]; [ 2.0; -5.0 ] ])
            let makeComplex () =
                let real, imag = makeReal (), makeImag ()
                let complexImage = Image.toComplex real imag
                real.decRefCount()
                imag.decRefCount()
                complexImage

            let complexImage =
                scalarPlan [ makeReal (), makeImag () ]
                >=> StackProcessing.toComplex
                |> drain

            try
                Expect.equal complexImage.[0, 0] (System.Numerics.Complex(3.0, 4.0)) "toComplex should compose real and imaginary streams."
            finally
                complexImage.decRefCount()

            let realPart =
                scalarPlan [ makeComplex () ]
                >=> StackProcessing.Re
                |> drain

            try
                Expect.floatClose Accuracy.high realPart.[0, 0] 3.0 "Re should extract the real stream."
            finally
                realPart.decRefCount()

            let imagPart =
                scalarPlan [ makeComplex () ]
                >=> StackProcessing.Im
                |> drain

            try
                Expect.floatClose Accuracy.high imagPart.[0, 0] 4.0 "Im should extract the imaginary stream."
            finally
                imagPart.decRefCount()

            let magnitude =
                scalarPlan [ makeComplex () ]
                >=> StackProcessing.modulus
                |> drain

            try
                Expect.floatClose Accuracy.high magnitude.[0, 0] 5.0 "modulus should compute complex magnitude."
            finally
                magnitude.decRefCount()

            let phase =
                scalarPlan [ makeComplex () ]
                >=> StackProcessing.arg
                |> drain

            try
                Expect.floatClose Accuracy.high phase.[0, 0] (Math.Atan2(4.0, 3.0)) "arg should compute complex phase."
            finally
                phase.decRefCount()

            let conjugated =
                scalarPlan [ makeComplex () ]
                >=> StackProcessing.conjugate
                |> drain

            try
                Expect.equal conjugated.[0, 0] (System.Numerics.Complex(3.0, -4.0)) "conjugate should negate the imaginary stream."
            finally
                conjugated.decRefCount()

            let fromPolar =
                scalarPlan [
                    Image<float>.ofArray2D (array2D [ [ 5.0 ] ]),
                    Image<float>.ofArray2D (array2D [ [ Math.Atan2(4.0, 3.0) ] ])
                ]
                >=> StackProcessing.polarToComplex
                |> drain

            try
                Expect.floatClose Accuracy.high fromPolar.[0, 0].Real 3.0 "polarToComplex should recover the real component."
                Expect.floatClose Accuracy.high fromPolar.[0, 0].Imaginary 4.0 "polarToComplex should recover the imaginary component."
            finally
                fromPolar.decRefCount()

        testCase "StackProcessing FFT invFFT and shiftFFT stream through chunk workspace" <| fun _ ->
            let makeSlice z =
                Image<float>.ofArray2D (
                    Array2D.init 2 2 (fun x y ->
                        if x = 0 && y = 0 && z = 0 then 1.0 else 0.0))

            let inputSlices = [ makeSlice 0; makeSlice 1 ]
            let fftSlices =
                imagePlan inputSlices
                >=> StackProcessing.FFT<float> 1u 1u 1u
                |> drainList

            try
                Expect.equal fftSlices.Length 2 "FFT should emit one complex slice per input slice."
                for slice in fftSlices do
                    for x in 0 .. 1 do
                        for y in 0 .. 1 do
                            expectComplexClose slice.[x, y] System.Numerics.Complex.One "3D FFT of an impulse should be one everywhere."

                let recovered =
                    imagePlan fftSlices
                    >=> StackProcessing.invFFT 1u 1u 1u
                    |> drainList

                try
                    Expect.equal recovered.Length 2 "invFFT should emit one real slice per frequency slice."
                    Expect.floatClose Accuracy.high recovered[0].[0, 0] 1.0 "invFFT should recover the impulse."
                    Expect.floatClose Accuracy.high recovered[0].[1, 0] 0.0 "invFFT should recover zero pixels."
                    Expect.floatClose Accuracy.high recovered[1].[0, 0] 0.0 "invFFT should recover zero z-slices."
                finally
                    disposeImages recovered

                let shifted =
                    imagePlan fftSlices
                    >=> StackProcessing.shiftFFT 1u 1u 1u
                    |> drainList

                try
                    Expect.equal shifted.Length 2 "shiftFFT should preserve output slice count."
                    for slice in shifted do
                        for x in 0 .. 1 do
                            for y in 0 .. 1 do
                                expectComplexClose slice.[x, y] System.Numerics.Complex.One "Shifting a constant spectrum should remain constant."
                finally
                    disposeImages shifted
            finally
                disposeImages fftSlices
                disposeImages inputSlices

        testCase "StackProcessing facade exposes local image filter wrappers" <| fun _ ->
            let floatSlices = [ for z in 0 .. 2 -> makeFloatSlice 5 5 z ]
            let uintSlices = [ for z in 0 .. 2 -> makeSlice 5 5 z ]
            let binarySlices = [ for z in 0 .. 2 -> makeBinarySlice 6 6 z ]

            let expectSameShapeFloat32 label (input: Image<float32> list) (stage: Stage<Image<float32>, Image<float32>>) =
                let actual =
                    imagePlan input
                    >=> stage
                    |> drainList

                try
                    Expect.equal actual.Length input.Length $"{label} should preserve slice count."
                    actual |> List.iter (fun image -> Expect.equal (image.GetSize()) (input.Head.GetSize()) $"{label} should preserve x/y shape.")
                finally
                    disposeImages actual

            let expectSameShapeUint8 label (input: Image<uint8> list) (stage: Stage<Image<uint8>, Image<uint8>>) =
                let actual =
                    imagePlan input
                    >=> stage
                    |> drainList

                try
                    Expect.equal actual.Length input.Length $"{label} should preserve slice count."
                    actual |> List.iter (fun image -> Expect.equal (image.GetSize()) (input.Head.GetSize()) $"{label} should preserve x/y shape.")
                finally
                    disposeImages actual

            try
                let clampStage : Stage<Image<float32>, Image<float32>> = StackProcessing.clamp<float32> 0.0 120.0
                let shiftScaleStage : Stage<Image<float32>, Image<float32>> = StackProcessing.shiftScale<float32> 1.0 2.0
                let stretchStage : Stage<Image<float32>, Image<float32>> = StackProcessing.intensityStretch<float32> 0.0 200.0 0.0 1.0
                let medianStage : Stage<Image<uint8>, Image<uint8>> = StackProcessing.smoothWMedian<uint8> 1u 3u
                let gradientStage : Stage<Image<float32>, Image<float32>> = StackProcessing.gradientMagnitude<float32> 3u
                let sobelStage : Stage<Image<float32>, Image<float32>> = StackProcessing.sobelEdge<float32> 3u
                let laplacianStage : Stage<Image<float32>, Image<float32>> = StackProcessing.laplacian<float32> 3u
                let erodeStage : Stage<Image<uint8>, Image<uint8>> = StackProcessing.grayscaleErode<uint8> 1u 3u
                let dilateStage : Stage<Image<uint8>, Image<uint8>> = StackProcessing.grayscaleDilate<uint8> 1u 3u
                let openingStage : Stage<Image<uint8>, Image<uint8>> = StackProcessing.grayscaleOpening<uint8> 1u 3u
                let closingStage : Stage<Image<uint8>, Image<uint8>> = StackProcessing.grayscaleClosing<uint8> 1u 3u
                let whiteTopHatStage : Stage<Image<uint8>, Image<uint8>> = StackProcessing.whiteTopHat<uint8> 1u 3u
                let blackTopHatStage : Stage<Image<uint8>, Image<uint8>> = StackProcessing.blackTopHat<uint8> 1u 3u
                let morphGradientStage : Stage<Image<uint8>, Image<uint8>> = StackProcessing.morphologicalGradient<uint8> 1u 3u
                let binaryContourStage : Stage<Image<uint8>, Image<uint8>> = StackProcessing.binaryContour false 3u
                let binaryMedianStage : Stage<Image<uint8>, Image<uint8>> = StackProcessing.binaryMedian 1u 3u
                let labelContourStage : Stage<Image<uint8>, Image<uint8>> = StackProcessing.labelContour<uint8> false 3u
                let changeLabelStage : Stage<Image<uint8>, Image<uint8>> = StackProcessing.changeLabel<uint8> 255.0 128.0
                let saltAndPepperStage : Stage<Image<float32>, Image<float32>> = StackProcessing.addSaltAndPepperNoise 0.0
                let shotStage : Stage<Image<float32>, Image<float32>> = StackProcessing.addShotNoise 0.0
                let speckleStage : Stage<Image<float32>, Image<float32>> = StackProcessing.addSpeckleNoise 0.0

                expectSameShapeFloat32 "clamp" floatSlices clampStage
                expectSameShapeFloat32 "shiftScale" floatSlices shiftScaleStage
                expectSameShapeFloat32 "intensityStretch" floatSlices stretchStage
                expectSameShapeUint8 "smoothWMedian" uintSlices medianStage
                expectSameShapeFloat32 "gradientMagnitude" floatSlices gradientStage
                expectSameShapeFloat32 "sobelEdge" floatSlices sobelStage
                expectSameShapeFloat32 "laplacian" floatSlices laplacianStage
                expectSameShapeUint8 "grayscaleErode" uintSlices erodeStage
                expectSameShapeUint8 "grayscaleDilate" uintSlices dilateStage
                expectSameShapeUint8 "grayscaleOpening" uintSlices openingStage
                expectSameShapeUint8 "grayscaleClosing" uintSlices closingStage
                expectSameShapeUint8 "whiteTopHat" uintSlices whiteTopHatStage
                expectSameShapeUint8 "blackTopHat" uintSlices blackTopHatStage
                expectSameShapeUint8 "morphologicalGradient" uintSlices morphGradientStage
                expectSameShapeUint8 "binaryContour" binarySlices binaryContourStage
                expectSameShapeUint8 "binaryMedian" binarySlices binaryMedianStage
                expectSameShapeUint8 "labelContour" binarySlices labelContourStage
                expectSameShapeUint8 "changeLabel" binarySlices changeLabelStage
                expectSameShapeFloat32 "addSaltAndPepperNoise" floatSlices saltAndPepperStage
                expectSameShapeFloat32 "addShotNoise" floatSlices shotStage
                expectSameShapeFloat32 "addSpeckleNoise" floatSlices speckleStage
            finally
                disposeImages floatSlices
                disposeImages uintSlices
                disposeImages binarySlices

        testCase "StackProcessing facade exposes histogram and pair conversion helpers" <| fun _ ->
            let histogramMap = Map.ofList [ 1, 2UL; 3, 4UL ]
            let pairs =
                scalarPlan [ histogramMap ]
                >=> StackProcessing.map2pairs<int, uint64>
                |> drain

            Expect.equal pairs [ 1, 2UL; 3, 4UL ] "map2pairs should expose map entries through the facade."

            let floatPairs =
                scalarPlan [ pairs ]
                >=> StackProcessing.pairs2floats<int, uint64>
                |> drain

            Expect.equal floatPairs [ 1.0, 2.0; 3.0, 4.0 ] "pairs2floats should convert numeric pairs through the facade."

            let intPairs =
                scalarPlan [ pairs ]
                >=> StackProcessing.pairs2ints<int, uint64>
                |> drain

            Expect.equal intPairs [ 1, 2; 3, 4 ] "pairs2ints should convert numeric pairs through the facade."

            let image = Image<float32>.ofArray2D (array2D [ [ 1.0f; 2.0f ]; [ 3.0f; 4.0f ] ])
            let minValue, maxValue = StackProcessing.getMinMax image
            Expect.equal minValue 1.0 "getMinMax should be exposed through the facade."
            Expect.equal maxValue 4.0 "getMinMax should be exposed through the facade."

            Expect.throws
                (fun () -> StackProcessing.failTypeMismatch<uint8> "facade" [ typeof<float32> ])
                "failTypeMismatch should be exposed through the facade."

    ]
