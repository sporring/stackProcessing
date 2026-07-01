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

let private nonlinearVoxel x y z =
    float32 (x * x + 7 * y + 31 * z + 3 * x * y + 5 * y * z + 11 * x * z)

let private trilinearArraySample (background: float32) (voxels: float32[,,]) (c: V3) =
    let width = Array3D.length1 voxels
    let height = Array3D.length2 voxels
    let depth = Array3D.length3 voxels

    if c.x < 0.0 || c.y < 0.0 || c.z < 0.0 || c.x >= float (width - 1) || c.y >= float (height - 1) || c.z >= float (depth - 1) then
        background
    else
        let x0 = int (Math.Floor c.x)
        let y0 = int (Math.Floor c.y)
        let z0 = int (Math.Floor c.z)
        let x1 = x0 + 1
        let y1 = y0 + 1
        let z1 = z0 + 1
        let fx = float32 (c.x - float x0)
        let fy = float32 (c.y - float y0)
        let fz = float32 (c.z - float z0)
        let lerp a b t = a + (b - a) * t

        let c00 = lerp voxels[x0, y0, z0] voxels[x1, y0, z0] fx
        let c10 = lerp voxels[x0, y1, z0] voxels[x1, y1, z0] fx
        let c01 = lerp voxels[x0, y0, z1] voxels[x1, y0, z1] fx
        let c11 = lerp voxels[x0, y1, z1] voxels[x1, y1, z1] fx
        let c0 = lerp c00 c10 fy
        let c1 = lerp c01 c11 fy
        lerp c0 c1 fz

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

let private pairStage name f =
    Stage.map name (fun _ (left, right) -> f left right) (fun _ -> 0UL) id

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

        testCase "fork builds a synchronized fan-out stage" <| fun _ ->
            let stage =
                fork (scalarStage "increment" ((+) 1), scalarStage "double" ((*) 2))

            let actual =
                scalarPlan [ 1; 2; 3 ]
                >=> stage
                |> drainList

            Expect.equal actual [ 2, 2; 3, 4; 4, 6 ] "fork should produce paired branch outputs as a reusable stage."

        testCase "ignorePairs consumes arbitrary paired branch outputs" <| fun _ ->
            scalarPlan [ 1; 2; 3 ]
            >=> fork (scalarStage "increment" ((+) 1), scalarStage "double" ((*) 2))
            >=> ignorePairs ()
            |> sink

        testCase "-->> composes a stage with a synchronized fan-out" <| fun _ ->
            let stage =
                scalarStage "increment" ((+) 1)
                -->> (scalarStage "left" ((+) 10), scalarStage "right" ((*) 10))

            let actual =
                scalarPlan [ 1; 2; 3 ]
                >=> stage
                |> drainList

            Expect.equal actual [ 12, 20; 13, 30; 14, 40 ] "-->> should compose a stage with a reusable fan-out."

        testCase ">=>> forks a synchronized stream into two stages" <| fun _ ->
            let actual =
                scalarPlan [ 1; 2; 3 ]
                >=>> (scalarStage "increment" ((+) 1), scalarStage "double" ((*) 2))
                |> drainList

            Expect.equal actual [ 2, 2; 3, 4; 4, 6 ] ">=>> should produce paired branch outputs."

        testCase "identity passes stream elements through unchanged" <| fun _ ->
            let actual =
                scalarPlan [ 1; 2; 3 ]
                >=> identity<int>
                |> drainList

            Expect.equal actual [ 1; 2; 3 ] "identity should be available from StackProcessing and preserve values."

        testCase ">=>> does not deadlock when one branch waits for a window" <| fun _ ->
            let delayed =
                Stage.window "delayed window" 3u 1u (fun _ _ -> 0) 1u
                --> Stage.flattenWindow "delayed flatten"

            let actual =
                scalarPlan [ 1; 2; 3 ]
                >=>> (scalarStage "fast" ((+) 10), delayed)
                |> drainList

            Expect.equal actual [ 11, 1; 12, 2; 13, 3 ] ">=>> should queue early requests from the fast branch."

        testCase "fork does not deadlock when one branch waits for a window" <| fun _ ->
            let delayed =
                Stage.window "delayed fork window" 3u 1u (fun _ _ -> 0) 1u
                --> Stage.flattenWindow "delayed fork flatten"

            let actual =
                scalarPlan [ 1; 2; 3 ]
                >=> fork (scalarStage "fast fork" ((+) 10), delayed)
                |> drainList

            Expect.equal actual [ 11, 1; 12, 2; 13, 3 ] "fork should use synchronized distribution and queue early requests."

        testCase ">=>> rejects branches with different slice domains" <| fun _ ->
            Expect.throws
                (fun () ->
                    scalarPlan [ 1; 2; 3 ]
                    >=>> (scalarStage "preserve" id, Stage.skip "skip first" 1u)
                    |> ignore)
                ">=>> should reject synchronization when one branch skips slices."

        testCase ">=>> rejects mixed streaming and reducer branch outputs" <| fun _ ->
            Expect.throws
                (fun () ->
                    scalarPlan [ 1; 2; 3 ]
                    >=>> (scalarStage "streaming" id, Stage.fold "sum" (+) 0 id (fun _ -> 1UL))
                    |> ignore)
                ">=>> should reject synchronization between streaming and reducer branch outputs."

        testCase ">>=> maps synchronized pairs to a single stream" <| fun _ ->
            let actual =
                scalarPlan [ 1; 2; 3 ]
                >=>> (scalarStage "increment" ((+) 1), scalarStage "double" ((*) 2))
                >>=> pairStage "add pair" (+)
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
                Expect.isTrue plan.optimize "commandLineSource should keep the optimizer enabled by default."
                Expect.equal (rest |> Array.toList) [ "alpha" ] "commandLineSource should remove the debug arguments."
            finally
                DebugLevel.set 0u
                Image<uint8>.setDebugLevel 0u

        testCase "command line source parses optimizer control independently" <| fun _ ->
            try
                let plan, rest = commandLineSource 1024UL [| "--no-optimize"; "-d"; "1"; "alpha" |]

                Expect.isTrue plan.debug "Debug output should still be controlled by -d."
                Expect.equal plan.debugLevel 1u "The debug level should still come from -d."
                Expect.isFalse plan.optimize "Optimizer control should be independent from debug output."
                Expect.equal (rest |> Array.toList) [ "alpha" ] "Optimizer arguments should be consumed."
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
                "poissonNoise"
                (fun () ->
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> StackProcessing.poissonNoise<float> 4u 3u 2u 0.0)

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
                    |> readRange<uint8> 1u 2 UInt32.MaxValue inputDir suffix
                    |> drainList

                try
                    Expect.equal rangeSlices.Length 1 "readRange should read first, first+step, ... up to last."
                    Expect.equal (rangeSlices[0].[0, 0]) 3uy "readRange should preserve the requested sorted slice order."
                finally
                    disposeImages rangeSlices

                let clampedRangeSlices =
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> readRange<uint8> 0u 2 UInt32.MaxValue inputDir suffix
                    |> drainList

                try
                    Expect.equal clampedRangeSlices.Length 2 "readRange should clamp endpoints outside the stack."
                    Expect.equal (clampedRangeSlices[0].[0, 0]) 0uy "A clamped first endpoint should begin at slice zero."
                    Expect.equal (clampedRangeSlices[1].[0, 0]) 6uy "A clamped last endpoint should allow the final stepped slice."
                finally
                    disposeImages clampedRangeSlices

                let reverseRangeSlices =
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> readRange<uint8> UInt32.MaxValue -1 0u inputDir suffix
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

        testCase "StackIO TIFF stack backend roundtrips scalar slices" <| fun _ ->
            let simpleItkInputDir = tempDirectory "fast-tiff-input"
            let fastOutputDir = tempDirectory "fast-tiff-output"
            let suffix = ".tiff"
            let slices = [ for z in 0 .. 2 -> makeSlice 6 5 z ]

            try
                writeSlices simpleItkInputDir suffix slices

                let fastRead =
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> read<uint8> simpleItkInputDir suffix
                    |> drainList

                try
                    Expect.equal fastRead.Length slices.Length "TIFF read should emit every slice."
                    Expect.equal (fastRead[2].GetSize()) [ 6u; 5u ] "TIFF read should preserve x/y shape."
                    Expect.equal fastRead[2].[4, 3] slices[2].[4, 3] "TIFF read should preserve pixel values."

                    let written =
                        imagePlan fastRead
                        >=> write fastOutputDir suffix
                        |> drainList

                    try
                        disposeImages written

                        let simpleItkReadBack =
                            source (2UL * 1024UL * 1024UL * 1024UL)
                            |> read<uint8> fastOutputDir suffix
                            |> drainList

                        try
                            Expect.equal simpleItkReadBack.Length slices.Length "TIFF write should create a normal TIFF stack."
                            Expect.equal (simpleItkReadBack[1].GetSize()) [ 6u; 5u ] "TIFF-written slices should be readable."
                            Expect.equal simpleItkReadBack[1].[5, 4] slices[1].[5, 4] "TIFF write should preserve pixel values."
                        finally
                            disposeImages simpleItkReadBack
                    finally
                        ()
                finally
                    disposeImages fastRead
            finally
                disposeImages slices
                deleteDirectory simpleItkInputDir
                deleteDirectory fastOutputDir

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
                    >=> marchingCubes 0.5
                    |> drainList

                Expect.isTrue (triangleSets |> List.exists (fun triangleSet -> not triangleSet.Triangles.IsEmpty)) "marchingCubes should emit at least one triangle set for a crossing surface."

                source (2UL * 1024UL * 1024UL * 1024UL)
                |> read<uint8> inputDir suffix
                >=> marchingCubes 0.5
                >=> writeMesh outputPath "auto"
                |> drain

                let meshText = File.ReadAllText(outputPath)
                Expect.stringContains meshText "v " "OBJ mesh should contain vertices."
                Expect.stringContains meshText "f " "OBJ mesh should contain faces."

                let vertexZValues =
                    meshText.Split([| '\n'; '\r' |], StringSplitOptions.RemoveEmptyEntries)
                    |> Array.choose (fun line ->
                        let parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries)
                        if parts.Length = 4 && parts[0] = "v" then
                            Some (Double.Parse(parts[3], Globalization.CultureInfo.InvariantCulture))
                        else
                            None)

                Expect.isGreaterThan (vertexZValues |> Array.max) 1.0 "OBJ vertices should use the streamed slice index as the z coordinate."

                let stlPath = Path.Combine(outputDir, "surface.stl")
                source (2UL * 1024UL * 1024UL * 1024UL)
                |> read<uint8> inputDir suffix
                >=> marchingCubes 0.5
                >=> writeMesh stlPath "auto"
                |> drain

                let stlText = File.ReadAllText(stlPath)
                Expect.stringContains stlText "solid stackProcessing" "STL mesh should contain a solid header."
                Expect.stringContains stlText "facet normal" "STL mesh should contain streamed facets."
                Expect.stringContains stlText "endsolid stackProcessing" "STL mesh should contain a solid footer."

                let boundarySlice = Image<uint8>.ofArray2D(Array2D.create 2 2 1uy, "boundary", 0)
                let boundaryTriangles =
                    imagePlan [ boundarySlice ]
                    >=> marchingCubes 0.5
                    |> drainList
                    |> List.collect _.Triangles

                let boundaryPoints =
                    boundaryTriangles
                    |> List.collect (fun triangle -> [ triangle.A; triangle.B; triangle.C ])

                Expect.isTrue (boundaryPoints |> List.exists (fun point -> point.X < 0.0)) "marchingCubes should close objects touching the left x boundary."
                Expect.isTrue (boundaryPoints |> List.exists (fun point -> point.X > 1.0)) "marchingCubes should close objects touching the right x boundary."
                Expect.isTrue (boundaryPoints |> List.exists (fun point -> point.Y < 0.0)) "marchingCubes should close objects touching the lower y boundary."
                Expect.isTrue (boundaryPoints |> List.exists (fun point -> point.Y > 1.0)) "marchingCubes should close objects touching the upper y boundary."
                Expect.isTrue (boundaryPoints |> List.exists (fun point -> point.Z < 0.0)) "marchingCubes should close objects touching the first z slice."
                Expect.isTrue (boundaryPoints |> List.exists (fun point -> point.Z > 0.0)) "marchingCubes should close objects touching the last z slice."
                boundarySlice.decRefCount()
            finally
                disposeImages slices
                deleteDirectory inputDir
                deleteDirectory outputDir

        testCase "objectSurfaceArea scales triangle coordinates before summing areas" <| fun _ ->
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
                >=> objectSurfaceArea 2.0 3.0 5.0
                |> drain

            Expect.floatClose Accuracy.high actual 10.5 "objectSurfaceArea should sum areas after anisotropic x/y/z scaling."

        testCase "objectVolume reduces UInt8 0-1 mask slices to physical volume" <| fun _ ->
            let slices =
                [ Image<uint8>.ofArray2D(array2D [ [ 1uy; 0uy ]; [ 1uy; 1uy ] ])
                  Image<uint8>.ofArray2D(array2D [ [ 0uy; 1uy ]; [ 0uy; 0uy ] ]) ]

            try
                let actual =
                    imagePlan slices
                    >=> objectVolume 2.0 3.0 4.0
                    |> drain

                Expect.floatClose Accuracy.high actual 96.0 "objectVolume should multiply foreground voxel count by physical voxel volume."
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

                let wrapperPath = Path.Combine(outputDir, "distances-wrapper")
                scalarPlan [ distances ]
                >=> writeCSVMatrix wrapperPath
                |> sink

                Expect.isTrue (File.Exists(wrapperPath + ".csv")) "writeCSVMatrix should append the CSV suffix."
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
                    >=> siftKeypoints 0.5 1.2 4u 0.0001 1u
                    |> drainList

                let points = pointSets |> List.collect _.Points
                Expect.isTrue (points |> List.exists (fun p -> p.X = 4.0 && p.Y = 4.0 && p.Z = 4.0)) "A centered impulse should produce a SIFT-style keypoint near the impulse coordinate."
            finally
                disposeImages slices

        testCase "streaming-friendly 3D keypoint detectors emit local point candidates" <| fun _ ->
            let makeImpulseSlices () =
                [ for z in 0 .. 10 ->
                    let image =
                        Array2D.init 11 11 (fun x y ->
                            let centeredImpulse = x = 5 && y = 5 && z = 5
                            let diagonalCorner = x >= 5 && y >= 5 && z >= 5
                            if centeredImpulse || diagonalCorner then 255uy else 0uy)
                        |> Image<uint8>.ofArray2D
                    image.index <- z
                    image ]

            let detectorCases =
                [ "logBlobKeypoints", logBlobKeypoints<uint8> 0.5 0.0 1u, true
                  "hessianKeypoints", hessianKeypoints 0.5 "Blob" 0.0 1u, true
                  "harris3DKeypoints", harris3DKeypoints 0.5 0.75 0.04 -1.0e9 1u, false
                  "forstner3DKeypoints", forstner3DKeypoints 0.5 0.75 0.0 1u, true
                  "phaseCongruencyKeypoints", phaseCongruencyKeypoints<uint8> 0.5 0.0 1u, true ]

            for name, detector, shouldEmit in detectorCases do
                let slices = makeImpulseSlices ()

                try
                    let pointSets =
                        imagePlan slices
                        >=> detector
                        |> drainList

                    let points = pointSets |> List.collect _.Points
                    if shouldEmit then
                        Expect.isNonEmpty points $"{name} should emit at least one bounded-window keypoint."
                    Expect.isTrue (points |> List.forall (fun p -> p.Z >= 0.0 && p.Z <= 10.0)) $"{name} should preserve stack z coordinates."
                finally
                    disposeImages slices

        testCase "coordinate sources emit x y and z slice coordinates" <| fun _ ->
            let xs =
                source 1024UL
                |> coordinateX<float> 3u 2u 2u
                |> drainList

            let ys =
                source 1024UL
                |> coordinateY<float> 3u 2u 2u
                |> drainList

            let zs =
                source 1024UL
                |> coordinateZ<float> 3u 2u 2u
                |> drainList

            try
                Expect.equal xs.Length 2 "coordinateX should emit one image per z slice."
                Expect.floatClose Accuracy.high (xs[0].Get [ 2u; 1u ]) 2.0 "coordinateX should store x positions."
                Expect.floatClose Accuracy.high (ys[0].Get [ 2u; 1u ]) 1.0 "coordinateY should store y positions."
                Expect.floatClose Accuracy.high (zs[1].Get [ 2u; 1u ]) 1.0 "coordinateZ should store slice indices."
            finally
                disposeImages xs
                disposeImages ys
                disposeImages zs

        testCase "polynomial bias model fits sampled coordinates and corrects the full stream" <| fun _ ->
            let makeSlices () =
                [ for z in 0 .. 3 ->
                    let image =
                        Array2D.init 4 3 (fun x y -> 10.0 + 2.0 * float x - 3.0 * float y + 5.0 * float z)
                        |> Image<float>.ofArray2D
                    image.index <- z
                    image ]

            let sampled = makeSlices ()
            let model =
                try
                    imagePlan sampled
                    >=> fitBiasModel 1
                    |> drain
                finally
                    disposeImages sampled

            let full = makeSlices ()
            let corrected =
                try
                    imagePlan full
                    >=> correctBias model
                    |> drainList
                finally
                    disposeImages full

            try
                for image in corrected do
                    let arr = image.toArray2D()
                    for y in 0 .. arr.GetLength(1) - 1 do
                        for x in 0 .. arr.GetLength(0) - 1 do
                            Expect.floatClose Accuracy.medium arr[x, y] 0.0 "Linear polynomial bias should subtract to numerical zero."
            finally
                disposeImages corrected

        testCase "serial section polynomial correction works slice-wise" <| fun _ ->
            let slices =
                [ for z in 0 .. 2 ->
                    let image =
                        Array2D.init 4 3 (fun x y -> 3.0 + float z + 2.0 * float x - float y)
                        |> Image<float>.ofArray2D
                    image.index <- z
                    image ]

            let corrected =
                try
                    imagePlan slices
                    >=> serialPolynomialBiasCorrect 1
                    |> drainList
                finally
                    disposeImages slices

            try
                for image in corrected do
                    let arr = image.toArray2D()
                    for y in 0 .. arr.GetLength(1) - 1 do
                        for x in 0 .. arr.GetLength(0) - 1 do
                            Expect.floatClose Accuracy.medium arr[x, y] 0.0 "Slicewise linear polynomial bias should subtract to numerical zero."
            finally
                disposeImages corrected

        testCase "serial section polynomial correction supports Float32" <| fun _ ->
            let slices =
                [ for z in 0 .. 1 ->
                    let image =
                        Array2D.init 4 3 (fun x y -> float32 (3.0 + float z + 2.0 * float x - float y))
                        |> Image<float32>.ofArray2D
                    image.index <- z
                    image ]

            let corrected =
                try
                    imagePlan slices
                    >=> serialPolynomialBiasCorrect 1
                    |> drainList
                finally
                    disposeImages slices

            try
                for image in corrected do
                    let arr = image.toArray2D()
                    for y in 0 .. arr.GetLength(1) - 1 do
                        for x in 0 .. arr.GetLength(0) - 1 do
                            Expect.floatClose Accuracy.medium (float arr[x, y]) 0.0 "Slicewise Float32 polynomial bias should subtract to numerical zero."
            finally
                disposeImages corrected

        testCase "serial section transform stream accumulates pairwise translations and applies on original canvas" <| fun _ ->
            let makeSlice z impulseX =
                let image =
                    Array2D.init 5 5 (fun x y -> if x = impulseX && y = 2 then 10.0 else 0.0)
                    |> Image<float>.ofArray2D
                image.index <- z
                image

            let sampled = [ makeSlice 0 2; makeSlice 1 3 ]
            let estimated =
                try
                    imagePlan sampled
                    >=> serialEstTrans 2 "SSDAffine" 0.5 0.1
                    |> drainList
                finally
                    disposeImages sampled

            try
                let manifests = estimated |> List.map snd
                Expect.equal manifests.Length 2 "The serial estimator should emit one manifest per slice."
                Expect.floatClose Accuracy.high manifests.[1].Transforms.[0].Matrix.[0].[2] -1.0 "The second slice should be translated back toward the first slice."
            finally
                estimated |> List.map fst |> disposeImages

            let input = [ makeSlice 0 2; makeSlice 1 3 ]
            let transformed =
                try
                    imagePlan input
                    >=> serialEstTrans 2 "SSDAffine" 0.5 0.1
                    >=> serialApplyTrans 0.0 None
                    |> drainList
                finally
                    disposeImages input

            try
                Expect.equal (transformed[0].GetWidth()) 5u "Serial application should keep the original canvas width."
                Expect.floatClose Accuracy.medium (transformed[1].Get [ 2u; 2u ]) 10.0 "The shifted impulse should align with the first slice on the original canvas."
            finally
                disposeImages transformed

        testCase "serial section SSD affine mode is available as an intensity-based option" <| fun _ ->
            let makeSlice z impulseX =
                let image =
                    Array2D.init 5 5 (fun x y -> if x = impulseX && y = 2 then 10.0 else 0.0)
                    |> Image<float>.ofArray2D
                image.index <- z
                image

            let sampled = [ makeSlice 0 2; makeSlice 1 3 ]
            let estimated =
                try
                    imagePlan sampled
                    >=> serialEstTrans 2 "SSDAffine" 0.5 0.1
                    |> drainList
                finally
                    disposeImages sampled

            try
                let manifests = estimated |> List.map snd
                Expect.floatClose Accuracy.high manifests.[1].Transforms.[0].Matrix.[0].[2] -1.0 "SSD mode should recover the same pure translation."
                Expect.floatClose Accuracy.high manifests.[1].Transforms.[0].Matrix.[0].[0] 1.0 "SSD mode should emit affine matrices."
            finally
                estimated |> List.map fst |> disposeImages

        testCase "serial section SSD affine reuses coarse scale for downsampled estimation" <| fun _ ->
            let makeSlice z offsetX =
                let image =
                    Array2D.init 16 16 (fun x y ->
                        if x >= 5 + offsetX && x <= 8 + offsetX && y >= 6 && y <= 9 then
                            float (10 + x + 2 * y)
                        else
                            0.0)
                    |> Image<float>.ofArray2D
                image.index <- z
                image

            let sampled = [ makeSlice 0 0; makeSlice 1 4 ]
            let estimated =
                try
                    imagePlan sampled
                    >=> serialEstTrans 6 "SSDAffine" 2.0 0.25
                    |> drainList
                finally
                    disposeImages sampled

            try
                let manifests = estimated |> List.map snd
                Expect.floatClose Accuracy.medium manifests.[1].Transforms.[0].Matrix.[0].[2] -4.0 "SSD scale should support coarse downsampled estimation and lift translation back to full resolution."
            finally
                estimated |> List.map fst |> disposeImages

        testCase "serial bounding box geometry expands serial apply canvas" <| fun _ ->
            let makeSlice z impulseX =
                let image =
                    Array2D.init 5 5 (fun x y -> if x = impulseX && y = 2 then 10.0 else 0.0)
                    |> Image<float>.ofArray2D
                image.index <- z
                image

            let geometryInput = [ makeSlice 0 2; makeSlice 1 3 ]
            let geometry =
                try
                    imagePlan geometryInput
                    >=> serialEstTrans 2 "SSDAffine" 0.5 0.1
                    >=> serialEstBoundingBox
                    |> drain
                finally
                    disposeImages geometryInput

            Expect.equal geometry.MinX -1.0 "The transformed stack geometry should include the shifted slice boundary."
            Expect.equal geometry.Width 6u "The transformed stack geometry should widen enough for all slice boundaries."
            Expect.equal geometry.Depth 2u "The transformed stack geometry should count streamed slices."

            let input = [ makeSlice 0 2; makeSlice 1 3 ]
            let transformed =
                try
                    imagePlan input
                    >=> serialEstTrans 2 "SSDAffine" 0.5 0.1
                    >=> serialApplyTrans 0.0 (Some geometry)
                    |> drainList
                finally
                    disposeImages input

            try
                Expect.equal (transformed[0].GetWidth()) 6u "Serial application should use the estimated geometry width."
                Expect.floatClose Accuracy.medium (transformed[1].Get [ 3u; 2u ]) 10.0 "The shifted impulse should align on the expanded canvas."
            finally
                disposeImages transformed

        testCase "serial 2D keypoints emit per-slice point sets" <| fun _ ->
            let slices =
                [ for z in 0 .. 1 ->
                    let image =
                        Array2D.init 7 7 (fun x y -> if x = 3 && y = 3 then 255uy else 0uy)
                        |> Image<uint8>.ofArray2D
                    image.index <- z
                    image ]

            try
                let pointSets =
                    imagePlan slices
                    >=> StackSerialSections.serialKeypoints2D<uint8> 0.5 0.0
                    |> drainList

                let points = pointSets |> List.collect _.Points
                Expect.isTrue (points |> List.exists (fun p -> p.X = 3.0 && p.Y = 3.0 && p.Z = 0.0)) "Slicewise 2D keypoints should preserve slice z."
            finally
                disposeImages slices

        testCase "writeVolume and readVolume stream multipage TIFF slices" <| fun _ ->
            let outputDir = tempDirectory "volume-tiff"
            let volumePath = Path.Combine(outputDir, "volume.tiff")
            let slices =
                [ for z in 0 .. 2 ->
                    let image =
                        Array2D.init 4 3 (fun x y -> uint16 (x + 10 * y + 100 * z))
                        |> Image<uint16>.ofArray2D
                    image.index <- z
                    image ]

            try
                imagePlan slices
                >=> writeVolume<uint16> volumePath
                |> drain

                let roundTripped =
                    source 1024UL
                    |> readVolume<uint16> volumePath
                    |> drainList

                try
                    Expect.equal roundTripped.Length 3 "readVolume should emit one slice per TIFF page."
                    Expect.equal (roundTripped[2].Get [ 3u; 2u ]) 223us "readVolume should preserve page pixel values."
                    Expect.equal roundTripped[2].index 2 "readVolume should assign slice indices from page order."
                finally
                    disposeImages roundTripped
            finally
                disposeImages slices
                deleteDirectory outputDir

        testCase "readVolume casts TIFF pages to the requested output type" <| fun _ ->
            let outputDir = tempDirectory "volume-tiff-cast"
            let volumePath = Path.Combine(outputDir, "volume.tiff")
            let slices =
                [ for z in 0 .. 1 ->
                    let image =
                        Array2D.init 3 2 (fun x y -> uint8 (x + 10 * y + 100 * z))
                        |> Image<uint8>.ofArray2D
                    image.index <- z
                    image ]

            try
                imagePlan slices
                >=> writeVolume<uint8> volumePath
                |> drain

                let asFloat32 =
                    source 1024UL
                    |> readVolume<float32> volumePath
                    |> drainList
                let asFloat64 =
                    source 1024UL
                    |> readVolume<float> volumePath
                    |> drainList

                try
                    Expect.equal asFloat32.Length 2 "readVolume should preserve the TIFF page count while casting."
                    Expect.floatClose Accuracy.high (float asFloat32[1].[2, 1]) 112.0 "readVolume should cast UInt8 TIFF pixels to Float32 output."
                    Expect.equal asFloat64.Length 2 "readVolume should preserve the TIFF page count while casting to Float64."
                    Expect.floatClose Accuracy.high asFloat64[1].[2, 1] 112.0 "readVolume should cast UInt8 TIFF pixels to Float64 output."
                finally
                    disposeImages asFloat64
                    disposeImages asFloat32
            finally
                disposeImages slices
                deleteDirectory outputDir

        testCase "volumeFilePath resolves hidden TIFF suffix aliases" <| fun _ ->
            let outputDir = tempDirectory "volume-file-path"
            let tifPath = Path.Combine(outputDir, "volumedata.tif")

            try
                File.WriteAllText(tifPath, "dummy")

                Expect.equal
                    (StackIO.volumeFilePath (Path.Combine(outputDir, "volumedata")) ".tiff")
                    tifPath
                    "A hidden .tiff file type should resolve an existing .tif file."

                Expect.equal
                    (StackIO.volumeFilePath tifPath ".tiff")
                    tifPath
                    "An explicit extension should be left unchanged."
            finally
                deleteDirectory outputDir

        testCase "meshFilePath resolves hidden OBJ and STL suffixes" <| fun _ ->
            Expect.equal (meshFilePath "surface" ".obj") "surface.obj" "OBJ format should append .obj when the output name has no extension."
            Expect.equal (meshFilePath "surface" ".stl") "surface.stl" "STL format should append .stl when the output name has no extension."
            Expect.equal (meshFilePath "surface.obj" ".stl") "surface.obj" "An explicit output extension should be left unchanged."

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
	                  T = TinyLinAlg.v3 2.0 -3.0 0.5 }

            let original: PointSet =
                { Points = [ point 4.0 -2.0 3.0; point -1.0 5.0 2.5 ] }
            let roundTripped =
                original
                |> transformPointSet affine
                |> transformPointSet (inverseAffine affine)

	            Expect.isLessThan (earthMoversDistance original.Points roundTripped.Points) 1.0e-10 "inverseAffine should invert affinePoint for resampler-style backward transforms."
	
	            let homogeneous = toHomogeneousMatrix affine |> unvectorizeMatrix
	
	            Expect.floatClose Accuracy.high homogeneous[0, 3] affine.T.x "The homogeneous matrix fourth column should hold the X translation term."
	            Expect.floatClose Accuracy.high homogeneous[1, 3] affine.T.y "The homogeneous matrix fourth column should hold the Y translation term."
	            Expect.floatClose Accuracy.high homogeneous[2, 3] affine.T.z "The homogeneous matrix fourth column should hold the Z translation term."
            Expect.floatClose Accuracy.high homogeneous[3, 0] 0.0 "The homogeneous matrix last row should keep affine point weight unchanged."
            Expect.floatClose Accuracy.high homogeneous[3, 1] 0.0 "The homogeneous matrix last row should keep affine point weight unchanged."
            Expect.floatClose Accuracy.high homogeneous[3, 2] 0.0 "The homogeneous matrix last row should keep affine point weight unchanged."
            Expect.floatClose Accuracy.high homogeneous[3, 3] 1.0 "The homogeneous matrix last row should keep affine point weight unchanged."

        testCase "affine registration aligns translated point sets and exposes resampler-compatible inverse" <| fun _ ->
            let fixedPoints =
                [ point 0.0 0.0 0.0
                  point 1.0 0.0 0.0
                  point 0.0 1.0 0.0
                  point 0.0 0.0 1.0 ]

            let movingPoints =
                fixedPoints
                |> List.map (fun p -> point (p.X + 2.0) (p.Y - 1.0) (p.Z + 0.5))

            let shuffledFixed =
                [ fixedPoints[2]; fixedPoints[0]; fixedPoints[3]; fixedPoints[1] ]

            let emd = earthMoversDistance fixedPoints shuffledFixed
            Expect.floatClose Accuracy.high emd 0.0 "EMD should match equal point sets independent of order."

            let result =
                affineRegistration
                    { defaultAffineRegistrationOptions with MaxIterations = 5 }
                    fixedPoints
                    movingPoints

            Expect.isLessThan result.Distance 1.0e-8 "The optimizer should find the centroid translation immediately for a pure translation."

            let transformedFixed =
                transformPointSet result.Transform { Points = fixedPoints }
                |> _.Points

            Expect.isLessThan (earthMoversDistance movingPoints transformedFixed) 1.0e-8 "The forward registration transform should map fixed points to moving points."

            let movingAsFixedCoordinates =
                transformPointSet result.InverseTransform { Points = movingPoints }
                |> _.Points

            Expect.isLessThan (earthMoversDistance fixedPoints movingAsFixedCoordinates) 1.0e-8 "The inverse transform should map moving-grid coordinates back to fixed coordinates for StackAffineResampler."

            let transformMatrix = toHomogeneousMatrix result.Transform
            let matrixAffine = ofHomogeneousMatrix transformMatrix
            let matrixTransformedFixed =
                transformPointSet matrixAffine { Points = fixedPoints }
                |> _.Points

            Expect.equal transformMatrix.Rows 4u "toHomogeneousMatrix should produce a 4x4 homogeneous matrix."
            Expect.equal transformMatrix.Columns 4u "toHomogeneousMatrix should produce a 4x4 homogeneous matrix."
            Expect.isLessThan (earthMoversDistance movingPoints matrixTransformedFixed) 1.0e-8 "ofHomogeneousMatrix should preserve the transform represented by toHomogeneousMatrix."

            let matchedBatches =
                [ ({ Points = fixedPoints |> List.take 2 } : PointSet),
                  ({ Points = movingPoints |> List.take 2 } : PointSet);
                  ({ Points = fixedPoints |> List.skip 2 } : PointSet),
                  ({ Points = movingPoints |> List.skip 2 } : PointSet) ]

            let forwardMatrix =
                scalarPlan matchedBatches
                >=> affineRegistrationMatrix { defaultAffineRegistrationOptions with Regularizer = 1.0e-12 }
                |> drain

            let inverseMatrix =
                scalarPlan matchedBatches
                >=> affineRegistrationInverseMatrix { defaultAffineRegistrationOptions with Regularizer = 1.0e-12 }
                |> drain

            Expect.equal forwardMatrix.Rows 4u "The forward transform matrix should be 4x4."
            Expect.equal inverseMatrix.Rows 4u "The inverse transform matrix should be 4x4."

            let streamedAffine = ofHomogeneousMatrix forwardMatrix
            let streamedTransformedFixed =
                transformPointSet streamedAffine { Points = fixedPoints }
                |> _.Points

            Expect.isLessThan (earthMoversDistance movingPoints streamedTransformedFixed) 1.0e-8 "The streaming affine reducer should update from multiple point-set batches."

            let streamedInverse = ofHomogeneousMatrix inverseMatrix
            let streamedMovingAsFixed =
                transformPointSet streamedInverse { Points = movingPoints }
                |> _.Points

            Expect.isLessThan (earthMoversDistance fixedPoints streamedMovingAsFixed) 1.0e-8 "The inverse affine reducer should emit the resampler direction."

        testCase "2D affine RANSAC ignores outlier point matches" <| fun _ ->
            let matches: PointMatch2D list =
                [ { FixedX = 2.0; FixedY = -1.0; MovingX = 0.0; MovingY = 0.0 }
                  { FixedX = 3.0; FixedY = -1.0; MovingX = 1.0; MovingY = 0.0 }
                  { FixedX = 2.0; FixedY = 0.0; MovingX = 0.0; MovingY = 1.0 }
                  { FixedX = 3.0; FixedY = 0.0; MovingX = 1.0; MovingY = 1.0 }
                  { FixedX = 40.0; FixedY = -30.0; MovingX = 2.0; MovingY = 2.0 } ]

            let matrix =
                affine2DRansac 80 0.25 0.5 123 matches
                |> Option.defaultWith (fun () -> failwith "RANSAC should find the dominant affine match set.")

            Expect.floatClose Accuracy.high matrix.[0].[0] 1.0 "The fitted affine should preserve x scale."
            Expect.floatClose Accuracy.high matrix.[1].[1] 1.0 "The fitted affine should preserve y scale."
            Expect.floatClose Accuracy.medium matrix.[0].[2] 2.0 "The fitted affine should recover x translation despite an outlier."
            Expect.floatClose Accuracy.medium matrix.[1].[2] -1.0 "The fitted affine should recover y translation despite an outlier."

        testCase "image-set manifest round-trips spatial data items as independent JSON sidecar" <| fun _ ->
            let outputDir = tempDirectory "image-set-manifest"
            let manifestPath = Path.Combine(outputDir, "imageset.json")
	            let affine: Affine =
	                { A =
	                    { m00 = 1.0; m01 = 0.0; m02 = 0.0
	                      m10 = 0.0; m11 = 1.0; m12 = 0.0
	                      m20 = 0.0; m21 = 0.0; m22 = 1.0 }
	                  T = v3 2.0 -1.0 0.5 }

            try
                let scalarItem =
                    scalarImageSetItem
                        "tile_001"
                        "tile_001.tiff"
                        ".tiff"
                        [ 128UL; 64UL; 32UL ]
                        [ 0.2; 0.2; 0.5 ]
                        (imageSetTransformFromAffine affine)
                        []

                let vectorItem =
                    vectorImageSetItem
                        "gradient_001"
                        "gradient_001.mha"
                        ".mha"
                        [ 128UL; 64UL; 32UL ]
                        [ 0.2; 0.2; 0.5 ]
                        (imageSetTransformFromAffine affine)
                        [ "tile_001" ]

                let pointItem =
                    pointSetManifestItem
                        "dog_001"
                        "dog_001.csv"
                        ".csv"
                        [ 128UL; 64UL; 32UL ]
                        [ 0.2; 0.2; 0.5 ]
                        (imageSetTransformFromAffine affine)
                        [ "tile_001" ]

                let meshItem =
                    triangleMeshManifestItem
                        "surface_001"
                        "surface_001.obj"
                        ".obj"
                        [ 128UL; 64UL; 32UL ]
                        [ 0.2; 0.2; 0.5 ]
                        (imageSetTransformFromAffine affine)
                        [ "tile_001" ]

                let manifest =
                    createImageSetManifest "sample-registered" "micrometer"
                    |> addImageSetItem scalarItem
                    |> addImageSetItem vectorItem
                    |> addImageSetItem pointItem
                    |> addImageSetItem meshItem

                writeImageSetManifest manifestPath manifest
                let reread = readImageSetManifest manifestPath

                Expect.equal reread.Version 1 "Image-set manifests should carry a version."
                Expect.equal reread.CoordinateSystem.Name "sample-registered" "Coordinate-system metadata should round-trip."
                Expect.equal reread.Items.Length 4 "Manifest items should round-trip."
                Expect.equal reread.Items[0].Id "tile_001" "Item ids should round-trip."
                Expect.equal reread.Items[0].Kind "ScalarImage" "Scalar images should keep their manifest kind."
                Expect.equal reread.Items[1].Kind "VectorImage" "Vector images should keep their manifest kind."
                Expect.equal reread.Items[2].Kind "PointSet" "Point sets should keep their manifest kind."
                Expect.equal reread.Items[3].Kind "TriangleMesh" "Triangle meshes should keep their manifest kind."
                Expect.equal reread.Items[1].Sources [ "tile_001" ] "Derived vector images should record their source item."
                Expect.equal reread.Items[2].Sources [ "tile_001" ] "Derived point sets should record their source item."
                Expect.equal reread.Items[3].Sources [ "tile_001" ] "Derived triangle meshes should record their source item."

                let transform = imageSetTransformToAffine reread.Items[0].TransformToWorld
                let transformed =
                    transformPointSet transform { Points = [ point 0.0 0.0 0.0 ] }
                    |> _.Points

                Expect.isLessThan (earthMoversDistance [ point 2.0 -1.0 0.5 ] transformed) 1.0e-10 "Manifest transforms should convert back to an affine."

                let updated =
                    reread
                    |> replaceImageSetItemTransform "tile_001" identityImageSetTransform

                Expect.equal updated.Items[0].TransformToWorld.Matrix identityImageSetTransform.Matrix "Manifest item transforms should be replaceable without touching data files."
            finally
                deleteDirectory outputDir

        testCase "identity manifest and registration composition update moving item transforms" <| fun _ ->
	            let fixedToWorld =
	                { A = identity3
	                  T = v3 10.0 0.0 0.0 }
	                |> imageSetTransformFromAffine
	
	            let fixedFromMoving =
	                { A = identity3
	                  T = v3 3.0 0.0 0.0 }
	                |> imageSetTransformFromAffine

            let fixedItem =
                scalarImageSetItem "fixed" "fixed" ".tiff" [ 2UL; 2UL; 1UL ] [ 1.0; 1.0; 1.0 ] fixedToWorld []

            let movingItem =
                scalarImageSetItem "moving" "moving" ".tiff" [ 2UL; 2UL; 1UL ] [ 1.0; 1.0; 1.0 ] identityImageSetTransform []

            let manifest =
                identityImageSetManifest "joint" "voxel"
                |> addImageSetItem fixedItem
                |> addImageSetItem movingItem
                |> updateMovingImageSetItemTransformFromRegistration "fixed" "moving" fixedFromMoving

            let movingToWorld =
                manifest.Items
                |> List.find (fun item -> item.Id = "moving")
                |> _.TransformToWorld

            Expect.floatClose Accuracy.high movingToWorld.Matrix.[0].[3] 13.0 "Moving item should compose as fixedToWorld * fixedFromMoving."
            Expect.floatClose Accuracy.high movingToWorld.Matrix.[1].[3] 0.0 "Composition should preserve unrelated translation components."

        testCase "image-set grid uses unit-spaced index tile coordinates" <| fun _ ->
            let grid = imageSetGrid [ 2UL; 3UL; 4UL ]
            let transform = imageSetGridIndexTransform grid [ 1; 2; 3 ]

            Expect.floatClose Accuracy.high transform.Matrix.[0].[3] 2.0 "Grid X offset should be index * tile width in voxel units."
            Expect.floatClose Accuracy.high transform.Matrix.[1].[3] 6.0 "Grid Y offset should be index * tile height in voxel units."
            Expect.floatClose Accuracy.high transform.Matrix.[2].[3] 12.0 "Grid Z offset should be index * tile depth in voxel units."

            let item =
                gridImageSetItem
                    "tile_100"
                    "tile_100"
                    ".tiff"
                    [ 2UL; 3UL; 4UL ]
                    [ 1.0; 1.0; 1.0 ]
                    [ 1; 0; 0 ]
                    identityImageSetTransform
                    []

            let manifest =
                identityImageSetManifest "grid" "voxel"
                |> withImageSetGrid grid
                |> addImageSetItem item

            Expect.equal manifest.Grid.Value.TileSize [ 2UL; 3UL; 4UL ] "Grid tile size should live on the manifest."
            Expect.equal manifest.Items[0].GridIndex (Some [ 1; 0; 0 ]) "Image items should carry their integer grid index."

        testCase "manifest item construction rejects non-affine homogeneous transforms" <| fun _ ->
            let invalidTransform =
                { identityImageSetTransform with
                    Matrix =
                        [ [ 1.0; 0.0; 0.0; 0.0 ]
                          [ 0.0; 1.0; 0.0; 0.0 ]
                          [ 0.0; 0.0; 1.0; 0.0 ]
                          [ 0.0; 0.0; 0.0; 2.0 ] ] }

            Expect.throws
                (fun () ->
                    scalarImageSetItem "bad" "bad" ".tiff" [ 1UL; 1UL; 1UL ] [ 1.0; 1.0; 1.0 ] invalidTransform []
                    |> ignore)
                "Manifest image items should reject transforms whose last row is not affine homogeneous form."

        testCase "stitch planning enforces equal grid image size" <| fun _ ->
            let grid = imageSetGrid [ 2UL; 2UL; 1UL ]

            let tile0 =
                gridImageSetItem "tile0" "tile0" ".tiff" [ 2UL; 2UL; 1UL ] [ 1.0; 1.0; 1.0 ] [ 0; 0; 0 ] identityImageSetTransform []

            let tile1 =
                gridImageSetItem "tile1" "tile1" ".tiff" [ 2UL; 2UL; 1UL ] [ 1.0; 1.0; 1.0 ] [ 1; 0; 0 ] identityImageSetTransform []

            let manifest =
                identityImageSetManifest "grid" "voxel"
                |> withImageSetGrid grid
                |> addImageSetItem tile0
                |> addImageSetItem tile1

            let plan = createStitchPlan manifest [ "tile0"; "tile1" ]
            Expect.equal plan.Size [ 4UL; 2UL; 1UL ] "Two touching grid tiles should produce a bounding box in voxel units."

            let mismatched =
                manifest
                |> addImageSetItem (gridImageSetItem "bad" "bad" ".tiff" [ 3UL; 2UL; 1UL ] [ 1.0; 1.0; 1.0 ] [ 2; 0; 0 ] identityImageSetTransform [])

            Expect.throws
                (fun () -> createStitchPlan mismatched [ "tile0"; "bad" ] |> ignore)
                "Stitch planning should reject image sets where selected image sizes differ."

        testCase "stitchManifestImages streams an identity manifest image stack" <| fun _ ->
            let rootDir = tempDirectory "manifest-stitch"
            let inputDir = Path.Combine(rootDir, "tile")
            let manifestPath = Path.Combine(rootDir, "imageset.json")
            Directory.CreateDirectory(inputDir) |> ignore

            let slices =
                [ array2D [ [ 1.0f; 2.0f ]; [ 3.0f; 4.0f ] ] |> Image<float32>.ofArray2D
                  array2D [ [ 5.0f; 6.0f ]; [ 7.0f; 8.0f ] ] |> Image<float32>.ofArray2D ]

            try
                writeSlices inputDir ".tiff" slices

                let item =
                    scalarImageSetItem
                        "tile"
                        inputDir
                        ".tiff"
                        [ 2UL; 2UL; 2UL ]
                        [ 1.0; 1.0; 1.0 ]
                        identityImageSetTransform
                        []

                let manifest =
                    identityImageSetManifest "joint" "voxel"
                    |> addImageSetItem item

                writeImageSetManifest manifestPath manifest

                let stitched =
                    source 1024UL
                    |> stitchManifestImages<float32> manifestPath [ "tile" ] 1.0
                    |> drainList

                Expect.equal stitched.Length 2 "Identity stitching should emit one slice per source slice."
                Expect.equal (stitched[0].toArray2D()) (slices[0].toArray2D()) "Identity stitching should preserve the first slice."
                Expect.equal (stitched[1].toArray2D()) (slices[1].toArray2D()) "Identity stitching should preserve the second slice."

                stitched |> List.iter (fun image -> image.decRefCount())
            finally
                disposeImages slices
                deleteDirectory rootDir

        testCase "resampleAffineFromChunks samples chunked slabs with the supplied output-to-input affine" <| fun _ ->
            let chunkDirectory = tempDirectory "affine-resampler-chunks"
            let slices =
                [ for z in 0 .. 3 ->
                    let image = makeFloatSlice 5 4 z
                    image.index <- z
                    image ]

            try
                imagePlan slices
                >=> writeChunks chunkDirectory ".tiff" 2u 2u 2u
                >=> ignoreSingles ()
                |> sink

                let lerp (a: float32) (b: float32) (t: float32) =
                    a + (b - a) * t

	                let transform: Affine =
	                    { A = identity3
	                      T = v3 0.5 1.0 0.0 }

                let output =
                    resampleAffineFromChunks
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

        testCase "resampleAffineFromChunks matches direct sampling across uneven chunk boundaries" <| fun _ ->
            let chunkDirectory = tempDirectory "affine-resampler-uneven-chunks"
            let voxels = Array3D.init 5 4 3 nonlinearVoxel
            let slices =
                [ for z in 0 .. 2 ->
                    let image =
                        Array2D.init 5 4 (fun x y -> voxels[x, y, z])
                        |> Image<float32>.ofArray2D

                    image.index <- z
                    image ]

            try
                imagePlan slices
                >=> writeChunks chunkDirectory ".tiff" 2u 2u 2u
                >=> ignoreSingles ()
                |> sink

                let lerp a b t = a + (b - a) * t
	                let transform: Affine =
	                    { A = identity3
	                      T = v3 0.25 0.5 0.5 }

                let output =
                    resampleAffineFromChunks
                        chunkDirectory
                        ".tiff"
                        lerp
                        2
                        (imageGeom 5 4 3)
                        (imageGeom 4 3 2)
                        transform
                        -999.0f
                    |> Seq.toList

                Expect.equal (output |> List.map fst) [ 0; 1 ] "The uneven-chunk resampler should emit the requested output slices in order."

                for k, image in output do
                    for y in 0 .. int (image.GetHeight()) - 1 do
                        for x in 0 .. int (image.GetWidth()) - 1 do
                            let c = v3 (float x + 0.25) (float y + 0.5) (float k + 0.5)
                            let expected = trilinearArraySample -999.0f voxels c
                            expectFloat32Close image[x, y] expected $"Chunked resampling should match direct array sampling at ({x},{y},{k})."

                output |> List.map snd |> disposeImages
            finally
                disposeImages slices
                deleteDirectory chunkDirectory

        testCase "resampleAffineFromChunks returns background when the trilinear footprint is outside" <| fun _ ->
            let chunkDirectory = tempDirectory "affine-resampler-background"
            let voxels = Array3D.init 4 4 3 nonlinearVoxel
            let slices =
                [ for z in 0 .. 2 ->
                    let image =
                        Array2D.init 4 4 (fun x y -> voxels[x, y, z])
                        |> Image<float32>.ofArray2D

                    image.index <- z
                    image ]

            try
                imagePlan slices
                >=> writeChunks chunkDirectory ".tiff" 2u 2u 2u
                >=> ignoreSingles ()
                |> sink

                let background = -1234.0f
                let lerp a b t = a + (b - a) * t
	                let transform: Affine =
	                    { A = identity3
	                      T = v3 -0.25 0.5 0.5 }

                let output =
                    resampleAffineFromChunks
                        chunkDirectory
                        ".tiff"
                        lerp
                        2
                        (imageGeom 4 4 3)
                        (imageGeom 2 2 1)
                        transform
                        background
                    |> Seq.toList

                Expect.equal output.Length 1 "The background test should emit one output slice."
                let _, image = output[0]

                for y in 0 .. int (image.GetHeight()) - 1 do
                    for x in 0 .. int (image.GetWidth()) - 1 do
                        let c = v3 (float x - 0.25) (float y + 0.5) 0.5
                        let expected = trilinearArraySample background voxels c
                        expectFloat32Close image[x, y] expected $"The resampler should use background consistently at ({x},{y},0)."

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
                    >=> streamConnectedObjects ObjectConnectivity.Six
                    |> drainList

                Expect.equal (batches |> List.map (fun batch -> batch.Objects.Length)) [ 0; 1; 0; 1 ] "Objects should be emitted when the next slice proves they no longer continue."

                let objects = batches |> List.collect _.Objects
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
                    >=> streamConnectedObjects ObjectConnectivity.Six
                    |> drainList
                    |> List.collect _.Objects

                let twentySixObjects =
                    imagePlan slices
                    >=> streamConnectedObjects ObjectConnectivity.TwentySix
                    |> drainList
                    |> List.collect _.Objects

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
                    >=> fillSmallHoles 1UL ObjectConnectivity.Six None None
                    |> drainList

                let z2 = filled |> List.find (fun image -> image.index = 2)
                Expect.equal z2[3, 3] 1uy "The one-voxel enclosed hole should be filled."
                Expect.equal z2[0, 3] 0uy "Background touching the x-y image border is exterior and should be preserved."
            finally
                disposeImages filled
                disposeImages slices

        testCase "fillSmallHoles fills z-spanning enclosed holes up to maximum volume" <| fun _ ->
            let slices =
                [ for z in 0 .. 5 ->
                    let image =
                        Array2D.init 7 7 (fun x y ->
                            let enclosedHole = x = 3 && y = 3 && z >= 1 && z <= 4
                            if enclosedHole then 0uy else 1uy)
                        |> Image<uint8>.ofArray2D
                    image.index <- z
                    image ]
            let mutable filled: Image<uint8> list = []

            try
                filled <-
                    imagePlan slices
                    >=> fillSmallHoles 4UL ObjectConnectivity.Six None None
                    |> drainList

                for z in 1 .. 4 do
                    let image = filled |> List.find (fun image -> image.index = z)
                    Expect.equal image[3, 3] 1uy $"The enclosed z-spanning hole should be filled on slice {z}."
            finally
                disposeImages filled
                disposeImages slices

        testCase "fillSmallHoles preserves UInt8 foreground value when filling holes" <| fun _ ->
            let slices =
                [ for z in 0 .. 4 ->
                    let image =
                        Array2D.init 7 7 (fun x y ->
                            let enclosedHole = x = 3 && y = 3 && z = 2
                            if enclosedHole then 0uy else 255uy)
                        |> Image<uint8>.ofArray2D
                    image.index <- z
                    image ]
            let mutable filled: Image<uint8> list = []

            try
                filled <-
                    imagePlan slices
                    >=> fillSmallHoles 1UL ObjectConnectivity.Six (Some 0.0) (Some 255.0)
                    |> drainList

                let z2 = filled |> List.find (fun image -> image.index = 2)
                Expect.equal z2[3, 3] 255uy "The filled pixel should use the observed foreground value, not hard-coded 1."
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
                scalarPlan [ ({ Objects = objects }: ObjectStream<uint8>) ]
                >=> paintObjects 5u 4u 3u (Some 2.0) (Some 7.0)
                |> drainList

            try
                Expect.equal (painted |> List.map _.index) [ 0; 1; 2 ] "Painting should emit empty slices for z gaps inside the object span."
                Expect.equal (painted[0][1, 2]) 7uy "First painted position should use the foreground value."
                Expect.equal (painted[0][3, 1]) 7uy "Second painted position should use the foreground value."
                Expect.equal (painted[0][0, 0]) 2uy "Background should use the background value."
                Expect.equal (painted[1][2, 2]) 2uy "The z=1 gap should be emitted as a background slice."
                Expect.equal (painted[2][2, 2]) 7uy "The z=2 position should be painted in the third emitted image."
            finally
                disposeImages painted

        testCase "paintObjectsCropped turns streamed objects into minimal local UInt8 masks" <| fun _ ->
            let objects: StreamedObject list =
                [ { Label = 7UL
                    Positions = [ { X = 4; Y = 5; Z = 10 }; { X = 6; Y = 5; Z = 10 }; { X = 5; Y = 6; Z = 11 } ]
                    Bounds = { MinX = 4; MaxX = 6; MinY = 5; MaxY = 6; MinZ = 10; MaxZ = 11 }
                    Size = 3UL } ]

            let painted =
                scalarPlan [ ({ Objects = objects }: ObjectStream<uint8>) ]
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

        testCase "object CSV write and read returns objects ordered by first z" <| fun _ ->
            let outputDir = tempDirectory "objects-csv"
            let earlyObject =
                { Label = 11UL
                  Positions = [ { X = 1; Y = 2; Z = 3 }; { X = 2; Y = 2; Z = 4 } ]
                  Bounds = { MinX = 1; MaxX = 2; MinY = 2; MaxY = 2; MinZ = 3; MaxZ = 4 }
                  Size = 2UL }
            let lateObject =
                { Label = 12UL
                  Positions = [ { X = 4; Y = 5; Z = 8 } ]
                  Bounds = { MinX = 4; MaxX = 4; MinY = 5; MaxY = 5; MinZ = 8; MaxZ = 8 }
                  Size = 1UL }

            try
                scalarPlan [ ({ Objects = [ lateObject ] }: ObjectStream<uint8>); ({ Objects = [ earlyObject ] }: ObjectStream<uint8>) ]
                >=> writeObjects outputDir ".csv"
                |> sink

                let files = Directory.GetFiles(outputDir, "*.csv") |> Array.map Path.GetFileName |> Array.sort
                Expect.equal files.Length 2 "writeObjects should write one CSV file per object."
                Expect.stringContains files[0] "z0000000003_z0000000004" "Object file names should include first and last z."
                Expect.stringContains files[1] "z0000000008_z0000000008" "Object file names should include first and last z."

                let reread =
                    source 1024UL
                    |> readObjects<uint8> outputDir ".csv"
                    |> drainList
                    |> List.collect _.Objects

                Expect.equal (reread |> List.map _.Label) [ 11UL; 12UL ] "readObjects should stream objects sorted by first z."
            finally
                deleteDirectory outputDir

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

        testCase "windowedThreshold preserves the slice-wise threshold result through the tail window" <| fun _ ->
            let makeSlices () =
                [ for z in 0 .. 4 ->
                    let image =
                        Array2D.init 4 3 (fun x y -> uint8 (x + 2 * y + 5 * z))
                        |> Image<uint8>.ofArray2D
                    image.index <- z
                    image ]

            let sliceInputs = makeSlices ()
            let slabInputs = makeSlices ()
            let mutable sliceWise: Image<uint8> list = []
            let mutable slabWise: Image<uint8> list = []

            try
                sliceWise <-
                    imagePlan sliceInputs
                    >=> thresholdRange 7.0 255.0
                    |> drainList

                slabWise <-
                    imagePlan slabInputs
                    >=> windowedThreshold<uint8> 3u 7.0 255.0
                    |> drainList

                Expect.equal slabWise.Length sliceWise.Length "Windowed threshold should emit one output per input slice, including the final partial window."
                Expect.equal (slabWise |> List.map _.index) (sliceWise |> List.map _.index) "Windowed threshold should preserve slice indices."

                for expected, actual in List.zip sliceWise slabWise do
                    Expect.equal (actual.toArray2D()) (expected.toArray2D()) $"Windowed threshold slice {actual.index} should match the slice-wise threshold."
            finally
                disposeImages slabWise
                disposeImages sliceWise
                disposeImages slabInputs
                disposeImages sliceInputs

        testCase "streamed objects expose dimensions and reducers summarize sizes" <| fun _ ->
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

            Expect.equal firstBatch.[0].Width 2UL "Width should be derived from x bounds."
            Expect.equal firstBatch.[1].Height 3UL "Height should be derived from y bounds."
            Expect.equal secondBatch.[0].Depth 1UL "Depth should be derived from z bounds."

            let stats =
                scalarPlan [ ({ Objects = firstBatch }: ObjectStream<uint8>); ({ Objects = secondBatch }: ObjectStream<uint8>) ]
                >=> objectSizes
                >=> stats ()
                |> drain

            Expect.equal stats.Count 3UL "Three objects should be summarized."
            Expect.floatClose Accuracy.high stats.Mean 3.0 "Mean size should be calculated online."
            Expect.floatClose Accuracy.high stats.Variance 1.0 "Sample variance of sizes 2,3,4 should be one."
            Expect.equal stats.Minimum 2UL "Minimum size should be tracked."
            Expect.equal stats.Maximum 4UL "Maximum size should be tracked."

            let histogram =
                scalarPlan [ ({ Objects = firstBatch }: ObjectStream<uint8>); ({ Objects = secondBatch }: ObjectStream<uint8>) ]
                >=> objectSizes
                >=> histogram ()
                |> drain

            Expect.equal histogram.Counts (Map.ofList [ 2UL, 2UL; 4UL, 1UL ]) "Size histogram should count exact object sizes."

        testCase "writeChunks creates chunk files and chunk metadata can be read" <| fun _ ->
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
                >=> writeChunks chunkDir chunkSuffix 2u 2u 2u
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
                Expect.equal info.size [ 5u; 4u; 4u ] "Zarr metadata should expose x/y/z image size."
                Expect.equal info.componentType "uint8" "Zarr metadata should expose the dataset dtype."

                rereadSlabs <-
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> readZarrSlabStacked<uint8> zarrPath 0 0 0 0 0
                    |> drainList

                Expect.equal rereadSlabs.Length 2 "readZarrSlabStacked should emit one stacked slab per storage z chunk."
                Expect.equal (rereadSlabs[0].GetSize()) [ 5u; 4u; 2u ] "The first Zarr slab should retain x/y and storage z chunk depth."

                rereadSlices <-
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> readZarrSlab<uint8> zarrPath 0 0 0 0 0
                    |> drainList

                Expect.equal rereadSlices.Length 4 "readZarrSlab should unstack slabs into a normal slice stream."
                let pixels = rereadSlices[3].toArray2D()
                Expect.equal pixels[4, 3] (uint8 ((4 + 2 * 3 + 3 * 3) % 251)) "Round-tripped Zarr pixel values should match the source stack."

                let castSlices =
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> readZarrSlab<float32> zarrPath 0 0 0 0 0
                    |> drainList

                try
                    Expect.equal castSlices.Length 4 "readZarrSlab should preserve slice count while casting."
                    Expect.floatClose Accuracy.high (float castSlices[3].[4, 3]) (float ((4 + 2 * 3 + 3 * 3) % 251)) "readZarrSlab should cast native UInt8 pixels to Float32 output."
                finally
                    disposeImages castSlices

                Expect.isFalse (File.Exists zarrDebugPath) "ZarrNET debug logging should not create a Windows-style biolog.txt side-effect."
            finally
                disposeImages rereadSlices
                disposeImages rereadSlabs
                disposeImages slices
                deleteDirectory inputDir
                deleteDirectory rootDir

        testCase "writeZarr and readZarrSlab support Float64 and complex dtypes" <| fun _ ->
            let rootDir = tempDirectory "zarr-wide-types"
            let float64Path = Path.Combine(rootDir, "float64.zarr")
            let complex64Path = Path.Combine(rootDir, "complex64.zarr")
            let complex128Path = Path.Combine(rootDir, "complex128.zarr")
            let mutable float64Slices: Image<float> list = []
            let mutable complex64Slices: Image<ComplexFloat32> list = []
            let mutable complex128Slices: Image<System.Numerics.Complex> list = []
            let mutable float64Slabs: Image<float> list = []
            let mutable complex64Slabs: Image<ComplexFloat32> list = []
            let mutable complex128Slabs: Image<System.Numerics.Complex> list = []
            let mutable complex128SlicesRead: Image<System.Numerics.Complex> list = []

            try
                float64Slices <-
                    [ for z in 0 .. 1 ->
                        Array2D.init 3 2 (fun x y -> float x + 10.0 * float y + 100.0 * float z + 0.25)
                        |> Image<float>.ofArray2D ]
                complex64Slices <-
                    [ for z in 0 .. 1 ->
                        Array2D.init 3 2 (fun x y ->
                            ComplexFloat32(
                                float32 (x + 10 * y + 100 * z),
                                float32 (1000 + x + 10 * y + 100 * z)))
                        |> Image<ComplexFloat32>.ofComplexFloat32Array2D ]
                complex128Slices <-
                    [ for z in 0 .. 1 ->
                        Array2D.init 3 2 (fun x y ->
                            System.Numerics.Complex(
                                float (x + 10 * y + 100 * z) + 0.5,
                                float (1000 + x + 10 * y + 100 * z) + 0.75))
                        |> Image<System.Numerics.Complex>.ofComplexArray2D ]

                imagePlan float64Slices
                >=> writeZarr float64Path "float64" 2u 3u 2u 2u 1.0 1.0 1.0 0
                |> sink

                imagePlan complex64Slices
                >=> writeZarr complex64Path "complex64" 2u 3u 2u 2u 1.0 1.0 1.0 0
                |> sink

                imagePlan complex128Slices
                >=> writeZarr complex128Path "complex128" 2u 3u 2u 2u 1.0 1.0 1.0 0
                |> sink

                Expect.equal (getZarrInfo float64Path 0 0).componentType "float64" "Float64 Zarr metadata should expose float64 dtype."
                Expect.equal (getZarrInfo complex64Path 0 0).componentType "complex64" "Complex64 Zarr metadata should expose complex64 dtype."
                Expect.equal (getZarrInfo complex128Path 0 0).componentType "complex128" "Complex128 Zarr metadata should expose complex128 dtype."

                float64Slabs <-
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> readZarrSlabStacked<float> float64Path 0 0 0 0 0
                    |> drainList
                complex64Slabs <-
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> readZarrSlabStacked<ComplexFloat32> complex64Path 0 0 0 0 0
                    |> drainList
                complex128Slabs <-
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> readZarrSlabStacked<System.Numerics.Complex> complex128Path 0 0 0 0 0
                    |> drainList

                let float64Values = float64Slabs[0].toArray3D()
                Expect.floatClose Accuracy.high float64Values[2, 1, 1] 112.25 "Float64 Zarr roundtrip should preserve payload values."

                let complex64Values = complex64Slabs[0].toComplexFloat32Array3D()
                Expect.floatClose Accuracy.high (float complex64Values[2, 1, 1].Real) 112.0 "Complex64 Zarr roundtrip should preserve real part."
                Expect.floatClose Accuracy.high (float complex64Values[2, 1, 1].Imaginary) 1112.0 "Complex64 Zarr roundtrip should preserve imaginary part."

                let complex128Values = complex128Slabs[0].toComplexArray3D()
                Expect.floatClose Accuracy.high complex128Values[2, 1, 1].Real 112.5 "Complex128 Zarr roundtrip should preserve real part."
                Expect.floatClose Accuracy.high complex128Values[2, 1, 1].Imaginary 1112.75 "Complex128 Zarr roundtrip should preserve imaginary part."

                complex128SlicesRead <-
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> readZarrSlab<System.Numerics.Complex> complex128Path 0 0 0 0 0
                    |> drainList
                Expect.equal complex128SlicesRead.Length 2 "Complex128 readZarrSlab should unstack into individual slices."
                let complex128SliceValues = complex128SlicesRead[1].toComplexArray2D()
                Expect.floatClose Accuracy.high complex128SliceValues[2, 1].Real 112.5 "Complex128 readZarrSlab should preserve real part."
                Expect.floatClose Accuracy.high complex128SliceValues[2, 1].Imaginary 1112.75 "Complex128 readZarrSlab should preserve imaginary part."
            finally
                disposeImages float64Slabs
                disposeImages complex64Slabs
                disposeImages complex128Slabs
                disposeImages complex128SlicesRead
                disposeImages float64Slices
                disposeImages complex64Slices
                disposeImages complex128Slices
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
                Expect.equal info.size [ 5u; 3u; 4u ] "NeXus metadata should expose x/y/z image size according to the axis mapping."

                rereadSlabs <-
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> readNexusSlabStacked<uint16> nexusPath datasetPath 0 1 2
                    |> drainList

                Expect.equal rereadSlabs.Length 1 "readNexusSlabStacked should emit stacked slabs using the dataset frame-axis chunk depth."
                Expect.equal (rereadSlabs[0].GetSize()) [ 5u; 3u; 4u ] "The NeXus slab should retain x/y and inferred z depth."

                rereadSlices <-
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> readNexusSlab<uint16> nexusPath datasetPath 0 1 2
                    |> drainList

                Expect.equal rereadSlices.Length 4 "readNexusSlab should unstack slabs into a normal slice stream."
                let pixels = rereadSlices[3].toArray2D()
                Expect.equal pixels[4, 2] (uint16 (4 + 10 * 2 + 100 * 3)) "NeXus pixel values should match the source HDF5 dataset."

                let castSlices =
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> readNexusSlab<float32> nexusPath datasetPath 0 1 2
                    |> drainList

                try
                    Expect.equal castSlices.Length 4 "readNexusSlab should preserve slice count while casting."
                    Expect.floatClose Accuracy.high (float castSlices[3].[4, 2]) (float (4 + 10 * 2 + 100 * 3)) "readNexusSlab should cast native UInt16 pixels to Float32 output."
                finally
                    disposeImages castSlices
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
                Expect.equal info.size [ 5u; 3u; 4u ] "writeNexus metadata should expose x/y/z image size."

                rereadSlices <-
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> readNexusSlab<uint16> nexusPath datasetPath 0 1 2
                    |> drainList

                Expect.equal rereadSlices.Length 4 "writeNexus output should be readable by readNexusSlab."
                let pixels = rereadSlices[2].toArray2D()
                Expect.equal pixels[4, 2] (uint16 (4 + 10 * 2 + 100 * 2)) "writeNexus should preserve pixel values."
            finally
                disposeImages rereadSlices
                disposeImages slices
                deleteDirectory inputDir
                deleteDirectory rootDir

        testCase "writeNexusSlab writes inferred-depth HDF5 detector slabs incrementally" <| fun _ ->
            let rootDir = tempDirectory "nexus-slab-write"
            let inputPath = Path.Combine(rootDir, "input.h5")
            let outputPath = Path.Combine(rootDir, "written-slabs.h5")
            let datasetPath = "/entry/data/data"
            let data =
                Array3D.init 4 3 5 (fun z y x -> uint16 (x + 10 * y + 100 * z))
            let mutable writtenSlabs: Image<uint16> list = []
            let mutable rereadSlices: Image<uint16> list = []

            try
                writeNexusStack inputPath datasetPath data

                writtenSlabs <-
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> readNexusSlabStacked<uint16> inputPath datasetPath 0 1 2
                    |> writeNexusSlab outputPath datasetPath 5u 3u 0 1 2
                    |> drainList

                let info = getNexusInfo outputPath datasetPath 0 1 2
                Expect.equal info.size [ 5u; 3u; 4u ] "writeNexusSlab metadata should expose x/y/z image size."

                rereadSlices <-
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> readNexusSlab<uint16> outputPath datasetPath 0 1 2
                    |> drainList

                Expect.equal writtenSlabs.Length 1 "The contiguous test input should be forwarded as one inferred slab."
                Expect.equal rereadSlices.Length 4 "writeNexusSlab output should be readable by readNexusSlab."
                let pixels = rereadSlices[3].toArray2D()
                Expect.equal pixels[4, 2] (uint16 (4 + 10 * 2 + 100 * 3)) "writeNexusSlab should preserve pixel values."
            finally
                disposeImages writtenSlabs
                disposeImages rereadSlices
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
                    |> resize 5u 4u 5u "NearestNeighbor"
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
                    |> resample 0.5 2.0 2.0 "Linear"
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
                a.index <- 7
                b.index <- 99
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
                Expect.equal (results |> List.map _.index) [ 7; 99; 7; 99; 7; 7 ] "Two-image operators should preserve the first input image index."

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
                    >=> imHistogram ()
                    |> drain

                Expect.equal actual.Counts (Map.ofList [ 0uy, 1UL; 1uy, 2UL; 2uy, 3UL; 3uy, 2UL ]) "histogram should fold all streamed slices."
            finally
                disposeImages slices
                deleteDirectory inputDir

        testCase "Chunk read feeds dense histogram reducer from TIFF ArrayPool storage" <| fun _ ->
            let inputDir = tempDirectory "chunk-histogram-input"
            let suffix = ".tiff"
            let slices =
                [ array2D [ [ 0uy; 1uy ]; [ 1uy; 2uy ] ] |> Image<uint8>.ofArray2D
                  array2D [ [ 2uy; 2uy ]; [ 3uy; 3uy ] ] |> Image<uint8>.ofArray2D ]

            try
                writeSlices inputDir suffix slices

                let actual =
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> read<uint8> inputDir suffix
                    >=> ChunkFunctions.histogramDenseReducer<uint8> ()
                    |> drain

                Expect.equal actual.Counts (Map.ofList [ 0uy, 1UL; 1uy, 2UL; 2uy, 3UL; 3uy, 2UL ]) "Chunk dense histogram should fold pixels read directly into pooled slice chunks."
            finally
                disposeImages slices
                deleteDirectory inputDir

        testCase "Chunk read and write round-trip TIFF slices" <| fun _ ->
            let inputDir = tempDirectory "chunk-tiff-input"
            let outputDir = tempDirectory "chunk-tiff-output"
            let suffix = ".tiff"
            let slices =
                [ array2D [ [ 4uy; 5uy; 6uy ]; [ 7uy; 8uy; 9uy ] ] |> Image<uint8>.ofArray2D
                  array2D [ [ 10uy; 11uy; 12uy ]; [ 13uy; 14uy; 15uy ] ] |> Image<uint8>.ofArray2D ]
            let mutable reread: Image<uint8> list = []

            try
                writeSlices inputDir suffix slices

                let written =
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> read<uint8> inputDir suffix
                    >=> write<uint8> outputDir suffix
                    |> drainList

                Expect.hasLength written 2 "write should emit one completion per consumed chunk."

                reread <-
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> read<uint8> outputDir suffix
                    |> drainList

                Expect.hasLength reread 2 "Chunk-written TIFF stack should be readable as normal images."
                Expect.equal (reread[0].toArray2D()) (slices[0].toArray2D()) "First chunk-written slice should match the source pixels."
                Expect.equal (reread[1].toArray2D()) (slices[1].toArray2D()) "Second chunk-written slice should match the source pixels."
            finally
                disposeImages reread
                disposeImages slices
                deleteDirectory inputDir
                deleteDirectory outputDir

        testCase "Chunk TIFF write options can emit compressed slices read by native decoded path" <| fun _ ->
            let inputDir = tempDirectory "chunk-tiff-compressed-input"
            let outputDir = tempDirectory "chunk-tiff-compressed-output"
            let suffix = ".tiff"
            let options =
                { Compression = StackIO.TiffCompression.Lzw
                  ByteOrder = StackIO.TiffByteOrder.Native }
            let slices =
                [ array2D [ [ 4uy; 5uy; 6uy ]; [ 7uy; 8uy; 9uy ] ] |> Image<uint8>.ofArray2D
                  array2D [ [ 10uy; 11uy; 12uy ]; [ 13uy; 14uy; 15uy ] ] |> Image<uint8>.ofArray2D ]
            let mutable reread: Image<uint8> list = []

            try
                writeSlices inputDir suffix slices

                source (2UL * 1024UL * 1024UL * 1024UL)
                |> read<uint8> inputDir suffix
                >=> writeTiffWithOptions<uint8> options outputDir suffix
                |> drain

                let info = getFileInfo (Path.Combine(outputDir, "image_000.tiff"))
                Expect.equal info.componentType "UInt8" "Compressed TIFF output should preserve the scalar component type."

                reread <-
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> read<uint8> outputDir suffix
                    |> drainList

                Expect.hasLength reread 2 "Compressed TIFF stack should be readable through the native decoded path."
                Expect.equal (reread[0].toArray2D()) (slices[0].toArray2D()) "First compressed slice should match the source pixels."
                Expect.equal (reread[1].toArray2D()) (slices[1].toArray2D()) "Second compressed slice should match the source pixels."
            finally
                disposeImages reread
                disposeImages slices
                deleteDirectory inputDir
                deleteDirectory outputDir

        testCase "Chunk read casts TIFF slices to the requested output type" <| fun _ ->
            let inputDir = tempDirectory "chunk-tiff-cast-input"
            let suffix = ".tiff"
            let slices =
                [ array2D [ [ 4uy; 5uy; 6uy ]; [ 7uy; 8uy; 9uy ] ] |> Image<uint8>.ofArray2D
                  array2D [ [ 10uy; 11uy; 12uy ]; [ 13uy; 14uy; 15uy ] ] |> Image<uint8>.ofArray2D ]
            let mutable asFloat32: Image<float32> list = []
            let mutable asFloat64: Image<float> list = []
            let mutable asInt64: Image<int64> list = []
            let mutable asUInt64: Image<uint64> list = []

            try
                writeSlices inputDir suffix slices

                asFloat32 <-
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> read<float32> inputDir suffix
                    |> drainList

                Expect.hasLength asFloat32 2 "Read should preserve the TIFF slice count while casting."
                Expect.equal (asFloat32[0].Get [ 2u; 1u ]) 9.0f "Read should cast UInt8 TIFF pixels to Float32 output."
                Expect.equal asFloat32[0].index 0 "Read should preserve slice indices when casting."

                asFloat64 <-
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> read<float> inputDir suffix
                    |> drainList

                Expect.floatClose Accuracy.high asFloat64[0].[2, 1] 9.0 "Read should cast UInt8 TIFF pixels to Float64 output."

                asInt64 <-
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> read<int64> inputDir suffix
                    |> drainList

                Expect.equal asInt64[0].[2, 1] 9L "Read should cast UInt8 TIFF pixels to Int64 output."

                asUInt64 <-
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> read<uint64> inputDir suffix
                    |> drainList

                Expect.equal asUInt64[0].[2, 1] 9UL "Read should cast UInt8 TIFF pixels to UInt64 output."
            finally
                disposeImages asUInt64
                disposeImages asInt64
                disposeImages asFloat64
                disposeImages asFloat32
                disposeImages slices
                deleteDirectory inputDir

        testCase "histogramEqualization streams Float64 CDF values from an estimated 3D histogram" <| fun _ ->
            let input =
                array2D [ [ 0uy; 10uy ]; [ 15uy; 20uy ] ]
                |> Image<uint8>.ofArray2D

            let histogram = StackCore.Histogram.ofMap (Map.ofList [ 0uy, 1UL; 10uy, 1UL; 20uy, 2UL ])
            let mutable equalized: Image<float> list = []

            try
                equalized <-
                    imagePlan [ input ]
                    >=> histogramEqualization histogram
                    |> drainList

                Expect.hasLength equalized 1 "Equalization should preserve slice cardinality."
                Expect.floatClose Accuracy.high equalized[0].[0, 0] 0.0 "The lowest occupied histogram bin should map to zero."
                Expect.floatClose Accuracy.high equalized[0].[0, 1] (1.0 / 3.0) "Exact sampled keys should map through the histogram CDF."
                Expect.floatClose Accuracy.medium equalized[0].[1, 0] (2.0 / 3.0) "Unsampled input values should interpolate between histogram keys."
                Expect.floatClose Accuracy.high equalized[0].[1, 1] 1.0 "The highest occupied histogram bin should map to one."
            finally
                input.decRefCount()
                disposeImages equalized

        testCase "writeCSVHistogram writes sorted key-count rows" <| fun _ ->
            let outputDir = tempDirectory "histogram-csv"
            let outputPath = Path.Combine(outputDir, "histogram")

            try
                scalarPlan [ Map.ofList [ 3uy, 2UL; 1uy, 4UL ]; Map.ofList [ 2uy, 5UL; 1uy, 1UL ] ]
                >=> writeCSVHistogram<uint8> outputPath
                |> sink

                let rows = File.ReadAllLines(outputPath + ".csv")
                Expect.equal rows [| "key,count"; "1,5"; "2,5"; "3,2" |] "Histogram CSV output should combine maps and keep keys sorted."
            finally
                deleteDirectory outputDir

        testCase "histogram threshold estimators return scalar thresholds for the standard threshold stage" <| fun _ ->
            let histogram = StackCore.Histogram.ofMap (Map.ofList [ 0.0f, 4UL; 10.0f, 4UL ])

            let otsu =
                scalarPlan [ histogram ]
                >=> otsuThreshold ()
                |> drain
            let moments = momentsThresholdFromHistogram histogram

            Expect.isGreaterThan otsu 0.0 "Otsu threshold should lie between the two modes."
            Expect.isLessThan otsu 10.0 "Otsu threshold should lie between the two modes."
            Expect.isGreaterThan moments 0.0 "Moments threshold should lie between the two modes."
            Expect.isLessThan moments 10.0 "Moments threshold should lie between the two modes."

        testCase "histogram threshold stages accept histograms and histogram estimates" <| fun _ ->
            let histogram = StackCore.Histogram.ofMap (Map.ofList [ 0uy, 4UL; 10uy, 4UL ])
            let estimate: HistogramEstimate<uint8> =
                { Histogram = histogram
                  Samples = 8UL
                  SlicesRead = 1u
                  CdfHalfWidth = 0.01
                  HoldoutMaxCdfDelta = 0.01
                  Method = "DKWAndHoldout"
                  Confidence = 0.95 }

            let otsuFromHistogram =
                scalarPlan [ histogram ]
                >=> otsuThreshold ()
                |> drain

            let otsuFromEstimate =
                scalarPlan [ estimate ]
                >=> otsuThreshold ()
                |> drain

            let fromHistogram =
                scalarPlan [ histogram ]
                >=> momentsThreshold ()
                |> drain

            let fromEstimate =
                scalarPlan [ estimate ]
                >=> momentsThreshold ()
                |> drain

            Expect.equal otsuFromEstimate otsuFromHistogram "otsuThreshold should use the embedded histogram from HistogramEstimate."
            Expect.isGreaterThan otsuFromHistogram 0.0 "Otsu threshold should lie between the two modes."
            Expect.isLessThan otsuFromHistogram 10.0 "Otsu threshold should lie between the two modes."
            Expect.equal fromEstimate fromHistogram "momentsThreshold should use the embedded histogram from HistogramEstimate."
            Expect.isGreaterThan fromHistogram 0.0 "Moments threshold should lie between the two modes."
            Expect.isLessThan fromHistogram 10.0 "Moments threshold should lie between the two modes."

        testCase "sumProjection reduces a stack to one transformed Float64 image" <| fun _ ->
            let slices =
                [ array2D [ [ 1s; -2s ]; [ 3s; -4s ] ] |> Image<int16>.ofArray2D
                  array2D [ [ 2s; -3s ]; [ 4s; -5s ] ] |> Image<int16>.ofArray2D ]
            let mutable projections: Image<float> list = []

            try
                projections <-
                    imagePlan slices
                    >=> sumProjection "Abs"
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
                    >=> imHistogram ()
                    >=> histogram2pairs<uint8>
                    >=> pairs2floats<uint8, uint64>
                    |> drain

                Expect.equal pairs [ 1.0, 2.0; 3.0, 6.0 ] "histogram2pairs followed by pairs2floats should expose histogram coordinates."
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
                    >=> connectedComponents (Some 2u)
                    >=> teeFst (writeSlabSlices labelDir labelSuffix 2u)
                    >=> makeConnectedComponentTranslationTable (Some 2u)
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
                expectFloat32Close (run (StackProcessing.scalarAdd<float32> 3.0)) 5.0f "scalarAdd should add scalar on the left."
                expectFloat32Close (run (StackProcessing.addScalar<float32> 3.0)) 5.0f "addScalar should add scalar on the right."
                expectFloat32Close (run (StackProcessing.scalarSub<float32> 10.0)) 8.0f "scalarSub should subtract the image from the scalar."
                expectFloat32Close (run (StackProcessing.subScalar<float32> 1.0)) 1.0f "subScalar should subtract the scalar from the image."
                expectFloat32Close (run (StackProcessing.scalarMul<float32> 4.0)) 8.0f "scalarMul should multiply."
                expectFloat32Close (run (StackProcessing.mulScalar<float32> 4.0)) 8.0f "mulScalar should multiply."
                expectFloat32Close (run (StackProcessing.scalarDiv<float32> 8.0)) 4.0f "scalarDiv should divide scalar by image."
                expectFloat32Close (run (StackProcessing.divScalar<float32> 2.0)) 1.0f "divScalar should divide image by scalar."
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

        testCase "StackProcessing structureTensor emits a 12-component eigensystem matrix per input slice" <| fun _ ->
            let makeDoubleSlice z =
                let image =
                    Array2D.init 5 5 (fun x _ -> float x + 0.0 * float z)
                    |> Image<float>.ofArray2D
                image.index <- z
                image

            let inputSlices = [ 0 .. 4 ] |> List.map makeDoubleSlice
            let actual =
                imagePlan inputSlices
                >=> StackProcessing.structureTensor 0.0 0.0
                |> drainList

            try
                Expect.equal actual.Length inputSlices.Length "structureTensor should preserve slice count."
                actual |> List.iter (fun image -> Expect.equal (image.GetNumberOfComponentsPerPixel()) 12u "structureTensor outputs should be 12-component vectorized eigensystems.")
                Expect.equal actual[2].[2, 2] [ 1.0; 0.0; 0.0; 1.0; 0.0; 0.0; 0.0; 1.0; 0.0; 0.0; 0.0; 1.0 ] "The eigensystem should store eigenvalues followed by the three eigenvectors."
            finally
                disposeImages actual

            let eigenvector0 =
                imagePlan ([ 0 .. 4 ] |> List.map makeDoubleSlice)
                >=> StackProcessing.structureTensor 0.0 0.0
                >=> StackProcessing.vectorRange<float> 3u 3u
                |> drainList

            try
                Expect.equal eigenvector0.Length inputSlices.Length "vectorRange should preserve the selected stream length."
                Expect.equal (eigenvector0 |> List.map _.index) [ 0 .. 4 ] "vectorRange should preserve the input image index even when selecting later components."
                Expect.equal eigenvector0[2].[2, 2] [ 1.0; 0.0; 0.0 ] "vectorRange 3 3 should select eigenvector 0."
            finally
                disposeImages eigenvector0

            let smoothed =
                imagePlan ([ 0 .. 4 ] |> List.map makeDoubleSlice)
                >=> StackProcessing.structureTensor 0.0 0.5
                |> drainList

            try
                Expect.equal smoothed.Length inputSlices.Length "structureTensor smoothing should preserve the input slice count."
                smoothed |> List.iter (fun image -> Expect.equal (image.GetNumberOfComponentsPerPixel()) 12u "Window padding must preserve vector component counts.")
            finally
                disposeImages smoothed

        testCase "StackProcessing converts 3-vector images to and from color images" <| fun _ ->
            let vector = new Image<float list>([ 2u; 1u ], 3u, "vector-color", 0)
            vector.[0, 0] <- [ -1.0; 0.0; 1.0 ]
            vector.[1, 0] <- [ 1.0; 0.5; -1.0 ]

            let color =
                imagePlan [ vector ]
                >=> StackProcessing.intensityStretch -1.0 1.0 0.0 255.0
                >=> StackProcessing.vectorCast<_, uint8>
                |> drain

            try
                Expect.equal (color.GetNumberOfComponentsPerPixel()) 3u "vectorCast should emit 3-component color pixels."
                Expect.equal color.[0, 0] [ 0uy; 128uy; 255uy ] "vectorCast should map the configured range to byte color channels."

                let roundTrip =
                    imagePlan [ color ]
                    >=> StackProcessing.colorToVector3 -1.0 1.0
                    |> drain

                try
                    let recovered = roundTrip.[0, 0]
                    Expect.isLessThan (Math.Abs(recovered.[0] + 1.0)) 0.01 "colorToVector3 should recover the low endpoint."
                    Expect.isLessThan (Math.Abs(recovered.[1])) 0.01 "colorToVector3 should recover the midpoint within byte quantization."
                    Expect.isLessThan (Math.Abs(recovered.[2] - 1.0)) 0.01 "colorToVector3 should recover the high endpoint."
                finally
                    roundTrip.decRefCount()
            finally
                color.decRefCount()

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

        testCase "StackProcessing FFTFloat32 invFFTFloat32 stream through native FFTW chunk workspace" <| fun _ ->
            let makeSlice z =
                Image<float32>.ofArray2D (
                    Array2D.init 2 2 (fun x y ->
                        if x = 0 && y = 0 && z = 0 then 1.0f else 0.0f))

            let inputSlices = [ makeSlice 0; makeSlice 1 ]
            let fftSlices =
                imagePlan inputSlices
                >=> StackProcessing.FFTFloat32<float32> 1u 1u 1u
                |> drainList

            try
                Expect.equal fftSlices.Length 2 "FFTFloat32 should emit one complex slice per input slice."
                for slice in fftSlices do
                    let values = slice.toComplexFloat32Array2D()
                    for x in 0 .. 1 do
                        for y in 0 .. 1 do
                            Expect.floatClose Accuracy.medium (float values[x, y].Real) 1.0 "3D FFTFloat32 of an impulse should be one everywhere, real part."
                            Expect.floatClose Accuracy.medium (float values[x, y].Imaginary) 0.0 "3D FFTFloat32 of an impulse should be real everywhere."

                let recovered =
                    imagePlan fftSlices
                    >=> StackProcessing.invFFTFloat32 1u 1u 1u
                    |> drainList

                try
                    Expect.equal recovered.Length 2 "invFFTFloat32 should emit one real slice per frequency slice."
                    Expect.floatClose Accuracy.medium (float recovered[0].[0, 0]) 1.0 "invFFTFloat32 should recover the impulse."
                    Expect.floatClose Accuracy.medium (float recovered[0].[1, 0]) 0.0 "invFFTFloat32 should recover zero pixels."
                    Expect.floatClose Accuracy.medium (float recovered[1].[0, 0]) 0.0 "invFFTFloat32 should recover zero z-slices."
                finally
                    disposeImages recovered
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
                let clampStage : Stage<Image<float32>, Image<float32>> = StackProcessing.clamp 0.0 120.0
                let shiftScaleStage : Stage<Image<float32>, Image<float32>> = StackProcessing.shiftScale<float32> 1.0 2.0
                let stretchStage : Stage<Image<float32>, Image<float32>> = StackProcessing.intensityStretch 0.0 200.0 0.0 1.0
                let medianStage : Stage<Image<uint8>, Image<uint8>> = StackProcessing.smoothWMedian<uint8> 1u (Some 3u)
                let gradientStage : Stage<Image<float32>, Image<float32>> = StackProcessing.gradientMagnitudeSquared 1.0 (Some 3u)
                let sobelStage : Stage<Image<float32>, Image<float32>> = StackProcessing.sobelEdge<float32> (Some 3u)
                let laplacianStage : Stage<Image<float32>, Image<float32>> = StackProcessing.laplacian<float32> (Some 3u)
                let erodeStage : Stage<Image<uint8>, Image<uint8>> = StackProcessing.grayscaleErode<uint8> 1u (Some 3u)
                let dilateStage : Stage<Image<uint8>, Image<uint8>> = StackProcessing.grayscaleDilate<uint8> 1u (Some 3u)
                let openingStage : Stage<Image<uint8>, Image<uint8>> = StackProcessing.grayscaleOpening<uint8> 1u (Some 3u)
                let closingStage : Stage<Image<uint8>, Image<uint8>> = StackProcessing.grayscaleClosing<uint8> 1u (Some 3u)
                let whiteTopHatStage : Stage<Image<uint8>, Image<uint8>> = StackProcessing.whiteTopHat<uint8> 1u (Some 3u)
                let blackTopHatStage : Stage<Image<uint8>, Image<uint8>> = StackProcessing.blackTopHat<uint8> 1u (Some 3u)
                let morphGradientStage : Stage<Image<uint8>, Image<uint8>> = StackProcessing.morphologicalGradient<uint8> 1u (Some 3u)
                let binaryContourStage : Stage<Image<uint8>, Image<uint8>> = StackProcessing.binaryContour false (Some 3u)
                let binaryMedianStage : Stage<Image<uint8>, Image<uint8>> = StackProcessing.binaryMedian 1u (Some 3u)
                let labelContourStage : Stage<Image<uint8>, Image<uint8>> = StackProcessing.labelContour<uint8> false (Some 3u)
                let changeLabelStage : Stage<Image<uint8>, Image<uint8>> = StackProcessing.changeLabel<uint8> 255.0 128.0
                let saltAndPepperStage : Stage<Image<float32>, Image<float32>> = StackProcessing.addSaltAndPepperNoise 0.0 None None
                let shotStage : Stage<Image<float32>, Image<float32>> = StackProcessing.addPoissonNoise 0.0
                let speckleStage : Stage<Image<float32>, Image<float32>> = StackProcessing.addSpeckleNoise 0.0

                expectSameShapeFloat32 "clamp" floatSlices clampStage
                expectSameShapeFloat32 "shiftScale" floatSlices shiftScaleStage
                expectSameShapeFloat32 "intensityStretch" floatSlices stretchStage
                expectSameShapeUint8 "smoothWMedian" uintSlices medianStage
                expectSameShapeFloat32 "gradientMagnitudeSquared" floatSlices gradientStage
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
                expectSameShapeFloat32 "addPoissonNoise" floatSlices shotStage
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

        testCase "chunk fan-out retains one reference for the second consuming branch" <| fun _ ->
            StackProcessing.Chunk.resetStats()

            let left, right =
                source 1024UL
                |> zero<uint8> 4u 4u 1u
                >=>> (objectVolume 1.0 1.0 1.0, objectVolume 1.0 1.0 1.0)
                |> drain

            Expect.equal left right "Both fan-out branches should see the same chunk data."

            let stats = StackProcessing.Chunk.stats()
            Expect.equal stats.Live 0L "Fan-out should not leave retained chunks alive."
            Expect.equal stats.Created stats.Released "Every chunk created by the fan-out test should be released."

    ]
