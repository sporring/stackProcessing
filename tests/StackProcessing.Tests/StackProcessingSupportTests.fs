module Tests.StackProcessingSupportTests

open System
open System.IO
open Expecto
open Image
open PureHDF
open SlimPipeline
open StackProcessing

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

        testCase "marchingCubes streams mesh chunks and writeMesh writes OBJ" <| fun _ ->
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

                let chunks =
                    source (2UL * 1024UL * 1024UL * 1024UL)
                    |> read<uint8> inputDir suffix
                    >=> marchingCubes<uint8> 0.5
                    |> drainList

                Expect.isTrue (chunks |> List.exists (fun chunk -> not chunk.Triangles.IsEmpty)) "marchingCubes should emit at least one triangle chunk for a crossing surface."

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

    ]
