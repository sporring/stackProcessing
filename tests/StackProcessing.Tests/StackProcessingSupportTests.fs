module Tests.StackProcessingSupportTests

open System
open System.IO
open Expecto
open Image
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
            finally
                disposeImages slices
                deleteDirectory inputDir

        testCase "writeInChunks creates chunk files and chunk metadata can be read" <| fun _ ->
            let inputDir = tempDirectory "chunks-input"
            let chunkDir = tempDirectory "chunks-output"
            let suffix = ".tiff"
            let chunkSuffix = ".mha"
            let slices = [ for z in 0 .. 3 -> makeSlice 4 4 z ]

            try
                writeSlices inputDir suffix slices

                source (2UL * 1024UL * 1024UL * 1024UL)
                |> read<uint8> inputDir suffix
                >=> writeInChunks chunkDir chunkSuffix 2u 2u 2u
                |> sink

                let info = getChunkInfo chunkDir chunkSuffix
                Expect.equal info.chunks [ 2; 2; 2 ] "4x4x4 data written as 2x2x2 chunks should create a 2x2x2 chunk grid."
                Expect.equal info.size [ 4UL; 4UL; 4UL ] "Chunk metadata should reconstruct full volume size."
                Expect.isTrue (File.Exists(getChunkFilename chunkDir chunkSuffix 1 1 1)) "The final chunk file should exist."
            finally
                disposeImages slices
                deleteDirectory inputDir
                deleteDirectory chunkDir

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
                    >=> teeFst (writeChunkSlices labelDir labelSuffix 2u)
                    >=> makeConnectedComponentTranslationTable 2u
                    |> drain

                Expect.isNonEmpty table "Connected component translation table should contain label mappings."
                Expect.isTrue (File.Exists(Path.Combine(labelDir, "image_000.mha"))) "writeChunkSlices should write label slices from the tuple stream."
                Expect.isTrue (table |> List.exists (fun (_, sourceLabel, targetLabel) -> sourceLabel <> 0UL && targetLabel <> 0UL)) "The table should contain foreground label mappings."
            finally
                disposeImages slices
                deleteDirectory inputDir
                deleteDirectory labelDir

    ]
