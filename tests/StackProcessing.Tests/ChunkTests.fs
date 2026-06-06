module Tests.ChunkTests

open System
open System.Runtime.InteropServices
open Expecto
open FSharp.Control
open SlimPipeline
open StackCore

let private manualChunk size bytes byteLength release =
    { Size = size
      Bytes = bytes
      ByteLength = byteLength
      Release = release
      RefCount = ref 1 }

let private runStageList (stage: SlimPipeline.Stage<'S, 'T>) (items: 'S seq) =
    (stage.Build()).Apply false (AsyncSeq.ofSeq items)
    |> AsyncSeq.toListAsync
    |> Async.RunSynchronously

let private chunkFromPixels width height (pixels: uint8[]) =
    let chunk = Chunk.create<uint8> (uint64 width, uint64 height, 1UL)
    let values = Chunk.span<uint8> chunk
    if values.Length <> pixels.Length then
        invalidArg "pixels" $"Expected {values.Length} pixels, got {pixels.Length}."
    pixels.CopyTo(values)
    chunk

let chunkSuite =
    testList "Chunk" [
        testCase "create rents byte-backed storage with typed span over requested size" <| fun _ ->
            let chunk = Chunk.create<uint16> (3UL, 2UL, 1UL)
            try
                Expect.equal chunk.ByteLength 12 "ByteLength should be element count times sizeof<uint16>."
                Expect.isGreaterThanOrEqual chunk.Bytes.Length chunk.ByteLength "ArrayPool may rent more than requested, but not less."

                let values = Chunk.span<uint16> chunk
                Expect.equal values.Length 6 "Span length should reflect ByteLength, not the physical ArrayPool length."

                values[0] <- 11us
                values[5] <- 17us
                let reread = Chunk.span<uint16> chunk
                Expect.equal reread[0] 11us "Span should expose the chunk backing bytes."
                Expect.equal reread[5] 17us "Span should expose the last valid element."
            finally
                Chunk.decRef chunk

        testCase "incRef and decRef release only when the final reference is dropped" <| fun _ ->
            let releaseCount = ref 0
            let chunk = manualChunk (4UL, 1UL, 1UL) (Array.zeroCreate<byte> 4) 4 (fun () -> releaseCount.Value <- releaseCount.Value + 1)

            let retained = Chunk.incRef chunk
            Expect.isTrue (obj.ReferenceEquals(chunk, retained)) "incRef should return the same chunk value."
            Expect.equal chunk.RefCount.Value 2 "incRef should increment the reference count."

            Chunk.decRef chunk
            Expect.equal releaseCount.Value 0 "First decRef should not release while another reference exists."
            Expect.equal chunk.RefCount.Value 1 "First decRef should leave one reference."

            Chunk.decRef retained
            Expect.equal releaseCount.Value 1 "Final decRef should call the release hook exactly once."
            Expect.equal chunk.RefCount.Value 0 "Final decRef should leave the count at zero."

            Expect.throws (fun () -> Chunk.decRef chunk) "decRef after release should fail."
            Expect.equal releaseCount.Value 1 "A failing extra decRef should not call release again."

        testCase "incRef rejects already released chunks" <| fun _ ->
            let chunk = manualChunk (1UL, 1UL, 1UL) (Array.zeroCreate<byte> 1) 1 ignore
            Chunk.decRef chunk
            Expect.throws (fun () -> Chunk.incRef chunk |> ignore) "incRef should reject released chunks."

        testCase "mapInto writes input-length values into a larger output chunk" <| fun _ ->
            let input = Chunk.create<uint8> (3UL, 1UL, 1UL)
            let output = Chunk.create<uint16> (5UL, 1UL, 1UL)
            try
                let inputValues = Chunk.span<uint8> input
                inputValues[0] <- 2uy
                inputValues[1] <- 4uy
                inputValues[2] <- 6uy

                let outputValues = Chunk.span<uint16> output
                for i in 0 .. outputValues.Length - 1 do
                    outputValues[i] <- 99us

                Chunk.mapInto (fun value -> uint16 value + 1us) input output

                Expect.sequenceEqual (outputValues.ToArray()) [| 3us; 5us; 7us; 99us; 99us |] "mapInto should only overwrite the input-length prefix."
            finally
                Chunk.decRef output
                Chunk.decRef input

        testCase "map creates a pooled output chunk" <| fun _ ->
            let input = Chunk.create<uint8> (4UL, 1UL, 1UL)
            let mutable outputOpt : Chunk<uint16> option = None
            try
                let inputValues = Chunk.span<uint8> input
                for i in 0 .. inputValues.Length - 1 do
                    inputValues[i] <- uint8 (i + 1)

                let output = Chunk.map (fun value -> uint16 value * 10us) input
                outputOpt <- Some output

                Expect.equal output.Size input.Size "map should preserve the input shape."
                Expect.equal output.RefCount.Value 1 "map should return a newly owned chunk."
                Expect.sequenceEqual ((Chunk.span<uint16> output).ToArray()) [| 10us; 20us; 30us; 40us |] "map should transform all values."
            finally
                outputOpt |> Option.iter Chunk.decRef
                Chunk.decRef input

        testCase "mapi receives flat indices in row-major order" <| fun _ ->
            let input = Chunk.create<uint8> (2UL, 2UL, 1UL)
            let mutable outputOpt : Chunk<int> option = None
            try
                let values = Chunk.span<uint8> input
                for i in 0 .. values.Length - 1 do
                    values[i] <- 10uy

                let output = Chunk.mapi (fun i value -> i + int value) input
                outputOpt <- Some output

                Expect.sequenceEqual ((Chunk.span<int> output).ToArray()) [| 10; 11; 12; 13 |] "mapi should pass monotonically increasing flat indices."
            finally
                outputOpt |> Option.iter Chunk.decRef
                Chunk.decRef input

        testCase "iter iteri fold and foldi traverse in span order" <| fun _ ->
            let chunk = Chunk.create<uint8> (4UL, 1UL, 1UL)
            try
                let values = Chunk.span<uint8> chunk
                values[0] <- 3uy
                values[1] <- 5uy
                values[2] <- 7uy
                values[3] <- 11uy

                let seen = ResizeArray<uint8>()
                Chunk.iter seen.Add chunk
                Expect.sequenceEqual (seen.ToArray()) [| 3uy; 5uy; 7uy; 11uy |] "iter should visit all values in order."

                let seenWithIndex = ResizeArray<int * uint8>()
                Chunk.iteri (fun i value -> seenWithIndex.Add(i, value)) chunk
                Expect.sequenceEqual (seenWithIndex.ToArray()) [| 0, 3uy; 1, 5uy; 2, 7uy; 3, 11uy |] "iteri should include flat indices."

                let sum = Chunk.fold (fun acc value -> acc + int value) 0 chunk
                Expect.equal sum 26 "fold should accumulate values."

                let weighted = Chunk.foldi (fun acc i value -> acc + i * int value) 0 chunk
                Expect.equal weighted 52 "foldi should include flat indices."
            finally
                Chunk.decRef chunk

        testCase "histogram counts chunk values" <| fun _ ->
            let chunk = Chunk.create<uint8> (6UL, 1UL, 1UL)
            try
                let values = Chunk.span<uint8> chunk
                values[0] <- 4uy
                values[1] <- 2uy
                values[2] <- 4uy
                values[3] <- 9uy
                values[4] <- 2uy
                values[5] <- 4uy

                let expected =
                    Map.ofList [
                        2uy, 2UL
                        4uy, 3UL
                        9uy, 1UL
                    ]

                Expect.equal (ChunkFunctions.histogram chunk) expected "histogram should count repeated values."
            finally
                Chunk.decRef chunk

        testCase "histogramDense counts UInt8 values with dense byte bins" <| fun _ ->
            let chunk = Chunk.create<uint8> (5UL, 1UL, 1UL)
            try
                let values = Chunk.span<uint8> chunk
                values[0] <- 255uy
                values[1] <- 0uy
                values[2] <- 255uy
                values[3] <- 7uy
                values[4] <- 0uy

                let expected = Map.ofList [ 0uy, 2UL; 7uy, 1UL; 255uy, 2UL ]
                Expect.equal (ChunkFunctions.histogramDense chunk) expected "histogramDense should count dense byte bins."
            finally
                Chunk.decRef chunk

        testCase "histogramDense counts UInt16 values with dense UInt16 bins" <| fun _ ->
            let chunk = Chunk.create<uint16> (5UL, 1UL, 1UL)
            try
                let values = Chunk.span<uint16> chunk
                values[0] <- 65535us
                values[1] <- 0us
                values[2] <- 512us
                values[3] <- 65535us
                values[4] <- 512us

                let expected = Map.ofList [ 0us, 1UL; 512us, 2UL; 65535us, 2UL ]
                Expect.equal (ChunkFunctions.histogramDense chunk) expected "histogramDense should count dense UInt16 bins."
            finally
                Chunk.decRef chunk

        testCase "histogramDense preserves signed integer abscissae" <| fun _ ->
            let chunk8 = Chunk.create<int8> (4UL, 1UL, 1UL)
            let chunk16 = Chunk.create<int16> (4UL, 1UL, 1UL)
            try
                let values8 = Chunk.span<int8> chunk8
                values8[0] <- SByte.MinValue
                values8[1] <- -1y
                values8[2] <- -1y
                values8[3] <- SByte.MaxValue

                let expected8 = Map.ofList [ SByte.MinValue, 1UL; -1y, 2UL; SByte.MaxValue, 1UL ]
                Expect.equal (ChunkFunctions.histogramDense chunk8) expected8 "histogramDense should return original int8 values, not shifted dense indices."

                let values16 = Chunk.span<int16> chunk16
                values16[0] <- Int16.MinValue
                values16[1] <- -17s
                values16[2] <- -17s
                values16[3] <- Int16.MaxValue

                let expected16 = Map.ofList [ Int16.MinValue, 1UL; -17s, 2UL; Int16.MaxValue, 1UL ]
                Expect.equal (ChunkFunctions.histogramDense chunk16) expected16 "histogramDense should return original int16 values, not shifted dense indices."
            finally
                Chunk.decRef chunk16
                Chunk.decRef chunk8

        testCase "histogramDense rejects non-small-integer chunks" <| fun _ ->
            let chunk = Chunk.create<float32> (1UL, 1UL, 1UL)
            try
                Expect.throws (fun () -> ChunkFunctions.histogramDense chunk |> ignore) "histogramDense should reject floating-point chunks."
            finally
                Chunk.decRef chunk

        testCase "histogramLeftEdges bins values and puts underflow in first bin" <| fun _ ->
            let chunk = Chunk.create<float32> (8UL, 1UL, 1UL)
            try
                let values = Chunk.span<float32> chunk
                values[0] <- -10.0f
                values[1] <- 0.0f
                values[2] <- 0.9f
                values[3] <- 1.0f
                values[4] <- 1.5f
                values[5] <- 2.0f
                values[6] <- 100.0f
                values[7] <- Single.NaN

                let expected =
                    Map.ofList [
                        0.0, 3UL
                        1.0, 2UL
                        2.0, 2UL
                    ]

                Expect.equal (ChunkFunctions.histogramLeftEdges [ 0.0; 1.0; 2.0 ] chunk) expected "left-edge histogram should clamp underflow to first bin and overflow to last bin."
            finally
                Chunk.decRef chunk

        testCase "histogramLeftEdges rejects empty non-finite and unsorted edges" <| fun _ ->
            let chunk = Chunk.create<uint8> (1UL, 1UL, 1UL)
            try
                Expect.throws (fun () -> ChunkFunctions.histogramLeftEdges [] chunk |> ignore) "Empty edge lists should be rejected."
                Expect.throws (fun () -> ChunkFunctions.histogramLeftEdges [ 0.0; Double.NaN ] chunk |> ignore) "NaN edges should be rejected."
                Expect.throws (fun () -> ChunkFunctions.histogramLeftEdges [ 1.0; 1.0 ] chunk |> ignore) "Duplicate edges should be rejected."
                Expect.throws (fun () -> ChunkFunctions.histogramLeftEdges [ 2.0; 1.0 ] chunk |> ignore) "Descending edges should be rejected."
            finally
                Chunk.decRef chunk

        testCase "ChunkFunctions.histogram counts byte-backed values directly" <| fun _ ->
            let bytes = Array.zeroCreate<byte> 6
            bytes[0] <- 3uy
            bytes[1] <- 1uy
            bytes[2] <- 3uy
            bytes[3] <- 9uy
            bytes[4] <- 1uy
            bytes[5] <- 3uy

            let expected = Map.ofList [ 1uy, 2UL; 3uy, 3UL; 9uy, 1UL ]
            Expect.equal (ChunkFunctions.histogramBytes<uint8> bytes bytes.Length) expected "ChunkFunctions.histogramBytes should count values from the valid byte prefix."

        testCase "ChunkFunctions.histogramDense preserves signed abscissae from byte storage" <| fun _ ->
            let values = [| Int16.MinValue; -2s; -2s; 0s; Int16.MaxValue |]
            let bytes = MemoryMarshal.AsBytes(values.AsSpan()).ToArray()

            let expected =
                Map.ofList [
                    Int16.MinValue, 1UL
                    -2s, 2UL
                    0s, 1UL
                    Int16.MaxValue, 1UL
                ]

            Expect.equal (ChunkFunctions.histogramDenseBytes<int16> bytes bytes.Length) expected "Dense histograms should expose original signed bin values, not shifted storage indices."

        testCase "ChunkFunctions.addDenseInto adds compatible dense count arrays" <| fun _ ->
            let target = ChunkFunctions.UInt8Counts [| 1UL; 2UL; 3UL |]
            let source = ChunkFunctions.UInt8Counts [| 10UL; 20UL; 30UL |]

            ChunkFunctions.addDenseInto target source

            match target with
            | ChunkFunctions.UInt8Counts counts ->
                Expect.sequenceEqual counts [| 11UL; 22UL; 33UL |] "Dense add should mutate the target counts."
            | _ ->
                failtest "Expected UInt8 dense counts."

            Expect.throws (fun () -> ChunkFunctions.addDenseInto target (ChunkFunctions.Int8Counts [| 1UL; 2UL; 3UL |])) "Dense add should reject incompatible integer domains."

        testCase "ChunkFunctions.histogramDenseCounts returns implicit-abscissa dense counts" <| fun _ ->
            let chunk = Chunk.create<int8> (4UL, 1UL, 1UL)
            try
                let values = Chunk.span<int8> chunk
                values[0] <- SByte.MinValue
                values[1] <- -1y
                values[2] <- -1y
                values[3] <- SByte.MaxValue

                match ChunkFunctions.histogramDenseCounts chunk with
                | ChunkFunctions.Int8Counts counts ->
                    Expect.equal counts[0] 1UL "Int8.MinValue should live at dense index 0."
                    Expect.equal counts[127] 2UL "-1 should live at dense index 127."
                    Expect.equal counts[128] 0UL "Zero should not have been observed."
                    Expect.equal counts[255] 1UL "Int8.MaxValue should live at dense index 255."
                | _ ->
                    failtest "Expected Int8 dense counts."
            finally
                Chunk.decRef chunk

        testCase "ChunkFunctions.histogramLeftEdges includes empty bin abscissae" <| fun _ ->
            let values = [| -10.0f; 0.5f; 100.0f |]
            let bytes = MemoryMarshal.AsBytes(values.AsSpan()).ToArray()

            let expected =
                Map.ofList [
                    0.0, 2UL
                    10.0, 0UL
                    20.0, 1UL
                ]

            Expect.equal (ChunkFunctions.histogramLeftEdgesBytes<float32> [ 0.0; 10.0; 20.0 ] bytes bytes.Length) expected "Left-edge histograms should keep all edge abscissae, including zero-count bins."

        testCase "ChunkFunctions.addLeftEdgesInto adds only compatible left-edge histograms" <| fun _ ->
            let target: ChunkFunctions.LeftEdgeHistogram =
                { ChunkFunctions.LeftEdges = [| 0.0; 10.0; 20.0 |]
                  Counts = [| 1UL; 2UL; 3UL |] }

            let source: ChunkFunctions.LeftEdgeHistogram =
                { ChunkFunctions.LeftEdges = [| 0.0; 10.0; 20.0 |]
                  Counts = [| 10UL; 20UL; 30UL |] }

            ChunkFunctions.addLeftEdgesInto target source
            Expect.sequenceEqual target.Counts [| 11UL; 22UL; 33UL |] "Left-edge add should mutate the target counts when edges match exactly."

            let incompatible: ChunkFunctions.LeftEdgeHistogram =
                { ChunkFunctions.LeftEdges = [| 0.0; 5.0; 20.0 |]
                  Counts = [| 1UL; 1UL; 1UL |] }

            Expect.throws (fun () -> ChunkFunctions.addLeftEdgesInto target incompatible) "Left-edge add should reject incompatible edge arrays."

        testCase "ChunkFunctions.histogramLeftEdgeCounts returns reusable edge and count arrays" <| fun _ ->
            let chunk = Chunk.create<float32> (3UL, 1UL, 1UL)
            try
                let values = Chunk.span<float32> chunk
                values[0] <- -1.0f
                values[1] <- 11.0f
                values[2] <- 99.0f

                let histogram = ChunkFunctions.histogramLeftEdgeCounts [ 0.0; 10.0; 20.0 ] chunk
                Expect.sequenceEqual histogram.LeftEdges [| 0.0; 10.0; 20.0 |] "Left-edge count histograms should preserve explicit abscissae."
                Expect.sequenceEqual histogram.Counts [| 1UL; 1UL; 1UL |] "Left-edge count histograms should store ordinates in a reusable count array."
            finally
                Chunk.decRef chunk

        testCase "ChunkFunctions dictionary helpers merge sparse histograms" <| fun _ ->
            let target = System.Collections.Generic.Dictionary<int, uint64>()
            target[1] <- 2UL
            target[3] <- 4UL

            let source = System.Collections.Generic.Dictionary<int, uint64>()
            source[1] <- 5UL
            source[2] <- 7UL

            ChunkFunctions.addDictionaryInto target source
            Expect.equal (ChunkFunctions.dictionaryToMap target) (Map.ofList [ 1, 7UL; 2, 7UL; 3, 4UL ]) "Sparse dictionary add should update existing bins and add new bins."

        testCase "ChunkFunctions.histogramReducer serially reduces sparse chunk histograms and releases chunks" <| fun _ ->
            let releaseCount = ref 0
            let chunk1 = manualChunk (3UL, 1UL, 1UL) [| 1uy; 2uy; 1uy |] 3 (fun () -> releaseCount.Value <- releaseCount.Value + 1)
            let chunk2 = manualChunk (4UL, 1UL, 1UL) [| 2uy; 3uy; 3uy; 3uy |] 4 (fun () -> releaseCount.Value <- releaseCount.Value + 1)

            let results = runStageList (ChunkFunctions.histogramReducer<uint8> ()) [ chunk1; chunk2 ]

            Expect.equal results.Length 1 "Histogram reducer should emit one final histogram."
            Expect.equal results[0].Counts (Map.ofList [ 1uy, 2UL; 2uy, 2UL; 3uy, 3UL ]) "Sparse chunk histogram reducer should merge all chunk counts."
            Expect.equal releaseCount.Value 2 "Histogram reducer should decRef every consumed chunk."
            Expect.equal chunk1.RefCount.Value 0 "First consumed chunk should have ref count zero."
            Expect.equal chunk2.RefCount.Value 0 "Second consumed chunk should have ref count zero."

        testCase "ChunkFunctions.histogramDenseReducer serially reduces dense chunk histograms" <| fun _ ->
            let releaseCount = ref 0
            let chunk1 = manualChunk (3UL, 1UL, 1UL) [| 0uy; 255uy; 255uy |] 3 (fun () -> releaseCount.Value <- releaseCount.Value + 1)
            let chunk2 = manualChunk (2UL, 1UL, 1UL) [| 7uy; 255uy |] 2 (fun () -> releaseCount.Value <- releaseCount.Value + 1)

            let results = runStageList (ChunkFunctions.histogramDenseReducer<uint8> ()) [ chunk1; chunk2 ]

            Expect.equal results.Length 1 "Dense histogram reducer should emit one final histogram."
            Expect.equal results[0].Counts (Map.ofList [ 0uy, 1UL; 7uy, 1UL; 255uy, 3UL ]) "Dense chunk histogram reducer should add compatible dense count arrays."
            Expect.equal releaseCount.Value 2 "Dense histogram reducer should decRef every consumed chunk."

        testCase "ChunkFunctions.histogramReducerParallel reduces sparse chunk histograms with window size two" <| fun _ ->
            let releaseCount = ref 0
            let chunks =
                [ manualChunk (3UL, 1UL, 1UL) [| 1uy; 2uy; 1uy |] 3 (fun () -> releaseCount.Value <- releaseCount.Value + 1)
                  manualChunk (2UL, 1UL, 1UL) [| 2uy; 4uy |] 2 (fun () -> releaseCount.Value <- releaseCount.Value + 1)
                  manualChunk (4UL, 1UL, 1UL) [| 4uy; 4uy; 8uy; 8uy |] 4 (fun () -> releaseCount.Value <- releaseCount.Value + 1)
                  manualChunk (1UL, 1UL, 1UL) [| 8uy |] 1 (fun () -> releaseCount.Value <- releaseCount.Value + 1)
                  manualChunk (2UL, 1UL, 1UL) [| 1uy; 8uy |] 2 (fun () -> releaseCount.Value <- releaseCount.Value + 1) ]

            let results = runStageList (ChunkFunctions.histogramReducerParallel<uint8> 2) chunks

            Expect.equal results.Length 1 "Parallel sparse histogram reducer should emit one final histogram."
            Expect.equal results[0].Counts (Map.ofList [ 1uy, 3UL; 2uy, 2UL; 4uy, 3UL; 8uy, 4UL ]) "Parallel sparse histogram reducer should merge worker dictionaries."
            Expect.equal releaseCount.Value chunks.Length "Parallel sparse histogram reducer should decRef every consumed chunk."

        testCase "ChunkFunctions.histogramDenseReducerParallel reduces dense chunk histograms with window size two" <| fun _ ->
            let releaseCount = ref 0
            let chunks =
                [ manualChunk (3UL, 1UL, 1UL) [| 0uy; 255uy; 255uy |] 3 (fun () -> releaseCount.Value <- releaseCount.Value + 1)
                  manualChunk (2UL, 1UL, 1UL) [| 7uy; 255uy |] 2 (fun () -> releaseCount.Value <- releaseCount.Value + 1)
                  manualChunk (3UL, 1UL, 1UL) [| 7uy; 7uy; 0uy |] 3 (fun () -> releaseCount.Value <- releaseCount.Value + 1)
                  manualChunk (1UL, 1UL, 1UL) [| 42uy |] 1 (fun () -> releaseCount.Value <- releaseCount.Value + 1)
                  manualChunk (2UL, 1UL, 1UL) [| 42uy; 255uy |] 2 (fun () -> releaseCount.Value <- releaseCount.Value + 1) ]

            let results = runStageList (ChunkFunctions.histogramDenseReducerParallel<uint8> 2) chunks

            Expect.equal results.Length 1 "Parallel dense histogram reducer should emit one final histogram."
            Expect.equal results[0].Counts (Map.ofList [ 0uy, 2UL; 7uy, 3UL; 42uy, 2UL; 255uy, 4UL ]) "Parallel dense histogram reducer should merge worker count arrays."
            Expect.equal releaseCount.Value chunks.Length "Parallel dense histogram reducer should decRef every consumed chunk."

        testCase "parallelCollect maps non-overlapping windows and collects emitted items" <| fun _ ->
            let stage =
                Stage.parallelCollect
                    "parallel-window-sums"
                    3
                    3
                    3
                    0
                    (fun _ item -> item)
                    (fun _ window -> [ window.Items |> List.sum ])
                    (fun _ -> 1UL)
                    id

            let results = runStageList stage [ 1; 2; 3; 4; 5; 6; 7; 8 ]

            Expect.equal results [ 6; 15; 15 ] "parallelCollect should cover full non-overlapping windows plus the trailing partial window."

        testCase "parallelCollect separates rolling window size from parallel batch size" <| fun _ ->
            let stage =
                Stage.parallelCollect
                    "parallel-rolling-window-sums"
                    3
                    2
                    1
                    0
                    (fun _ item -> item)
                    (fun _ window -> [ window.Items |> List.sum ])
                    (fun _ -> 1UL)
                    id

            let results = runStageList stage [ 1; 2; 3; 4; 5 ]

            Expect.equal results [ 6; 9; 12; 9 ] "parallelCollect should keep rolling-window semantics while batching windows separately."

        testCase "ChunkFunctions.binaryDilateZonohedral expands a UInt8 mask through chunk slices" <| fun _ ->
            let width = 3
            let height = 3
            let depth = 3
            let plane = width * height
            let chunks =
                [ for z in 0 .. depth - 1 ->
                    let pixels = Array.zeroCreate<uint8> plane
                    if z = 1 then
                        pixels[1 * width + 1] <- 1uy
                    chunkFromPixels width height pixels ]

            let outputs = runStageList (ChunkFunctions.binaryDilateZonohedral 1u) chunks
            try
                Expect.equal outputs.Length depth "Chunk zonohedral dilation should preserve slice count."

                for z in 0 .. depth - 1 do
                    let values = Chunk.span<uint8> outputs[z]
                    for y in 0 .. height - 1 do
                        for x in 0 .. width - 1 do
                            let expected =
                                if x <= 1 && y >= 1 && y <= 2 && z <= 1 then 1uy else 0uy
                            Expect.equal values[y * width + x] expected $"Unexpected dilation value at ({x},{y},{z})."
            finally
                outputs |> List.iter Chunk.decRef

        testCase "ChunkFunctions.binaryDilateZonohedralParallel matches the serial chunk dilation" <| fun _ ->
            let width = 5
            let height = 4
            let depth = 6
            let plane = width * height

            let makeChunks () =
                [ for z in 0 .. depth - 1 ->
                    let pixels = Array.zeroCreate<uint8> plane
                    if z = 1 then
                        pixels[1 * width + 2] <- 1uy
                    if z = 3 then
                        pixels[2 * width + 3] <- 1uy
                    chunkFromPixels width height pixels ]

            let serialOutputs = runStageList (ChunkFunctions.binaryDilateZonohedral 1u) (makeChunks ())
            let parallelOutputs = runStageList (ChunkFunctions.binaryDilateZonohedralParallel 1u 3) (makeChunks ())

            try
                Expect.equal parallelOutputs.Length serialOutputs.Length "Parallel chunk dilation should preserve the serial output count."

                for z in 0 .. serialOutputs.Length - 1 do
                    let serialValues = (Chunk.span<uint8> serialOutputs[z]).ToArray()
                    let parallelValues = (Chunk.span<uint8> parallelOutputs[z]).ToArray()
                    Expect.sequenceEqual parallelValues serialValues $"Parallel chunk dilation should match serial output slice {z}."
            finally
                parallelOutputs |> List.iter Chunk.decRef
                serialOutputs |> List.iter Chunk.decRef

        testCase "ChunkFunctions.binaryErodeZonohedral erodes a UInt8 mask through chunk slices" <| fun _ ->
            let width = 3
            let height = 3
            let depth = 3
            let plane = width * height
            let chunks =
                [ for _z in 0 .. depth - 1 ->
                    Array.create plane 1uy
                    |> chunkFromPixels width height ]

            let outputs = runStageList (ChunkFunctions.binaryErodeZonohedral 1u) chunks
            try
                Expect.equal outputs.Length depth "Chunk zonohedral erosion should preserve slice count."

                for z in 0 .. depth - 1 do
                    let values = Chunk.span<uint8> outputs[z]
                    for y in 0 .. height - 1 do
                        for x in 0 .. width - 1 do
                            let expected =
                                if x <= 1 && y >= 1 && z <= 1 then 1uy else 0uy
                            Expect.equal values[y * width + x] expected $"Unexpected erosion value at ({x},{y},{z})."
            finally
                outputs |> List.iter Chunk.decRef

        testCase "ChunkFunctions parallel morphology stages match serial chunk morphology" <| fun _ ->
            let width = 5
            let height = 4
            let depth = 6
            let plane = width * height

            let makeChunks () =
                [ for z in 0 .. depth - 1 ->
                    let pixels = Array.zeroCreate<uint8> plane
                    for y in 1 .. 2 do
                        for x in 1 .. 3 do
                            if z >= 1 && z <= 4 then
                                pixels[y * width + x] <- 1uy
                    if z = 2 then
                        pixels[0] <- 1uy
                    chunkFromPixels width height pixels ]

            let compareStages name serialStage parallelStage =
                let serialOutputs = runStageList serialStage (makeChunks ())
                let parallelOutputs = runStageList parallelStage (makeChunks ())
                try
                    Expect.equal parallelOutputs.Length serialOutputs.Length $"{name} should preserve the serial output count."

                    for z in 0 .. serialOutputs.Length - 1 do
                        let serialValues = (Chunk.span<uint8> serialOutputs[z]).ToArray()
                        let parallelValues = (Chunk.span<uint8> parallelOutputs[z]).ToArray()
                        Expect.sequenceEqual parallelValues serialValues $"{name} parallel output should match serial output slice {z}."
                finally
                    parallelOutputs |> List.iter Chunk.decRef
                    serialOutputs |> List.iter Chunk.decRef

            compareStages
                "erosion"
                (ChunkFunctions.binaryErodeZonohedral 1u)
                (ChunkFunctions.binaryErodeZonohedralParallel 1u 3)

            compareStages
                "opening"
                (ChunkFunctions.binaryOpeningZonohedral 1u)
                (ChunkFunctions.binaryOpeningZonohedralParallel 1u 3)

            compareStages
                "closing"
                (ChunkFunctions.binaryClosingZonohedral 1u)
                (ChunkFunctions.binaryClosingZonohedralParallel 1u 3)

        testCase "ChunkFunctions.histogramLeftEdgesReducer serially reduces compatible binned chunk histograms" <| fun _ ->
            let releaseCount = ref 0
            let values1 = [| -1.0f; 0.5f; 11.0f |]
            let values2 = [| 19.0f; 20.0f; 99.0f |]
            let bytes1 = MemoryMarshal.AsBytes(values1.AsSpan()).ToArray()
            let bytes2 = MemoryMarshal.AsBytes(values2.AsSpan()).ToArray()
            let chunk1 = manualChunk (3UL, 1UL, 1UL) bytes1 bytes1.Length (fun () -> releaseCount.Value <- releaseCount.Value + 1)
            let chunk2 = manualChunk (3UL, 1UL, 1UL) bytes2 bytes2.Length (fun () -> releaseCount.Value <- releaseCount.Value + 1)

            let results = runStageList (ChunkFunctions.histogramLeftEdgesReducer<float32> [ 0.0; 10.0; 20.0 ]) [ chunk1; chunk2 ]

            Expect.equal results.Length 1 "Left-edge histogram reducer should emit one final histogram."
            Expect.equal results[0].Counts (Map.ofList [ 0.0, 2UL; 10.0, 2UL; 20.0, 2UL ]) "Left-edge chunk histogram reducer should add compatible ordinate arrays."
            Expect.equal results[0].Binning (Some(FixedEdges(0.0, 20.0, 3u))) "Left-edge reducer should preserve bin metadata."
            Expect.equal releaseCount.Value 2 "Left-edge histogram reducer should decRef every consumed chunk."

        testCase "toIndex and ofIndex are inverse for row-major xyz indexing" <| fun _ ->
            let width = 5
            let height = 4

            for z in 0 .. 2 do
                for y in 0 .. height - 1 do
                    for x in 0 .. width - 1 do
                        let index = Chunk.toIndex width height x y z
                        Expect.equal (Chunk.ofIndex width height index) (x, y, z) $"ofIndex should invert toIndex for ({x},{y},{z})."
    ]
