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

let private chunkFromInt8Pixels width height (pixels: int8[]) =
    let chunk = Chunk.create<int8> (uint64 width, uint64 height, 1UL)
    let values = Chunk.span<int8> chunk
    if values.Length <> pixels.Length then
        invalidArg "pixels" $"Expected {values.Length} pixels, got {pixels.Length}."
    pixels.CopyTo(values)
    chunk

let private chunkFromInt16Pixels width height (pixels: int16[]) =
    let chunk = Chunk.create<int16> (uint64 width, uint64 height, 1UL)
    let values = Chunk.span<int16> chunk
    if values.Length <> pixels.Length then
        invalidArg "pixels" $"Expected {values.Length} pixels, got {pixels.Length}."
    pixels.CopyTo(values)
    chunk

let private chunkFromUInt16Pixels width height (pixels: uint16[]) =
    let chunk = Chunk.create<uint16> (uint64 width, uint64 height, 1UL)
    let values = Chunk.span<uint16> chunk
    if values.Length <> pixels.Length then
        invalidArg "pixels" $"Expected {values.Length} pixels, got {pixels.Length}."
    pixels.CopyTo(values)
    chunk

let private chunkFromFloat32Pixels width height (pixels: float32[]) =
    let chunk = Chunk.create<float32> (uint64 width, uint64 height, 1UL)
    let values = Chunk.span<float32> chunk
    if values.Length <> pixels.Length then
        invalidArg "pixels" $"Expected {values.Length} pixels, got {pixels.Length}."
    pixels.CopyTo(values)
    chunk

let private chunkFromInt32Pixels width height (pixels: int32[]) =
    let chunk = Chunk.create<int32> (uint64 width, uint64 height, 1UL)
    let values = Chunk.span<int32> chunk
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

        testCase "ofImage and toImageWith bridge scalar image buffers through Chunk storage" <| fun _ ->
            let image = Image<float32>.ofFlatArray([ 2u; 2u ], [| 1.0f; 2.0f; 3.0f; 4.0f |], "chunkBridge", 17)
            let chunk = Chunk.ofImage image
            let roundTrip = Chunk.toImageWith "chunkBridge.roundTrip" 19 chunk
            try
                Expect.equal chunk.Size (2UL, 2UL, 1UL) "ofImage should represent a 2D image as a depth-1 chunk."
                Expect.sequenceEqual ((Chunk.span<float32> chunk).ToArray()) [| 1.0f; 2.0f; 3.0f; 4.0f |] "ofImage should copy image pixels into the valid chunk span."
                Expect.equal (roundTrip.GetSize()) [ 2u; 2u ] "toImageWith should convert depth-1 chunks back to 2D images."
                Expect.equal roundTrip.index 19 "toImageWith should carry the requested index."
                Expect.sequenceEqual (roundTrip.toFlatArray()) [| 1.0f; 2.0f; 3.0f; 4.0f |] "toImageWith should preserve scalar pixel values."
            finally
                roundTrip.decRefCount()
                Chunk.decRef chunk
                image.decRefCount()

        testCase "toSlab and ofSlab bridge chunk windows through emitted slab slices" <| fun _ ->
            let chunks =
                [ for z in 0 .. 2 ->
                    [| for i in 0 .. 3 -> uint16 (z * 10 + i) |]
                    |> chunkFromUInt16Pixels 2 2 ]
            let window =
                { Items = chunks
                  EmitRange = 1u, 1u
                  ReleaseCount = 1u }
            let slab = Chunk.toSlabWith "chunkSlabBridge" window
            let outputs = Chunk.ofSlab slab
            try
                Expect.equal (slab.Image.GetSize()) [ 2u; 2u; 3u ] "toSlab should stack 2D slice chunks into a 3D slab image."
                Expect.equal outputs.Length 1 "ofSlab should emit only the requested slice range."
                Expect.equal outputs[0].Size (2UL, 2UL, 1UL) "ofSlab should return 2D emitted slices as depth-1 chunks."
                Expect.sequenceEqual ((Chunk.span<uint16> outputs[0]).ToArray()) [| 10us; 11us; 12us; 13us |] "ofSlab should preserve the emitted center slice pixels."
                chunks |> List.iter (fun chunk -> Expect.equal chunk.RefCount.Value 1 "Bridge helpers should not own or release source chunks.")
            finally
                outputs |> List.iter Chunk.decRef
                slab.Image.decRefCount()
                chunks |> List.iter Chunk.decRef

        testCase "ChunkFunctions scalar arithmetic and pair arithmetic release inputs" <| fun _ ->
            let scalarInput = chunkFromInt32Pixels 4 1 [| 1; 2; 3; 4 |]
            let scalarOutput = runStageList (ChunkFunctions.addScalar 10) [ scalarInput ]
            try
                Expect.equal scalarInput.RefCount.Value 0 "Scalar arithmetic should release the consumed input chunk."
                Expect.equal scalarOutput.Length 1 "Scalar arithmetic should emit one chunk."
                Expect.sequenceEqual ((Chunk.span<int32> scalarOutput[0]).ToArray()) [| 11; 12; 13; 14 |] "addScalar should transform every element."
            finally
                scalarOutput |> List.iter Chunk.decRef

            let left = chunkFromFloat32Pixels 3 1 [| 1.0f; 2.0f; 3.0f |]
            let right = chunkFromFloat32Pixels 3 1 [| 10.0f; 20.0f; 30.0f |]
            let pairOutput = runStageList (ChunkFunctions.add<float32>) [ left, right ]
            try
                Expect.equal left.RefCount.Value 0 "Pair arithmetic should release the left input chunk."
                Expect.equal right.RefCount.Value 0 "Pair arithmetic should release the right input chunk."
                Expect.sequenceEqual ((Chunk.span<float32> pairOutput[0]).ToArray()) [| 11.0f; 22.0f; 33.0f |] "add should combine matching elements."
            finally
                pairOutput |> List.iter Chunk.decRef

        testCase "ChunkFunctions comparisons and mask operations produce UInt8 binary chunks" <| fun _ ->
            let left = chunkFromInt16Pixels 4 1 [| 1s; 5s; 7s; 9s |]
            let right = chunkFromInt16Pixels 4 1 [| 1s; 6s; 6s; 10s |]
            let greater = runStageList (ChunkFunctions.greater<int16>) [ left, right ]
            try
                Expect.sequenceEqual ((Chunk.span<uint8> greater[0]).ToArray()) [| 0uy; 0uy; 1uy; 0uy |] "greater should emit 0/1 mask values."
            finally
                greater |> List.iter Chunk.decRef

            let maskA = chunkFromPixels 4 1 [| 1uy; 0uy; 1uy; 0uy |]
            let maskB = chunkFromPixels 4 1 [| 1uy; 1uy; 0uy; 0uy |]
            let anded = runStageList ChunkFunctions.maskAnd [ maskA, maskB ]
            try
                Expect.sequenceEqual ((Chunk.span<uint8> anded[0]).ToArray()) [| 1uy; 0uy; 0uy; 0uy |] "maskAnd should preserve the 0/1 binary convention."
            finally
                anded |> List.iter Chunk.decRef

            let maskC = chunkFromPixels 4 1 [| 1uy; 0uy; 2uy; 0uy |]
            let notted = runStageList ChunkFunctions.maskNot [ maskC ]
            try
                Expect.sequenceEqual ((Chunk.span<uint8> notted[0]).ToArray()) [| 0uy; 1uy; 0uy; 1uy |] "maskNot should map zero to one and any non-zero value to zero."
            finally
                notted |> List.iter Chunk.decRef

        testCase "ChunkFunctions Float32 intensity stages use span-sized outputs" <| fun _ ->
            let shiftedInput = chunkFromFloat32Pixels 4 1 [| -1.0f; 0.0f; 1.0f; 2.0f |]
            let shifted = runStageList (ChunkFunctions.shiftScaleFloat32 (1.0: double) (2.0: double)) [ shiftedInput ]
            try
                Expect.sequenceEqual ((Chunk.span<float32> shifted[0]).ToArray()) [| 0.0f; 2.0f; 4.0f; 6.0f |] "shiftScaleFloat32 should apply (x + shift) * scale."
            finally
                shifted |> List.iter Chunk.decRef

            let windowInput = chunkFromFloat32Pixels 4 1 [| -10.0f; 0.0f; 5.0f; 10.0f |]
            let windowed = runStageList (ChunkFunctions.intensityWindowFloat32 (0.0: double) (10.0: double) (0.0: double) (1.0: double)) [ windowInput ]
            try
                Expect.sequenceEqual ((Chunk.span<float32> windowed[0]).ToArray()) [| 0.0f; 0.0f; 0.5f; 1.0f |] "intensityWindowFloat32 should clamp and scale the configured window."
            finally
                windowed |> List.iter Chunk.decRef

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

        testCase "ChunkFunctions.convolveFixedKernel applies a z-window kernel to UInt8 slices" <| fun _ ->
            let kernel = Array3D.zeroCreate<float32> 1 1 3
            kernel[0, 0, 0] <- 1.0f
            kernel[0, 0, 1] <- 2.0f
            kernel[0, 0, 2] <- 3.0f

            let chunks =
                [ chunkFromPixels 1 1 [| 10uy |]
                  chunkFromPixels 1 1 [| 20uy |]
                  chunkFromPixels 1 1 [| 30uy |] ]

            let outputs = runStageList (ChunkFunctions.convolveFixedKernel<uint8> kernel) chunks
            try
                Expect.equal outputs.Length 3 "Chunk convolution should preserve slice count for same-size zero-boundary convolution."
                let values = outputs |> List.map (fun chunk -> (Chunk.span<uint8> chunk)[0])
                Expect.sequenceEqual values [ 80uy; 140uy; 80uy ] "Chunk convolution should use zero padding before the first and after the last slice."
                chunks |> List.iter (fun chunk -> Expect.equal chunk.RefCount.Value 0 "Convolution should release consumed input chunks.")
            finally
                outputs |> List.iter Chunk.decRef

        testCase "ChunkFunctions.convolveFixedKernelParallel matches serial float32 convolution" <| fun _ ->
            let width = 40
            let height = 3
            let depth = 5
            let kernel = Array3D.zeroCreate<float32> 3 1 3
            kernel[0, 0, 0] <- 0.125f
            kernel[1, 0, 1] <- 0.5f
            kernel[2, 0, 2] <- 0.25f

            let makeChunks () =
                [ for z in 0 .. depth - 1 ->
                    [| for y in 0 .. height - 1 do
                           for x in 0 .. width - 1 do
                               float32 (z * 1000 + y * width + x) |]
                    |> chunkFromFloat32Pixels width height ]

            let serialOutputs = runStageList (ChunkFunctions.convolveFixedKernel<float32> kernel) (makeChunks ())
            let parallelOutputs = runStageList (ChunkFunctions.convolveFixedKernelParallel<float32> kernel 3) (makeChunks ())

            try
                Expect.equal parallelOutputs.Length serialOutputs.Length "Parallel chunk convolution should preserve the serial output count."

                for z in 0 .. serialOutputs.Length - 1 do
                    let serialValues = (Chunk.span<float32> serialOutputs[z]).ToArray()
                    let parallelValues = (Chunk.span<float32> parallelOutputs[z]).ToArray()
                    Expect.sequenceEqual parallelValues serialValues $"Parallel chunk convolution should match serial output slice {z}."
            finally
                parallelOutputs |> List.iter Chunk.decRef
                serialOutputs |> List.iter Chunk.decRef

        testCase "ChunkFunctions rolling PH median matches dense PH median" <| fun _ ->
            let width = 5
            let height = 4
            let depth = 6
            let radius = 1

            let makeChunks () =
                [ for z in 0 .. depth - 1 ->
                    [| for y in 0 .. height - 1 do
                           for x in 0 .. width - 1 do
                               uint8 ((z * 37 + y * 11 + x * 7 + (x * y)) % 251) |]
                    |> chunkFromPixels width height ]

            let denseInputs = makeChunks ()
            let rollingInputs = makeChunks ()
            let treeInputs = makeChunks ()
            let blockedZInputs = makeChunks ()
            let transposedInputs = makeChunks ()
            let xFirstInputs = makeChunks ()
            let xBlockInputs = makeChunks ()
            let yBandInputs = makeChunks ()
            let denseOutputs = runStageList (ChunkFunctions.medianPerreaultHebertUInt8Dense radius) denseInputs
            let rollingOutputs = runStageList (ChunkFunctions.medianPerreaultHebertUInt8DenseRolling radius) rollingInputs
            let treeOutputs = runStageList (ChunkFunctions.medianPerreaultHebertUInt8DenseRollingTree radius) treeInputs
            let blockedZOutputs = runStageList (ChunkFunctions.medianPerreaultHebertUInt8DenseRollingBlockedZ radius) blockedZInputs
            let transposedOutputs = runStageList (ChunkFunctions.medianPerreaultHebertUInt8DenseRollingTransposedXBlock radius) transposedInputs
            let xFirstOutputs = runStageList (ChunkFunctions.medianPerreaultHebertUInt8DenseXFirstMaterialized radius) xFirstInputs
            let xBlockOutputs = runStageList (ChunkFunctions.medianPerreaultHebertUInt8DenseXBlock radius) xBlockInputs
            let yBandOutputs = runStageList (ChunkFunctions.medianPerreaultHebertUInt8DenseRollingYBands radius 3) yBandInputs

            try
                Expect.equal rollingOutputs.Length denseOutputs.Length "Rolling PH median should preserve the dense PH output count."
                Expect.equal treeOutputs.Length denseOutputs.Length "Tree PH median should preserve the dense PH output count."
                Expect.equal blockedZOutputs.Length denseOutputs.Length "Blocked-z PH median should preserve the dense PH output count."
                Expect.equal transposedOutputs.Length denseOutputs.Length "Transposed x-block PH median should preserve the dense PH output count."
                Expect.equal xFirstOutputs.Length denseOutputs.Length "X-first PH median should preserve the dense PH output count."
                Expect.equal xBlockOutputs.Length denseOutputs.Length "X-block PH median should preserve the dense PH output count."
                Expect.equal yBandOutputs.Length denseOutputs.Length "Y-band PH median should preserve the dense PH output count."
                Expect.equal rollingOutputs.Length depth "Rolling PH median should emit one slice per input slice."

                for z in 0 .. denseOutputs.Length - 1 do
                    let denseValues = (Chunk.span<uint8> denseOutputs[z]).ToArray()
                    let rollingValues = (Chunk.span<uint8> rollingOutputs[z]).ToArray()
                    let treeValues = (Chunk.span<uint8> treeOutputs[z]).ToArray()
                    let blockedZValues = (Chunk.span<uint8> blockedZOutputs[z]).ToArray()
                    let transposedValues = (Chunk.span<uint8> transposedOutputs[z]).ToArray()
                    let xFirstValues = (Chunk.span<uint8> xFirstOutputs[z]).ToArray()
                    let xBlockValues = (Chunk.span<uint8> xBlockOutputs[z]).ToArray()
                    let yBandValues = (Chunk.span<uint8> yBandOutputs[z]).ToArray()
                    Expect.sequenceEqual rollingValues denseValues $"Rolling PH median should match dense PH output slice {z}."
                    Expect.sequenceEqual treeValues denseValues $"Tree PH median should match dense PH output slice {z}."
                    Expect.sequenceEqual blockedZValues denseValues $"Blocked-z PH median should match dense PH output slice {z}."
                    Expect.sequenceEqual transposedValues denseValues $"Transposed x-block PH median should match dense PH output slice {z}."
                    Expect.sequenceEqual xFirstValues denseValues $"X-first PH median should match dense PH output slice {z}."
                    Expect.sequenceEqual xBlockValues denseValues $"X-block PH median should match dense PH output slice {z}."
                    Expect.sequenceEqual yBandValues denseValues $"Y-band PH median should match dense PH output slice {z}."

                denseInputs |> List.iter (fun chunk -> Expect.equal chunk.RefCount.Value 0 "Dense PH median should release consumed input chunks.")
                rollingInputs |> List.iter (fun chunk -> Expect.equal chunk.RefCount.Value 0 "Rolling PH median should release consumed input chunks.")
                treeInputs |> List.iter (fun chunk -> Expect.equal chunk.RefCount.Value 0 "Tree PH median should release consumed input chunks.")
                blockedZInputs |> List.iter (fun chunk -> Expect.equal chunk.RefCount.Value 0 "Blocked-z PH median should release consumed input chunks.")
                transposedInputs |> List.iter (fun chunk -> Expect.equal chunk.RefCount.Value 0 "Transposed x-block PH median should release consumed input chunks.")
                xFirstInputs |> List.iter (fun chunk -> Expect.equal chunk.RefCount.Value 0 "X-first PH median should release consumed input chunks.")
                xBlockInputs |> List.iter (fun chunk -> Expect.equal chunk.RefCount.Value 0 "X-block PH median should release consumed input chunks.")
                yBandInputs |> List.iter (fun chunk -> Expect.equal chunk.RefCount.Value 0 "Y-band PH median should release consumed input chunks.")
            finally
                yBandOutputs |> List.iter Chunk.decRef
                xBlockOutputs |> List.iter Chunk.decRef
                xFirstOutputs |> List.iter Chunk.decRef
                transposedOutputs |> List.iter Chunk.decRef
                blockedZOutputs |> List.iter Chunk.decRef
                treeOutputs |> List.iter Chunk.decRef
                rollingOutputs |> List.iter Chunk.decRef
                denseOutputs |> List.iter Chunk.decRef

        testCase "ChunkFunctions nth-element median stages match sorted zero-padded reference" <| fun _ ->
            let width = 5
            let height = 4
            let depth = 5
            let radius = 1
            let windowLength = 2 * radius + 1
            let sampleCount = windowLength * windowLength * windowLength
            let medianIndex = sampleCount / 2

            let inline reference zero (sliceValues: _[][]) =
                [ for z in 0 .. depth - 1 ->
                    [| for y in 0 .. height - 1 do
                           for x in 0 .. width - 1 do
                               let values =
                                   [| for dz in -radius .. radius do
                                          for dy in -radius .. radius do
                                              for dx in -radius .. radius do
                                                  let zz = z + dz
                                                  let yy = y + dy
                                                  let xx = x + dx
                                                  if zz >= 0 && zz < depth && yy >= 0 && yy < height && xx >= 0 && xx < width then
                                                      yield sliceValues[zz][yy * width + xx]
                                                  else
                                                      yield zero |]
                               Array.sortInPlace values
                               yield values[medianIndex] |] ]

            let uint16Slices =
                [| for z in 0 .. depth - 1 ->
                    [| for y in 0 .. height - 1 do
                           for x in 0 .. width - 1 ->
                               uint16 ((z * 1009 + y * 211 + x * 17 + x * y * 13) % 65521) |] |]
            let uint8Slices =
                [| for z in 0 .. depth - 1 ->
                    [| for y in 0 .. height - 1 do
                           for x in 0 .. width - 1 ->
                               uint8 ((z * 101 + y * 29 + x * 17 + x * y * 7) % 251) |] |]
            let int16Slices =
                [| for z in 0 .. depth - 1 ->
                    [| for y in 0 .. height - 1 do
                           for x in 0 .. width - 1 ->
                               int16 (((z * 997 + y * 127 + x * 23 + x * y * 11) % 60001) - 30000) |] |]
            let float32Slices =
                [| for z in 0 .. depth - 1 ->
                    [| for y in 0 .. height - 1 do
                           for x in 0 .. width - 1 ->
                               float32 ((z * 997 + y * 127 + x * 23 + x * y * 11) % 60001) / 7.0f - 3000.0f |] |]

            let uint8QuickInputs = uint8Slices |> Array.map (chunkFromPixels width height) |> Array.toList
            let uint16Inputs = uint16Slices |> Array.map (chunkFromUInt16Pixels width height) |> Array.toList
            let int16Inputs = int16Slices |> Array.map (chunkFromInt16Pixels width height) |> Array.toList
            let float32Inputs = float32Slices |> Array.map (chunkFromFloat32Pixels width height) |> Array.toList
            let uint8QuickOutputs = runStageList (ChunkFunctions.medianQuickselectUInt8 radius) uint8QuickInputs
            let uint16Outputs = runStageList (ChunkFunctions.medianNthElementUInt16 radius) uint16Inputs
            let int16Outputs = runStageList (ChunkFunctions.medianNthElementInt16 radius) int16Inputs
            let float32Outputs = runStageList (ChunkFunctions.medianNthElementFloat32 radius) float32Inputs
            let uint16QuickInputs = uint16Slices |> Array.map (chunkFromUInt16Pixels width height) |> Array.toList
            let uint16QuickParallelInputs = uint16Slices |> Array.map (chunkFromUInt16Pixels width height) |> Array.toList
            let uint16SortInputs = uint16Slices |> Array.map (chunkFromUInt16Pixels width height) |> Array.toList
            let uint16SortParallelInputs = uint16Slices |> Array.map (chunkFromUInt16Pixels width height) |> Array.toList
            let int16QuickInputs = int16Slices |> Array.map (chunkFromInt16Pixels width height) |> Array.toList
            let uint16QuickOutputs = runStageList (ChunkFunctions.medianQuickselectUInt16 radius) uint16QuickInputs
            let uint16QuickParallelOutputs = runStageList (ChunkFunctions.medianQuickselectUInt16ParallelCollect radius 3) uint16QuickParallelInputs
            let uint16SortOutputs = runStageList (ChunkFunctions.medianSortUInt16 radius) uint16SortInputs
            let uint16SortParallelOutputs = runStageList (ChunkFunctions.medianSortUInt16ParallelCollect radius 3) uint16SortParallelInputs
            let int16QuickOutputs = runStageList (ChunkFunctions.medianQuickselectInt16 radius) int16QuickInputs

            try
                let expectedUInt8 = reference 0uy uint8Slices
                let expectedUInt16 = reference 0us uint16Slices
                let expectedInt16 = reference 0s int16Slices
                let expectedFloat32 = reference 0.0f float32Slices

                Expect.equal uint8QuickOutputs.Length depth "UInt8 quickselect median should emit one slice per input slice."
                Expect.equal uint16Outputs.Length depth "UInt16 nth-element median should emit one slice per input slice."
                Expect.equal int16Outputs.Length depth "Int16 nth-element median should emit one slice per input slice."
                Expect.equal float32Outputs.Length depth "Float32 nth-element median should emit one slice per input slice."
                Expect.equal uint16QuickOutputs.Length depth "UInt16 quickselect median should emit one slice per input slice."
                Expect.equal uint16QuickParallelOutputs.Length depth "Parallel UInt16 quickselect median should emit one slice per input slice."
                Expect.equal uint16SortOutputs.Length depth "UInt16 sort median should emit one slice per input slice."
                Expect.equal uint16SortParallelOutputs.Length depth "Parallel UInt16 sort median should emit one slice per input slice."
                Expect.equal int16QuickOutputs.Length depth "Int16 quickselect median should emit one slice per input slice."

                for z in 0 .. depth - 1 do
                    Expect.sequenceEqual ((Chunk.span<uint8> uint8QuickOutputs[z]).ToArray()) expectedUInt8[z] $"UInt8 quickselect median should match reference slice {z}."
                    Expect.sequenceEqual ((Chunk.span<uint16> uint16Outputs[z]).ToArray()) expectedUInt16[z] $"UInt16 nth-element median should match reference slice {z}."
                    Expect.sequenceEqual ((Chunk.span<int16> int16Outputs[z]).ToArray()) expectedInt16[z] $"Int16 nth-element median should match reference slice {z}."
                    Expect.sequenceEqual ((Chunk.span<float32> float32Outputs[z]).ToArray()) expectedFloat32[z] $"Float32 nth-element median should match reference slice {z}."
                    Expect.sequenceEqual ((Chunk.span<uint16> uint16QuickOutputs[z]).ToArray()) expectedUInt16[z] $"UInt16 quickselect median should match reference slice {z}."
                    Expect.sequenceEqual ((Chunk.span<uint16> uint16QuickParallelOutputs[z]).ToArray()) expectedUInt16[z] $"Parallel UInt16 quickselect median should match reference slice {z}."
                    Expect.sequenceEqual ((Chunk.span<uint16> uint16SortOutputs[z]).ToArray()) expectedUInt16[z] $"UInt16 sort median should match reference slice {z}."
                    Expect.sequenceEqual ((Chunk.span<uint16> uint16SortParallelOutputs[z]).ToArray()) expectedUInt16[z] $"Parallel UInt16 sort median should match reference slice {z}."
                    Expect.sequenceEqual ((Chunk.span<int16> int16QuickOutputs[z]).ToArray()) expectedInt16[z] $"Int16 quickselect median should match reference slice {z}."

                uint8QuickInputs |> List.iter (fun chunk -> Expect.equal chunk.RefCount.Value 0 "UInt8 quickselect median should release consumed input chunks.")
                uint16Inputs |> List.iter (fun chunk -> Expect.equal chunk.RefCount.Value 0 "UInt16 nth-element median should release consumed input chunks.")
                int16Inputs |> List.iter (fun chunk -> Expect.equal chunk.RefCount.Value 0 "Int16 nth-element median should release consumed input chunks.")
                float32Inputs |> List.iter (fun chunk -> Expect.equal chunk.RefCount.Value 0 "Float32 nth-element median should release consumed input chunks.")
                uint16QuickInputs |> List.iter (fun chunk -> Expect.equal chunk.RefCount.Value 0 "UInt16 quickselect median should release consumed input chunks.")
                uint16QuickParallelInputs |> List.iter (fun chunk -> Expect.equal chunk.RefCount.Value 0 "Parallel UInt16 quickselect median should release consumed input chunks.")
                uint16SortInputs |> List.iter (fun chunk -> Expect.equal chunk.RefCount.Value 0 "UInt16 sort median should release consumed input chunks.")
                uint16SortParallelInputs |> List.iter (fun chunk -> Expect.equal chunk.RefCount.Value 0 "Parallel UInt16 sort median should release consumed input chunks.")
                int16QuickInputs |> List.iter (fun chunk -> Expect.equal chunk.RefCount.Value 0 "Int16 quickselect median should release consumed input chunks.")
            finally
                int16QuickOutputs |> List.iter Chunk.decRef
                uint16SortParallelOutputs |> List.iter Chunk.decRef
                uint16SortOutputs |> List.iter Chunk.decRef
                uint16QuickParallelOutputs |> List.iter Chunk.decRef
                uint16QuickOutputs |> List.iter Chunk.decRef
                float32Outputs |> List.iter Chunk.decRef
                int16Outputs |> List.iter Chunk.decRef
                uint16Outputs |> List.iter Chunk.decRef
                uint8QuickOutputs |> List.iter Chunk.decRef

        testCase "ChunkFunctions ITK-wrapped median parallelCollect matches one-worker wrapper" <| fun _ ->
            let width = 5
            let height = 4
            let depth = 5
            let radius = 1

            let makeInputs () =
                [ for z in 0 .. depth - 1 ->
                    [| for y in 0 .. height - 1 do
                           for x in 0 .. width - 1 ->
                               uint16 ((z * 1009 + y * 211 + x * 17 + x * y * 13) % 65521) |]
                    |> chunkFromUInt16Pixels width height ]

            let serialInputs = makeInputs ()
            let parallelInputs = makeInputs ()
            let serialOutputs = runStageList (ChunkFunctions.medianItkWrapped<uint16> radius) serialInputs
            let parallelOutputs = runStageList (ChunkFunctions.medianItkWrappedParallelCollect<uint16> radius 3) parallelInputs

            try
                Expect.equal serialOutputs.Length depth "ITK-wrapped median should emit one slice per input slice."
                Expect.equal parallelOutputs.Length serialOutputs.Length "Parallel ITK-wrapped median should preserve serial output count."

                for z in 0 .. depth - 1 do
                    Expect.sequenceEqual ((Chunk.span<uint16> parallelOutputs[z]).ToArray()) ((Chunk.span<uint16> serialOutputs[z]).ToArray()) $"Parallel ITK-wrapped median should match one-worker slice {z}."

                serialInputs |> List.iter (fun chunk -> Expect.equal chunk.RefCount.Value 0 "One-worker ITK-wrapped median should release consumed input chunks.")
                parallelInputs |> List.iter (fun chunk -> Expect.equal chunk.RefCount.Value 0 "Parallel ITK-wrapped median should release consumed input chunks.")
            finally
                parallelOutputs |> List.iter Chunk.decRef
                serialOutputs |> List.iter Chunk.decRef

        testCase "ChunkFunctions castToFloat32 widens signed and unsigned integer chunk spans" <| fun _ ->
            let width = 2 * System.Numerics.Vector<byte>.Count + 3
            let bytes = [| for i in 0 .. width - 1 -> uint8 ((i * 7) % 251) |]
            let signedBytes = [| for i in 0 .. width - 1 -> int8 ((i % 101) - 50) |]
            let uint8Chunk = chunkFromPixels width 1 bytes
            let int8Chunk = chunkFromInt8Pixels width 1 signedBytes
            let uint16Chunk = Chunk.create<uint16> (uint64 width, 1UL, 1UL)
            let int16Chunk = Chunk.create<int16> (uint64 width, 1UL, 1UL)
            try
                let uint16Values = Chunk.span<uint16> uint16Chunk
                let int16Values = Chunk.span<int16> int16Chunk
                for i in 0 .. width - 1 do
                    uint16Values[i] <- uint16 (i * 257)
                    int16Values[i] <- int16 (i * 257 - 30000)
                let expectedUInt16 = uint16Values.ToArray() |> Array.map float32
                let expectedInt16 = int16Values.ToArray() |> Array.map float32

                let uint8Output = runStageList ChunkFunctions.castToFloat32<uint8> [ uint8Chunk ]
                let int8Output = runStageList ChunkFunctions.castToFloat32<int8> [ int8Chunk ]
                let uint16Output = runStageList ChunkFunctions.castToFloat32<uint16> [ uint16Chunk ]
                let int16Output = runStageList ChunkFunctions.castToFloat32<int16> [ int16Chunk ]
                try
                    Expect.sequenceEqual ((Chunk.span<float32> uint8Output[0]).ToArray()) (bytes |> Array.map float32) "UInt8 cast should widen all byte values to Float32."
                    Expect.sequenceEqual ((Chunk.span<float32> int8Output[0]).ToArray()) (signedBytes |> Array.map float32) "Int8 cast should preserve negative values while widening to Float32."
                    Expect.sequenceEqual ((Chunk.span<float32> uint16Output[0]).ToArray()) expectedUInt16 "UInt16 cast should widen all UInt16 values to Float32."
                    Expect.sequenceEqual ((Chunk.span<float32> int16Output[0]).ToArray()) expectedInt16 "Int16 cast should preserve signed values while widening to Float32."
                finally
                    uint8Output |> List.iter Chunk.decRef
                    int8Output |> List.iter Chunk.decRef
                    uint16Output |> List.iter Chunk.decRef
                    int16Output |> List.iter Chunk.decRef
            finally
                if uint8Chunk.RefCount.Value > 0 then Chunk.decRef uint8Chunk
                if int8Chunk.RefCount.Value > 0 then Chunk.decRef int8Chunk
                if uint16Chunk.RefCount.Value > 0 then Chunk.decRef uint16Chunk
                if int16Chunk.RefCount.Value > 0 then Chunk.decRef int16Chunk

        testCase "ChunkFunctions castFromFloat32 narrows signed integer chunks" <| fun _ ->
            let input = chunkFromFloat32Pixels 4 1 [| -200.0f; -5.4f; 12.6f; 200.0f |]
            let output = runStageList ChunkFunctions.castFromFloat32<int8> [ input ]
            try
                Expect.sequenceEqual ((Chunk.span<int8> output[0]).ToArray()) [| SByte.MinValue; -5y; 13y; SByte.MaxValue |] "Int8 narrowing should round and clamp signed values."
            finally
                output |> List.iter Chunk.decRef

        testCase "ChunkFunctions.convolveFixedKernel supports Int32 chunks" <| fun _ ->
            let kernel = Array3D.zeroCreate<float32> 1 1 3
            kernel[0, 0, 0] <- 1.0f
            kernel[0, 0, 1] <- 2.0f
            kernel[0, 0, 2] <- 3.0f

            let chunks =
                [ chunkFromInt32Pixels 1 1 [| -10 |]
                  chunkFromInt32Pixels 1 1 [| 20 |]
                  chunkFromInt32Pixels 1 1 [| -30 |] ]

            let outputs = runStageList (ChunkFunctions.convolveFixedKernel<int32> kernel) chunks
            try
                let values = outputs |> List.map (fun chunk -> (Chunk.span<int32> chunk)[0])
                Expect.sequenceEqual values [ 40; -60; -40 ] "Int32 chunk convolution should preserve signed output values."
                chunks |> List.iter (fun chunk -> Expect.equal chunk.RefCount.Value 0 "Convolution should release consumed Int32 input chunks.")
            finally
                outputs |> List.iter Chunk.decRef

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
