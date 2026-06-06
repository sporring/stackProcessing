module Tests.ChunkTests

open Expecto
open StackCore

let private manualChunk size bytes byteLength release =
    { Size = size
      Bytes = bytes
      ByteLength = byteLength
      Release = release
      RefCount = ref 1 }

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

                Expect.equal (Chunk.histogram chunk) expected "histogram should count repeated values."
            finally
                Chunk.decRef chunk

        testCase "toIndex and ofIndex are inverse for row-major xyz indexing" <| fun _ ->
            let width = 5
            let height = 4

            for z in 0 .. 2 do
                for y in 0 .. height - 1 do
                    for x in 0 .. width - 1 do
                        let index = Chunk.toIndex width height x y z
                        Expect.equal (Chunk.ofIndex width height index) (x, y, z) $"ofIndex should invert toIndex for ({x},{y},{z})."
    ]
