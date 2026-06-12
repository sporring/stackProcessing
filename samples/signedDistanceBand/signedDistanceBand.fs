// Compute a band-limited signed distance map from a binary stack.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/signedDistanceBand"
        | _ -> "../data/rotatingBoxes", "../tmp/signedDistanceBand"

    src
    |> readChunkSlices<uint8> input ".tiff"
    >=> chunkSignedDistanceBandNativeParallelCollect 8u 4u 4
    >=> writeChunkSlices output ".tiff"
    |> sink

    0
