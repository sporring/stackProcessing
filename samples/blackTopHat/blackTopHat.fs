// Extract small background holes with a binary zonohedral black top-hat filter.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/blackTopHat"
        | _ -> "../data/volume", "../tmp/blackTopHat"

    src
    |> readChunkSlices<uint8> input ".tiff"
    >=> chunkThresholdRange<uint8> 1 255
    >=> chunkBinaryBlackTopHatZonohedral 3u
    >=> writeChunkSlices output ".tiff"
    |> sink

    0
