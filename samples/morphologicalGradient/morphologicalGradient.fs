// Compute the binary zonohedral morphological gradient of a mask stack.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/morphologicalGradient"
        | _ -> "../data/volume", "../tmp/morphologicalGradient"

    src
    |> readChunkSlices<uint8> input ".tiff"
    >=> chunkThresholdRange<uint8> 1 255
    >=> chunkBinaryGradientZonohedral 3u
    >=> writeChunkSlices output ".tiff"
    |> sink

    0
