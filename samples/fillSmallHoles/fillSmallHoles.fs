// Fill small enclosed background regions in a UInt8 mask.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/fillSmallHoles"
        | _ -> "../data/rotatingBoxes", "../tmp/fillSmallHoles"

    src
    |> readChunkSlices<uint8> input ".tiff"
    >=> chunkFillSmallHoles 128UL ObjectConnectivity.TwentySix
    >=> writeChunkSlices output ".tiff"
    |> sink

    0
