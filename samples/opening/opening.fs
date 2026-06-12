// Binary opening removes small foreground details from a UInt8 mask.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/opening"
        | _ -> "../data/rotatingBoxes", "../tmp/opening"

    src
    |> readChunkSlices<uint8> input ".tiff"
    >=> chunkBinaryOpeningZonohedral 2u
    >=> writeChunkSlices output ".tiff"
    |> sink

    0
