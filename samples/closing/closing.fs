// Binary closing fills small background gaps in a 0/1 UInt8 mask.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/closing"
        | _ -> "../data/rotatingBoxes", "../tmp/closing"

    src
    |> readChunkSlices<uint8> input ".tiff"
    >=> chunkBinaryClosingZonohedral 3u
    >=> writeChunkSlices output ".tiff"
    |> sink

    0
