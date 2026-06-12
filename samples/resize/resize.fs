// Resize a stack to an explicit x/y/z size.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/resize"
        | _ -> "../data/rotatingBoxes", "../tmp/resize"

    src
    |> readChunkSlices<uint8> input ".tiff"
    |> chunkResize<uint8> 96u 96u 96u "Linear"
    >=> writeChunkSlices output ".tiff"
    |> sink

    0
