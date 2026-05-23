// Close a grayscale stack to suppress small dark structures.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/grayscaleClosing"
        | _ -> "../data/volume", "../tmp/grayscaleClosing"

    src
    |> read<uint8> input ".tiff"
    >=> grayscaleClosing<uint8> 3u None
    >=> write output ".tiff"
    |> sink

    0
