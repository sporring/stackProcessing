// Erode bright structures in a grayscale stack.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/grayscaleErode"
        | _ -> "../data/volume", "../tmp/grayscaleErode"

    src
    |> read<uint8> input ".tiff"
    >=> grayscaleErode<uint8> 3u None
    >=> write output ".tiff"
    |> sink

    0
