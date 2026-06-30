// Extract small foreground structures with a binary zonohedral white top-hat filter.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/whiteTopHat"
        | _ -> "../data/volume", "../tmp/whiteTopHat"

    src
    |> read<uint8> input ".tiff"
    >=> thresholdRange 1 255
    >=> binaryWhiteTopHat 3u
    >=> write output ".tiff"
    |> sink

    0
