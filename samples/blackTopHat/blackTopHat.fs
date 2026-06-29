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
        | _ -> "../data/rotatingBoxes", "../tmp/blackTopHat"

    // read a binary image and apply the top-hat mathematical morphology operation to it. 
    src
    |> read<uint8> input ".tiff"
    >=> binaryBlackTopHat 3u
    >=> intensityStretch 0.0 1.0 0.0 255.0
    >=> write output ".tiff"
    |> sink

    0
