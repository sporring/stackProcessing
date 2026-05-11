// Remove small connected foreground components from a binary stack.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output, maximumVolume =
        match args with
        | [| input; output; maximumVolume |] -> input, output, uint64 maximumVolume
        | [| input; output |] -> input, output, 1000UL
        | [| input |] -> input, "../tmp/removeSmallObjects", 1000UL
        | _ -> "../data/rotatingBoxes", "../tmp/removeSmallObjects", 1000UL

    src
    |> read<uint8> input ".tiff"
    >=> removeSmallObjects maximumVolume ObjectConnectivity.TwentySix
    >=> write output ".tiff"
    |> sink

    0
