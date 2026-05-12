// Extract small bright structures with a white top-hat filter.
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
    >=> whiteTopHat<uint8> 1u 5u
    >=> write output ".tiff"
    |> sink

    0
