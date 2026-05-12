// Extract small dark structures with a black top-hat filter.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/blackTopHat"
        | _ -> "../data/volume", "../tmp/blackTopHat"

    src
    |> read<uint8> input ".tiff"
    >=> blackTopHat<uint8> 1u 5u
    >=> write output ".tiff"
    |> sink

    0
