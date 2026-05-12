// Dilate bright structures in a grayscale stack.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/grayscaleDilate"
        | _ -> "../data/volume", "../tmp/grayscaleDilate"

    src
    |> read<uint8> input ".tiff"
    >=> grayscaleDilate<uint8> 1u 5u
    >=> write output ".tiff"
    |> sink

    0
