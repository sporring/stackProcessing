// Highlight edges by computing gradient magnitude.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/gradientMagnitude"
        | _ -> "../data/volume", "../tmp/gradientMagnitude"

    src
    |> read<float> input ".tiff"
    >=> gradientMagnitude<float> None
    >=> intensityStretch<float> 0.0 255.0 0.0 255.0
    >=> cast<float, uint8>
    >=> write output ".tiff"
    |> sink

    0
