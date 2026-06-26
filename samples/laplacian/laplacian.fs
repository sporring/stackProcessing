// Enhance second-derivative structure with a Laplacian filter.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/laplacian"
        | _ -> "../data/volume", "../tmp/laplacian"

    src
    |> read<float32> input ".tiff"
    >=> laplacian 1.0 3
    >=> intensityStretch 0.0 255.0 0.0 255.0
    >=> cast<_, uint8>
    >=> write output ".tiff"
    |> sink

    0
