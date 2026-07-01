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

    // Calculate the laplacian and scale intensities for better viewing. The min and max values are estimated by computeStats()
    let stats = 
        src
        |> readRange<float32> 20u 1 30u input ".tiff"
        >=> laplacian 1.0 (Some 7u)
        >=> computeStats ()
        |> drain

    src
    |> read<float32> input ".tiff"
    >=> laplacian 1.0 (Some 7u)
    >=> intensityStretch stats.Min stats.Max 0.0 255.0
    >=> cast<_, uint8>
    >=> write output ".tiff"
    |> sink

    0
