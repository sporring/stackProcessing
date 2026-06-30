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
        | _ -> "../data/rotatingBoxes", "../tmp/gradientMagnitude"

    // calculate the gradient magnitude squared, smoothing it with a normal distribution of std 1.0
    // It's calculated twice, first to get the statistics. For larger than memory imageprocessing,
    // sweeping twice should probably be avoided.
    let stats = 
        src
        |> read<float32> input ".tiff"
        >=> gradientMagnitudeSquared 1.0 (Some 7u)
        >=> computeStats ()
        |> drain

    src
    |> read<float32> input ".tiff"
    >=> gradientMagnitudeSquared 1.0 (Some 7u)
    >=> intensityStretch stats.Min stats.Max 0.0 255.0
    >=> cast<_, uint8>
    >=> write output ".tiff"
    |> sink

    0
