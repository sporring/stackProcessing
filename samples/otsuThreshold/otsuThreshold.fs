// Estimate an Otsu threshold from sampled slices, then threshold the full stack.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/otsuThreshold"
        | _ -> "../data/volume", "../tmp/otsuThreshold"

    // Estimate an Otsu threshold from sampled slices, then threshold the full stack.
    let thresholdValue =
        src
        |> histogramEstimate<uint8> 16u input ".tiff" None None None
        >=> otsuThreshold ()
        |> drain

    src
    |> read<uint8> input ".tiff"
    >=> threshold thresholdValue
    >=> intensityStretch 0.0 1.0 0.0 255.0
    >=> write output ".tiff"
    |> sink

    printfn "Estimated Otsu threshold: %.6f" thresholdValue
    0
