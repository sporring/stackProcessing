// Estimate robust intensity limits from a histogram, then stretch/clamp outliers for inspection.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/quantileClamp"
        | _ -> "../data/volume", "../tmp/quantileClamp"

    let histogram =
        src
        |> readRandom<uint8> 24u input ".tiff"
        >=> imageHistogramFixedBins<uint8> 0.0 255.0 256u
        |> drain

    let limits = quantiles [0.01; 0.99] (histogram :> obj)

    src
    |> read<float32> input ".tiff"
    >=> intensityStretch limits[0] limits[1] 0.0 255.0
    >=> clamp<float32> 0.0 255.0
    >=> cast<_, uint8>
    >=> write output ".tiff"
    |> sink

    0
