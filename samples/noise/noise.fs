// Generate synthetic noise, smooth it, and display its image histogram.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, _ = commandLineSource availableMemory args

    let histogram =
        src
        |> normalNoise<float> 256u 256u 256u 0.0 50.0
        >=> smoothWGauss 1.0 (Some ImageFunctions.Valid) None None
        >=> cast<float, uint8>
        >=> imHistogramFixedBins 0.0 255.0 256u
        >=> histogramCounts
        |> drain

    showChartWithLabels "Column" "Noise histogram" "Intensity" "Pixel count" histogram

    0
