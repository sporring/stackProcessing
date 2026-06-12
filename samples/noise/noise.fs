// Generate synthetic noise, smooth it, and display its image histogram.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, _ = commandLineSource availableMemory args

    let histogram =
        src
        |> zero<float32> 64u 64u 64u
        >=> addNormalNoise<float32> 0.0 50.0
        >=> gaussianFilter<float32> 3.0 9 4
        >=> cast<float32, uint8>
        >=> imageHistogramFixedBins<uint8> 0.0 255.0 256u
        >=> histogramCounts
        |> drain

    showChartWithLabels "Column" "Noise histogram" "Intensity" "Pixel count" histogram

    0
