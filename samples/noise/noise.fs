// Generate synthetic noise, smooth it, and display its image histogram.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, _ = commandLineSource availableMemory args

    let histogram =
        src
        |> chunkZero<float32> 64u 64u 64u
        >=> chunkAddNormalNoise<float32> 0.0 50.0
        >=> gaussianFilterNativeParallelCollect<float32> 3.0 9 4
        >=> chunkCast<float32, uint8>
        >=> chunkHistogramFixedBins<uint8> 0.0 255.0 256u
        >=> histogramCounts
        |> drain

    showChartWithLabels "Column" "Noise histogram" "Intensity" "Pixel count" histogram

    0
