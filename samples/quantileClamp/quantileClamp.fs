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
        |> readChunkSlicesRandom<uint8> 24u input ".tiff"
        >=> chunkHistogramFixedBins<uint8> 0.0 255.0 256u
        |> drain

    let limits = chunkQuantiles [0.01; 0.99] (histogram :> obj)

    src
    |> readChunkSlices<float32> input ".tiff"
    >=> chunkIntensityWindow<float32> limits[0] limits[1] 0.0 255.0
    >=> chunkClamp<float32> 0.0 255.0
    >=> chunkCast<float32, uint8>
    >=> writeChunkSlices output ".tiff"
    |> sink

    0
