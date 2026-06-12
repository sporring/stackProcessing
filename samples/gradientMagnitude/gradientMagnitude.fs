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
    |> readChunkSlices<float32> input ".tiff"
    >=> gradientMagnitudeNativeParallelCollect 1.0 3 4
    >=> chunkIntensityWindow<float32> 0.0 255.0 0.0 255.0
    >=> chunkCast<float32, uint8>
    >=> writeChunkSlices output ".tiff"
    |> sink

    0
