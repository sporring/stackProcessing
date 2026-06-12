// Detect edges with the Sobel operator.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/sobelEdge"
        | _ -> "../data/volume", "../tmp/sobelEdge"

    src
    |> readChunkSlices<float32> input ".tiff"
    >=> sobelMagnitudeNativeParallelCollect 4
    >=> chunkCast<float32, uint8>
    >=> writeChunkSlices output ".tiff"
    |> sink

    0
