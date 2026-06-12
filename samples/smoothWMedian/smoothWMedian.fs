// Remove impulse-like noise with a windowed median filter.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/smoothWMedian"
        | _ -> "../data/volume", "../tmp/smoothWMedian"

    src
    |> readChunkSlices<float32> input ".tiff"
    >=> chunkMedianNativeNthElementFloat32ParallelCollect 1 4
    >=> chunkCast<float32, uint8>
    >=> writeChunkSlices output ".tiff"
    |> sink

    0
