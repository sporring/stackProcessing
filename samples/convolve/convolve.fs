// Custom convolution with a small normalized 3D averaging kernel.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/convolve"
        | _ -> "../data/volume", "../tmp/convolve"

    src
    |> readChunkSlices<float32> input ".tiff"
    >=> boxFilterNativeParallelCollectXYZ<float32> 1 1 1 4
    >=> chunkIntensityWindow<float32> 0.0 255.0 0.0 255.0
    >=> chunkCast<float32, uint8>
    >=> writeChunkSlices output ".tiff"
    |> sink

    0
