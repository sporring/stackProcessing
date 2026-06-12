// Write the x/color component of the dominant gradient direction estimated with Chunk PCA.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/pcaGradientDirection"
        | _ -> "../data/volume", "../tmp/pcaGradientDirection"

    src
    |> readChunkSlices<float32> input ".tiff"
    >=> gradientVectorNativeParallelCollect 1.0 7 4
    >=> chunkPCAFloat32 3u
    >=> chunkSelectGroupedVectorOutput 4u 1u
    >=> chunkVector3ToColorFloat32 -1.0f 1.0f
    >=> chunkVectorElement<uint8> 0u
    >=> writeChunkSlices output ".tiff"
    |> sink

    0
