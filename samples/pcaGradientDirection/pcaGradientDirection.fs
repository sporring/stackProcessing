// Write the dominant gradient direction estimated with Chunk PCA as RGB color slices.
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
    |> read<float32> input ".tiff"
    >=> gradientVector 1.0 7 4
    >=> pcaFloat32 3u
    >=> selectGroupedVectorOutput 4u 1u
    >=> vector3ToColorFloat32 -1.0f 1.0f
    >=> writeColor output ".tiff"
    |> sink

    0
