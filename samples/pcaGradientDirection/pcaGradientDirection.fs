// Colorize the dominant gradient direction estimated with PCA.
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
    |> read<float> input ".tiff"
    >=> gradient 1u (Some 7u)
    >=> PCA 3u
    >=> selectGroupedOutput 4u 1u
    >=> vector3ToColor -1.0 1.0
    >=> write output ".tiff"
    |> sink

    0
