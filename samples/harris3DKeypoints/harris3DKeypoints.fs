// Detect Harris 3D keypoints and write them as a CSV point set.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let output =
        match args with
        | [| output |] -> output
        | _ -> "../tmp/harris3DKeypoints"

    src
    |> zero<float32> 64u 64u 64u
    >=> addNormalNoise<float32> 128.0 25.0
    >=> harris3DKeypoints<float32> 3.0 1.5 0.04 0.1 16u
    >=> writePointSet output ".csv"
    |> sink

    0
