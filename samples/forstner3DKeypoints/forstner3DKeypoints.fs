// Detect Forstner 3D keypoints and write them as a CSV point set.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let output =
        match args with
        | [| output |] -> output
        | _ -> "../tmp/forstner3DKeypoints"

    src
    |> chunkZero<float32> 64u 64u 64u
    >=> chunkAddNormalNoise<float32> 128.0 25.0
    >=> chunkForstner3DKeypoints<float32> 3.0 1.5 0.1 16u
    >=> writePointSet output ".csv"
    |> sink

    0
