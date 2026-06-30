// Detect Hessian keypoints and write them as a CSV point set.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let output =
        match args with
        | [| output |] -> output
        | _ -> "../tmp/hessianKeypoints"

    src
    |> zero<float32> 64u 64u 64u
    >=> addNormalNoise 128.0 25.0
    >=> hessianKeypoints 3.0 "Blob" 0.1 16u
    >=> writePointSet output ".csv"
    |> sink

    0
