// Detect several kinds of 3D keypoints and write them as CSV point sets.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let outputRoot =
        match args with
        | [| outputRoot |] -> outputRoot
        | _ -> "../tmp/keypoints"

    src
    |> normalNoise<float> 32u 32u 16u 128.0 25.0
    >=> siftKeypoints<float> 1.0 1.6 4u 0.1 16u
    >=> writePointSet (outputRoot + "/sift") ".csv"
    |> sink

    src
    |> normalNoise<float> 32u 32u 16u 128.0 25.0
    >=> hessianKeypoints<float> 1.0 "Blob" 0.1 16u
    >=> writePointSet (outputRoot + "/hessian") ".csv"
    |> sink

    src
    |> normalNoise<float> 32u 32u 16u 128.0 25.0
    >=> harris3DKeypoints<float> 1.0 1.5 0.04 0.1 16u
    >=> writePointSet (outputRoot + "/harris") ".csv"
    |> sink

    src
    |> normalNoise<float> 32u 32u 16u 128.0 25.0
    >=> forstner3DKeypoints<float> 1.0 1.5 0.1 16u
    >=> writePointSet (outputRoot + "/forstner") ".csv"
    |> sink

    0
