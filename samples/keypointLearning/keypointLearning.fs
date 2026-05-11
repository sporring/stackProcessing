// Detect several kinds of 3D keypoints and write them as CSV point sets.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, outputRoot =
        match args with
        | [| input; outputRoot |] -> input, outputRoot
        | [| input |] -> input, "../tmp/keypoints"
        | _ -> "../data/volume", "../tmp/keypoints"

    src
    |> read<float> input ".tiff"
    >=> siftKeypoints<float> 1.0 1.6 4u 0.03 8u
    >=> writePointSet (outputRoot + "/sift") ".csv"
    |> sink

    src
    |> read<float> input ".tiff"
    >=> hessianKeypoints<float> 1.0 "Blob" 0.03 8u
    >=> writePointSet (outputRoot + "/hessian") ".csv"
    |> sink

    src
    |> read<float> input ".tiff"
    >=> harris3DKeypoints<float> 1.0 1.5 0.04 0.03 8u
    >=> writePointSet (outputRoot + "/harris") ".csv"
    |> sink

    src
    |> read<float> input ".tiff"
    >=> forstner3DKeypoints<float> 1.0 1.5 0.03 8u
    >=> writePointSet (outputRoot + "/forstner") ".csv"
    |> sink

    0
