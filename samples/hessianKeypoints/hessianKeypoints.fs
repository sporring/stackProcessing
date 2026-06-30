// Detect Hessian keypoints and write them as a CSV point set.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output = 
        if args.Length > 0 then
            "../data/objects", "../tmp/hessianKeypoints"
        else
            "../data/objects", "../tmp/hessianKeypoints"

    // read an image and calculate its keypoints
    src
    |> read<uint8> input ".tiff"
    >=> hessianKeypoints 3.0 "Blob" 0.1 16u
    >=> writePointSet output ".csv"
    |> sink

    0
