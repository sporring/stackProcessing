// Estimate a coarse output envelope, then estimate and apply slice-wise affine transforms.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/serialTransform"
        | _ -> "../data/volume", "../tmp/serialTransform"

    let geometry =
        src
        |> readRange<float32> 0u 16 64u input ".tiff"
        >=> serialEstTrans 8 "dogAffine" 1.6 0.1
        >=> serialEstBoundingBox
        |> drain

    src
    |> read<float32> input ".tiff"
    >=> serialEstTrans 8 "dogAffine" 1.6 0.1
    >=> serialApplyTrans 0.0 (Some geometry)
    >=> cast<_, uint8>
    >=> write output ".tiff"
    |> sink

    0
