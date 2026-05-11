// Estimate and apply slice-wise affine transforms in one streaming pipeline.
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

    src
    |> read<float32> input ".tiff"
    >=> serialEstTrans<float32> 8 "dogAffine" 1.6 0.1
    >=> serialApplyTrans<float32> 0.0 None
    >=> cast<float32, uint8>
    >=> write output ".tiff"
    |> sink

    0
