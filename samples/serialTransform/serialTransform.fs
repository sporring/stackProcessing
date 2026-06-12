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
        |> readChunkSlicesRange<float32> 0u 16 64u input ".tiff"
        >=> chunkSerialEstTrans<float32> 8 "dogAffine" 1.6 0.1
        >=> chunkSerialEstBoundingBox<float32>
        |> drain

    src
    |> readChunkSlices<float32> input ".tiff"
    >=> chunkSerialEstTrans<float32> 8 "dogAffine" 1.6 0.1
    >=> chunkSerialApplyTrans<float32> 0.0f (Some geometry)
    >=> chunkCast<float32, uint8>
    >=> writeChunkSlices output ".tiff"
    |> sink

    0
