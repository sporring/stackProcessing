// Fit a simple polynomial bias model and apply it with and without a mask.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, mask, outputRoot =
        match args with
        | [| input; mask; outputRoot |] -> input, mask, outputRoot
        | [| input; mask |] -> input, mask, "../tmp/biasCorrection"
        | [| input |] -> input, input, "../tmp/biasCorrection"
        | _ -> "../data/rotatingBoxes", "../data/rotatingBoxes", "../tmp/biasCorrection"

    // Generate random indices from the input image.
    let sampledZ = src |> sliceIndicesRandom 10u input ".tiff"

    // read the sampled slices and fit a 2nd degree polynomial bias model to them.
    let model =
        sampledZ
        >=> readAtIndices<float32> input ".tiff"
        >=> fitBiasModel 2
        |> drain

    // Apply the bias model to the full image and write the result.
    src
    |> readRange<float32> 0u 1 255u input ".tiff"
    >=> correctBias model
    >=> cast<_, uint8>
    >=> write (outputRoot + "/unmasked") ".tiff"
    |> sink

    // read the sample slices together with a mask image and fit a polyomial model to the image.
    let maskedModel =
        sampledZ
        >=>> (readAtIndices<float32> input ".tiff", readAtIndices<uint8> mask ".tiff")
        >>=> fitBiasModelMasked 2
        |> drain

    // Apply the bias model to the masked pixel values of the entire image.
    let fullImage =
        src |> read<float32> input ".tiff"
    let fullMask =
        src |> read<uint8> mask ".tiff"

    zip fullImage fullMask
    >=> correctBiasMasked maskedModel
    >=> cast<_, uint8>
    >=> write (outputRoot + "/masked") ".tiff"
    |> sink

    0
