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

    let model =
        src
        |> readRange<float32> 0u 1 255u input ".tiff"
        >=> fitBiasModel<float32> 2 256u
        |> drain

    src
    |> readRange<float32> 0u 1 255u input ".tiff"
    >=> correctBias<float32> model
    >=> cast<_, uint8>
    >=> write (outputRoot + "/unmasked") ".tiff"
    |> sink

    let image = src |> readRange<float32> 0u 1 255u input ".tiff"
    let maskStream = src |> read<uint8> mask ".tiff"
    let maskedModel =
        zip image maskStream
        >=> fitBiasModelMasked<float32> 2 256u
        |> drain

    zip image maskStream
    >=> correctBiasMasked<float32> maskedModel
    >=> cast<_, uint8>
    >=> write (outputRoot + "/masked") ".tiff"
    |> sink

    0
