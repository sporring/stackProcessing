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
        | [| input |] -> input, "../data/rotatingBoxes", "../tmp/biasCorrection"
        | _ -> "../data/volume", "../data/rotatingBoxes", "../tmp/biasCorrection"

    let model =
        src
        |> read<float> input ".tiff"
        >=> fitBiasModel<float> 2 256u
        |> drain

    src
    |> read<float> input ".tiff"
    >=> correctBias<float> model
    >=> intensityStretch<float> 0.0 255.0 0.0 255.0
    >=> cast<float, uint8>
    >=> write (outputRoot + "/unmasked") ".tiff"
    |> sink

    let image = src |> read<float> input ".tiff"
    let maskStream = src |> read<uint8> mask ".tiff"
    let maskedModel =
        zip image maskStream
        >=> fitBiasModelMasked<float> 2 256u
        |> drain

    zip image maskStream
    >=> correctBiasMasked<float> maskedModel
    >=> intensityStretch<float> 0.0 255.0 0.0 255.0
    >=> cast<float, uint8>
    >=> write (outputRoot + "/masked") ".tiff"
    |> sink

    0
