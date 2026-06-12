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
        |> readChunkSlicesRange<float32> 0u 1 255u input ".tiff"
        >=> fitBiasModelChunk<float32> 2 256u
        |> drain

    src
    |> readChunkSlicesRange<float32> 0u 1 255u input ".tiff"
    >=> correctBiasChunk<float32> model
    >=> chunkCast<float, uint8>
    >=> writeChunkSlices (outputRoot + "/unmasked") ".tiff"
    |> sink

    let image = src |> readChunkSlicesRange<float32> 0u 1 255u input ".tiff"
    let maskStream = src |> readChunkSlices<uint8> mask ".tiff"
    let maskedModel =
        zip image maskStream
        >=> fitBiasModelChunkMasked<float32> 2 256u
        |> drain

    zip image maskStream
    >=> correctBiasChunkMasked<float32> maskedModel
    >=> chunkCast<float, uint8>
    >=> writeChunkSlices (outputRoot + "/masked") ".tiff"
    |> sink

    0
