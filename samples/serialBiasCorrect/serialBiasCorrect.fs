// Correct slowly varying slice-wise intensity bias in an image stack.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/serialBiasCorrect"
        | _ -> "../data/volume", "../tmp/serialBiasCorrect"

    src
    |> readChunkSlices<float32> input ".tiff"
    >=> chunkSerialPolynomialBiasCorrect<float32> 2
    >=> writeChunkSlices output ".tiff"
    |> sink

    0
