// Compute the binary zonohedral morphological gradient of a mask stack.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/morphologicalGradient"
        | _ -> "../data/volume", "../tmp/morphologicalGradient"

    // Compute the binary zonohedral morphological gradient of a mask stack. The morphological gradient is the difference between the dilation and erosion of a binary mask.
    src
    |> read<uint8> input ".tiff"
    >=> threshold 1.0
    >=> binaryGradient 3u
    >=> write output ".tiff"
    |> sink

    0
