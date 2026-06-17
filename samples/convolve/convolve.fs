// Custom convolution with a small normalized 3D averaging kernel.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/convolve"
        | _ -> "../data/volume", "../tmp/convolve"

    src
    |> read<float32> input ".tiff"
    >=> boxFilterXYZ<float32> 1 1 1 4
    >=> intensityWindow<float32> 0.0 255.0 0.0 255.0
    >=> cast<float32, uint8>
    >=> write output ".tiff"
    |> sink

    0
