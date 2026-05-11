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

    let kernel = Image<float>.ofArray3D(Array3D.create 3 3 3 (1.0 / 27.0), "average3x3x3")

    src
    |> read<float> input ".tiff"
    >=> convolve kernel (Some ImageFunctions.Same) (Some ImageFunctions.ZeroFluxNeumannPad) (Some 8u)
    >=> intensityStretch<float> 0.0 255.0 0.0 255.0
    >=> cast<float, uint8>
    >=> write output ".tiff"
    |> sink

    0
