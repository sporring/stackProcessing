// Write the dominant gradient direction estimated from the gradient covariance matrix as RGB color slices.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/pcaGradientDirection"
        | _ -> "../data/volume", "../tmp/pcaGradientDirection"

    let covariance =
        src
        |> read<float32> input ".tiff"
        >=> gradientVector 1.0 7
        >=> covarianceMatrix
        |> drain

    let eigenbasis = symmetricMatrixEigenbasis covariance

    src
    |> read<float32> input ".tiff"
    >=> gradientVector 1.0 7
    >=> projectVectorBasis eigenbasis
    >=> intensityStretch -1.0 1.0 0.0 255.0
    >=> vectorCast<_, uint8>
    >=> writeColor output ".tiff"
    |> sink

    0
