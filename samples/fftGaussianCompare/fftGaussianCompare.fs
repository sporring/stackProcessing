// Compare a direct Gaussian smoothing result with an FFT round-trip result.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, smoothedOutput, roundTripOutput =
        match args with
        | [| input; smoothedOutput; roundTripOutput |] -> input, smoothedOutput, roundTripOutput
        | [| input |] -> input, "../tmp/fftGaussianSmooth", "../tmp/fftRoundTrip"
        | _ -> "../data/volume", "../tmp/fftGaussianSmooth", "../tmp/fftRoundTrip"

    src
    |> read<float> input ".tiff"
    >=> smoothWGauss 1.25 None None (Some 16u)
    >=> write smoothedOutput ".tiff"
    |> sink

    src
    |> read<float> input ".tiff"
    >=> FFT<float> 32u 32u 16u
    >=> invFFT 32u 32u 16u
    >=> write roundTripOutput ".tiff"
    |> sink

    0
