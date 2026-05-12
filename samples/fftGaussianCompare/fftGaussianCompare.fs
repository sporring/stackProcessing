// Compare a direct Gaussian smoothing result with an FFT round-trip result on a
// small synthetic volume.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let smoothedOutput, roundTripOutput =
        match args with
        | [| smoothedOutput; roundTripOutput |] -> smoothedOutput, roundTripOutput
        | _ -> "../tmp/fftGaussianSmooth", "../tmp/fftRoundTrip"

    src
    |> normalNoise<float> 32u 32u 16u 128.0 25.0
    >=> smoothWGauss 1.25 None None (Some 8u)
    >=> cast<float,float32>
    >=> write smoothedOutput ".tiff"
    |> sink

    src
    |> normalNoise<float> 32u 32u 16u 128.0 25.0
    >=> FFT<float> 16u 16u 8u
    >=> invFFT 16u 16u 8u
    >=> cast<float,float32>
    >=> write roundTripOutput ".tiff"
    |> sink

    0
