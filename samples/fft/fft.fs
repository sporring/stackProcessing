// Run an FFT round-trip on a small synthetic volume.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let output =
        match args with
        | [| output |] -> output
        | _ -> "../tmp/fft"

    src
    |> normalNoise<float> 64u 64u 64u 128.0 25.0
    >=> FFT<float> 16u 16u 8u
    >=> invFFT 16u 16u 8u
    >=> cast<float,float32>
    >=> write output ".tiff"
    |> sink

    0
