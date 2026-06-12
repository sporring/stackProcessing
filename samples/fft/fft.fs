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
    |> zero<float32> 8u 8u 4u
    >=> addNormalNoise<float32> 128.0 25.0
    >=> fft
    >=> fftShift3D
    >=> fftShift3D
    >=> invFft
    >=> write output ".tiff"
    |> sink

    0
