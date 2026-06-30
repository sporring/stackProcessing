// Run an FFT round-trip on a small synthetic volume.
open System
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let output =
        match args with
        | [| output |] -> output
        | _ -> "../tmp/fft"

    // Fourier transform a random image, then shift the zero frequency to the center and save.
    src
    |> zero<float32> 64u 64u 64u
    >=> addNormalNoise<float32> 128.0 25.0
    >=> fft
    >=> fftShift3D
    >=> length
    >=> addScalar 1.0
    >=> log<float32>
    >=> intensityStretch 0.0 (Math.Log(128.0 * 64.0 * 64.0 * 64.0 + 1.0)) 0.0 255.0
    >=> cast<_,uint8>
    >=> write output ".tiff"
    |> sink

    0
