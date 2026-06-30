// Run an FFT round-trip on a small synthetic volume.
open System
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/fft"
        | _ -> "../data/rotatingBoxes", "../tmp/fft"

    // Runs a separable uniform filter on an input image

    // Fourier transform a random image, then shift the zero frequency to the center and save.

    src
    |> read<float32> input ".tiff"
    >=> fft
    >=> fftShift3D
    >=> length
    >=> addScalar 1.0
    >=> log
    >=> intensityStretch 0.0 (Math.Log(255.0 * 3.0 * 12.0 * 12.0 * 12.0 + 1.0)) 0.0 255.0
    >=> cast<_,uint8>
    >=> write output ".tiff"
    |> sink

    0
