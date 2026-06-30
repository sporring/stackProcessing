// Generate synthetic noise, smooth it, and display its image histogram.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let output =
        match args with
        | [| output |] -> output
        | _ -> "../tmp/randomGaussianField"

    src
    |> zero<float32> 64u 64u 64u
    >=> addNormalNoise<float32> 128.0 50.0
    >=> gaussianFilter<float32> 3.0 None // kernel size is default, here ceil (2.0*3.0+1.0) = 7.0
    >=> cast<_, uint8>
    >=> write output ".tiff"
    |> sink

    0
