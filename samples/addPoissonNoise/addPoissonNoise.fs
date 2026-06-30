// Generate a Poisson noise stack.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let output =
        match args with
        | [| output |] -> output
        | _ -> "../tmp/addPoissonNoise"

    // Generate a Poisson noise stack as double and with mean 128.0, cast to uint8, and save.
    src
    |> zero<float> 64u 64u 64u
    >=> addPoissonNoise<float> 128.0
    >=> cast<_, uint8>
    >=> write output ".tiff"
    |> sink

    0
