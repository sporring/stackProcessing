// Generate a shot noise stack.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let output =
        match args with
        | [| output |] -> output
        | _ -> "../tmp/shotNoise"

    src
    |> shotNoise<float> 64u 64u 16u 2.0
    >=> cast<float, uint8>
    >=> write output ".tiff"
    |> sink

    0
