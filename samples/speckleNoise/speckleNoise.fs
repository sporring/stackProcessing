// Generate a speckle noise stack.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let output =
        match args with
        | [| output |] -> output
        | _ -> "../tmp/speckleNoise"

    src
    |> speckleNoise<float> 64u 64u 16u 0.5
    >=> cast<float, uint8>
    >=> write output ".tiff"
    |> sink

    0
