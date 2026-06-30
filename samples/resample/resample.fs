// Resample a synthetic stack by scale factors.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let output =
        match args with
        | [| output |] -> output
        | _ -> "../tmp/resample"

    src
    |> zero<float32> 64u 64u 64u
    >=> addNormalNoise 128.0 25.0
    |> resample 1.5 1.5 1.5 "Linear"
    >=> cast<_, uint8>
    >=> write output ".tiff"
    |> sink

    0
