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
    |> chunkZero<float> 64u 64u 64u
    >=> chunkAddShotNoise<float> 2.0
    >=> chunkCast<float, uint8>
    >=> writeChunkSlices output ".tiff"
    |> sink

    0
