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
    |> chunkZero<float32> 64u 64u 64u
    >=> chunkAddNormalNoise<float32> 128.0 25.0
    |> chunkResample<float32> 1.5 1.5 1.5 "Linear"
    >=> chunkCast<float32, uint8>
    >=> writeChunkSlices output ".tiff"
    |> sink

    0
