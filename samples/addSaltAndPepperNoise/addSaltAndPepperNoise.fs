// Add salt-and-pepper noise to a zero-valued stack.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let output =
        match args with
        | [| output |] -> output
        | _ -> "../tmp/addSaltAndPepperNoise"

    src
    |> chunkZero<uint8> 64u 64u 64u
    >=> chunkAddSaltAndPepperNoise<uint8> 0.02
    >=> writeChunkSlices output ".tiff"
    |> sink

    0
