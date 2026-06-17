// Generate a salt-and-pepper noise stack.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let output =
        match args with
        | [| output |] -> output
        | _ -> "../tmp/saltAndPepperNoise"

    src
    |> zero<uint8> 64u 64u 64u
    >=> addSaltAndPepperNoise<uint8> 0.02
    >=> write output ".tiff"
    |> sink

    0
