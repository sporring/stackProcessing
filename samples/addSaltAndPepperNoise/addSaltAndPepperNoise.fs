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

    // generate an image of zeros and add salt and peper noise to it.
    src
    |> zero<uint8> 64u 64u 64u
    >=> addSaltAndPepperNoise<uint8> 0.02 None None
    >=> write output ".tiff"
    |> sink

    0
