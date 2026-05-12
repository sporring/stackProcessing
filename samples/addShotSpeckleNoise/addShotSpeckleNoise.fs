// Add shot noise and then speckle noise to a zero-valued stack.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let output =
        match args with
        | [| output |] -> output
        | _ -> "../tmp/addShotSpeckleNoise"

    src
    |> zero<float> 64u 64u 16u
    >=> addShotNoise 2.0
    >=> addSpeckleNoise 0.5
    >=> cast<float, uint8>
    >=> write output ".tiff"
    |> sink

    0
