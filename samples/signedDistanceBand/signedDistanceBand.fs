// Compute a band-limited signed distance map from a binary stack.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/signedDistanceBand"
        | _ -> "../data/rotatingBoxes", "../tmp/signedDistanceBand"

    src
    |> readRange<uint8> 0u 1 63u input ".tiff"
    >=> imageDivScalar<uint8> 255uy
    >=> signedDistanceBand 8u 4u
    >=> cast<float, float32>
    >=> write output ".tiff"
    |> sink

    0
