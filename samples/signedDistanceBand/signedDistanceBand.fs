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
    |> read<uint8> input ".tiff"
    >=> signedDistanceBand 8u 4u
    >=> write output ".mha"
    |> sink

    0
