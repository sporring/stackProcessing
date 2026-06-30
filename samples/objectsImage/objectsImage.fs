// Repaint connected objects from a binary stack into an object-label image.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/objectsImage"
        | _ -> "../data/rotatingBoxes", "../tmp/objectsImage"

    src
    |> read<uint8> input ".tiff"
    >=> streamConnectedObjects ObjectConnectivity.TwentySix
    >=> paintObjects 512u 384u
    >=> write output ".tiff"
    |> sink

    0
