// Binary opening removes small foreground details from a UInt8 mask.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/opening"
        | _ -> "../data/rotatingBoxes", "../tmp/opening"

    // Binary opening removes small foreground details from a UInt8 mask.
    src
    |> read<uint8> input ".tiff"
    >=> binaryOpening 2u
    >=> intensityStretch 0.0 1.0 0.0 255.0
    >=> write output ".tiff"
    |> sink

    0
