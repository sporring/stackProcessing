// Binary contour extracts foreground boundaries from a UInt8 mask.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/binaryContour"
        | _ -> "../data/rotatingBoxes", "../tmp/binaryContour"

    src
    |> read<uint8> input ".tiff"
    >=> binaryContour false 5u
    >=> write output ".tiff"
    |> sink

    0
