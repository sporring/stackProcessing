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

    // Read an image and apply the binary contour algorithm to it.
    src
    |> read<uint8> input ".tiff"
    >=> binaryContour false
    >=> intensityStretch 0.0 1.0 0.0 255.0
    >=> write output ".tiff"
    |> sink

    0
