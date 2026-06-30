// Binary dilation grows foreground regions in a UInt8 mask.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/dilate"
        | _ -> "../data/rotatingBoxes", "../tmp/dilate"

    // Dilate a binary image with a sphere of radius 2. Stretch intensities for easy viewing.
    src
    |> read<uint8> input ".tiff"
    >=> binaryDilate 2u
    >=> intensityStretch 0.0 1.0 0.0 255.0
    >=> write output ".tiff"
    |> sink

    0
