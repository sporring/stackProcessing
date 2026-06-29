// Binary closing fills small background gaps in a 0/1 UInt8 mask.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/closing"
        | _ -> "../data/rotatingBoxes", "../tmp/closing"

    // Read the image an perform mathematical morphology closing on it, then write the result to disk.
    src
    |> read<uint8> input ".tiff"
    >=> binaryClosing 3u
    >=> intensityStretch 0.0 1.0 0.0 255.0
    >=> write output ".tiff"
    |> sink

    0
