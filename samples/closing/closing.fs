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

    src
    |> read<uint8> input ".tiff"
    >=> binaryClosing 3u
    >=> write output ".tiff"
    |> sink

    0
