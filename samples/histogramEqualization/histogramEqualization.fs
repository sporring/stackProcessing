// Equalize stack intensities from a histogram estimated from random slices.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/histogramEqualization"
        | _ -> "../data/volume", "../tmp/histogramEqualization"

    let histogram =
        src
        |> readRandom<uint8> 8u input ".tiff"
        >=> imHistogram ()
        |> drain

    src
    |> read<uint8> input ".tiff"
    >=> histogramEqualization histogram
    >=> cast<float, uint8>
    >=> write output ".tiff"
    |> sink

    0
