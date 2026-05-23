// Remove impulse-like noise with a windowed median filter.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/smoothWMedian"
        | _ -> "../data/volume", "../tmp/smoothWMedian"

    src
    |> read<float> input ".tiff"
    >=> smoothWMedian<float> 1u None
    >=> cast<float, uint8>
    >=> write output ".tiff"
    |> sink

    0
