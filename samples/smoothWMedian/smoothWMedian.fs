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
    |> read<float32> input ".tiff"
    >=> medianFloat32 1
    >=> cast<_, uint8>
    >=> write output ".tiff"
    |> sink

    0
