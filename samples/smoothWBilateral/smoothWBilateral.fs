// Smooth while preserving edges with a bilateral filter.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/smoothWBilateral"
        | _ -> "../data/volume", "../tmp/smoothWBilateral"

    src
    |> read<float> input ".tiff"
    >=> smoothWBilateral<float> 1.5 30.0 5u
    >=> cast<float, uint8>
    >=> write output ".tiff"
    |> sink

    0
