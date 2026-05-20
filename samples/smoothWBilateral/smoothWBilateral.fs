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
    |> readRange<float> 0u 1 31u input ".tiff"
    >=> smoothWBilateral<float> 1.5 30.0 5u
    >=> intensityStretch<float> 0.0 255.0 0.0 255.0
    >=> cast<float, uint8>
    >=> write output ".tiff"
    |> sink

    0
