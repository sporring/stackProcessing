// Detect edges with the Sobel operator.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input, output =
        match args with
        | [| input; output |] -> input, output
        | [| input |] -> input, "../tmp/sobelEdge"
        | _ -> "../data/volume", "../tmp/sobelEdge"

    src
    |> read<float> input ".tiff"
    >=> sobelEdge<float> 5u
    >=> cast<float, uint8>
    >=> write output ".tiff"
    |> sink

    0
