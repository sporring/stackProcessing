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
    |> read<float32> input ".tiff"
    >=> sobelMagnitude ()
    >=> cast<_, uint8>
    >=> write output ".tiff"
    |> sink

    0
