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

    // Estiamte a dense histogram
    let histogram =
        src
        |> readRandom<uint8> 16u input ".tiff"
        >=> imageHistogramDense ()
        |> drain
    // plot the histogram before equalization
    showHistogramWithLabels "Before equalization" "gray values" "count" histogram

    // Equalize the stack using the estimated histogram and plot the histogram after equalization
    src
    |> read<uint8> input ".tiff"
    >=> histogramEqualization histogram
    >=> imageHistogram ()
    >=> plotHistogramWithLabels "Histogram" "gray values" "count"
    |> sink

    0
