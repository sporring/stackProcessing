// Streams connected objects from the synthetic object image and plots object-size histogram.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL // 2GB for example

    let src, args = commandLineSource availableMemory args

    let input, binWidth =
        match args with
        | [| input; binWidth |] -> input, uint64 binWidth
        | [| input |] -> input, 100UL
        | _ -> "../data/rotatingBoxes", 100UL

    let measured =
        src
        |> readChunkSlices<uint8> input ".tiff"
        >=> streamConnectedObjectsChunk<uint8> ObjectConnectivity.TwentySix
        >=> measureObjects

    let stats =
        measured
        >=> objectSizeStats
        |> drain

    let histogram =
        measured
        >=> objectSizes
        >=> histogram binWidth
        |> drain

    printfn "Object size stats: count=%d mean=%g variance=%g min=%d max=%d" stats.Count stats.Mean stats.Variance stats.Minimum stats.Maximum
    printfn "Histogram bins: %A" histogram
    showChartWithLabels "Column" "Streamed object-size histogram" "size bin" "object count" histogram.Counts
    0
