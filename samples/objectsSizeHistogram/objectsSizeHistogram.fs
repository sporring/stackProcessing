// Streams connected objects from the synthetic object image and plots object-size histogram.
open StackProcessing
open Plotly.NET

let private plotHistogram (histogram: Map<uint64, uint64>) =
    let bins =
        histogram
        |> Map.toList
        |> List.map (fun (bin, count) -> float bin, float count)

    let keys = bins |> List.map fst
    let values = bins |> List.map snd

    Chart.Column(values = values, Keys = keys)
    |> Chart.withTitle "Streamed object-size histogram"
    |> Chart.withXAxisStyle "size bin"
    |> Chart.withYAxisStyle "object count"
    |> Chart.show

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL

    let src, args = commandLineSource availableMemory args

    let input, binWidth =
        match args with
        | [| input; binWidth |] -> input, uint64 binWidth
        | [| input |] -> input, 100UL
        | _ -> "../objectsImage", 100UL

    let measured =
        src
        |> read<uint8> input ".tiff"
        >=> streamConnectedObjects<uint8> ObjectConnectivity.TwentySix
        >=> measureObjects

    let stats =
        measured
        >=> objectSizeStats
        |> drain

    let histogram =
        measured
        >=> objectSizeHistogram binWidth
        |> drain

    printfn "Object size stats: count=%d mean=%g variance=%g min=%d max=%d" stats.Count stats.Mean stats.Variance stats.Minimum stats.Maximum
    printfn "Histogram bins: %A" histogram

    plotHistogram histogram
    0
