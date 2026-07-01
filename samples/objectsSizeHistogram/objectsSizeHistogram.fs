// Streams connected objects from the synthetic object image and plots object-size histogram.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL // 2GB for example

    let src, args = commandLineSource availableMemory args

    let input =
        match args with
        | [| input |] -> input
        | _ -> "../data/objects"

    // Streams connected objects from the synthetic object image and calculate object related statistics in two different ways.
    let stats, sizeHistogram =
        src
        |> read<uint8> input ".tiff"
        >=> streamConnectedObjects ObjectConnectivity.TwentySix
        >=> objectSizes
        >=>> (stats (), histogram ())
        |> drain

    printfn "Object size stats: count=%d mean=%g variance=%g min=%d max=%d" stats.Count stats.Mean stats.Variance stats.Minimum stats.Maximum
    printfn "Histogram bins: %A" sizeHistogram

    0
