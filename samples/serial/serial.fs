// Estimate a volume histogram from a random subset of slices.
open StackProcessing

[<EntryPoint>]
let main args =
    let availableMemory = 2UL * 1024UL * 1024UL * 1024UL
    let src, args = commandLineSource availableMemory args

    let input =
        match args with
        | [| input |] -> input
        | _ -> "../data/volume"

    let histogram =
        src
        |> readChunkSlicesRandom<uint8> 16u input ".tiff"
        >=> chunkHistogram<uint8> ()
        |> drain

    printfn "Sampled histogram bins: %d" histogram.Counts.Count
    showChartWithLabels "Column" "Sampled volume histogram" "intensity" "pixel count" histogram.Counts

    0
