open StackProcessing

[<EntryPoint>]
let main arg =
    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 2GB for example

    let src, arg = commandLineSource availableMemory arg
    let input = 
        if arg.Length > 0 then
            "../data/volume"
        else
            "../data/volume"

    // Demonstrate the >=>> operator, which fans out the measured histogram
    src
    |> read<uint8> input ".tiff"
    >=> imageHistogram ()
    >=>> (print (), plotHistogramWithLabels "Histogram" "gray values" "count")
    |> sink

    0
