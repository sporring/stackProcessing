// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net10.0
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

    // Estimate a sparse histogram and plot it.
    src
    |> read<uint8> input ".tiff"
    >=> imageHistogram ()
    >=> plotHistogramWithLabels "Histogram" "gray values" "count"
    |> sink

    0
