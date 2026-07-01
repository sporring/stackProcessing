// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net10.0
open StackProcessing

[<EntryPoint>]
let main arg =
    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 2GB for example

    let src, arg = commandLineSource availableMemory arg
    let input,output = 
        if arg.Length > 0 then
            "../data/volume","../tmp/normalize"
        else
            "../data/volume","../tmp/normalize"

    // A pipeline implementation of imageStretch.
    let stat =
        src
        |> readRandom<float32> 12u input ".tiff"
        >=> computeStats ()
        |> drain

    src
    |> read<float32> input ".tiff"
    >=> subScalar stat.Min
    >=> mulScalar (255.0 / (stat.Max - stat.Min))
    >=> cast<_, uint8>
    >=> write output ".tiff"
    |> sink

    0
