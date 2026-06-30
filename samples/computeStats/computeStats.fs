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

    // Read full image statistics and estimate the same from a few random images. Result is drained for printing.
    let stats = 
        src
        |> read<uint8> input ".tiff"
        >=> computeStats ()
        |> drain

    let partialStats = 
        src
        |> readRandom<uint8> 5u input ".tiff"
        >=> computeStats ()
        |> drain

    printfn "Full image statistics: %A" stats
    printfn "Partial image statistics: %A" partialStats

    0
