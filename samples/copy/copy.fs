// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net10.0
open StackProcessing

[<EntryPoint>]
let main arg =
    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 2GB for example

    let src, arg = commandLineSource availableMemory arg
    let input,output = 
        if arg.Length > 0 then
            "../data/volume", "../tmp/copy"
        else
            "../data/volume", "../tmp/copy"

    // Demonstrating the basic scaffold of a pipeline: copy an image.
    src
    |> read<uint8> input ".tiff"
    >=> write output ".tiff"
    |> sink

    0
