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

    src
    |> readRandom<uint8> 1u input ".tiff"
    >=> show (fun chunk -> showChunkWithLabels<uint8> "Viridis" "An Image" "" "" chunk)
    |> sink

    0
